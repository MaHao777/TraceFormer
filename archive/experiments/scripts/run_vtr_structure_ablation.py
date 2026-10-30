"""冻结缓存上的轨迹结构干预；验证集选单阈值，测试集只复评。"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from traceformer.postprocessing import tracking
from traceformer.postprocessing.contracts import EventCache, density_group, sha256_file
from traceformer.postprocessing.direct_verifier import event_branches
from traceformer.postprocessing.learning import blend_scores
from traceformer.postprocessing.metrics import BinaryCounts, score_binary
from traceformer.postprocessing.submission import load_prediction, write_prediction_npz
from traceformer.primary import load_primary_manifest

REPO = Path(__file__).resolve().parents[3]
CACHE_ROOT = (REPO / "outputs/server_results_20261018/complete/"
              "traceformer-speed-audit-20261018_222449")
IDENTITY = (REPO / "outputs/server_results_20261018/complete/"
            "traceformer-prefix-14t-20261018_214457/dataset_identity.json")
RELEASE = REPO / "outputs/main_model/event_release/release_report.json"
GRID = (0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.65, 0.7, 0.75, 0.85,
        0.9, 0.94, 0.96, 0.98, 0.99, 0.995)
VARIANTS = ("neural", "direct", "tracks", "full", "no_extension", "no_duration",
            "permuted", "frame_local")
FAMILIES = ("track", "direct", "direct_or_track")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def permute_frames(locations, seed=37):
    """只置换完整50ms帧编号，保持帧内偏移、空间、事件行与概率的对应。"""
    result = np.asarray(locations, dtype=np.int64).copy()
    frames = result[:, 2] // 50
    first, last = int(frames.min()), int(frames.max())
    mapping = np.random.default_rng(seed).permutation(last - first + 1) + first
    result[:, 2] = mapping[frames - first] * 50 + result[:, 2] % 50
    return result, mapping


def frame_local_prediction(locations, probability, settings):
    """保留热点/空间/回填，只把每个节点作为独立单帧轨迹。"""
    settings = {**settings, "minimum_track_frames": 1, "extension_frames": 0}

    def singletons(nodes, *args):
        return [[node] for frame in sorted(nodes) for node in nodes[frame]]

    # 临时替换仅作用于本次串行实验，退出时自动恢复生产函数。
    with patch.object(tracking, "_link_tracks", singletons):
        return tracking.track_prediction(locations, probability, **settings)


def control_masks(cache, scores, contract):
    """共享冻结的两套融合分数，分别干预轨迹而不重训任何模型。"""
    started = time.perf_counter()
    track_score, direct_score, tracked, direct = event_branches(cache, scores, contract)
    settings = contract["tracking"][density_group(cache.locations)[0]]
    masks = {"neural": cache.prediction, "direct": direct, "tracks": tracked,
             "full": direct | tracked}
    times = {name: 0.0 for name in masks}
    times["full"] = time.perf_counter() - started
    for name, overrides in (("no_extension", {"extension_frames": 0}),
                            ("no_duration", {"minimum_track_frames": 1})):
        started = time.perf_counter()
        mask = tracking.track_prediction(cache.locations, track_score, **{**settings, **overrides})
        masks[name] = direct | mask
        times[name] = time.perf_counter() - started
    started = time.perf_counter()
    permuted, mapping = permute_frames(cache.locations)
    masks["permuted"] = direct | tracking.track_prediction(permuted, track_score, **settings)
    times["permuted"] = time.perf_counter() - started
    started = time.perf_counter()
    masks["frame_local"] = direct | frame_local_prediction(cache.locations, track_score, settings)
    times["frame_local"] = time.perf_counter() - started
    return masks, (track_score, direct_score, direct), times, mapping


def threshold_mask(family, threshold, track_score, direct_score, direct):
    if family == "track":
        return track_score >= threshold
    if family == "direct":
        return direct_score >= threshold
    if family == "direct_or_track":
        return direct | (track_score >= threshold)
    raise ValueError(f"unknown threshold family: {family}")


def select_threshold(validation_grid, *, split):
    """接口显式拒绝测试集寻优，平分时按预先固定的网格顺序取首项。"""
    if split != "val":
        raise ValueError("threshold selection is validation-only")
    return max(validation_grid, key=lambda row: row["aggregate"]["score"])


class PreparedScoring:
    """预先分组原始标注；重复消融评分免去逐阈值全事件扫描。"""

    def __init__(self, locations, labels, target_ids):
        self.locations = np.asarray(locations, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=bool)
        target_ids = np.asarray(target_ids, dtype=np.int64)
        times = self.locations[:, 2]
        self.frames = int((times[-1] - times[0]) / 50)
        frame_ids = times // 50
        strict = (times % 50 != 0) & (frame_ids >= 0) & (frame_ids <= self.frames)
        positive = strict & self.labels & (target_ids != 0)
        self.positive_rows = np.flatnonzero(positive)
        pairs = np.column_stack((frame_ids[positive], target_ids[positive]))
        _, self.target_inverse, self.target_counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True)
        self.background_groups = []
        for frame in range(self.frames + 1):
            rows = np.flatnonzero(strict & ~self.labels & (frame_ids == frame))
            if len(rows):
                self.background_groups.append(rows)

    def score(self, prediction):
        prediction = np.asarray(prediction, dtype=bool)
        if prediction.shape != self.labels.shape:
            raise ValueError("prediction does not align with original annotations")
        tp = int(np.sum(prediction & self.labels))
        fp = int(np.sum(prediction & ~self.labels))
        hit = np.bincount(self.target_inverse,
                          weights=prediction[self.positive_rows],
                          minlength=len(self.target_counts))
        counts = BinaryCounts(tp=tp, fp=fp, fn=int(self.labels.sum()) - tp,
                              tn=int((~self.labels).sum()) - fp,
                              detected_objects=int(np.sum(hit >= self.target_counts * 0.0001)),
                              objects=len(self.target_counts), frames=self.frames)
        for rows in self.background_groups:
            points = self.locations[rows[prediction[rows]], :2]
            if not len(points):
                continue
            points, multiplicities = np.unique(points, axis=0, return_counts=True)
            # 官方像素计数按uint8回绕；只裁掉全零边缘，不改变连通关系。
            points = points[multiplicities % 256 != 0]
            if not len(points):
                continue
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            canvas = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            canvas[points[:, 1] - top, points[:, 0] - left] = True
            counts.false_components += int(ndimage.label(canvas, np.ones((3, 3)))[1])
        return counts


def verified_inputs(split, source_root, cache_manifest, identity):
    paths = sorted((CACHE_ROOT / "cache" / split).glob(f"{split}_[0-9][0-9][0-9].npz"))
    expected = identity[split]
    if len(paths) != 24 or {path.name for path in paths} != set(expected):
        raise ValueError("requires all 24 source sequences per split")
    for path in paths:
        score_path = path.with_name(path.stem + "_scores.npz")
        source_path = source_root / path.name
        hashes = {}
        for item in (path, score_path):
            key = item.relative_to(CACHE_ROOT).as_posix()
            hashes[item.name] = sha256_file(item)
            if hashes[item.name] != cache_manifest[key]["sha256"]:
                raise ValueError(f"cache hash mismatch: {item}")
        hashes["source"] = sha256_file(source_path)
        if hashes["source"] != expected[path.name]["sha256"]:
            raise ValueError(f"source hash mismatch: {source_path}")
        cache = EventCache.load(path, allow_labels=False)
        with np.load(score_path, allow_pickle=False) as saved:
            scores = {name: saved[name].astype(np.float32) for name in ("small", "large")}
        for name, values in {"base": cache.base_probability, **scores}.items():
            if values.shape != (len(cache.locations),) or not np.all(np.isfinite(values)):
                raise ValueError(f"unaligned or nonfinite {name} scores")
            if np.any((values < 0) | (values > 1)):
                raise ValueError(f"invalid {name} probability")
        with np.load(source_path, allow_pickle=False) as raw:
            locations = raw["ev_loc"].astype(np.int64)
            labels = raw["evs_norm"][:, 4].astype(bool)
            ids = raw["evs_norm"][:, 5].astype(np.int64)
        if not np.array_equal(locations, cache.locations):
            raise ValueError("event order differs from original source")
        yield path.name, cache, scores, locations, labels, ids, hashes


def production_mask(split, name):
    return load_prediction(REPO / "outputs/main_model/integration/local" /
                           f"{split}_replayed" / name)


def aggregate(files):
    total = BinaryCounts()
    for metrics in files.values():
        total += BinaryCounts(**{key: int(metrics[key]) for key in asdict(total)})
    return total.metrics()


def run(args):
    started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    _, main_manifest = load_primary_manifest()
    release = read_json(RELEASE)
    contract = copy.deepcopy(release["event_contract"])
    report = {"status": "running", "protocol": {
        "frozen_checkpoint_sha256": release["provenance"]["checkpoint_sha256"],
        "release_report_sha256": sha256_file(RELEASE), "event_contract": contract,
        "main_manifest_sha256": sha256_file(REPO / "configs/main_model.json"),
        "main_model_selection": main_manifest["selection"],
        "cache_manifest_sha256": sha256_file(CACHE_ROOT / "cache_manifest.json"),
        "source_identity_sha256": sha256_file(IDENTITY),
        "script_sha256": sha256_file(Path(__file__)),
        "permutation_seed": 37, "threshold_grid": list(GRID),
        "threshold_families": list(FAMILIES),
        "selection_split": "val", "test_threshold_selection": False,
        "neural_retraining": False, "cache_labels_allowed": False,
        "scope": "exploratory frozen-system interventions; test has previously been examined",
        "no_duration": "minimum linked-node-count gate removed; association/interpolation/extension retained",
        "frame_local": "compound removal of linking/minimum support/interpolation/extension; hot filter retained",
        "permuted": "only downstream trajectory time is permuted; neural/tree scores stay original",
        "metric": "original integer-ms strict frame boundaries and modulo-256 pixel counts",
    }, "results": {}, "audits": {"inputs": {}, "production_masks_exact": {},
                                  "official_scoring_exact": {}}, "threshold_selection": {}}
    cache_manifest = read_json(CACHE_ROOT / "cache_manifest.json")
    identity = read_json(IDENTITY)
    grid_rows = [{"family": family, "threshold": threshold, "files": {}}
                 for family in FAMILIES for threshold in GRID]
    selected = None
    for split in ("val", "test"):
        source = REPO / "data/ev-uav/val" if split == "val" else REPO.parent / "EV-UAV-dataset/test"
        results = {name: {"files": {}, "prediction_sha256": {}, "walltime_seconds": 0.0}
                   for name in (*VARIANTS, "threshold")}
        report["results"][split] = results
        report["audits"]["inputs"][split] = {}
        report["audits"]["production_masks_exact"][split] = {}
        report["audits"]["official_scoring_exact"][split] = {}
        for index, item in enumerate(verified_inputs(split, source, cache_manifest, identity)):
            name, cache, scores, locations, labels, ids, hashes = item
            file_started = time.perf_counter()
            masks, probabilities, times, mapping = control_masks(cache, scores, contract)
            if not np.array_equal(masks["full"], production_mask(split, name)):
                raise ValueError(f"production mask differs: {split}/{name}")
            prepared = PreparedScoring(locations, labels, ids)
            official = score_binary(locations, labels, ids, masks["full"])
            if asdict(official) != asdict(prepared.score(masks["full"])):
                raise ValueError(f"optimized scorer differs: {split}/{name}")
            report["audits"]["official_scoring_exact"][split][name] = True
            report["audits"]["production_masks_exact"][split][name] = True
            report["audits"]["inputs"][split][name] = {**hashes, "frame_mapping": mapping.tolist()}
            if split == "val":
                for row in grid_rows:
                    mask = threshold_mask(row["family"], row["threshold"], *probabilities)
                    row["files"][name] = prepared.score(mask).metrics()
            else:
                masks["threshold"] = threshold_mask(selected["family"], selected["threshold"],
                                                     *probabilities)
                times["threshold"] = 0.0
            for variant, mask in masks.items():
                path = output / "masks" / split / variant / name
                write_prediction_npz(path, mask)
                result = results[variant]
                result["files"][name] = prepared.score(mask).metrics()
                result["prediction_sha256"][name] = sha256_file(path)
                result["walltime_seconds"] += times[variant]
            print(f"{split} {index + 1:02d}/24 {name}: {time.perf_counter()-file_started:.3f}s",
                  flush=True)
            if args.smoke:
                report["status"] = "smoke_complete"
                report["elapsed_seconds"] = time.perf_counter() - started
                write_json(output / "report.json", report)
                return report
        if split == "val":
            for row in grid_rows:
                row["aggregate"] = aggregate(row["files"])
            selected = select_threshold(grid_rows, split="val")
            report["threshold_selection"] = {
                "winner": {key: selected[key] for key in ("family", "threshold", "aggregate")},
                "validation_grid": grid_rows,
                "tie_break": "first in fixed family/grid order", "test_candidates_scored": 1,
            }
            # 选好全局阈值后再写验证掩码，不依据测试结果改变选择。
            for item in verified_inputs(split, source, cache_manifest, identity):
                name, cache, scores, _, _, _, _ = item
                track_score = blend_scores(cache.base_probability, scores["small"], scores["large"],
                                           tuple(contract["track_weights"]))
                direct_score = blend_scores(cache.base_probability, scores["small"], scores["large"],
                                            tuple(contract["direct_weights"]))
                direct = direct_score >= contract["direct_threshold"]
                mask = threshold_mask(selected["family"], selected["threshold"],
                                      track_score, direct_score, direct)
                path = output / "masks/val/threshold" / name
                write_prediction_npz(path, mask)
                results["threshold"]["files"][name] = selected["files"][name]
                results["threshold"]["prediction_sha256"][name] = sha256_file(path)
        for result in results.values():
            result["aggregate"] = aggregate(result["files"])
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "threshold": report["threshold_selection"]["winner"]}, ensure_ascii=False),
          flush=True)
    return report


def rescore(output):
    """另进程只读保存掩码及原始NPZ，复核所有整数计数与哈希。"""
    started = time.perf_counter()
    report = read_json(output / "report.json")
    checked = 0
    for split, variants in report["results"].items():
        source = REPO / "data/ev-uav/val" if split == "val" else REPO.parent / "EV-UAV-dataset/test"
        for name, audit in report["audits"]["inputs"][split].items():
            if sha256_file(source / name) != audit["source"]:
                raise ValueError("raw source changed before independent scoring")
            with np.load(source / name, allow_pickle=False) as raw:
                locations = raw["ev_loc"].astype(np.int64)
                labels = raw["evs_norm"][:, 4].astype(bool)
                ids = raw["evs_norm"][:, 5].astype(np.int64)
                for variant, result in variants.items():
                    path = output / "masks" / split / variant / name
                    if sha256_file(path) != result["prediction_sha256"][name]:
                        raise ValueError("saved mask hash changed")
                    mask = load_prediction(path)
                    actual = score_binary(locations, labels, ids, mask).metrics()
                    if actual != result["files"][name]:
                        raise ValueError(f"independent metric mismatch: {split}/{variant}/{name}")
                    checked += 1
        for result in variants.values():
            if aggregate(result["files"]) != result["aggregate"]:
                raise ValueError("global aggregation differs")
    audit = {"status": "verified", "mask_files": checked,
             "elapsed_seconds": time.perf_counter() - started,
             "report_sha256": sha256_file(output / "report.json"),
             "method": "new process, original NPZ and saved masks; official score_binary verifies all saved variants"}
    write_json(output / "independent_report.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/core_structure_20261023/run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    rescore(args.output) if args.rescore else run(args)
