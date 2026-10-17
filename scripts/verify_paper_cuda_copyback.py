"""核验下载的 CUDA 产物，并从本地原始 24 序列重新计算分数。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from traceformer.postprocessing.metrics import BinaryCounts, score_binary
from traceformer.postprocessing.submission import evaluate_predictions, load_prediction


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def competition_counts(actual: dict) -> dict:
    # 神经报告沿用旧评分字段名，显式转换后再逐项比较。
    return {"intersection": actual["tp"], "union": actual["tp"] + actual["fp"] + actual["fn"],
            "true_positive": actual["tp"], "gt_positive": actual["tp"] + actual["fn"],
            "detected_objects": actual["detected_objects"], "objects": actual["objects"],
            "false_alarms": actual["false_components"], "frames": actual["frames"]}


def verify(root: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "data/ev-uav/val"
    sources = sorted(dataset.glob("*.npz"))
    if len(sources) != 24:
        raise ValueError("requires exactly 24 local validation sources")
    manifest = read(root / "transfer_manifest.json")
    checked_inputs = {}
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.exists():
            # 原始数据与未修改的上游源码已有本地副本，不重复传输。
            path = (repo / "outputs/baselines" / relative if relative.startswith("upstream/")
                    else repo / relative)
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"input/source hash differs: {relative}")
        checked_inputs[relative] = actual

    arrays = {}
    for path in sources:
        with np.load(path, allow_pickle=False) as bundle:
            events = bundle["ev"]
            locations = np.column_stack((events["x"], events["y"], events["t"])).astype(np.int64)
            arrays[path.stem] = (locations, events["label"].astype(bool), events["name"].astype(np.int64))

    audit = read(root / "postflight_audit.json")
    result = {"status": "verified", "input_sha256": checked_inputs, "stages": {}}
    for name in ("TraceFormer-main", "PACT", "TraceFormer-complete"):
        folder = root / "results" / name
        record = audit["stages"][name]
        if record["status"] != "verified":
            raise ValueError(f"remote stage is unverified: {name}")
        for filename, key in (("report.json", "report_sha256"),
                              ("independent_report.json", "independent_report_sha256")):
            if sha256(folder / filename) != record[key]:
                raise ValueError(f"downloaded report differs: {name}/{filename}")
        report = read(folder / "report.json")
        independent = read(folder / "independent_report.json")
        if name == "TraceFormer-complete":
            stages = {}
            for stage in ("neural", "direct", "event_track", "complete"):
                if {p.name for p in (folder / stage).glob("*.npz")} != {p.name for p in sources}:
                    raise ValueError(f"binary prediction file set differs: {stage}")
                stages[stage] = evaluate_predictions(folder / stage, dataset)
                if stages[stage] != report["stages"][stage]:
                    raise ValueError(f"local binary scoring differs: {stage}")
            if independent != stages["complete"]:
                raise ValueError("complete-system independent report differs")
            for source in sources:
                evidence = report["files"][source.name]
                for path, key in ((source, "source_sha256"),
                    (root / "results/TraceFormer-main/cache" / source.name, "cache_sha256"),
                    (folder / "complete" / source.name, "final_sha256")):
                    if sha256(path) != evidence[key]:
                        raise ValueError(f"complete-system input/output differs: {path}")
            # 比较掩码本身，避免只靠相同全局分数声称与旧发布完全一致。
            archived = repo / "outputs/postprocess_event_release_dense_rescue_20260829/validation_release99_dense_rescue/event_release"
            differences = {p.name: int(np.count_nonzero(
                load_prediction(folder / "complete" / p.name) != load_prediction(archived / p.name)))
                for p in sources}
            result["stages"][name] = {"metrics": stages, "archived_mask_differences": differences}
            continue
        if {p.stem for p in folder.glob("val_*.pt")} != set(arrays):
            raise ValueError(f"probability file set differs: {name}")
        probabilities = {}
        for path in sources:
            saved_path = folder / f"{path.stem}.pt"
            if sha256(saved_path) != independent["prediction_sha256"][saved_path.name]:
                raise ValueError(f"prediction hash differs: {saved_path}")
            saved = torch.load(saved_path, map_location="cpu", weights_only=True)
            probability = saved["event_probabilities"].numpy()
            if saved["source_sha256"] != sha256(path):
                raise ValueError("prediction source hash differs")
            if probability.shape != arrays[path.stem][1].shape or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("probability shape/range differs")
            probabilities[path.stem] = probability
        metrics = {}
        points = {point: report[point]["threshold"]
                  for point in ("fixed_threshold", "selected_threshold", "release_threshold") if point in report}
        if name == "PACT":
            # 固定上游 utils/eval.py 的三个评估入口默认值均为 0.9。
            points["native_threshold"] = 0.9
        for point, threshold in points.items():
            pooled, per_sequence = BinaryCounts(), {}
            for stem, (locations, labels, identities) in arrays.items():
                counts = score_binary(locations, labels, identities, probabilities[stem] >= threshold)
                pooled += counts
                per_sequence[f"{stem}.npz"] = counts.metrics()
            actual = pooled.metrics()
            if point in report:
                expected = report[point]["competition"]
                if competition_counts(actual) != expected["counts"] or abs(actual["score"] - expected["score"]) > 1e-12:
                    raise ValueError(f"local probability scoring differs: {name}/{point}")
            if point == "selected_threshold":
                for filename, values in per_sequence.items():
                    before = independent["per_sequence_selected"][filename]["competition"]
                    if competition_counts(values) != before["counts"] or abs(values["score"] - before["score"]) > 1e-12:
                        raise ValueError(f"local per-sequence scoring differs: {name}/{filename}")
            metrics[point] = {"threshold": threshold, "aggregate": actual, "files": per_sequence}
        result["stages"][name] = {"metrics": metrics}
        if name == "PACT":
            result["stages"][name]["native_threshold_source"] = "upstream/PACT-02e9e4d20250/utils/eval.py; default thresh=0.9"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verified = verify(args.root)
    (args.root / "local_verification.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": verified["status"], "stages": list(verified["stages"])}))
