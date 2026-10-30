"""读取冻结预测，分析四组注意力在测试集上的事件、目标帧与虚警差异。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, precision_recall_curve

from traceformer.postprocessing.metrics import BinaryCounts
from traceformer.postprocessing.submission import _source_arrays, load_prediction

VARIANTS = ("A0", "A1", "A2", "A3")
COLORS = ("#3074a8", "#8c6bb1", "#d28b26", "#bd3c38")
FIELDS = tuple(asdict(BinaryCounts()))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def components(xy, target_xy, *, details):
    """复现官方8连通和像素计数溢出；距离仅作为误报空间位置诊断。"""
    if not len(xy):
        return 0, {"sizes": [], "near3": 0, "far3": 0, "no_gt": 0}
    positions, counts = np.unique(xy[:, 1] * 346 + xy[:, 0], return_counts=True)
    positions = positions[(counts % 256) != 0]
    mask = np.zeros((260, 346), dtype=bool)
    mask.ravel()[positions] = True
    connected, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    result = {"sizes": [], "near3": 0, "far3": 0, "no_gt": 0}
    if details and count:
        ids = connected.ravel()[positions]
        result["sizes"] = np.bincount(ids, minlength=count + 1)[1:].tolist()
        if len(target_xy):
            points = np.column_stack((positions % 346, positions // 346))
            distance = cKDTree(target_xy).query(points)[0]
            nearest = np.full(count + 1, np.inf)
            np.minimum.at(nearest, ids, distance)
            result["near3"] = int(np.sum(nearest[1:] <= 3))
            result["far3"] = int(np.sum(nearest[1:] > 3))
        else:
            result["no_gt"] = int(count)
    return int(count), result


def metric_difference(a, b):
    return {key: a[key] - b[key] for key in ("score", "iou", "seg_acc", "pd", "fa")}


def terms(metrics):
    return {"IoU": 0.2 * metrics["iou"], "SegAcc": 0.1 * metrics["seg_acc"],
            "Pd": 0.4 * metrics["pd"], "Fa": 0.3 * math.exp(-10000 * metrics["fa"])}


def strata(records, classify):
    groups = defaultdict(list)
    for record in records:
        groups[classify(record)].append(record)
    output = {}
    for name, rows in sorted(groups.items()):
        total = sum(r["n_gt"] for r in rows)
        output[name] = {"target_frames": len(rows), "gt_events": total, "variants": {}}
        for variant in VARIANTS:
            hits = sum(r[variant]["detected"] for r in rows)
            tp = sum(r[variant]["tp"] for r in rows)
            output[name]["variants"][variant] = {
                "detected": hits, "pd": hits / len(rows), "tp": tp,
                "event_recall": tp / total,
            }
    return output


def add_speed(records):
    """用相邻最多5个标注质心估计局部速度；这是有噪声的诊断量。"""
    tracks = defaultdict(list)
    for row in records:
        tracks[(row["sequence"], row["id"])].append(row)
    for track in tracks.values():
        for i, row in enumerate(track):
            nearby = [p for p in track[max(0, i - 2):i + 3]
                      if abs(p["frame"] - row["frame"]) <= 2]
            row["speed"] = None
            if len(nearby) >= 3:
                time = np.array([p["frame"] * 0.05 for p in nearby])
                time -= time.mean()
                slope = time @ np.array([p["centroid"] for p in nearby]) / (time @ time)
                row["speed"] = float(np.linalg.norm(slope))


def temporal_gaps(records):
    """仅在连续存在标注的目标帧内统计漏检段，不跨越无标注区间。"""
    tracks = defaultdict(list)
    for row in records:
        tracks[(row["sequence"], row["id"])].append(row)
    result = {}
    for variant in VARIANTS:
        runs = []
        for track in tracks.values():
            length, previous = 0, -10
            for row in track:
                if row["frame"] != previous + 1 and length:
                    runs.append(length)
                    length = 0
                if row[variant]["detected"]:
                    if length:
                        runs.append(length)
                        length = 0
                else:
                    length += 1
                previous = row["frame"]
            if length:
                runs.append(length)
        result[variant] = {"runs": len(runs), "missed_frames": sum(runs),
                           "max_frames": max(runs, default=0),
                           "runs_at_least4": sum(v >= 4 for v in runs),
                           "frames_in_runs_at_least4": sum(v for v in runs if v >= 4)}
    return result


def analyze(root, output):
    suite = read(root / "suite.json")
    reports = {v: read(root / f"test/{v}/independent_report.json") for v in VARIANTS}
    raw_reports = {v: read(root / f"test/{v}/raw_independent_report.json") for v in VARIANTS}
    manifest = read(root / "test/dataset_manifest.json")["files"]
    assert len(manifest) == 24
    global_counts = {v: BinaryCounts() for v in VARIANTS}
    fixed = {f"{v}@{t:.2f}": BinaryCounts() for v in VARIANTS for t in (0.9, 0.95)}
    profiles = {v: {"sizes": [], "near3": 0, "far3": 0, "no_gt": 0} for v in VARIANTS}
    pairs = {f"{a}-{b}": defaultdict(int) for a, b in combinations(VARIANTS, 2)}
    labels_all, probability_all = [], {v: [] for v in VARIANTS}
    sequences, objects, frame_rows, identities = {}, [], [], []
    for source in manifest:
        path = Path(source["path"])
        assert digest(path) == source["sha256"], path
        name = path.stem
        loc, gt, target_ids, _ = _source_arrays(path)
        assert np.all(np.diff(loc[:, 2]) >= 0)
        n_frames = int((loc[-1, 2] - loc[0, 2]) / 50)
        probabilities, masks = {}, {}
        for variant in VARIANTS:
            prob_path = root / f"test/{variant}/probabilities/{name}.pt"
            mask_path = root / f"test/{variant}/predictions/{name}.npz"
            saved = torch.load(prob_path, map_location="cpu", weights_only=True)
            assert saved["source_sha256"] == source["sha256"]
            probabilities[variant] = saved["event_probabilities"].numpy()
            masks[variant] = load_prediction(mask_path)
            assert masks[variant].shape == gt.shape
            assert np.array_equal(masks[variant], probabilities[variant] >= reports[variant]["threshold"])
            assert digest(mask_path) == reports[variant]["files"][path.name]["prediction_sha256"]
            identities.append({"path": str(prob_path), "sha256": digest(prob_path)})
            probability_all[variant].append(probabilities[variant])
        labels_all.append(gt)
        for variant in VARIANTS:
            for threshold in (0.9, 0.95):
                masks[f"{variant}@{threshold:.2f}"] = probabilities[variant] >= threshold
        counts = {key: BinaryCounts(tp=int((mask & gt).sum()), fp=int((mask & ~gt).sum()),
                                    fn=int((~mask & gt).sum()), tn=int((~mask & ~gt).sum()),
                                    frames=n_frames) for key, mask in masks.items()}
        for a, b in combinations(VARIANTS, 2):
            pair = pairs[f"{a}-{b}"]
            for key, select in {"a_tp_b_fn": gt & masks[a] & ~masks[b],
                                "a_fn_b_tp": gt & ~masks[a] & masks[b],
                                "a_fp_b_tn": ~gt & masks[a] & ~masks[b],
                                "a_tn_b_fp": ~gt & ~masks[a] & masks[b]}.items():
                pair[key] += int(select.sum())
        # 官方评分先转整数毫秒，再排除每个50ms帧的边界事件。
        frame_ids = loc[:, 2] // 50
        valid = (loc[:, 2] % 50 != 0) & (frame_ids <= n_frames)
        valid_indices = np.flatnonzero(valid)
        groups = np.split(valid_indices, np.flatnonzero(np.diff(frame_ids[valid_indices])) + 1)
        for indices in groups:
            if not len(indices):
                continue
            positive = indices[gt[indices]]
            row = {"sequence": name, "frame": int(frame_ids[indices[0]]),
                   "n_gt": len(positive)}
            frame_objects = []
            for target_id in np.unique(target_ids[positive]):
                if target_id == 0:
                    continue
                target = positive[target_ids[positive] == target_id]
                record = {"sequence": name, "frame": row["frame"], "id": int(target_id),
                          "n_gt": len(target), "centroid": loc[target, :2].mean(0).tolist()}
                for key, mask in masks.items():
                    tp = int(mask[target].sum())
                    detected = tp / len(target) >= 0.0001
                    counts[key].objects += 1
                    counts[key].detected_objects += int(detected)
                    if key in VARIANTS:
                        record[key] = {"tp": tp, "detected": int(detected)}
                for a, b in combinations(VARIANTS, 2):
                    pair = pairs[f"{a}-{b}"]
                    hit_a, hit_b = record[a]["detected"], record[b]["detected"]
                    pair["a_detect_b_miss"] += int(hit_a and not hit_b)
                    pair["a_miss_b_detect"] += int(hit_b and not hit_a)
                frame_objects.append(record)
                objects.append(record)
            for key, mask in masks.items():
                false = indices[~gt[indices] & mask[indices]]
                count, detail = components(loc[false, :2], loc[positive, :2], details=key in VARIANTS)
                counts[key].false_components += count
                if key in VARIANTS:
                    profiles[key]["sizes"].extend(detail["sizes"])
                    for field in ("near3", "far3", "no_gt"):
                        profiles[key][field] += detail[field]
                    row[key] = {"tp": int(mask[positive].sum()), "fp": len(false),
                                "detected": sum(r[key]["detected"] for r in frame_objects)}
            frame_rows.append(row)
        record = {"events": len(gt), "gt_events": int(gt.sum()), "frames": n_frames,
                  "variants": {}}
        for key, count in counts.items():
            if key in VARIANTS:
                metrics = count.metrics()
                expected = raw_reports[key]["files"][path.name]
                assert all(metrics[field] == expected[field] for field in FIELDS), (name, key)
                assert all(abs(metrics[field] - expected[field]) < 1e-12
                           for field in ("score", "iou", "seg_acc", "pd", "fa"))
                global_counts[key] += count
                record["variants"][key] = metrics
            else:
                fixed[key] += count
        sequences[name] = record
        print(f"audited {name}: {len(gt):,} events", flush=True)
    assert len(objects) == global_counts["A0"].objects == 5228
    global_metrics = {v: count.metrics() for v, count in global_counts.items()}
    for variant in VARIANTS:
        assert all(global_metrics[variant][key] == raw_reports[variant]["aggregate"][key] for key in FIELDS)
        sizes = np.array(profiles[variant].pop("sizes"))
        profiles[variant].update(components=len(sizes), singleton=int((sizes == 1).sum()),
                                 two_pixels=int((sizes == 2).sum()), three_plus=int((sizes >= 3).sum()),
                                 mean_pixels=float(sizes.mean()))
        assert profiles[variant]["components"] == global_metrics[variant]["false_components"]
    add_speed(objects)
    strength = strata(objects, lambda r: "1-4" if r["n_gt"] < 5 else "5-9" if r["n_gt"] < 10
                      else "10-19" if r["n_gt"] < 20 else "20+")
    speed = strata(objects, lambda r: "unknown" if r["speed"] is None else "<32"
                   if r["speed"] < 32 else "32-128" if r["speed"] < 128 else ">=128")
    for variant in VARIANTS:
        assert sum(g["variants"][variant]["detected"] for g in strength.values()) == global_metrics[variant]["detected_objects"]
    overlap = {}
    for hit_a0, hit_a3 in ((1, 1), (1, 0), (0, 1), (0, 0)):
        rows = [r for r in objects if (r["A0"]["detected"], r["A3"]["detected"]) == (hit_a0, hit_a3)]
        overlap[f"A0_{hit_a0}_A3_{hit_a3}"] = {
            "target_frames": len(rows), "gt_events": sum(r["n_gt"] for r in rows),
            "A0_tp": sum(r["A0"]["tp"] for r in rows), "A3_tp": sum(r["A3"]["tp"] for r in rows),
        }
    density = {}
    for label, dense in (("below_300k", False), ("at_least_300k", True)):
        names = [n for n, r in sequences.items() if (r["events"] >= 300000) == dense]
        density[label] = {"sequences": names, "variants": {}}
        for variant in VARIANTS:
            pooled = BinaryCounts()
            for name in names:
                pooled += BinaryCounts(**{k: sequences[name]["variants"][variant][k] for k in FIELDS})
            density[label]["variants"][variant] = pooled.metrics()
    contribution = {}
    a0 = global_metrics["A0"]
    for name, record in sequences.items():
        zero, three = record["variants"]["A0"], record["variants"]["A3"]
        replace = BinaryCounts(**{k: a0[k] - zero[k] + three[k] for k in FIELDS}).metrics()
        excluded = {v: BinaryCounts(**{k: global_metrics[v][k] - record["variants"][v][k]
                                      for k in FIELDS}).metrics() for v in ("A0", "A3")}
        contribution[name] = {"standalone_delta": metric_difference(three, zero),
                              "replace_in_A0_score_effect": replace["score"] - a0["score"],
                              "leave_one_out_delta": metric_difference(excluded["A3"], excluded["A0"])}
    ap, pr = {}, {}
    gt_all = np.concatenate(labels_all)
    for variant in VARIANTS:
        probability = np.concatenate(probability_all[variant])
        precision, recall, _ = precision_recall_curve(gt_all, probability, drop_intermediate=True)
        ap[variant] = {"average_precision": float(average_precision_score(gt_all, probability)),
                       "positive_probability_quantiles": np.quantile(probability[gt_all], [.1, .5, .9]).tolist()}
        pr[variant] = (precision, recall)
    histories = {}
    for job in suite["jobs"]:
        done = read(root / job["id"] / "done.json")
        history = read(Path(done["checkpoint"]).parent / "history.json")
        best = max(history, key=lambda r: r["validation"]["competition"]["score"])
        histories[job["result_id"]] = {
            "best_epoch": best["epoch"], "scores": [r["validation"]["competition"]["score"] for r in history],
            "threshold": reports[job["result_id"]]["threshold"],
        }
    result = {"global": global_metrics, "thresholds": {v: reports[v]["threshold"] for v in VARIANTS},
              "weighted_score_terms": {v: terms(global_metrics[v]) for v in VARIANTS},
              "sequences": sequences, "pairwise": {k: dict(v) for k, v in pairs.items()},
              "target_strength": strength, "centroid_speed_diagnostic": speed, "density": density,
              "A0_A3_target_frame_overlap": overlap, "temporal_miss_runs": temporal_gaps(objects),
              "false_components": profiles, "fixed_threshold_diagnostic": {k: v.metrics() for k, v in fixed.items()},
              "ranking": ap, "A3_A0_sequence_contribution": contribution, "training": histories,
              "audit": {"source_files": 24, "events": len(gt_all), "target_frames": len(objects),
                        "scored_target_events": sum(r["n_gt"] for r in objects),
                        "full_count_match": True, "source_manifest_sha256": digest(root / "test/dataset_manifest.json"),
                        "probabilities": identities, "script_sha256": digest(__file__)}}
    save(output / "analysis.json", result)
    save(output / "target_frames.json", objects)
    fields = ["sequence", "events", "gt_events", "variant", *FIELDS, "score", "iou", "seg_acc", "pd", "fa"]
    with (output / "per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in sequences.items():
            for variant in VARIANTS:
                metrics = row["variants"][variant]
                writer.writerow({"sequence": name, "events": row["events"], "gt_events": row["gt_events"],
                                 "variant": variant, **{k: metrics[k] for k in fields if k in metrics}})
    render(output, result, pr, frame_rows, root)
    print(json.dumps({"global": global_metrics, "pairwise": result["pairwise"],
                      "strength": strength, "components": profiles, "AP": ap}, indent=2), flush=True)


def render(output, result, pr, frames, root):
    """保存标准科研图；案例由固定的差异排序规则选出，并标注选择依据。"""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = list(result["sequences"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, metric, title, factor in zip(axes, ("score", "iou", "pd"),
                                       ("Score difference", "IoU difference (pp)", "Pd difference (pp)"), (1, 100, 100)):
        values = [result["A3_A0_sequence_contribution"][n]["standalone_delta"][metric] * factor for n in names]
        ax.bar(range(24), values, color=[COLORS[3] if v >= 0 else COLORS[0] for v in values])
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel(title)
    axes[0].set_title("A3 minus A0 on each of all 24 test sequences (frozen thresholds)")
    axes[-1].set_xticks(range(24), [n[-3:] for n in names], rotation=45)
    axes[-1].set_xlabel("Test sequence ID")
    fig.savefig(output / "per_sequence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    keys = ("1-4", "5-9", "10-19", "20+")
    for i, variant in enumerate(VARIANTS):
        for ax, metric in zip(axes[:2], ("pd", "event_recall")):
            ax.plot(range(4), [100 * result["target_strength"][k]["variants"][variant][metric] for k in keys],
                    marker="o", color=COLORS[i], label=variant)
            ax.set_xticks(range(4), [f"{k}\nn={result['target_strength'][k]['target_frames']}" for k in keys])
            ax.set_xlabel("GT events per target-frame")
        precision, recall = pr[variant]
        axes[2].plot(recall, precision, label=f"{variant} AP={result['ranking'][variant]['average_precision']:.4f}",
                     color=COLORS[i])
    axes[0].set_ylabel("Target-frame Pd (%)")
    axes[1].set_ylabel("GT event recall in scored frames (%)")
    axes[2].set(xlabel="Event recall", ylabel="Event precision", xlim=(.7, 1), ylim=(.65, 1))
    axes[0].legend()
    axes[2].legend(fontsize=8)
    fig.savefig(output / "strength_and_ranking.png", dpi=180)
    plt.close(fig)
    candidates = [r for r in frames if r["n_gt"] >= 3]
    rules = [("A3 recovers target-frame(s) missed by A0", [r for r in candidates if r["A3"]["detected"] > r["A0"]["detected"]],
              lambda r: r["A3"]["tp"] - r["A0"]["tp"]),
             ("A0 recovers target-frame(s) missed by A3", [r for r in candidates if r["A0"]["detected"] > r["A3"]["detected"]],
              lambda r: r["A0"]["tp"] - r["A3"]["tp"]),
             ("Largest A3 minus A0 false-event increase", candidates,
              lambda r: r["A3"]["fp"] - r["A0"]["fp"])]
    selected = [(label, max(rows, key=score)) for label, rows, score in rules if rows]
    save(output / "case_selection.json", [{"rule": label, **row} for label, row in selected])
    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.6 * len(selected)), constrained_layout=True)
    for row_index, (rule, frame) in enumerate(selected):
        path = Path(read(root / "suite.json")["test_dataset_root"]) / "test" / f"{frame['sequence']}.npz"
        loc, gt, _, _ = _source_arrays(path)
        select = (loc[:, 2] > frame["frame"] * 50) & (loc[:, 2] < (frame["frame"] + 1) * 50)
        xy, target = loc[select, :2], gt[select]
        for col, variant in enumerate(("GT", *VARIANTS)):
            ax = axes[row_index, col]
            ax.scatter(xy[~target, 0], xy[~target, 1], s=2, c="#d3d6db", rasterized=True)
            if variant == "GT":
                ax.scatter(xy[target, 0], xy[target, 1], s=10, c="#15945f")
                ax.set_title(f"{frame['sequence']} / {frame['frame'] * 50}-{(frame['frame'] + 1) * 50}ms\nGT events: {int(target.sum())}")
                ax.set_ylabel(rule, fontsize=9)
            else:
                prediction = load_prediction(root / f"test/{variant}/predictions/{frame['sequence']}.npz")[select]
                for mask, color, marker in ((target & prediction, "#15945f", "o"),
                                            (target & ~prediction, "#df9c22", "x"),
                                            (~target & prediction, "#c83479", "+")):
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=13, c=color, marker=marker)
                ax.set_title(f"{variant}: TP={int((target & prediction).sum())}, FN={int((target & ~prediction).sum())}\nFP={int((~target & prediction).sum())}")
            ax.set(xlim=(0, 346), ylim=(260, 0), aspect="equal")
            ax.tick_params(labelsize=7)
    fig.suptitle("All event coordinates in selected 50ms frames: green TP, orange FN, magenta FP", fontsize=12)
    fig.savefig(output / "prediction_cases.png", dpi=200)
    for row_index, (_, frame) in enumerate(selected):
        path = Path(read(root / "suite.json")["test_dataset_root"]) / "test" / f"{frame['sequence']}.npz"
        loc, gt, _, _ = _source_arrays(path)
        selected_gt = gt & (loc[:, 2] > frame["frame"] * 50) & (loc[:, 2] < (frame["frame"] + 1) * 50)
        center = loc[selected_gt, :2].mean(0)
        extent = np.maximum(np.ptp(loc[selected_gt, :2], axis=0) / 2 + 12, 20)
        for ax in axes[row_index]:
            ax.set(xlim=(center[0] - extent[0], center[0] + extent[0]),
                   ylim=(center[1] + extent[1], center[1] - extent[1]))
    fig.suptitle("Target ROI zoom; TP/FN/FP counts in titles refer to the full frame", fontsize=12)
    fig.savefig(output / "prediction_cases_zoom.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    destination = args.suite.resolve() / "analysis_20261023"
    destination.mkdir(exist_ok=True)
    analyze(args.suite.resolve(), destination)
