"""只读完整训练/验证缓存，审计严格事件记忆、混合Token和IoU理论上界。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from traceformer.experiments.event_memory import SamePixelEventMemory
from traceformer.postprocessing.contracts import sha256_file

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "outputs/neural_event_memory_20261024"
DEFAULT_SOURCE = REPO / "outputs/cone_attention_cpu_20261017_235917/seed37/cone_B/attempt_001"
BINS = ("le_1ms", "gt_1_le_10ms", "gt_10_le_50ms", "gt_50ms_finite", "missing")
VIEWS = (*SamePixelEventMemory.gap_names, "nearest_same", "nearest_opposite")
CLASSES = ("background", "foreground")
STRATA = ("pure_token", "mixed_token")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                          + "\n", encoding="utf-8")


def token_oracle(positive, negative):
    """任意逐Token恒定二分类器的最大事件IoU，不是Score或Pd的上界。"""
    positive = np.asarray(positive, dtype=np.int64)
    negative = np.asarray(negative, dtype=np.int64)
    total_positive = int(positive.sum())
    if total_positive == 0:
        return {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "selected_tokens": 0,
                "event_recall": 0.0, "exact_integer_optimality_checked": True}
    purity = positive / (positive + negative)
    order = np.argsort(-purity, kind="stable")
    tp = np.cumsum(positive[order], dtype=np.int64)
    fp = np.cumsum(negative[order], dtype=np.int64)
    iou = tp / (total_positive + fp)
    chosen = int(np.argmax(iou))
    true_positive, false_positive = int(tp[chosen]), int(fp[chosen])
    # 最优比值I要求纳入p-I*n>=0的Token；整数交叉相乘避免浮点证明误差。
    margins = positive * (total_positive + false_positive) - negative * true_positive
    assert np.all(margins[order[:chosen + 1]] >= 0)
    assert np.all(margins[order[chosen + 1:]] <= 0)
    return {"iou": float(iou[chosen]), "tp": true_positive, "fp": false_positive,
            "fn": total_positive - true_positive, "selected_tokens": chosen + 1,
            "event_recall": true_positive / total_positive,
            "exact_integer_optimality_checked": True}


def distributions(counts):
    """同时保存绝对计数和各FG/BG类内归一化比例。"""
    result = {}
    for view_index, view in enumerate(VIEWS):
        rows = counts[view_index]
        result[view] = {}
        for stratum, values in (("all", rows.sum(axis=0)), *zip(STRATA, rows)):
            result[view][stratum] = {}
            for class_index, name in enumerate(CLASSES):
                values_by_bin = values[class_index]
                total = int(values_by_bin.sum())
                result[view][stratum][name] = {
                    "events": total,
                    "bin_counts": {key: int(value) for key, value in zip(BINS, values_by_bin)},
                    "bin_fractions": {key: int(value) / total if total else 0.0
                                      for key, value in zip(BINS, values_by_bin)},
                }
    return result


def analyze_file(path, identity, extractor):
    if sha256_file(path) != identity["cache_sha256"]:
        raise ValueError(f"cache hash changed: {path.name}")
    item = torch.load(path, weights_only=True, mmap=True)
    xy, timestamps = item["xy"].numpy(), item["timestamps_ms"].numpy()
    local, indices = item["locals"].numpy(), item["indices"].numpy()
    labels = item["labels"].numpy()
    if xy.dtype != np.float32 or local.dtype != np.float32 or timestamps.dtype != np.float64:
        raise ValueError("expected FP32 decoder fields and original float64 milliseconds")
    if labels.shape != (len(xy),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("invalid diagnostic labels")
    if local.shape != (len(xy), 6) or indices.shape != labels.shape:
        raise ValueError("cache event fields are not aligned")
    if len(xy) != identity["events"] or len(item["tokens"]) != identity["tokens"]:
        raise ValueError("cache event/token counts differ from manifest")
    labels = labels.astype(bool)
    counts = np.bincount(indices, minlength=identity["tokens"])
    assert len(counts) == identity["tokens"] and np.all(counts > 0)
    positive = np.bincount(indices, weights=labels, minlength=len(counts)).astype(np.int64)
    negative = counts - positive
    mixed_tokens = (positive > 0) & (negative > 0)
    mixed = mixed_tokens[indices]

    gaps = extractor.gaps(xy, timestamps, local[:, 3])
    assert gaps.dtype == np.float64 and not np.isnan(gaps).any()
    assert np.all(gaps[np.isfinite(gaps)] > 0)
    # 重算结果与真实训练12维缓存逐值一致，诊断不使用另一套近邻定义。
    activations = np.exp(-gaps[:, :, None] / np.asarray(extractor.decay_ms))
    expected = activations.reshape(len(gaps), 12).astype(np.float32)
    assert np.array_equal(expected, item["memory"].numpy())
    views = [gaps[:, index] for index in range(4)]
    views.extend((np.minimum(gaps[:, 0], gaps[:, 1]), np.minimum(gaps[:, 2], gaps[:, 3])))
    result = np.zeros((len(VIEWS), 2, 2, len(BINS)), dtype=np.int64)
    # pure/mixed与BG/FG联合编码，所有分层严格互斥并覆盖原始事件。
    categories = mixed.astype(np.int64) * 2 + labels.astype(np.int64)
    for index, values in enumerate(views):
        bins = np.searchsorted((1.0, 10.0, 50.0), values, side="left")
        bins[~np.isfinite(values)] = 4
        result[index] = np.bincount(categories * 5 + bins, minlength=20).reshape(2, 2, 5)
        assert int(result[index].sum()) == len(labels)
        np.testing.assert_array_equal(result[index].sum(axis=(0, 2)),
                                      [int((~labels).sum()), int(labels.sum())])
        for stratum, chosen in enumerate((~mixed, mixed)):
            np.testing.assert_array_equal(result[index, stratum].sum(axis=1),
                                          [int(np.sum(chosen & ~labels)),
                                           int(np.sum(chosen & labels))])
    totals = {
        "events": len(labels), "foreground": int(labels.sum()), "background": int((~labels).sum()),
        "tokens": len(counts), "positive_tokens": int(np.sum(positive > 0)),
        "mixed_tokens": int(mixed_tokens.sum()),
        "pure_foreground_tokens": int(np.sum((positive > 0) & (negative == 0))),
        "pure_background_tokens": int(np.sum(positive == 0)),
        "foreground_in_mixed_tokens": int(positive[mixed_tokens].sum()),
        "background_in_mixed_tokens": int(negative[mixed_tokens].sum()),
        "foreground_in_background_dominated_tokens": int(positive[positive < negative].sum()),
    }
    assert totals["mixed_tokens"] + totals["pure_foreground_tokens"] + totals["pure_background_tokens"] == len(counts)
    summary = {"source_sha256": identity["source_sha256"],
               "cache_sha256": identity["cache_sha256"], "totals": totals,
               "token_constant_iou_oracle": token_oracle(positive, negative),
               "gap_distributions": distributions(result), "cached_memory_exact": True}
    return summary, result, positive, negative


def analyze_split(root, source, split, extractor):
    started = time.perf_counter()
    directory = root / "cache" / split
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    expected_files = 99 if split == "train" else 24
    if (manifest.get("status") != "complete" or manifest["split"] != split
            or len(manifest["files"]) != expected_files):
        raise ValueError(f"requires the complete {expected_files}-sequence {split} cache")
    for filename, key in (("best.pt", "checkpoint_sha256"), ("config.yaml", "config_sha256")):
        if sha256_file(source / filename) != manifest[key]:
            raise ValueError(f"source {key} changed")
    for relative, expected in manifest["feature_sources"].items():
        if sha256_file(REPO / relative) != expected:
            raise ValueError(f"cached feature source changed: {relative}")
    original_manifest_path = source / "dataset_manifest.json"
    original = read(original_manifest_path)
    originals = {Path(row["path"]).stem: row["sha256"] for row in original["files"]
                 if row["split"] == split}
    if set(originals) != set(manifest["files"]):
        raise ValueError("cache set differs from the pretrained source data manifest")
    files, positive_parts, negative_parts = {}, [], []
    total_counts = np.zeros((len(VIEWS), 2, 2, len(BINS)), dtype=np.int64)
    for index, (name, identity) in enumerate(sorted(manifest["files"].items())):
        if identity["source_sha256"].lower() != originals[name].lower():
            raise ValueError(f"source data identity differs: {name}")
        summary, counts, positive, negative = analyze_file(directory / f"{name}.pt", identity, extractor)
        files[name] = summary
        total_counts += counts
        positive_parts.append(positive)
        negative_parts.append(negative)
        if (index + 1) % 20 == 0 or index + 1 == expected_files:
            print(f"{split} {index + 1}/{expected_files}: {time.perf_counter()-started:.2f}s", flush=True)
    total = {key: sum(row["totals"][key] for row in files.values())
             for key in next(iter(files.values()))["totals"]}
    result = distributions(total_counts)
    for view in VIEWS:
        for name in CLASSES:
            assert result[view]["all"][name]["events"] == total[name]
            for key in BINS:
                assert result[view]["all"][name]["bin_counts"][key] == sum(
                    result[view][stratum][name]["bin_counts"][key] for stratum in STRATA)
                assert result[view]["all"][name]["bin_counts"][key] == sum(
                    file["gap_distributions"][view]["all"][name]["bin_counts"][key]
                    for file in files.values())
    oracle = token_oracle(np.concatenate(positive_parts), np.concatenate(negative_parts))
    return {"sequences": expected_files, "cache_manifest_sha256": sha256_file(manifest_path),
            "source_dataset_manifest_sha256": sha256_file(original_manifest_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_config_sha256": manifest["config_sha256"],
            "feature_sources": manifest["feature_sources"],
            "totals": total, "gap_distributions": result,
            "token_constant_iou_oracle": oracle, "files": files,
            "elapsed_seconds": time.perf_counter() - started}


def plot_fractions(split, payload, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 2, figsize=(11, 9.3), sharey=True, layout="constrained")
    positions = np.arange(len(BINS))
    labels = ("≤1", "(1,10]", "(10,50]", ">50 finite", "Missing")
    for row, stratum in enumerate(("all", "mixed_token", "pure_token")):
        for column, view in enumerate(("nearest_same", "nearest_opposite")):
            axis = axes[row, column]
            for class_index, (name, color) in enumerate((("foreground", "#0072B2"),
                                                       ("background", "#D55E00"))):
                entry = payload["gap_distributions"][view][stratum][name]
                values = [entry["bin_fractions"][key] for key in BINS]
                bars = axis.bar(positions + (class_index - .5) * .36, values, .36,
                                label=f"{name.title()} (n={entry['events']:,})", color=color)
                axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], fontsize=6, padding=2)
            axis.set_xticks(positions, labels)
            axis.set_ylim(0, 1.02)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
            axis.set_title(f"{stratum.replace('_', ' ').title()} · {view.replace('_', ' ')}")
            axis.legend(fontsize=7, loc="upper right")
            if column == 0:
                axis.set_ylabel("Fraction within foreground/background")
            if row == 2:
                axis.set_xlabel("Strict nearest same-pixel gap (ms)")
    figure.suptitle(f"Strict event-memory diagnostics: {split} ({payload['sequences']} sequences)\n"
                   "Equal-time events excluded; labels used for analysis only", fontsize=12)
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"{split}_fractions.{extension}", dpi=180)
    plt.close(figure)


def export_tables(payload, directory):
    with (directory / "counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "view", "stratum", "class", "bin", "count", "class_events", "fraction"))
        for split, result in payload.items():
            for view, strata in result["gap_distributions"].items():
                for stratum, classes in strata.items():
                    for name, values in classes.items():
                        for key in BINS:
                            writer.writerow((split, view, stratum, name, key, values["bin_counts"][key],
                                             values["events"], values["bin_fractions"][key]))
    lines = ["# 严格事件记忆诊断", "", "只分析99个训练与24个验证序列，不读取测试集，不训练模型。", "",
             "所有时间邻居严格早于/晚于查询事件，同时间事件全部排除；缺失支持为零。",
             "每个间隔视图的FG/BG与pure/mixed计数均已断言回加原始标签数量。", "",
             "|分区|正事件|正事件处于混合Token|Token常量IoU标签oracle|",
             "|---|---:|---:|---:|"]
    for split, result in payload.items():
        total = result["totals"]
        lines.append(f"|{split}|{total['foreground']:,}|"
                     f"{total['foreground_in_mixed_tokens']/total['foreground']:.4%}|"
                     f"{result['token_constant_iou_oracle']['iou']:.8f}|")
    lines.extend(("", "|分区/分层|同极性≤1ms FG/BG|同极性≤10ms FG/BG|同极性≤50ms FG/BG|",
                  "|---|---:|---:|---:|"))
    for split, result in payload.items():
        for stratum in ("all", "mixed_token", "pure_token"):
            classes = result["gap_distributions"]["nearest_same"][stratum]
            pairs = []
            for count in (1, 2, 3):
                values = [sum(classes[name]["bin_fractions"][key] for key in BINS[:count])
                          for name in ("foreground", "background")]
                pairs.append(f"{values[0]:.4%}/{values[1]:.4%}")
            lines.append(f"|{split}/{stratum}|" + "|".join(pairs) + "|")
    lines.extend(("", "解释边界：统计差异仅支持进一步检验事件记忆假设，不构成神经结构提分证据。",
                  "1ms内的密集响应可能偏背景，不能把支持强度解释为单调可靠性。",
                  "Token常量oracle使用真实标签为每个Token选择恒定输出，只约束事件IoU，",
                  "不是可部署模型，也不是官方Score/Pd/Fa上限；若高于实际神经成绩，",
                  "不能据混合Token比例断言量化已成为当前瓶颈。", ""))
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    # 新目录保存每次定义下的结果，绝不覆盖历史诊断。
    directory = args.root / "diagnostics" / datetime.now().strftime("strict_memory_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    extractor = SamePixelEventMemory()
    results = {split: analyze_split(args.root, args.source, split, extractor) for split in ("train", "val")}
    report = {"status": "complete", "protocol": {
        "script_sha256": sha256_file(Path(__file__)), "root": str(args.root.resolve()),
        "source": str(args.source.resolve()), "pid": os.getpid(), "torch_threads": 1,
        "partitions": {"train": 99, "val": 24}, "test_accessed": False, "training": False,
        "gap_definition": "strictly earlier/later at the same pixel; equal times excluded",
        "gap_columns": list(extractor.gap_names), "nearest": "minimum of strict past/future gap",
        "bins": list(BINS), "strata": list(STRATA),
        "labels_used_only_for_counts_and_oracle": True,
        "oracle": "exact maximum event IoU with one arbitrary binary output per occupied token",
        "source_identity": "cache source hashes also match the pretrained dataset manifest",
        "cache_memory_recomputed_exactly": True, "all_partition_assertions_passed": True,
    }, "results": results}
    export_tables(results, directory)
    for split, result in results.items():
        plot_fractions(split, result, directory)
    report["elapsed_total_seconds"] = time.perf_counter() - started
    write(directory / "report.json", report)
    print(json.dumps({"output": str(directory.resolve()), "elapsed_seconds": report["elapsed_total_seconds"],
                      "oracle_iou": {split: result["token_constant_iou_oracle"]["iou"]
                                     for split, result in results.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
