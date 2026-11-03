"""核查完整val/test结果，汇总基准指标和概率诊断，生成单个回传包。"""

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from benchmark_metrics import event_columns, from_counts, measure
from collect_results import collect
from event_io import THRESHOLD, sha256, write_json
from sklearn.metrics import roc_auc_score

BUNDLE = Path(__file__).resolve().parents[1]
MODELS = ("ev_spsegnet", "pact", "pointevent")


def auc(truth, probability):
    return float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None


def analyze_split(root, split, inventory):
    rows, totals = [], []
    expected = {r["file"]: (r["sha256"], r["events"]) for r in inventory}
    for model in MODELS:
        report_path = root / model / "report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "complete":
            continue
        actual = {r["file"]: (r["sha256"], r["events"]) for r in report["sequences"]}
        if actual != expected or report["split"] != split or report["threshold"] != THRESHOLD:
            raise ValueError(f"Incomplete split or changed input protocol: {split}/{model}")
        probabilities, labels, current = [], [], []
        for name in sorted(expected):
            with np.load(root / "inputs" / name, allow_pickle=False) as data:
                raw, truth = data["ev"], data["gt"].astype(bool)
            if not np.array_equal(event_columns(raw)[3], truth):
                raise ValueError(f"Exported GT differs from original event labels: {split}/{name}")
            with np.load(root / model / "predictions" / name, allow_pickle=False) as data:
                probability, prediction = data["probability"], data["prediction"]
            result = measure(raw, prediction)
            result.update(split=split, model=model, sequence=name, events=len(truth),
                          target_events=int(truth.sum()), predicted_events=int(prediction.sum()),
                          probability_max=float(probability.max()),
                          probability_p99=float(np.quantile(probability, 0.99)),
                          gt_probability_median=float(np.median(probability[truth]))
                          if truth.any() else None,
                          positives_at_05_diagnostic=int((probability >= 0.5).sum()),
                          auc=auc(truth, probability))
            current.append(result)
            probabilities.append(probability)
            labels.append(truth)
        # 全局计数先相加再求比值；逐序列平均值另外列出，避免混用。
        counts = {key: sum(row["counts"][key] for row in current)
                  for key in current[0]["counts"]}
        valid_auc = [r["auc"] for r in current if r["auc"] is not None]
        valid_iou = [r["iou"] for r in current if r["iou"] is not None]
        total = dict(split=split, model=model, sequences=len(current), counts=counts,
                     **from_counts(counts), threshold=THRESHOLD, backend=report.get("backend"),
                     auc_global=auc(np.concatenate(labels), np.concatenate(probabilities)),
                     auc_mean_sequence=float(np.mean(valid_auc)) if valid_auc else None,
                     iou_mean_sequence=float(np.mean(valid_iou)) if valid_iou else None,
                     zero_positive_sequences=[r["sequence"] for r in current
                                              if r["predicted_events"] == 0],
                     checkpoint_sha256=report["checkpoint_sha256"])
        totals.append(total)
        rows.extend(current)
    return totals, rows


def write_csv(path, rows):
    flat = [{**{key: value for key, value in row.items()
                if not isinstance(value, (dict, list))}, **row.get("counts", {}),
             **{f"paper_delta_{key}": value for key, value in row.get("paper_difference", {}).items()}}
            for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def collect_full(root, archive_path):
    root, archive_path = root.resolve(), archive_path.resolve()
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    inventory_path = root / "dataset_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    references = json.loads((BUNDLE / "paper_reference.json").read_text(encoding="utf-8"))
    totals, rows, split_summaries = [], [], {}
    with tempfile.TemporaryDirectory(prefix="external_full_return_") as temporary:
        stage = Path(temporary)
        for split in ("val", "test"):
            split_root = root / split
            completed = [p for p in split_root.glob("*/report.json")
                         if json.loads(p.read_text(encoding="utf-8"))["status"] == "complete"]
            if not completed:
                split_summaries[split] = dict(status="no_completed_models")
                continue
            part = stage / f"{split}.zip"
            # 复用昨天收集器，逐项校验概率、掩码、行号和跨模型输入身份。
            collect(split_root, part)
            split_totals, split_rows = analyze_split(split_root, split, inventory["splits"][split])
            totals.extend(split_totals)
            rows.extend(split_rows)
            present = [r["model"] for r in split_totals]
            split_summaries[split] = dict(models=present, missing_models=sorted(set(MODELS)-set(present)),
                                          sequences=len(inventory["splits"][split]))
        if not totals:
            raise ValueError("No completed full-split predictions to return")
        canonical_test = {f"test_{i:03d}.npz" for i in range(24)}
        full_test = {r["file"] for r in inventory["splits"]["test"]} == canonical_test
        for row in totals:
            if row["split"] == "test" and full_test:
                reference = references["models"][row["model"]]
                row["paper_difference"] = {
                    "iou_percentage_points": (row["iou"] - reference["iou"]) * 100,
                    "seg_acc_percentage_points": (row["seg_acc"] - reference["seg_acc"]) * 100,
                    "pd_percentage_points": (row["pd"] - reference["pd"]) * 100,
                    "fa_times_1e4": (row["fa"] - reference["fa"]) * 1e4,
                }
        metrics = dict(threshold=THRESHOLD, threshold_selection="none",
                       aggregation="sum event/object counts before computing split ratios",
                       fa_protocol="EV-SpSegNet reference, one 8-connected-component count per 50ms frame",
                       notes=references["notes"], summary=totals, per_sequence=rows)
        analysis = root / "analysis"
        analysis.mkdir(exist_ok=True)
        write_json(analysis / "metrics.json", metrics)
        write_csv(analysis / "metrics_summary.csv", totals)
        write_csv(analysis / "metrics_per_sequence.csv", rows)
        selected = {p: p.relative_to(root).as_posix() for p in analysis.iterdir() if p.is_file()}
        selected[inventory_path] = "dataset_inventory.json"
        for name in ("run_status.json", "environment_windows.json", "qualitative_selection.json"):
            path = root / name
            if path.is_file():
                selected[path] = name
        for path in (root / "logs").glob("*.log"):
            selected[path] = path.relative_to(root).as_posix()
        selected[BUNDLE / "paper_reference.json"] = "paper_reference.json"
        selected[BUNDLE / "FULL_RUN.md"] = "FULL_RUN.md"
        for path in stage.iterdir():
            selected[path] = path.name
        complete = all(not split_summaries[s].get("missing_models", ["all"])
                       for s in ("val", "test"))
        summary = dict(status="complete" if complete else "partial", splits=split_summaries,
                       prediction_files=len(rows), threshold=THRESHOLD,
                       gpu_operator_parity="not_established_with_original_fused_extensions")
        prefix = "external_full_results/"
        checksums = "".join(f"{sha256(path)}  {name}\n" for path, name in sorted(
            selected.items(), key=lambda item: item[1]))
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in sorted(selected.items(), key=lambda item: item[1]):
                archive.write(path, prefix + name)
            archive.writestr(prefix + "summary.json", json.dumps(summary, indent=2) + "\n")
            archive.writestr(prefix + "SHA256SUMS", checksums)
        archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
            f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(json.dumps(summary))
    print(f"Return: {archive_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    collect_full(args.results, args.archive)
