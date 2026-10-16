"""从已保存的逐事件概率独立复评，并检查源数据哈希和完整性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from evaluate_external_baseline import probability_metrics, select_threshold

from traceformer.config import load_config
from traceformer.training.engine import _dataset, _file_sha256, _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = _dataset(config, "val")
    if len(dataset) != 24:
        raise ValueError("requires all 24 validation files")
    expected = {path.stem for path in dataset.files}
    if {path.stem for path in args.predictions.glob("val_*.pt")} != expected:
        raise ValueError("prediction file set differs from the 24 validation files")
    results, hashes = [], {}
    for sequence in dataset:
        path = args.predictions / f"{sequence.source_path.stem}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        probability = saved["event_probabilities"]
        if saved["source_sha256"] != _file_sha256(sequence.source_path):
            raise ValueError("prediction source hash differs")
        if probability.shape != (sequence.length,) or not torch.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError("invalid aligned probabilities")
        results.append((probability, sequence))
        hashes[path.name] = _file_sha256(path)
    original = json.loads((args.predictions / "report.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise ValueError("initial report is not complete")
    report = {"status": "complete", "prediction_sha256": hashes,
              "fixed_threshold": probability_metrics(results, config, 0.5),
              "selected_threshold": select_threshold(results, config)}
    if "release_threshold" in original:
        report["release_threshold"] = probability_metrics(results, config, original["release_threshold"]["threshold"])
    for operating_point in ("fixed_threshold", "selected_threshold", "release_threshold"):
        if operating_point not in report:
            continue
        before, after = original[operating_point], report[operating_point]
        if before["competition"]["counts"] != after["competition"]["counts"] or abs(before["competition"]["score"] - after["competition"]["score"]) > 1e-12:
            raise ValueError(f"independent scoring mismatch: {operating_point}")
    threshold = report["selected_threshold"]["threshold"]
    report["per_sequence_selected"] = {sequence.source_path.name: probability_metrics([(p, sequence)], config, threshold)
                                       for p, sequence in results}
    _write_json(args.predictions / "independent_report.json", report)
    print(f"Independent rescore verified: {args.predictions}", flush=True)


if __name__ == "__main__":
    main()
