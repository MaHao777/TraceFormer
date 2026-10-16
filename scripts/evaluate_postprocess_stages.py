"""盲推并独立复评 99 序列事件发布模型与 dense 长轨迹救回。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from traceformer.gpu.runtime import add_backend_arguments, validate_backends
from traceformer.postprocessing.contracts import EventCache, density_group, sha256_file
from traceformer.postprocessing.direct_verifier import (
    apply_direct_verifier,
    direct_candidate_features,
    event_branches,
    load_direct_verifier,
)
from traceformer.postprocessing.learning import (
    load_event_models,
    positive_prediction_threads,
    predict_event_scores,
    required_event_models,
)
from traceformer.postprocessing.submission import (
    evaluate_predictions,
    load_prediction,
    write_prediction_npz,
)
from traceformer.postprocessing.tracking import rescue_reference_tracks


def _metric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    keys = (
        "tp",
        "fp",
        "fn",
        "detected_objects",
        "false_components",
        "iou",
        "seg_acc",
        "pd",
        "fa",
        "score",
    )
    return {key: float(after[key]) - float(before[key]) for key in keys}


def _load_original(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = payload.get("base", payload)
    if "aggregate" not in original or "files" not in original:
        raise ValueError("original report must contain aggregate and files")
    return original


def _load_named_prediction(directory: Path, name: str) -> np.ndarray:
    stem = Path(name).stem
    for candidate in (directory / name, directory / f"{stem}.txt"):
        if candidate.is_file():
            return load_prediction(candidate)
    raise FileNotFoundError(f"missing reference prediction for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="盲推并独立比较 99 序列事件发布模型与 dense 长轨迹救回"
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--event-release-dir", required=True)
    parser.add_argument("--direct-verifier-dir", required=True)
    parser.add_argument(
        "--diagnostic-allow-rejected-verifier",
        action="store_true",
        help="仅诊断开发门槛失败的门控器，结果不得用于正式提交",
    )
    parser.add_argument("--reference-prediction-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--original-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    add_backend_arguments(parser, trees=True)
    arguments = parser.parse_args()
    validate_backends(prediction_backend=arguments.prediction_backend, feature_backend=arguments.feature_backend)

    cache_dir = Path(arguments.cache_dir)
    training_dir = Path(arguments.training_dir)
    event_release_dir = Path(arguments.event_release_dir)
    direct_verifier_dir = Path(arguments.direct_verifier_dir)
    reference_dir = Path(arguments.reference_prediction_dir)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    event_dir = output_dir / "event_release"
    event_rescue_dir = output_dir / "event_release_dense_rescue"
    gate_dir = output_dir / "event_release_direct_gate"
    rescue_dir = output_dir / "event_release_direct_gate_dense_rescue"
    base_dir = output_dir / "objectness"
    report = json.loads(
        (training_dir / "training_report.json").read_text(encoding="utf-8")
    )
    if report["status"] != "rejected_component_stage":
        raise ValueError("expected a completed component-stage development report")

    # 验证缓存不含标签，模型选择后再从原始NPZ独立计分。
    release_report_path = event_release_dir / "release_report.json"
    release_report = json.loads(release_report_path.read_text(encoding="utf-8"))
    expected_training_files = len(report["training_names"]) + len(report["holdout_names"])
    if release_report["training_files"] != expected_training_files:
        raise ValueError("event release did not use every training cache")
    model_dir = event_release_dir / "events"
    for name, expected_sha256 in release_report["event_model_sha256"].items():
        if sha256_file(model_dir / name) != expected_sha256:
            raise ValueError(f"event release model hash mismatch: {name}")
    event_contract = release_report["event_contract"]
    models = load_event_models(model_dir, names=required_event_models(event_contract),
                               backend=arguments.prediction_backend)
    verifier_model, verifier_report = load_direct_verifier(direct_verifier_dir)
    verifier_released = verifier_report["status"] == "released"
    if not verifier_released and not arguments.diagnostic_allow_rejected_verifier:
        raise ValueError("direct verifier did not pass its development gate")
    verifier_threshold = float(verifier_report["threshold"])
    records: dict[str, Any] = {}
    for cache_path in sorted(cache_dir.glob("*.npz")):
        cache = EventCache.load(cache_path, allow_labels=False)
        scores = predict_event_scores(cache, models, num_threads=arguments.prediction_threads,
                                      feature_backend=arguments.feature_backend)
        group, density = density_group(cache.locations)
        _, direct_probability, tracked, direct = event_branches(
            cache, scores, event_contract
        )
        event_prediction = tracked | direct
        candidate_batch = direct_candidate_features(
            cache,
            direct_probability,
            tracked,
            direct,
            include_targets=False,
        )
        gate_prediction = apply_direct_verifier(
            candidate_batch,
            tracked,
            direct,
            verifier_model,
            verifier_threshold,
            density_group=group,
        )
        reference_prediction = _load_named_prediction(reference_dir, cache_path.name)
        # 先对正式事件发布结果做救回，保持与历史最佳相同的比较口径。
        event_rescue_prediction, event_rescue_report = rescue_reference_tracks(
            cache.locations,
            event_prediction,
            reference_prediction,
            density_group=group,
        )
        rescue_prediction, rescue_report = rescue_reference_tracks(
            cache.locations,
            gate_prediction,
            reference_prediction,
            density_group=group,
        )
        write_prediction_npz(base_dir / cache_path.name, cache.prediction)
        write_prediction_npz(event_dir / cache_path.name, event_prediction)
        write_prediction_npz(event_rescue_dir / cache_path.name, event_rescue_prediction)
        write_prediction_npz(gate_dir / cache_path.name, gate_prediction)
        write_prediction_npz(rescue_dir / cache_path.name, rescue_prediction)
        records[cache_path.name] = {
            "events": len(cache.prediction),
            "density_group": group,
            "events_per_frame": density,
            "objectness_positive": int(cache.prediction.sum()),
            "event_positive": int(event_prediction.sum()),
            "event_rescue_positive": int(event_rescue_prediction.sum()),
            "event_rescue": event_rescue_report,
            "gate_positive": int(gate_prediction.sum()),
            "gate": {
                "candidate_components": int(len(candidate_batch.features)),
                "candidate_events": int(np.sum(direct & ~tracked)),
                "deleted_events": int(np.sum(event_prediction & ~gate_prediction)),
            },
            "rescue_positive": int(rescue_prediction.sum()),
            "rescue": rescue_report,
        }
        print(f"{cache_path.name}: {len(cache.prediction):,} events", flush=True)

    dataset_root = Path(arguments.dataset_root)
    original = _load_original(Path(arguments.original_report))
    objectness = evaluate_predictions(base_dir, dataset_root)
    event_stage = evaluate_predictions(event_dir, dataset_root)
    event_rescue_stage = evaluate_predictions(event_rescue_dir, dataset_root)
    gate_stage = evaluate_predictions(gate_dir, dataset_root)
    rescue_stage = evaluate_predictions(rescue_dir, dataset_root)
    comparison = {
        "protocol": {
            "source_training_report": str(
                (training_dir / "training_report.json").resolve()
            ),
            "event_release_report": str(release_report_path.resolve()),
            "event_release_report_sha256": sha256_file(release_report_path),
            "event_model_sha256": release_report["event_model_sha256"],
            "prediction_threads": arguments.prediction_threads,
            "prediction_backend": arguments.prediction_backend,
            "feature_backend": arguments.feature_backend,
            "active_event_models": list(models),
            "direct_verifier_report": str(
                (direct_verifier_dir / "training_report.json").resolve()
            ),
            "direct_verifier_report_sha256": sha256_file(
                direct_verifier_dir / "training_report.json"
            ),
            "direct_verifier_model_sha256": verifier_report["release_model"]["sha256"],
            "direct_verifier_threshold": verifier_threshold,
            "direct_verifier_formally_eligible": verifier_released,
            "diagnostic_allow_rejected_verifier": bool(
                arguments.diagnostic_allow_rejected_verifier
            ),
            "reference_prediction_dir": str(reference_dir.resolve()),
            "validation_labels_accessed_during_inference": False,
            "development_training_files": len(report["training_names"]),
            "development_holdout_files": len(report["holdout_names"]),
            "release_training_files": release_report["training_files"],
        },
        "original": original,
        "objectness": objectness,
        "event_release": event_stage,
        "event_release_dense_rescue": event_rescue_stage,
        "event_release_direct_gate": gate_stage,
        "event_release_direct_gate_dense_rescue": rescue_stage,
        "objectness_vs_original_delta": _metric_delta(
            objectness["aggregate"], original["aggregate"]
        ),
        "event_release_vs_objectness_delta": _metric_delta(
            event_stage["aggregate"], objectness["aggregate"]
        ),
        "dense_rescue_vs_event_release_delta": _metric_delta(
            event_rescue_stage["aggregate"], event_stage["aggregate"]
        ),
        "direct_gate_vs_event_release_delta": _metric_delta(
            gate_stage["aggregate"], event_stage["aggregate"]
        ),
        "dense_rescue_vs_direct_gate_delta": _metric_delta(
            rescue_stage["aggregate"], gate_stage["aggregate"]
        ),
        "val_015": {
            "original": original["files"]["val_015.npz"],
            "objectness": objectness["files"]["val_015.npz"],
            "event_release": event_stage["files"]["val_015.npz"],
            "event_release_dense_rescue": event_rescue_stage["files"]["val_015.npz"],
            "event_release_direct_gate": gate_stage["files"]["val_015.npz"],
            "event_release_direct_gate_dense_rescue": rescue_stage["files"][
                "val_015.npz"
            ],
        },
        "val_023": {
            "original": original["files"]["val_023.npz"],
            "objectness": objectness["files"]["val_023.npz"],
            "event_release": event_stage["files"]["val_023.npz"],
            "event_release_dense_rescue": event_rescue_stage["files"]["val_023.npz"],
            "event_release_direct_gate": gate_stage["files"]["val_023.npz"],
            "event_release_direct_gate_dense_rescue": rescue_stage["files"][
                "val_023.npz"
            ],
        },
        "inference": records,
    }
    output_path = output_dir / "validation_comparison.json"
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
