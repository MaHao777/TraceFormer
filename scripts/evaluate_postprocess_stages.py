"""盲推并独立复评开发集事件级与组件级后处理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huav.postprocessing.contracts import EventCache, density_group, sha256_file
from huav.postprocessing.learning import (
    blend_scores,
    load_event_models,
    predict_event_scores,
    require_lightgbm,
)
from huav.postprocessing.refinement import (
    apply_component_model,
    component_features,
)
from huav.postprocessing.submission import (
    evaluate_predictions,
    write_prediction_npz,
)
from huav.postprocessing.tracking import track_prediction


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用79/20开发模型盲推验证集，并独立比较各后处理阶段"
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--original-report", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    cache_dir = Path(arguments.cache_dir)
    training_dir = Path(arguments.training_dir)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    event_dir = output_dir / "event_stage"
    component_dir = output_dir / "component_stage"
    base_dir = output_dir / "objectness"
    report = json.loads(
        (training_dir / "training_report.json").read_text(encoding="utf-8")
    )
    if report["status"] != "rejected_component_stage":
        raise ValueError("expected a completed component-stage development report")

    # 验证缓存不含标签，模型选择后再从原始NPZ独立计分。
    models = load_event_models(training_dir / "development" / "events")
    lgb = require_lightgbm()
    component_path = training_dir / "development" / "component.txt"
    component_model = lgb.Booster(model_file=str(component_path))
    event_contract = report["event_stage"]
    component_threshold = float(report["component_stage"]["threshold"])
    records: dict[str, Any] = {}
    for cache_path in sorted(cache_dir.glob("*.npz")):
        cache = EventCache.load(cache_path, allow_labels=False)
        scores = predict_event_scores(cache, models)
        group, density = density_group(cache.locations)
        track_probability = blend_scores(
            cache.base_probability,
            scores["small"],
            scores["large"],
            tuple(event_contract["track_weights"]),
        )
        direct_probability = blend_scores(
            cache.base_probability,
            scores["small"],
            scores["large"],
            tuple(event_contract["direct_weights"]),
        )
        event_prediction = track_prediction(
            cache.locations,
            track_probability,
            **event_contract["tracking"][group],
        ) | (direct_probability >= float(event_contract["direct_threshold"]))
        batch = component_features(
            cache,
            track_probability,
            event_prediction,
            include_targets=False,
        )
        component_prediction = apply_component_model(
            batch,
            event_prediction,
            component_model,
            component_threshold,
        )
        write_prediction_npz(base_dir / cache_path.name, cache.prediction)
        write_prediction_npz(event_dir / cache_path.name, event_prediction)
        write_prediction_npz(component_dir / cache_path.name, component_prediction)
        records[cache_path.name] = {
            "events": len(cache.prediction),
            "density_group": group,
            "events_per_frame": density,
            "objectness_positive": int(cache.prediction.sum()),
            "event_positive": int(event_prediction.sum()),
            "component_positive": int(component_prediction.sum()),
        }
        print(f"{cache_path.name}: {len(cache.prediction):,} events", flush=True)

    dataset_root = Path(arguments.dataset_root)
    original = _load_original(Path(arguments.original_report))
    objectness = evaluate_predictions(base_dir, dataset_root)
    event_stage = evaluate_predictions(event_dir, dataset_root)
    component_stage = evaluate_predictions(component_dir, dataset_root)
    comparison = {
        "protocol": {
            "source_training_report": str(
                (training_dir / "training_report.json").resolve()
            ),
            "event_model_sha256": {
                name: sha256_file(training_dir / "development" / "events" / name)
                for name in ("small.txt", "large.txt")
            },
            "component_model_sha256": sha256_file(component_path),
            "validation_labels_accessed_during_inference": False,
            "development_training_files": len(report["training_names"]),
            "development_holdout_files": len(report["holdout_names"]),
        },
        "original": original,
        "objectness": objectness,
        "objectness_event": event_stage,
        "objectness_component": component_stage,
        "objectness_vs_original_delta": _metric_delta(
            objectness["aggregate"], original["aggregate"]
        ),
        "event_vs_objectness_delta": _metric_delta(
            event_stage["aggregate"], objectness["aggregate"]
        ),
        "component_vs_objectness_delta": _metric_delta(
            component_stage["aggregate"], objectness["aggregate"]
        ),
        "component_vs_event_delta": _metric_delta(
            component_stage["aggregate"], event_stage["aggregate"]
        ),
        "val_015": {
            "original": original["files"]["val_015.npz"],
            "objectness": objectness["files"]["val_015.npz"],
            "objectness_event": event_stage["files"]["val_015.npz"],
            "objectness_component": component_stage["files"]["val_015.npz"],
        },
        "val_023": {
            "original": original["files"]["val_023.npz"],
            "objectness": objectness["files"]["val_023.npz"],
            "objectness_event": event_stage["files"]["val_023.npz"],
            "objectness_component": component_stage["files"]["val_023.npz"],
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
