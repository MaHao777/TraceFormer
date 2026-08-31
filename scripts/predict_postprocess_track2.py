"""用指定HUAV检查点与事件级后处理生成Track 2提交包。"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from huav.config import load_config
from huav.data import NPZEventDataset
from huav.postprocessing.contracts import EventCache, density_group, sha256_file
from huav.postprocessing.direct_verifier import (
    apply_direct_verifier,
    direct_candidate_features,
    event_branches,
    load_direct_verifier,
)
from huav.postprocessing.learning import load_event_models, predict_event_scores
from huav.postprocessing.submission import load_prediction
from huav.postprocessing.tracking import rescue_reference_tracks
from huav.training.engine import build_model, forward_with_sparse_mixture


def _serializable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    return value


def _write_track2_txt(
    source_path: Path,
    output_path: Path,
    prediction: np.ndarray,
) -> None:
    with np.load(source_path, allow_pickle=False) as archive:
        source = archive["ev"]
    if len(source) != len(prediction):
        raise ValueError(f"{source_path.name}: prediction row count mismatch")

    # 完全复刻官方test_Challenge2.py的字段类型与九位小数时间戳。
    rows = np.empty(
        len(source),
        dtype=[
            ("x", source.dtype["x"]),
            ("y", source.dtype["y"]),
            ("t", source.dtype["t"]),
            ("p", source.dtype["p"]),
            ("label", np.int64),
        ],
    )
    for field in ("x", "y", "t", "p"):
        rows[field] = source[field]
    rows["label"] = prediction.astype(np.int64, copy=False)
    np.savetxt(
        output_path,
        rows,
        fmt=["%d", "%d", "%.9f", "%d", "%d"],
        delimiter=" ",
    )


def _load_named_prediction(directory: Path, name: str) -> np.ndarray:
    stem = Path(name).stem
    for candidate in (directory / name, directory / f"{stem}.txt"):
        if candidate.is_file():
            return load_prediction(candidate)
    raise FileNotFoundError(f"missing reference prediction for {name}")


def _validation_gate(
    validation_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    candidate = validation_report.get(
        "event_release_direct_gate_dense_rescue",
        validation_report.get("event_release_dense_rescue"),
    )
    if candidate is None:
        raise ValueError("validation report has no final event release stage")
    baseline = baseline_report.get("event_stage", baseline_report.get("objectness_event"))
    if baseline is None:
        raise ValueError("baseline validation report has no event stage")
    score_gain = float(candidate["aggregate"]["score"]) - float(
        baseline["aggregate"]["score"]
    )
    val_023_pd_gain = float(candidate["files"]["val_023.npz"]["pd"]) - float(
        baseline["files"]["val_023.npz"]["pd"]
    )
    fa = float(candidate["aggregate"]["fa"])
    gate = {
        "score_gain": score_gain,
        "val_023_pd_gain": val_023_pd_gain,
        "fa": fa,
        "passed": score_gain > 0 and val_023_pd_gain >= 0 and fa <= 6.0e-6,
    }
    if not gate["passed"]:
        raise ValueError(f"validation promotion gate failed: {gate}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用原模型与事件级后处理生成Track 2测试集提交"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--event-release-dir", type=Path, required=True)
    parser.add_argument("--direct-verifier-dir", type=Path)
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="仅使用已通过门槛的事件模型，不加载直接验证器",
    )
    parser.add_argument("--training-cache-manifest", type=Path, required=True)
    parser.add_argument("--reference-prediction-dir", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--baseline-validation-report", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") not in {5, 7}:
        raise ValueError("event-postprocess submission requires schema 5 or 7")
    for name, expected in (
        ("model_config", _serializable(asdict(config.model))),
        ("sparse_mixture_config", _serializable(asdict(config.sparse_mixture))),
        ("resolution", asdict(config.resolution)),
    ):
        if checkpoint.get(name) != expected:
            raise ValueError(f"checkpoint {name} does not match config")

    cache_manifest = json.loads(
        arguments.training_cache_manifest.read_text(encoding="utf-8")
    )
    checkpoint_sha256 = sha256_file(arguments.checkpoint)
    if cache_manifest["checkpoint_sha256"].lower() != checkpoint_sha256.lower():
        raise ValueError("event models were not trained from this checkpoint cache")

    training_report_path = arguments.training_dir / "training_report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report["status"] != "rejected_component_stage":
        raise ValueError("expected completed event and component development training")
    event_contract = training_report["event_stage"]
    if not event_contract["gate"]["passed"]:
        raise ValueError("event postprocess did not pass its development gate")

    release_report_path = arguments.event_release_dir / "release_report.json"
    release_report = json.loads(release_report_path.read_text(encoding="utf-8"))
    expected_training_files = len(cache_manifest["files"])
    if release_report["training_files"] != expected_training_files:
        raise ValueError("event release did not use every training cache")
    event_contract = release_report["event_contract"]
    model_dir = arguments.event_release_dir / "events"
    for name, expected_sha256 in release_report["event_model_sha256"].items():
        if sha256_file(model_dir / name) != expected_sha256:
            raise ValueError(f"event release model hash mismatch: {name}")
    verifier_model = None
    verifier_report = None
    verifier_threshold = None
    if not arguments.event_only:
        if arguments.direct_verifier_dir is None:
            raise ValueError("direct verifier directory is required without --event-only")
        verifier_model, verifier_report = load_direct_verifier(
            arguments.direct_verifier_dir
        )
        if verifier_report["status"] != "released":
            raise ValueError("direct verifier did not pass its development gate")
        verifier_threshold = float(verifier_report["threshold"])
    validation_report = json.loads(arguments.validation_report.read_text(encoding="utf-8"))
    baseline_validation_report = json.loads(
        arguments.baseline_validation_report.read_text(encoding="utf-8")
    )
    if arguments.event_only:
        candidate = validation_report["event_release_dense_rescue"]
        baseline = validation_report["original"]
        validation_gate = {
            "score_gain": float(candidate["aggregate"]["score"])
            - float(baseline["aggregate"]["score"]),
            "val_023_pd_gain": float(candidate["files"]["val_023.npz"]["pd"])
            - float(baseline["files"]["val_023.npz"]["pd"]),
            "fa": float(candidate["aggregate"]["fa"]),
        }
        validation_gate["passed"] = (
            validation_gate["score_gain"] > 0
            and validation_gate["val_023_pd_gain"] >= 0
            and validation_gate["fa"] <= 6.0e-6
        )
        if not validation_gate["passed"]:
            raise ValueError(f"validation promotion gate failed: {validation_gate}")
    else:
        validation_gate = _validation_gate(
            validation_report, baseline_validation_report
        )
    if (
        validation_report["protocol"]["event_release_report_sha256"]
        != sha256_file(release_report_path)
    ):
        raise ValueError("validation report does not match the event release")
    verifier_report_path = None
    if not arguments.event_only:
        verifier_report_path = arguments.direct_verifier_dir / "training_report.json"
        if (
            validation_report["protocol"]["direct_verifier_report_sha256"]
            != sha256_file(verifier_report_path)
        ):
            raise ValueError("validation report does not match the direct verifier")

    model = build_model(config)
    load_result = model.load_state_dict(checkpoint["model_state"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("checkpoint did not strict-load")
    model.eval()
    event_models = load_event_models(model_dir)
    dataset = NPZEventDataset(
        arguments.test_dir,
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    output_dir = arguments.output_dir
    txt_dir = output_dir / "submission_txt"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=False)
    txt_dir.mkdir()
    prediction_dir.mkdir()

    files: list[dict[str, Any]] = []
    total_rows = 0
    total_positives = 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            sequence = dataset[index]
            events = sequence.events
            batch_indices = torch.zeros(len(events), dtype=torch.long)
            output, sparse_view = forward_with_sparse_mixture(
                model,
                sequence,
                events,
                batch_indices,
                config,
                mixture_weight=1.0,
            )
            with np.load(sequence.source_path, allow_pickle=False) as archive:
                locations = archive["ev_loc"].astype(np.int64, copy=True)
                raw_features = archive["evs_norm"][:, :4].astype(
                    np.float32, copy=True
                )
            probability = torch.sigmoid(output["event_logits"]).cpu().numpy()
            cache = EventCache(
                name=sequence.source_path.name,
                locations=locations,
                raw_features=raw_features,
                base_probability=probability,
                token_features=output["token_features"].cpu().numpy(),
                event_token_indices=output["event_token_indices"].cpu().numpy(),
                prediction=probability
                >= float(
                    checkpoint.get(
                        "decision_threshold", config.inference.sigmoid_threshold
                    )
                ),
                labels=None,
                target_ids=None,
            )
            scores = predict_event_scores(cache, event_models)
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
            if arguments.event_only:
                # 未通过开发门槛的直接验证器不能进入正式测试预测。
                prediction = event_prediction
            else:
                prediction = apply_direct_verifier(
                    candidate_batch,
                    tracked,
                    direct,
                    verifier_model,
                    verifier_threshold,
                    density_group=group,
                )
            reference_prediction = _load_named_prediction(
                arguments.reference_prediction_dir,
                sequence.source_path.name,
            )
            prediction, rescue_report = rescue_reference_tracks(
                cache.locations,
                prediction,
                reference_prediction,
                density_group=group,
            )
            np.savez_compressed(
                prediction_dir / sequence.source_path.name,
                prediction=prediction.astype(np.uint8),
            )
            txt_path = txt_dir / f"{sequence.source_path.stem}.txt"
            _write_track2_txt(sequence.source_path, txt_path, prediction)
            positives = int(prediction.sum())
            total_rows += len(prediction)
            total_positives += positives
            files.append(
                {
                    "source": sequence.source_path.name,
                    "rows": len(prediction),
                    "positives": positives,
                    "density_group": group,
                    "events_per_frame": density,
                    "sparse_view": sparse_view is not None,
                    "direct_gate": {
                        "candidate_components": int(len(candidate_batch.features)),
                        "candidate_events": int(np.sum(direct & ~tracked)),
                        "deleted_events": int(np.sum(event_prediction & ~prediction)),
                    },
                    "rescue": rescue_report,
                    "txt_sha256": sha256_file(txt_path),
                }
            )
            print(
                f"{index + 1:02d}/{len(dataset):02d} {txt_path.name}: "
                f"{len(prediction):,} rows, {positives:,} positives",
                flush=True,
            )

    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            archive.write(txt_path, arcname=txt_path.name)
    manifest = {
        "scheme": (
            f"schema{checkpoint['schema_version']}_event_release99_dense_rescue"
            if arguments.event_only
            else "schema5_event_release99_direct_temporal_gate_dense_rescue"
        ),
        "validation": validation_report[
            "event_release_dense_rescue"
            if arguments.event_only
            else "event_release_direct_gate_dense_rescue"
        ]["aggregate"],
        "validation_gate": validation_gate,
        "validation_report": str(arguments.validation_report.resolve()),
        "validation_report_sha256": sha256_file(arguments.validation_report),
        "baseline_validation_report": str(arguments.baseline_validation_report.resolve()),
        "baseline_validation_report_sha256": sha256_file(
            arguments.baseline_validation_report
        ),
        "config": str(arguments.config.resolve()),
        "config_sha256": sha256_file(arguments.config),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema_version": int(checkpoint["schema_version"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_report": str(training_report_path.resolve()),
        "training_report_sha256": sha256_file(training_report_path),
        "event_release_report": str(release_report_path.resolve()),
        "event_release_report_sha256": sha256_file(release_report_path),
        "event_model_sha256": release_report["event_model_sha256"],
        "event_release_training_files": release_report["training_files"],
        "direct_verifier_report": (
            str(verifier_report_path.resolve()) if verifier_report_path else None
        ),
        "direct_verifier_report_sha256": (
            sha256_file(verifier_report_path) if verifier_report_path else None
        ),
        "direct_verifier_model_sha256": (
            verifier_report["release_model"]["sha256"] if verifier_report else None
        ),
        "direct_verifier_threshold": verifier_threshold,
        "reference_prediction_dir": str(arguments.reference_prediction_dir.resolve()),
        "dense_rescue_contract": {
            "enabled_group": "dense",
            "minimum_track_frames": 8,
            "cluster_radius": 1,
            "max_component_span": 12,
            "max_gap": 2,
            "max_distance": 4.0,
            "half_width": 2,
            "half_height": 2,
            "extension_frames": 0,
        },
        "test_dir": str(arguments.test_dir.resolve()),
        "timestamp_format": "%.9f",
        "file_count": len(files),
        "total_rows": total_rows,
        "total_positives": total_positives,
        "submission_zip": str(zip_path.resolve()),
        "submission_zip_sha256": sha256_file(zip_path),
        "files": files,
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"submission ZIP: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
