"""用指定TraceFormer检查点与事件级后处理生成Track 2提交包。"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.gpu.runtime import add_backend_arguments, validate_backends
from traceformer.postprocessing.cache import infer_event_cache
from traceformer.postprocessing.contracts import density_group, sha256_file
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
from traceformer.postprocessing.release import event_validation_gate, load_event_release
from traceformer.training.engine import load_inference_model


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


def _validation_gate(
    validation_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    candidate = validation_report.get(
        "event_release_direct_gate",
        validation_report.get("event_release"),
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


def _apply_optional_direct_verifier(
    cache, direct_probability, tracked, direct, model, threshold, *, group, event_only
):
    # 恒等分支不构造候选特征；dense 原验证器规则同样直接保留事件掩码。
    if event_only or group == "dense":
        return tracked | direct, None
    candidates = direct_candidate_features(
        cache, direct_probability, tracked, direct, include_targets=False
    )
    return apply_direct_verifier(
        candidates, tracked, direct, model, threshold, density_group=group
    ), candidates


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--baseline-validation-report", type=Path)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    add_backend_arguments(parser, trees=True)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if not arguments.event_only and arguments.baseline_validation_report is None:
        parser.error("baseline validation report is required without --event-only")
    validate_backends(prediction_backend=arguments.prediction_backend, feature_backend=arguments.feature_backend)

    config = load_config(arguments.config)
    torch.set_num_threads(config.training.num_threads)
    # 所有推理入口共用完整结构检查，包括精修头及消融契约。
    model, checkpoint = load_inference_model(
        config, arguments.checkpoint, device=torch.device(arguments.device),
    )

    cache_manifest = json.loads(
        arguments.training_cache_manifest.read_text(encoding="utf-8")
    )
    checkpoint_sha256 = sha256_file(arguments.checkpoint)
    if cache_manifest["checkpoint_sha256"].lower() != checkpoint_sha256.lower():
        raise ValueError("event models were not trained from this checkpoint cache")

    training_report_path = arguments.training_dir / "training_report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
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
    baseline_path = arguments.baseline_validation_report or arguments.validation_report
    baseline_validation_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    if arguments.event_only:
        candidate = validation_report["event_release"]
        baseline = validation_report["original"]
        validation_gate = event_validation_gate(candidate, baseline)
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

    if arguments.event_only:
        event_models, release_report = load_event_release(
            arguments.event_release_dir, checkpoint=arguments.checkpoint,
            config=arguments.config, backend=arguments.prediction_backend,
        )
    else:
        event_models = load_event_models(model_dir, names=required_event_models(event_contract),
                                        backend=arguments.prediction_backend)
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
            cache = infer_event_cache(
                model, config, sequence, checkpoint["decision_threshold"],
            )
            scores = predict_event_scores(
                cache, event_models, num_threads=arguments.prediction_threads,
                feature_backend=arguments.feature_backend,
            )
            group, density = density_group(cache.locations)
            _, direct_probability, tracked, direct = event_branches(
                cache, scores, event_contract
            )
            event_prediction = tracked | direct
            prediction, candidate_batch = _apply_optional_direct_verifier(
                cache, direct_probability, tracked, direct, verifier_model, verifier_threshold,
                group=group, event_only=arguments.event_only,
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
                    "direct_gate": {
                        "executed": candidate_batch is not None,
                        "candidate_components": (
                            int(len(candidate_batch.features)) if candidate_batch is not None else 0
                        ),
                        "candidate_events": int(np.sum(direct & ~tracked)),
                        "deleted_events": int(np.sum(event_prediction & ~prediction)),
                    },
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
            f"schema{checkpoint['schema_version']}_event_release99"
            if arguments.event_only
            else "schema5_event_release99_direct_temporal_gate"
        ),
        "validation": validation_report[
            "event_release"
            if arguments.event_only
            else "event_release_direct_gate"
        ]["aggregate"],
        "validation_gate": validation_gate,
        "validation_report": str(arguments.validation_report.resolve()),
        "validation_report_sha256": sha256_file(arguments.validation_report),
        "baseline_validation_report": str(baseline_path.resolve()),
        "baseline_validation_report_sha256": sha256_file(baseline_path),
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
        "prediction_threads": arguments.prediction_threads,
        "device": arguments.device,
        "prediction_backend": arguments.prediction_backend,
        "feature_backend": arguments.feature_backend,
        "active_event_models": list(event_models),
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
