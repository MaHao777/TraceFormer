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
from huav.postprocessing.learning import (
    blend_scores,
    load_event_models,
    predict_event_scores,
)
from huav.postprocessing.tracking import track_prediction
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用原模型与事件级后处理生成Track 2测试集提交"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--training-cache-manifest", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 5:
        raise ValueError("original event-postprocess submission requires schema 5")
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

    model = build_model(config)
    load_result = model.load_state_dict(checkpoint["model_state"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("checkpoint did not strict-load")
    model.eval()
    event_models = load_event_models(arguments.training_dir / "development" / "events")
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
            prediction = track_prediction(
                cache.locations,
                track_probability,
                **event_contract["tracking"][group],
            ) | (
                direct_probability >= float(event_contract["direct_threshold"])
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
        "scheme": "schema5_original_plus_event_postprocess",
        "validation_score": 0.946594016911914,
        "config": str(arguments.config.resolve()),
        "config_sha256": sha256_file(arguments.config),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema_version": 5,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_report": str(training_report_path.resolve()),
        "training_report_sha256": sha256_file(training_report_path),
        "event_model_sha256": {
            name: sha256_file(
                arguments.training_dir / "development" / "events" / name
            )
            for name in ("small.txt", "large.txt")
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
