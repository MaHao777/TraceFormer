"""CPU/GPU training, validation and artifacts for the sparse cone model."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from huav.config import HUAVConfig
from huav.data import EventSequence, LoadedEventSequence, NPZEventDataset
from huav.inference import (
    generate_submission_rows,
    pack_submission_zip,
    write_prediction_txt,
)
from huav.models import SpatioTemporalConeDetector

from .losses import EventFocalDiceLoss
from .metrics import (
    EventMetricAccumulator,
    EVUAVMetricAccumulator,
    competition_metrics,
)
from .sampling import sample_event_indices


def build_model(config: HUAVConfig) -> SpatioTemporalConeDetector:
    return SpatioTemporalConeDetector(
        width=config.resolution.width,
        height=config.resolution.height,
        spatial_block_size_x=config.model.spatial_block_size_x,
        spatial_block_size_y=config.model.spatial_block_size_y,
        num_time_blocks=config.model.num_time_blocks,
        token_dim=config.model.token_dim,
        event_embed_dim=config.model.event_embed_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        cone_heads=tuple(config.model.cone_heads),
        layer_time_scales=tuple(config.model.layer_time_scales),
        max_neighbors_per_cone=config.model.max_neighbors_per_cone,
        causal_mode=config.model.causal_mode,
        direction_dropout=config.model.direction_dropout,
        debug_attention_stats=config.model.debug_attention_stats,
    )


def _dataset(config: HUAVConfig, split: str) -> NPZEventDataset:
    limit = (
        config.training.max_train_files
        if split == "train"
        else config.training.max_val_files
    )
    return NPZEventDataset(
        Path(config.dataset.root) / split,
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
        limit_files=limit,
    )


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_threshold(
    logits: torch.Tensor, labels: torch.Tensor, *, steps: int, fallback: float
) -> float:
    candidates = torch.linspace(0.05, 0.95, steps=steps)
    candidates = torch.unique(torch.cat((candidates, torch.tensor([fallback]))))
    probabilities = torch.sigmoid(logits)
    target = labels > 0.5
    best_threshold = float(fallback)
    best_iou = -1.0
    for candidate in candidates.tolist():
        prediction = probabilities >= candidate
        intersection = int(torch.sum(prediction & target).item())
        union = int(torch.sum(prediction | target).item())
        iou = intersection / union if union else 0.0
        if iou > best_iou or (
            iou == best_iou
            and abs(candidate - fallback) < abs(best_threshold - fallback)
        ):
            best_iou = iou
            best_threshold = float(candidate)
    return best_threshold


def _amp_is_enabled(requested: bool, device: torch.device) -> bool:
    return bool(requested and device.type == "cuda")


@torch.no_grad()
def evaluate_model(
    model: SpatioTemporalConeDetector,
    dataset: NPZEventDataset,
    config: HUAVConfig,
    *,
    device: torch.device,
    progress: bool = False,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    model.eval()
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for index in range(len(dataset)):
        sequence = dataset[index]
        events = sequence.events.to(device)
        output = model(events)
        logits = output["event_logits"].detach().cpu()
        sequence_results.append((logits, sequence))
        all_logits.append(logits)
        all_labels.append(sequence.labels)
        if progress:
            print(
                f"  validation {index + 1:03d}/{len(dataset):03d} "
                f"{sequence.source_path.name}: {sequence.length:,} events",
                flush=True,
            )

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else config.inference.sigmoid_threshold
    )
    if threshold_override is None and config.evaluation.optimize_threshold:
        threshold = _select_threshold(
            logits,
            labels,
            steps=config.evaluation.threshold_steps,
            fallback=threshold,
        )
    event_accumulator = EventMetricAccumulator(threshold=threshold)
    event_accumulator.update(logits, labels)
    object_accumulator = EVUAVMetricAccumulator(
        width=config.resolution.width,
        height=config.resolution.height,
        frame_ms=config.evaluation.frame_ms,
        object_correct_fraction=config.evaluation.object_correct_fraction,
        threshold=threshold,
    )
    for sequence_logits, sequence in sequence_results:
        object_accumulator.update(
            sequence_logits,
            sequence.labels,
            sequence.object_ids,
            sequence.timestamps_ms,
            sequence.events[:, :2],
        )
    event = event_accumulator.compute()
    object_metrics = object_accumulator.compute()
    return {
        "threshold": float(threshold),
        "competition": competition_metrics(
            event,
            object_metrics,
            width=config.resolution.width,
            height=config.resolution.height,
        ),
        "event": event,
        "ev_uav_compatible": object_metrics,
        "validation_files": len(dataset),
    }


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    competition = metrics["competition"]
    counts = competition["counts"]
    event = metrics["event"]
    official = metrics["ev_uav_compatible"]
    return "\n".join(
        (
            "# HUAV Spatiotemporal-Cone Validation Metrics",
            "",
            f"- Decision threshold: `{metrics['threshold']:.6f}`",
            f"- Validation files: `{metrics['validation_files']}`",
            f"- Validation events: `{event['event_count']}`",
            "",
            "## Official JinSight competition score",
            "",
            "| Metric | Value | Definition |",
            "|---|---:|---|",
            f"| Score | {competition['score']:.12f} | Official weighted total |",
            f"| IoU | {competition['iou']:.12f} | Intersection / union |",
            f"| SegAcc | {competition['seg_acc']:.12f} | TP / GT-positive events |",
            f"| Pd | {competition['pd']:.12f} | Detected / object-frame instances |",
            f"| Fa | {competition['fa']:.12e} | False components / frame pixels |",
            f"| ScoreFa | {competition['score_fa']:.12f} | exp(-10000 * Fa) |",
            "",
            "`Score = 0.2 * IoU + 0.1 * SegAcc + 0.4 * Pd + 0.3 * ScoreFa`",
            "",
            f"- Counts: intersection=`{counts['intersection']}`, union=`{counts['union']}`, "
            f"true_positive=`{counts['true_positive']}`, "
            f"gt_positive=`{counts['gt_positive']}`",
            f"- Counts: detected_objects=`{counts['detected_objects']}`, "
            f"objects=`{counts['objects']}`, false_alarms=`{counts['false_alarms']}`, "
            f"frames=`{counts['frames']}`",
            "- Official 50 ms framing uses floor(t_ms) and strict open intervals.",
            "",
            "## Event-level diagnostics",
            "",
            "| Metric | Value | Definition |",
            "|---|---:|---|",
            f"| Pd | {event['pd']:.8f} | TP / (TP + FN), event-level |",
            f"| Fa | {event['fa']:.8f} | FP / (FP + TN), event-level |",
            f"| IoU | {event['iou']:.8f} | TP / (TP + FP + FN), positive class |",
            f"| Acc | {event['acc']:.8f} | (TP + TN) / all events |",
            "",
            "## EV-UAV-compatible protocol",
            "",
            f"- Object Pd: `{official['pd']:.8f}`",
            f"- Spatial-component Fa: `{official['fa']:.10f}`",
            f"- Detected/object instances: "
            f"`{official['detected_objects']}/{official['object_instances']}`",
            f"- False components / equivalent frames: "
            f"`{official['false_components']}/{official['frame_count']}`",
            "",
            "The compatibility values follow the sibling EV-UAV reference: "
            "50 ms temporal frames, object-level Pd, and false connected components "
            "normalized by frame count and sensor area.",
            "",
        )
    )


def _run_directory(
    config: HUAVConfig, output_dir: str | Path | None, *, device_name: str
) -> Path:
    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(config.training.output_root) / f"{device_name}_{timestamp}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dataset_manifest(*datasets: NPZEventDataset) -> dict[str, Any]:
    files = []
    for dataset in datasets:
        for path in dataset.files:
            stat = path.stat()
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append(
                {
                    "split": dataset.root.name,
                    "path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                    "sha256": digest.hexdigest(),
                }
            )
    return {"schema": "huav-structured-ev-v1", "files": files}


def _manifest_identity_digest(manifest: dict[str, Any]) -> str:
    identity = {
        "schema": manifest["schema"],
        "files": [
            {
                "split": item["split"],
                "name": Path(item["path"]).name,
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in manifest["files"]
        ],
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def run_training(
    config: HUAVConfig,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    config.validate()
    device_name = device or config.training.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch_device = torch.device(device_name)
    torch.set_num_threads(config.training.num_threads)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    run_dir = _run_directory(config, output_dir, device_name=torch_device.type)
    configuration = _serializable(asdict(config))
    _atomic_write_text(
        run_dir / "config.yaml",
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
    )
    train_dataset = _dataset(config, "train")
    validation_dataset = _dataset(config, "val")
    manifest = _dataset_manifest(train_dataset, validation_dataset)
    validation_manifest = {
        "schema": manifest["schema"],
        "files": [
            item
            for item in manifest["files"]
            if item["split"] == validation_dataset.root.name
        ],
    }
    validation_manifest_digest = _manifest_identity_digest(validation_manifest)
    _write_json(run_dir / "dataset_manifest.json", manifest)
    model = build_model(config).to(torch_device)
    criterion = EventFocalDiceLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        lambda_dice=config.loss.lambda_dice,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    order_generator = torch.Generator().manual_seed(config.training.seed)
    sampling_generator = torch.Generator(device=torch_device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, torch_device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None

    print(
        f"HUAV training on {torch_device.type}: {len(train_dataset)} train files, "
        f"{len(validation_dataset)} validation files, {config.training.epochs} epochs",
        flush=True,
    )
    for epoch in range(config.training.epochs):
        model.train()
        totals = {
            "total": 0.0,
            "focal": 0.0,
            "dice": 0.0,
        }
        order = torch.randperm(
            len(train_dataset), generator=order_generator
        ).tolist()
        for step, dataset_index in enumerate(order):
            sequence = train_dataset[dataset_index]
            events = sequence.events.to(torch_device)
            labels = sequence.labels.to(torch_device)
            query_indices = sample_event_indices(
                labels,
                background_ratio=config.training.background_ratio,
                generator=sampling_generator,
                max_background_without_positive=(
                    config.training.max_background_without_positive
                ),
            )
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=torch_device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(events, batch_indices)
                losses = criterion(
                    output["event_logits"][query_indices],
                    labels[query_indices],
                    batch_indices[query_indices],
                )
            if amp_enabled:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())
            if progress:
                print(
                    f"  epoch {epoch + 1}/{config.training.epochs} "
                    f"train {step + 1:03d}/{len(order):03d} "
                    f"loss={losses['total'].item():.5f} "
                    f"queries={len(query_indices):,}/{sequence.length:,}",
                    flush=True,
                )

        for name in totals:
            totals[name] /= len(train_dataset)
        metrics = evaluate_model(
            model,
            validation_dataset,
            config,
            device=torch_device,
            progress=progress,
        )
        epoch_record = {"epoch": epoch + 1, "train_loss": totals, "validation": metrics}
        history.append(epoch_record)
        _write_json(run_dir / "history.json", history)
        competition = metrics["competition"]
        score = float(competition["score"])
        print(
            f"epoch {epoch + 1}/{config.training.epochs}: "
            f"loss={totals['total']:.6f} Score={score:.6f} "
            f"IoU={competition['iou']:.6f} SegAcc={competition['seg_acc']:.6f} "
            f"Pd={competition['pd']:.6f} Fa={competition['fa']:.3e} "
            f"threshold={metrics['threshold']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            _atomic_torch_save(
                {
                    "schema_version": 2,
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "decision_threshold": float(metrics["threshold"]),
                    "model_config": configuration["model"],
                    "resolution": configuration["resolution"],
                    "validation_manifest_sha256": validation_manifest_digest,
                },
                run_dir / "best.pt",
            )
            _write_json(run_dir / "metrics.json", best_metrics)
            _atomic_write_text(
                run_dir / "metrics.md", _metrics_markdown(best_metrics)
            )

    if best_metrics is None:
        raise RuntimeError("training completed without validation metrics")
    summary = {
        "run_dir": str(run_dir.resolve()),
        "device": torch_device.type,
        "epochs": config.training.epochs,
        "mixed_precision": amp_enabled,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_score": best_score,
        "best_iou": float(best_metrics["competition"]["iou"]),
        "metrics": best_metrics,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def evaluate_checkpoint(
    config: HUAVConfig,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    progress: bool = False,
) -> dict[str, Any]:
    config.validate()
    torch_device = torch.device(device)
    torch.set_num_threads(config.training.num_threads)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    if checkpoint.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema version")
    expected_model = _serializable(asdict(config.model))
    expected_resolution = _serializable(asdict(config.resolution))
    if checkpoint.get("model_config") != expected_model:
        raise ValueError("checkpoint model configuration does not match the supplied config")
    if checkpoint.get("resolution") != expected_resolution:
        raise ValueError("checkpoint resolution does not match the supplied config")
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    if checkpoint.get("validation_manifest_sha256") != actual_validation_digest:
        raise ValueError("checkpoint validation dataset manifest does not match current data")
    model = build_model(config).to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = checkpoint.get("decision_threshold")
    return evaluate_model(
        model,
        validation_dataset,
        config,
        device=torch_device,
        progress=progress,
        threshold_override=threshold,
    )


def _loaded_to_event_sequence(
    loaded: LoadedEventSequence,
    config: HUAVConfig,
) -> EventSequence:
    """Bridge from the pipeline's *LoadedEventSequence* to *EventSequence* for submission."""
    return EventSequence(
        x=loaded.events[:, 0].round().to(torch.int64).numpy(),
        y=loaded.events[:, 1].round().to(torch.int64).numpy(),
        t=loaded.timestamps_ms.numpy(),
        p=loaded.events[:, 3].round().to(torch.int64).numpy(),
        width=config.resolution.width,
        height=config.resolution.height,
        source_path=loaded.source_path,
    )


def generate_submission_package(
    config: HUAVConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    progress: bool = False,
) -> Path:
    """Load a trained checkpoint and produce a competition-ready submission ZIP.

    Steps
    -----
    1. Validate *config* and the checkpoint metadata.
    2. Run inference on every file in the validation set.
    3. For each sequence write a ``val_XXX.txt`` with the five-column format
       ``x y t p predicted_label``.
    4. Bundle all TXT files into ``submission.zip``.

    Returns the path of the created ZIP archive.
    """
    config.validate()
    torch_device = torch.device(device)
    torch.set_num_threads(config.training.num_threads)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    if checkpoint.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema version")
    expected_model = _serializable(asdict(config.model))
    expected_resolution = _serializable(asdict(config.resolution))
    if checkpoint.get("model_config") != expected_model:
        raise ValueError("checkpoint model configuration does not match the supplied config")
    if checkpoint.get("resolution") != expected_resolution:
        raise ValueError("checkpoint resolution does not match the supplied config")
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    if checkpoint.get("validation_manifest_sha256") != actual_validation_digest:
        raise ValueError("checkpoint validation dataset manifest does not match current data")
    model = build_model(config).to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint.get("decision_threshold", config.inference.sigmoid_threshold))
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(validation_dataset)):
        sequence = validation_dataset[index]
        events = sequence.events.to(torch_device)
        with torch.no_grad():
            output = model(events)
        logits = output["event_logits"].detach().cpu()
        event_sequence = _loaded_to_event_sequence(sequence, config)
        rows = generate_submission_rows(event_sequence, logits, threshold)
        txt_name = f"{event_sequence.source_stem}.txt"
        write_prediction_txt(rows, output_dir, txt_name)
        if progress:
            print(
                f"  {index + 1:03d}/{len(validation_dataset):03d} "
                f"{txt_name}: {sequence.length:,} events",
                flush=True,
            )

    zip_path = output_dir / "submission.zip"
    pack_submission_zip(output_dir, zip_path)
    return zip_path


__all__ = [
    "_amp_is_enabled",
    "build_model",
    "evaluate_checkpoint",
    "evaluate_model",
    "run_training",
]
