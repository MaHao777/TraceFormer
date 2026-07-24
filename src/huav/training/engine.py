"""CPU-capable training, validation and artifact writing for the dual-view model."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import yaml

from huav.config import HUAVConfig
from huav.data import LoadedEventSequence, NPZEventDataset
from huav.models import DualViewEventClassifier
from huav.models.projection import build_view_targets

from .losses import DualViewLoss
from .metrics import EVUAVMetricAccumulator, EventMetricAccumulator
from .sampling import sample_event_indices


def build_model(config: HUAVConfig) -> DualViewEventClassifier:
    return DualViewEventClassifier(
        width=config.resolution.width,
        height=config.resolution.height,
        time_bins=tuple(config.model.time_bins),
        feature_channels=config.model.feature_channels,
        attention_heads=config.model.attention_heads,
        window_size=(config.model.window_time, config.model.window_space),
        cross_time_radius=config.model.cross_time_radius,
        dropout=config.model.dropout,
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


def _auxiliary_scale(epoch: int, *, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


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


@torch.no_grad()
def evaluate_model(
    model: DualViewEventClassifier,
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
    event_metrics = EventMetricAccumulator(threshold=threshold)
    event_metrics.update(logits, labels)
    official_metrics = EVUAVMetricAccumulator(
        width=config.resolution.width,
        height=config.resolution.height,
        frame_ms=config.evaluation.frame_ms,
        object_correct_fraction=config.evaluation.object_correct_fraction,
        threshold=threshold,
    )
    for sequence_logits, sequence in sequence_results:
        official_metrics.update(
            sequence_logits,
            sequence.labels,
            sequence.object_ids,
            sequence.timestamps_ms,
            sequence.events[:, :2],
        )
    return {
        "threshold": float(threshold),
        "event": event_metrics.compute(),
        "ev_uav_compatible": official_metrics.compute(),
        "validation_files": len(dataset),
    }


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    event = metrics["event"]
    official = metrics["ev_uav_compatible"]
    return "\n".join(
        (
            "# HUAV CPU Validation Metrics",
            "",
            f"- Decision threshold: `{metrics['threshold']:.6f}`",
            f"- Validation files: `{metrics['validation_files']}`",
            f"- Validation events: `{event['event_count']}`",
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
    criterion = DualViewLoss(
        view_event_weight=config.loss.view_event_weight,
        view_weight=config.loss.view_weight,
        consistency_weight=config.loss.consistency_weight,
        temporal_weight=config.loss.temporal_weight,
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
    history: list[dict[str, Any]] = []
    best_iou = -1.0
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
            "event": 0.0,
            "view_event": 0.0,
            "view": 0.0,
            "consistency": 0.0,
            "temporal": 0.0,
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
            )
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=torch_device
            )
            output = model(
                events,
                batch_indices,
                batch_size=1,
                query_indices=query_indices,
            )
            xt_target, yt_target = build_view_targets(
                events,
                labels,
                batch_indices,
                batch_size=1,
                width=config.resolution.width,
                height=config.resolution.height,
                time_bins=model.fine_time_bins,
                sigma=config.training.target_sigma,
            )
            auxiliary_scale = _auxiliary_scale(
                epoch,
                warmup_epochs=config.training.auxiliary_warmup_epochs,
                ramp_epochs=config.training.auxiliary_ramp_epochs,
            )
            losses = criterion(
                output,
                labels[query_indices],
                xt_target,
                yt_target,
                auxiliary_scale=auxiliary_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
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
        iou = float(metrics["event"]["iou"])
        print(
            f"epoch {epoch + 1}/{config.training.epochs}: "
            f"loss={totals['total']:.6f} Pd={metrics['event']['pd']:.6f} "
            f"Fa={metrics['event']['fa']:.6f} IoU={iou:.6f} "
            f"Acc={metrics['event']['acc']:.6f} threshold={metrics['threshold']:.3f}",
            flush=True,
        )
        if iou > best_iou:
            best_iou = iou
            best_metrics = metrics
            _atomic_torch_save(
                {
                    "schema_version": 1,
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
        "best_iou": best_iou,
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


__all__ = [
    "build_model",
    "_auxiliary_scale",
    "evaluate_checkpoint",
    "evaluate_model",
    "run_training",
]
