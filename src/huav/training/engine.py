"""CPU/GPU training, validation and artifacts for the sparse cone model."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from huav.config import HUAVConfig, ModelConfig
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
from .runtime import prepare_device
from .sampling import sample_event_indices

CHECKPOINT_SCHEMA_VERSION = 5
LEGACY_READ_ONLY_SCHEMA_VERSION = 4
ADAPTIVE_MODEL_FIELDS = {
    "adaptive_threshold_mode",
    "threshold_hidden_dim",
    "threshold_time_bins",
    "threshold_max_global_offset",
    "threshold_max_local_offset",
}
LOSS_NAMES = (
    "total",
    "focal",
    "dice",
    "object_frame",
    "false_alarm",
    "threshold_magnitude",
    "threshold_center",
    "threshold_smoothness",
)


def build_model(config: HUAVConfig) -> SpatioTemporalConeDetector:
    return SpatioTemporalConeDetector(
        width=config.resolution.width,
        height=config.resolution.height,
        spatial_block_size_x=config.model.spatial_block_size_x,
        spatial_block_size_y=config.model.spatial_block_size_y,
        num_time_blocks=config.model.num_time_blocks,
        token_dim=config.model.token_dim,
        event_embed_dim=config.model.event_embed_dim,
        num_pool_queries=config.model.num_pool_queries,
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
        adaptive_threshold_mode=config.model.adaptive_threshold_mode,
        threshold_hidden_dim=config.model.threshold_hidden_dim,
        threshold_time_bins=config.model.threshold_time_bins,
        threshold_max_global_offset=config.model.threshold_max_global_offset,
        threshold_max_local_offset=config.model.threshold_max_local_offset,
    )


def _configure_training_stage(
    model: SpatioTemporalConeDetector,
    training_stage: str,
) -> int:
    """Set module modes and gradients for one training epoch."""
    model.train()
    if training_stage == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif training_stage == "calibrator":
        if model.calibrator is None:
            raise ValueError("calibrator training requires an adaptive calibrator")
        for module in (model.tokenizer, model.encoder, model.decoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model.calibrator.train()
        for parameter in model.calibrator.parameters():
            parameter.requires_grad_(True)
    elif training_stage == "joint":
        if model.calibrator is None:
            raise ValueError("joint training requires an adaptive calibrator")
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unsupported training stage: {training_stage}")
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _make_optimizer(
    model: SpatioTemporalConeDetector,
    config: HUAVConfig,
) -> torch.optim.AdamW:
    training = config.training
    if training.training_stage == "joint":
        if model.calibrator is None or training.detector_learning_rate is None:
            raise ValueError("joint training requires detector and calibrator")
        detector_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("calibrator.") and parameter.requires_grad
        ]
        calibrator_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("calibrator.") and parameter.requires_grad
        ]
        groups = [
            {
                "params": detector_parameters,
                "lr": training.detector_learning_rate,
                "name": "detector",
            },
            {
                "params": calibrator_parameters,
                "lr": training.learning_rate,
                "name": "calibrator",
            },
        ]
    else:
        groups = [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                "lr": training.learning_rate,
                "name": training.training_stage,
            }
        ]
    return torch.optim.AdamW(
        groups,
        weight_decay=training.weight_decay,
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        progress = min(step, epochs) / max(1, epochs)
        return minimum_ratio + (1.0 - minimum_ratio) * (
            1.0 + math.cos(math.pi * progress)
        ) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _criterion(config: HUAVConfig) -> EventFocalDiceLoss:
    return EventFocalDiceLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        lambda_dice=config.loss.lambda_dice,
        lambda_object_frame=config.loss.lambda_object_frame,
        object_frame_logit_margin=config.loss.object_frame_logit_margin,
        lambda_false_alarm=config.loss.lambda_false_alarm,
        false_alarm_logit_margin=config.loss.false_alarm_logit_margin,
        false_alarm_topk_per_frame=config.loss.false_alarm_topk_per_frame,
        lambda_threshold_magnitude=config.loss.lambda_threshold_magnitude,
        lambda_threshold_smoothness=config.loss.lambda_threshold_smoothness,
        lambda_threshold_center=config.loss.lambda_threshold_center,
    )


def _threshold_loss_inputs(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "global_threshold_offsets": output["global_threshold_offsets"],
        "local_threshold_offsets": output["local_threshold_offsets"],
        "threshold_group_indices": output["threshold_group_indices"],
        "threshold_group_batch_indices": output[
            "threshold_group_batch_indices"
        ],
        "threshold_neighbor_edges": output["threshold_neighbor_edges"],
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    loaded = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint must be a mapping")
    return dict(loaded)


def _checkpoint_model_state(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError(
            "checkpoint model_state must map string names to tensors"
        )
    return dict(state)


def _normalized_legacy_model_config(model_config: object) -> dict[str, Any]:
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    normalized = dict(model_config)
    defaults = _serializable(asdict(ModelConfig()))
    for name in ADAPTIVE_MODEL_FIELDS:
        normalized.setdefault(name, defaults[name])
    return normalized


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    config: HUAVConfig,
    validation_manifest_digest: str,
    *,
    allow_legacy_read_only: bool,
) -> None:
    schema = checkpoint.get("schema_version")
    expected_model = _serializable(asdict(config.model))
    if schema == CHECKPOINT_SCHEMA_VERSION:
        _validate_schema5_metadata(checkpoint)
        actual_model = checkpoint.get("model_config")
    elif schema == LEGACY_READ_ONLY_SCHEMA_VERSION and allow_legacy_read_only:
        if config.model.adaptive_threshold_mode != "none":
            raise ValueError(
                "schema 4 checkpoints are supported read-only only in none mode"
            )
        actual_model = _normalized_legacy_model_config(
            checkpoint.get("model_config")
        )
    else:
        raise ValueError("unsupported checkpoint schema version")
    if schema == LEGACY_READ_ONLY_SCHEMA_VERSION:
        _checkpoint_model_state(checkpoint)
    _stored_decision_threshold(checkpoint)
    if actual_model != expected_model:
        raise ValueError(
            "checkpoint model configuration does not match the supplied config"
        )
    if checkpoint.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError(
            "checkpoint resolution does not match the supplied config"
        )
    if checkpoint.get("validation_manifest_sha256") != validation_manifest_digest:
        raise ValueError(
            "checkpoint validation dataset manifest does not match current data"
        )


def _stored_decision_threshold(checkpoint: dict[str, Any]) -> float:
    threshold = checkpoint.get("decision_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError(
            "checkpoint decision_threshold must be finite and in [0,1]"
        )
    return float(threshold)


def _validate_provenance_pair(
    checkpoint: Mapping[str, Any],
    path_key: str,
    hash_key: str,
) -> bool:
    path_present = path_key in checkpoint
    hash_present = hash_key in checkpoint
    if path_present != hash_present:
        raise ValueError(
            f"checkpoint {path_key} and {hash_key} must appear together"
        )
    if not path_present:
        return False
    source = checkpoint[path_key]
    source_hash = checkpoint[hash_key]
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"checkpoint {path_key} must be non-empty")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in source_hash
        )
    ):
        raise ValueError(f"checkpoint {hash_key} must be 64 hex characters")
    return True


def _expected_trainable_state_count(
    state: Mapping[str, torch.Tensor],
    training_stage: str,
) -> int:
    if training_stage == "calibrator":
        return sum(
            value.numel()
            for name, value in state.items()
            if name.startswith("calibrator.")
        )
    return sum(value.numel() for value in state.values())


def _validate_schema5_metadata(checkpoint: dict[str, Any]) -> None:
    state = _checkpoint_model_state(checkpoint)
    training_stage = checkpoint.get("training_stage")
    if (
        not isinstance(training_stage, str)
        or training_stage not in {"full", "calibrator", "joint"}
    ):
        raise ValueError(
            "checkpoint training_stage must be full, calibrator, or joint"
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config must be a mapping")
    adaptive_mode = model_config.get("adaptive_threshold_mode")
    if adaptive_mode not in {
        "none",
        "global",
        "local",
        "local_neighbor",
        "hierarchical",
    }:
        raise ValueError(
            "checkpoint adaptive_threshold_mode is missing or invalid"
        )
    if (
        training_stage in {"calibrator", "joint"}
        and adaptive_mode == "none"
    ):
        raise ValueError(
            "checkpoint calibrator/joint stage requires adaptive thresholding"
        )
    trainable_count = checkpoint.get("trainable_parameter_count")
    if (
        isinstance(trainable_count, bool)
        or not isinstance(trainable_count, Integral)
        or int(trainable_count) <= 0
    ):
        raise ValueError(
            "checkpoint trainable_parameter_count must be a positive integer"
        )
    expected_count = _expected_trainable_state_count(state, training_stage)
    if int(trainable_count) != expected_count:
        raise ValueError(
            "checkpoint trainable_parameter_count does not match model_state"
        )
    source_present = _validate_provenance_pair(
        checkpoint,
        "source_checkpoint",
        "source_checkpoint_sha256",
    )
    _validate_provenance_pair(
        checkpoint,
        "resumed_from_checkpoint",
        "resumed_from_sha256",
    )
    if training_stage in {"calibrator", "joint"} and not source_present:
        raise ValueError(
            "calibrator/joint checkpoint requires source checkpoint provenance"
        )
    if training_stage == "full" and source_present:
        raise ValueError(
            "full checkpoint must not contain source checkpoint provenance"
        )


def _base_model_config(model_config: object) -> dict[str, Any]:
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    return {
        key: value
        for key, value in model_config.items()
        if key not in ADAPTIVE_MODEL_FIELDS
    }


def _initialize_detector_weights(
    model: SpatioTemporalConeDetector,
    config: HUAVConfig,
    checkpoint_path: Path,
    validation_manifest_digest: str,
    *,
    device: torch.device,
) -> dict[str, str]:
    if model.calibrator is None:
        raise ValueError(
            "initialize_detector requires an adaptive target model"
        )
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if checkpoint.get("schema_version") not in {
        LEGACY_READ_ONLY_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported checkpoint schema version")
    if checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        _validate_schema5_metadata(checkpoint)
        _stored_decision_threshold(checkpoint)
    source_model_config = _normalized_legacy_model_config(
        checkpoint.get("model_config")
    )
    if source_model_config["adaptive_threshold_mode"] != "none":
        raise ValueError(
            "detector initialization requires a non-adaptive source checkpoint"
        )
    if (
        _base_model_config(checkpoint.get("model_config"))
        != _base_model_config(_serializable(asdict(config.model)))
    ):
        raise ValueError(
            "detector checkpoint model configuration does not match target"
        )
    if checkpoint.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError("detector checkpoint resolution does not match target")
    if checkpoint.get("validation_manifest_sha256") != validation_manifest_digest:
        raise ValueError(
            "detector checkpoint validation dataset manifest does not match target"
        )
    source_state = _checkpoint_model_state(checkpoint)
    if any(name.startswith("calibrator.") for name in source_state):
        raise ValueError(
            "detector initialization requires a non-adaptive source state"
        )
    target_state = model.state_dict()
    target_detector_keys = {
        name for name in target_state if not name.startswith("calibrator.")
    }
    source_detector_state = {
        name: value
        for name, value in source_state.items()
        if not name.startswith("calibrator.")
    }
    source_detector_keys = set(source_detector_state)
    if source_detector_keys != target_detector_keys:
        missing = sorted(target_detector_keys - source_detector_keys)
        unexpected = sorted(source_detector_keys - target_detector_keys)
        raise ValueError(
            "detector checkpoint keys do not match target: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatched_shapes = sorted(
        name
        for name, value in source_detector_state.items()
        if not isinstance(value, torch.Tensor)
        or value.shape != target_state[name].shape
    )
    if mismatched_shapes:
        raise ValueError(
            "detector checkpoint tensor shape mismatch: "
            f"{mismatched_shapes}"
        )
    incompatible = model.load_state_dict(source_detector_state, strict=False)
    expected_missing = sorted(
        name for name in target_state if name.startswith("calibrator.")
    )
    if (
        sorted(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "detector initialization must leave exactly calibrator parameters "
            "uninitialized"
        )
    return {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
    }


def _offset_statistics(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flattened.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
        }
    absolute = flattened.abs()
    return {
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "p95_abs": float(torch.quantile(absolute, 0.95).item()),
        "max_abs": float(absolute.max().item()),
    }


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


def _metrics_at_threshold(
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]],
    config: HUAVConfig,
    threshold: float,
) -> dict[str, Any]:
    event_accumulator = EventMetricAccumulator(threshold=threshold)
    object_accumulator = EVUAVMetricAccumulator(
        width=config.resolution.width,
        height=config.resolution.height,
        frame_ms=config.evaluation.frame_ms,
        object_correct_fraction=config.evaluation.object_correct_fraction,
        threshold=threshold,
    )
    for sequence_logits, sequence in sequence_results:
        event_accumulator.update(sequence_logits, sequence.labels)
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
        "validation_files": len(sequence_results),
    }


def _select_competition_threshold(
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]],
    config: HUAVConfig,
    *,
    steps: int,
    fallback: float,
) -> dict[str, Any]:
    coarse = torch.linspace(0.05, 0.95, steps=steps)
    coarse = torch.unique(torch.cat((coarse, torch.tensor([fallback]))))
    coarse_step = 0.9 / max(1, steps - 1)
    best_metrics: dict[str, Any] | None = None

    def consider(candidate: float) -> None:
        nonlocal best_metrics
        metrics = _metrics_at_threshold(sequence_results, config, candidate)
        if best_metrics is None:
            best_metrics = metrics
            return
        score = float(metrics["competition"]["score"])
        best_score = float(best_metrics["competition"]["score"])
        if score > best_score or (
            score == best_score
            and abs(candidate - fallback)
            < abs(float(best_metrics["threshold"]) - fallback)
        ):
            best_metrics = metrics

    for candidate in coarse.tolist():
        consider(float(candidate))
    if best_metrics is None:
        raise RuntimeError("threshold search produced no candidates")
    center = float(best_metrics["threshold"])
    lower = max(0.01, center - coarse_step)
    upper = min(0.99, center + coarse_step)
    refined = torch.linspace(lower, upper, steps=9)
    for candidate in refined.tolist():
        consider(float(candidate))
    if best_metrics is None:
        raise RuntimeError("threshold search produced no metrics")
    return best_metrics


def _amp_is_enabled(requested: bool, device: torch.device) -> bool:
    return bool(requested and device.type == "cuda")


def _autocast_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("mixed precision dtype must be 'float16' or 'bfloat16'")


def _grad_scaler_is_enabled(amp_enabled: bool, dtype: torch.dtype) -> bool:
    return bool(amp_enabled and dtype == torch.float16)


def _dense_sequence_loss_weight(
    events: torch.Tensor,
    labels: torch.Tensor,
    config: HUAVConfig,
) -> float:
    training = config.training
    if (
        len(events) < training.dense_sequence_event_threshold
        or training.dense_sequence_weight <= 1
    ):
        return 1.0
    background = labels <= 0.5
    if not bool(background.any()):
        return 1.0
    background_polarity = float(events[background, 3].float().mean().item())
    lower = 1.0 - training.dense_sequence_background_polarity_max
    upper = training.dense_sequence_background_polarity_max
    if lower <= background_polarity <= upper:
        return float(training.dense_sequence_weight)
    return 1.0


@torch.no_grad()
def evaluate_model(
    model: SpatioTemporalConeDetector,
    dataset: NPZEventDataset,
    config: HUAVConfig,
    *,
    device: torch.device,
    progress: bool = False,
    threshold_override: float | None = None,
    criterion: EventFocalDiceLoss | None = None,
    force_threshold_search: bool = False,
) -> dict[str, Any]:
    model.eval()
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]] = []
    global_offsets: list[torch.Tensor] = []
    local_offsets: list[torch.Tensor] = []
    per_sequence_global_mean: list[dict[str, Any]] = []
    loss_totals = (
        {name: 0.0 for name in LOSS_NAMES}
        if criterion is not None
        else None
    )
    validation_generator = torch.Generator(device=device.type).manual_seed(
        config.training.seed
    )
    for index in range(len(dataset)):
        sequence = dataset[index]
        events = sequence.events.to(device)
        labels = sequence.labels.to(device)
        object_ids = sequence.object_ids.to(device)
        timestamps_ms = sequence.timestamps_ms.to(device)
        batch_indices = torch.zeros(
            len(events), dtype=torch.long, device=device
        )
        output = model(events, batch_indices)
        if criterion is not None and loss_totals is not None:
            query_indices = sample_event_indices(
                labels,
                logits=output["event_logits"],
                background_ratio=config.training.background_ratio,
                hard_negative_fraction=config.training.hard_negative_fraction,
                generator=validation_generator,
                max_background_without_positive=(
                    config.training.max_background_without_positive
                ),
            )
            losses = criterion(
                output["event_logits"][query_indices],
                labels[query_indices],
                batch_indices[query_indices],
                object_ids=object_ids[query_indices],
                timestamps_ms=timestamps_ms[query_indices],
                frame_ms=config.evaluation.frame_ms,
                full_logits=output["event_logits"],
                full_targets=labels,
                full_batch_indices=batch_indices,
                full_timestamps_ms=timestamps_ms,
                full_coordinates=events[:, :2],
                **_threshold_loss_inputs(output),
            )
            losses["total"] = losses["total"] * _dense_sequence_loss_weight(
                events,
                labels,
                config,
            )
            for name in loss_totals:
                loss_totals[name] += float(losses[name].detach().item())
        logits = output["event_logits"].detach().cpu()
        sequence_global_offsets = (
            output["global_threshold_offsets"].detach().cpu()
        )
        sequence_local_offsets = (
            output["local_threshold_offsets"].detach().cpu()
        )
        global_offsets.append(sequence_global_offsets)
        local_offsets.append(sequence_local_offsets)
        per_sequence_global_mean.append(
            {
                "source": sequence.source_path.name,
                "mean": (
                    float(sequence_global_offsets.double().mean().item())
                    if sequence_global_offsets.numel()
                    else 0.0
                ),
            }
        )
        sequence_results.append((logits, sequence))
        if progress:
            print(
                f"  validation {index + 1:03d}/{len(dataset):03d} "
                f"{sequence.source_path.name}: {sequence.length:,} events",
                flush=True,
            )
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else config.inference.sigmoid_threshold
    )
    if threshold_override is None and (
        config.evaluation.optimize_threshold or force_threshold_search
    ):
        result = _select_competition_threshold(
            sequence_results,
            config,
            steps=config.evaluation.threshold_steps,
            fallback=threshold,
        )
    else:
        result = _metrics_at_threshold(sequence_results, config, threshold)
    result["calibration"] = {
        "global": _offset_statistics(torch.cat(global_offsets)),
        "local": _offset_statistics(torch.cat(local_offsets)),
        "per_sequence_global_mean": per_sequence_global_mean,
    }
    if loss_totals is not None:
        result["loss"] = {
            name: total / len(dataset) for name, total in loss_totals.items()
        }
    return result


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    competition = metrics["competition"]
    counts = competition["counts"]
    event = metrics["event"]
    official = metrics["ev_uav_compatible"]
    calibration = metrics["calibration"]
    per_sequence_rows = [
        f"| {item['source']} | {item['mean']:.8f} |"
        for item in sorted(
            calibration["per_sequence_global_mean"],
            key=lambda item: item["source"],
        )
    ]
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
            "## Adaptive-threshold calibration",
            "",
            "| Offset | Mean | Std | P95 absolute | Max absolute |",
            "|---|---:|---:|---:|---:|",
            f"| Global | {calibration['global']['mean']:.8f} | "
            f"{calibration['global']['std']:.8f} | "
            f"{calibration['global']['p95_abs']:.8f} | "
            f"{calibration['global']['max_abs']:.8f} |",
            f"| Local | {calibration['local']['mean']:.8f} | "
            f"{calibration['local']['std']:.8f} | "
            f"{calibration['local']['p95_abs']:.8f} | "
            f"{calibration['local']['max_abs']:.8f} |",
            "",
            "### Per-sequence global mean",
            "",
            "| File | Global mean |",
            "|---|---:|",
            *per_sequence_rows,
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
    initialize_detector: str | Path | None = None,
) -> dict[str, Any]:
    config.validate()
    if config.training.training_stage == "joint":
        raise ValueError(
            "run_training does not accept the joint training stage; "
            "use resume_training"
        )
    if (
        config.training.training_stage == "calibrator"
        and initialize_detector is None
    ):
        raise ValueError(
            "calibrator training requires initialize_detector"
        )
    if (
        initialize_detector is not None
        and config.training.training_stage != "calibrator"
    ):
        raise ValueError(
            "initialize_detector is only valid for the calibrator training stage"
        )
    device_name = device or config.training.device
    runtime = prepare_device(device_name, num_threads=config.training.num_threads)
    torch_device = runtime.device
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    configuration = _serializable(asdict(config))
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
    model = build_model(config).to(torch_device)
    initialization_metadata: dict[str, str] = {}
    if initialize_detector is not None:
        initialization_metadata = _initialize_detector_weights(
            model,
            config,
            Path(initialize_detector),
            validation_manifest_digest,
            device=torch_device,
        )
    _configure_training_stage(
        model, config.training.training_stage
    )
    trainable_parameter_count = _expected_trainable_state_count(
        model.state_dict(), config.training.training_stage
    )
    criterion = _criterion(config)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(
        optimizer,
        epochs=config.training.epochs,
        minimum_ratio=config.training.min_learning_rate_ratio,
    )
    run_dir = _run_directory(config, output_dir, device_name=torch_device.type)
    _atomic_write_text(
        run_dir / "config.yaml",
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
    )
    _write_json(run_dir / "dataset_manifest.json", manifest)
    if initialization_metadata:
        _atomic_write_text(
            run_dir / "initialized_from.txt",
            f"checkpoint: {initialization_metadata['source_checkpoint']}\n"
            f"sha256: "
            f"{initialization_metadata['source_checkpoint_sha256']}\n",
        )
    order_generator = torch.Generator().manual_seed(config.training.seed)
    sampling_generator = torch.Generator(device=torch_device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, torch_device)
    amp_dtype = _autocast_dtype(config.training.mixed_precision_dtype)
    scaler_enabled = _grad_scaler_is_enabled(amp_enabled, amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None

    print(
        f"HUAV training on {torch_device.type}: {len(train_dataset)} train files, "
        f"{len(validation_dataset)} validation files, {config.training.epochs} epochs",
        flush=True,
    )
    for epoch in range(config.training.epochs):
        _configure_training_stage(model, config.training.training_stage)
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        totals = {name: 0.0 for name in LOSS_NAMES}
        learning_rates = {
            str(group["name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        learning_rate = learning_rates.get(
            "calibrator", next(iter(learning_rates.values()))
        )
        order = torch.randperm(
            len(train_dataset), generator=order_generator
        ).tolist()
        for step, dataset_index in enumerate(order):
            sequence = train_dataset[dataset_index]
            events = sequence.events.to(torch_device)
            labels = sequence.labels.to(torch_device)
            object_ids = sequence.object_ids.to(torch_device)
            timestamps_ms = sequence.timestamps_ms.to(torch_device)
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=torch_device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(events, batch_indices)
                query_indices = sample_event_indices(
                    labels,
                    logits=output["event_logits"],
                    background_ratio=config.training.background_ratio,
                    hard_negative_fraction=(
                        config.training.hard_negative_fraction
                    ),
                    generator=sampling_generator,
                    max_background_without_positive=(
                        config.training.max_background_without_positive
                    ),
                )
                losses = criterion(
                    output["event_logits"][query_indices],
                    labels[query_indices],
                    batch_indices[query_indices],
                    object_ids=object_ids[query_indices],
                    timestamps_ms=timestamps_ms[query_indices],
                    frame_ms=config.evaluation.frame_ms,
                    full_logits=output["event_logits"],
                    full_targets=labels,
                    full_batch_indices=batch_indices,
                    full_timestamps_ms=timestamps_ms,
                    full_coordinates=events[:, :2],
                    **_threshold_loss_inputs(output),
                )
                sequence_weight = _dense_sequence_loss_weight(
                    events,
                    labels,
                    config,
                )
                losses["total"] = losses["total"] * sequence_weight
            if amp_enabled:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    trainable_parameters, config.training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                clip_grad_norm_(
                    trainable_parameters, config.training.gradient_clip_norm
                )
                optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())
            if progress:
                print(
                    f"  epoch {epoch + 1}/{config.training.epochs} "
                    f"train {step + 1:03d}/{len(order):03d} "
                    f"loss={losses['total'].item():.5f} "
                    f"weight={sequence_weight:.2f} "
                    f"queries={len(query_indices):,}/{sequence.length:,}",
                    flush=True,
                )

        for name in totals:
            totals[name] /= len(train_dataset)
        validation_result = evaluate_model(
            model,
            validation_dataset,
            config,
            device=torch_device,
            progress=progress,
            criterion=criterion,
        )
        validation_loss = validation_result.pop("loss")
        metrics = validation_result
        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "learning_rates": learning_rates,
            "train_loss": totals,
            "validation_loss": validation_loss,
            "validation": metrics,
        }
        history.append(epoch_record)
        _write_json(run_dir / "history.json", history)
        competition = metrics["competition"]
        score = float(competition["score"])
        print(
            f"epoch {epoch + 1}/{config.training.epochs}: "
            f"train_loss={totals['total']:.6f} "
            f"val_loss={validation_loss['total']:.6f} Score={score:.6f} "
            f"IoU={competition['iou']:.6f} SegAcc={competition['seg_acc']:.6f} "
            f"Pd={competition['pd']:.6f} Fa={competition['fa']:.3e} "
            f"threshold={metrics['threshold']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            checkpoint_payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "decision_threshold": float(metrics["threshold"]),
                "model_config": configuration["model"],
                "resolution": configuration["resolution"],
                "validation_manifest_sha256": validation_manifest_digest,
                "training_stage": config.training.training_stage,
                "trainable_parameter_count": trainable_parameter_count,
                **initialization_metadata,
            }
            _atomic_torch_save(checkpoint_payload, run_dir / "best.pt")
            _write_json(run_dir / "metrics.json", best_metrics)
            _atomic_write_text(
                run_dir / "metrics.md", _metrics_markdown(best_metrics)
            )
        scheduler.step()

    if best_metrics is None:
        raise RuntimeError("training completed without validation metrics")
    summary = {
        "run_dir": str(run_dir.resolve()),
        "device": torch_device.type,
        "epochs": config.training.epochs,
        "mixed_precision": amp_enabled,
        "mixed_precision_dtype": (
            config.training.mixed_precision_dtype if amp_enabled else None
        ),
        "gradient_scaling": scaler_enabled,
        "runtime": runtime.metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_stage": config.training.training_stage,
        "trainable_parameter_count": trainable_parameter_count,
        **initialization_metadata,
        "best_score": best_score,
        "best_iou": float(best_metrics["competition"]["iou"]),
        "metrics": best_metrics,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def resume_training(
    config: HUAVConfig,
    checkpoint_path: str | Path,
    *,
    additional_epochs: int = 10,
    output_dir: str | Path | None = None,
    device: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Resume training from a saved checkpoint for additional epochs.

    The checkpoint's model architecture (model config + resolution) must match
    the supplied *config*.  The optimizer and LR scheduler are freshly created,
    so the LR schedule restarts from the configured learning rate.

    Args:
        config: Fresh training configuration.  The ``training`` section controls
            the learning rate, scheduler, loss weights etc. for the resumed run.
            The ``epochs`` field is **not** used directly; use *additional_epochs*
            to control how many extra epochs to train.
        checkpoint_path: Path to a ``best.pt`` checkpoint saved by a prior
            :func:`run_training` call.
        additional_epochs: How many extra epochs to train.
        output_dir: If set, use this exact directory (must be empty or
            non-existent).  Otherwise a timestamped directory is created under
            ``config.training.output_root``.
        device: Override the configured training device.
        progress: Print per-file / per-epoch progress.

    Returns:
        A summary dict with keys ``run_dir``, ``best_score``, etc.
    """
    config.validate()
    if config.training.training_stage != "joint":
        raise ValueError("resume_training requires the joint training stage")
    if (
        isinstance(additional_epochs, bool)
        or not isinstance(additional_epochs, Integral)
        or int(additional_epochs) <= 0
    ):
        raise ValueError("additional_epochs must be a positive integer")
    device_name = device or config.training.device
    runtime = prepare_device(device_name, num_threads=config.training.num_threads)
    torch_device = runtime.device
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    # ── Validate checkpoint compatibility ──────────────────────────────
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path, torch_device)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    _validate_schema5_metadata(checkpoint)
    _stored_decision_threshold(checkpoint)
    expected_model = _serializable(asdict(config.model))
    expected_resolution = _serializable(asdict(config.resolution))
    if checkpoint.get("model_config") != expected_model:
        raise ValueError(
            "checkpoint model configuration does not match the supplied config"
        )
    if checkpoint.get("resolution") != expected_resolution:
        raise ValueError(
            "checkpoint resolution does not match the supplied config"
        )

    # ── Run directory ───────────────────────────────────────────────────
    configuration = _serializable(asdict(config))
    resume_lineage = {
        "resumed_from_checkpoint": str(checkpoint_path.resolve()),
        "resumed_from_sha256": _file_sha256(checkpoint_path),
    }

    # ── Datasets & manifest ─────────────────────────────────────────────
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
    if checkpoint.get("validation_manifest_sha256") != validation_manifest_digest:
        raise ValueError(
            "checkpoint validation dataset manifest does not match current data"
        )
    # ── Build model & load weights ──────────────────────────────────────
    model = build_model(config).to(torch_device)
    model.load_state_dict(_checkpoint_model_state(checkpoint))
    _configure_training_stage(
        model, config.training.training_stage
    )
    trainable_parameter_count = _expected_trainable_state_count(
        model.state_dict(), config.training.training_stage
    )

    # ── Loss, optimiser, scheduler ──────────────────────────────────────
    criterion = _criterion(config)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(
        optimizer,
        epochs=additional_epochs,
        minimum_ratio=config.training.min_learning_rate_ratio,
    )
    run_dir = _run_directory(config, output_dir, device_name=torch_device.type)
    _atomic_write_text(
        run_dir / "config.yaml",
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
    )
    _atomic_write_text(
        run_dir / "resumed_from.txt",
        f"checkpoint: {checkpoint_path.resolve()}\n"
        f"source_epoch: {checkpoint.get('epoch', 'unknown')}\n",
    )
    _write_json(run_dir / "dataset_manifest.json", manifest)
    order_generator = torch.Generator().manual_seed(config.training.seed)
    sampling_generator = torch.Generator(device=torch_device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, torch_device)
    amp_dtype = _autocast_dtype(config.training.mixed_precision_dtype)
    scaler_enabled = _grad_scaler_is_enabled(amp_enabled, amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None

    source_epoch = int(checkpoint.get("epoch", 0))
    total_epochs_display = source_epoch + additional_epochs

    print(
        f"HUAV resumed training on {torch_device.type}: "
        f"{len(train_dataset)} train files, "
        f"{len(validation_dataset)} validation files, "
        f"{additional_epochs} additional epochs "
        f"(resumed from checkpoint epoch {source_epoch})",
        flush=True,
    )

    for epoch in range(additional_epochs):
        _configure_training_stage(model, config.training.training_stage)
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        totals = {name: 0.0 for name in LOSS_NAMES}
        learning_rates = {
            str(group["name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        learning_rate = learning_rates.get(
            "calibrator", next(iter(learning_rates.values()))
        )
        order = torch.randperm(
            len(train_dataset), generator=order_generator
        ).tolist()
        for step, dataset_index in enumerate(order):
            sequence = train_dataset[dataset_index]
            events = sequence.events.to(torch_device)
            labels = sequence.labels.to(torch_device)
            object_ids = sequence.object_ids.to(torch_device)
            timestamps_ms = sequence.timestamps_ms.to(torch_device)
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=torch_device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(events, batch_indices)
                query_indices = sample_event_indices(
                    labels,
                    logits=output["event_logits"],
                    background_ratio=config.training.background_ratio,
                    hard_negative_fraction=(
                        config.training.hard_negative_fraction
                    ),
                    generator=sampling_generator,
                    max_background_without_positive=(
                        config.training.max_background_without_positive
                    ),
                )
                losses = criterion(
                    output["event_logits"][query_indices],
                    labels[query_indices],
                    batch_indices[query_indices],
                    object_ids=object_ids[query_indices],
                    timestamps_ms=timestamps_ms[query_indices],
                    frame_ms=config.evaluation.frame_ms,
                    full_logits=output["event_logits"],
                    full_targets=labels,
                    full_batch_indices=batch_indices,
                    full_timestamps_ms=timestamps_ms,
                    full_coordinates=events[:, :2],
                    **_threshold_loss_inputs(output),
                )
                sequence_weight = _dense_sequence_loss_weight(
                    events,
                    labels,
                    config,
                )
                losses["total"] = losses["total"] * sequence_weight
            if amp_enabled:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    trainable_parameters, config.training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                clip_grad_norm_(
                    trainable_parameters, config.training.gradient_clip_norm
                )
                optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())
            if progress:
                current_display = source_epoch + epoch + 1
                print(
                    f"  epoch {current_display}/{total_epochs_display} "
                    f"train {step + 1:03d}/{len(order):03d} "
                    f"loss={losses['total'].item():.5f} "
                    f"weight={sequence_weight:.2f} "
                    f"queries={len(query_indices):,}/{sequence.length:,}",
                    flush=True,
                )

        for name in totals:
            totals[name] /= len(train_dataset)
        validation_result = evaluate_model(
            model,
            validation_dataset,
            config,
            device=torch_device,
            progress=progress,
            criterion=criterion,
        )
        validation_loss = validation_result.pop("loss")
        metrics = validation_result
        epoch_record = {
            "epoch": source_epoch + epoch + 1,
            "resumed_epoch": epoch + 1,
            "learning_rate": learning_rate,
            "learning_rates": learning_rates,
            "train_loss": totals,
            "validation_loss": validation_loss,
            "validation": metrics,
        }
        history.append(epoch_record)
        _write_json(run_dir / "history.json", history)
        competition = metrics["competition"]
        score = float(competition["score"])
        print(
            f"epoch {source_epoch + epoch + 1}/{total_epochs_display} "
            f"(resumed {epoch + 1}/{additional_epochs}): "
            f"train_loss={totals['total']:.6f} "
            f"val_loss={validation_loss['total']:.6f} Score={score:.6f} "
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
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "model_state": model.state_dict(),
                    "epoch": source_epoch + epoch + 1,
                    "decision_threshold": float(metrics["threshold"]),
                    "model_config": configuration["model"],
                    "resolution": configuration["resolution"],
                    "validation_manifest_sha256": validation_manifest_digest,
                    "training_stage": config.training.training_stage,
                    "trainable_parameter_count": trainable_parameter_count,
                    **resume_lineage,
                    **{
                        name: checkpoint[name]
                        for name in (
                            "source_checkpoint",
                            "source_checkpoint_sha256",
                        )
                        if name in checkpoint
                    },
                },
                run_dir / "best.pt",
            )
            _write_json(run_dir / "metrics.json", best_metrics)
            _atomic_write_text(
                run_dir / "metrics.md", _metrics_markdown(best_metrics)
            )
        scheduler.step()

    if best_metrics is None:
        raise RuntimeError(
            "resumed training completed without validation metrics"
        )
    summary = {
        "run_dir": str(run_dir.resolve()),
        "device": torch_device.type,
        "resumed_from": str(checkpoint_path.resolve()),
        "source_epoch": source_epoch,
        "additional_epochs": additional_epochs,
        "mixed_precision": amp_enabled,
        "mixed_precision_dtype": (
            config.training.mixed_precision_dtype if amp_enabled else None
        ),
        "gradient_scaling": scaler_enabled,
        "runtime": runtime.metadata,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training_stage": config.training.training_stage,
        "trainable_parameter_count": trainable_parameter_count,
        **resume_lineage,
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
    reoptimize_threshold: bool = False,
) -> dict[str, Any]:
    config.validate()
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    torch_device = runtime.device
    checkpoint = _load_checkpoint(checkpoint_path, torch_device)
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    _validate_checkpoint_metadata(
        checkpoint,
        config,
        actual_validation_digest,
        allow_legacy_read_only=True,
    )
    model = build_model(config).to(torch_device)
    try:
        model.load_state_dict(_checkpoint_model_state(checkpoint))
    except RuntimeError as error:
        raise ValueError("checkpoint model state does not match model") from error
    threshold = (
        None if reoptimize_threshold else _stored_decision_threshold(checkpoint)
    )
    return evaluate_model(
        model,
        validation_dataset,
        config,
        device=torch_device,
        progress=progress,
        threshold_override=threshold,
        force_threshold_search=reoptimize_threshold,
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
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    torch_device = runtime.device
    checkpoint = _load_checkpoint(checkpoint_path, torch_device)
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    _validate_checkpoint_metadata(
        checkpoint,
        config,
        actual_validation_digest,
        allow_legacy_read_only=True,
    )
    model = build_model(config).to(torch_device)
    try:
        model.load_state_dict(_checkpoint_model_state(checkpoint))
    except RuntimeError as error:
        raise ValueError("checkpoint model state does not match model") from error
    threshold = _stored_decision_threshold(checkpoint)
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
    "resume_training",
    "run_training",
]
