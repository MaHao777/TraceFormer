"""CPU/GPU training, validation and artifacts for the sparse cone model."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from traceformer.artifacts import (
    create_run_directory,
    rebuild_output_index,
    record_run_finished,
    record_run_started,
)
from traceformer.config import TraceFormerConfig
from traceformer.data import (
    EventSequence,
    HighThresholdView,
    LoadedEventSequence,
    NPZEventDataset,
    is_dense_balanced_event_stream,
    simulate_high_threshold_events,
)
from traceformer.experiments.ablation import AblationSpec, apply_ablation
from traceformer.inference import (
    generate_submission_rows,
    pack_submission_zip,
    write_prediction_txt,
)
from traceformer.models import TraceFormer
from traceformer.visualization import render_training_curves

from .losses import (
    EventFocalDiceLoss,
    background_threshold_consistency_loss,
    frame_cell_objectness_loss,
    object_frame_threshold_consistency_loss,
    positive_logit_preservation_loss,
    sparse_block_false_peak_loss,
)
from .lr_schedules import LearningRateController
from .metrics import (
    EventMetricAccumulator,
    EVUAVMetricAccumulator,
    competition_metrics,
)
from .runtime import prepare_device
from .sampling import sample_event_indices

CHECKPOINT_SCHEMA_VERSION = 7
REFINEMENT_SOURCE_SCHEMA_VERSION = 5
TRAINING_STATE_SCHEMA_VERSION = 3

SPARSE_INFERENCE_FIELDS = (
    "min_events",
    "max_polarity_fraction",
    "threshold_ratio",
)
REFINEMENT_MODEL_FIELDS = ("context_dim", "scales")
LEGACY_FIXED_MODEL_FIELDS = {
    "causal_mode": False,
    "debug_attention_stats": False,
}


def _update_training_curves(
    run_dir: Path, history: list[dict[str, Any]]
) -> None:
    try:
        render_training_curves(
            history,
            run_dir / "training_curves.png",
            run_name=run_dir.name,
        )
    except Exception as error:
        print(
            f"warning: training curves were not updated: {error}",
            file=sys.stderr,
            flush=True,
        )


def build_model(
    config: TraceFormerConfig, *, ablation: AblationSpec | None = None
) -> TraceFormer:
    model = TraceFormer(
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
        direction_dropout=config.model.direction_dropout,
        sparse_mixture_router_bias=config.sparse_mixture.initial_router_bias,
        sparse_mixture_residual_scale=(
            config.sparse_mixture.initial_residual_scale
        ),
        frame_refinement_context_dim=config.frame_refinement.context_dim,
        frame_refinement_scales=tuple(config.frame_refinement.scales),
        duration_ms=config.dataset.duration_ms,
    )
    return apply_ablation(model, config, ablation) if ablation is not None else model


def _file_sha256(path: Path) -> str:
    """分块计算检查点哈希，避免一次性读入大文件。"""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_subset(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """只提取决定固定模型推理行为的配置字段。"""

    return {name: value.get(name) for name in fields}


def _fixed_model_config(value: Any) -> dict[str, Any] | None:
    """移除历史上固定为关闭状态的模型开关，再比较真实结构。"""

    if not isinstance(value, dict):
        return None
    result = dict(value)
    for name, fixed_value in LEGACY_FIXED_MODEL_FIELDS.items():
        if name in result and result.pop(name) != fixed_value:
            return None
    return result


def _initialize_full_path(
    model: TraceFormer,
    config: TraceFormerConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """从历史检查点严格加载 Full 主干并冻结。"""

    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if _fixed_model_config(checkpoint.get("model_config")) != _serializable(
        asdict(config.model)
    ):
        raise ValueError("initial Full checkpoint model configuration does not match")
    if checkpoint.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError("initial Full checkpoint resolution does not match")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("initial Full checkpoint has no model state")
    excluded_prefixes = ("sparse_mixture.", "event_refinement_head.")
    expected_full = {
        name
        for name in model.state_dict()
        if not name.startswith(excluded_prefixes)
    }
    source_full = {
        name: value
        for name, value in source_state.items()
        if not name.startswith(excluded_prefixes)
    }
    if set(source_full) != expected_full:
        missing = sorted(expected_full - set(source_full))
        extra = sorted(set(source_full) - expected_full)
        raise ValueError(
            f"initial Full checkpoint state mismatch: missing={missing}, extra={extra}"
        )
    incompatible = model.load_state_dict(source_full, strict=False)
    expected_missing = {
        name for name in model.state_dict() if name.startswith(excluded_prefixes)
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError("initial Full checkpoint could not be loaded strictly")
    model.configure_training_stage("sparse_mixture")
    return {
        "kind": "full_path",
        "path": str(path),
        "sha256": _file_sha256(path),
        "schema_version": checkpoint.get("schema_version"),
        "epoch": checkpoint.get("epoch"),
    }


def _initialize_refinement_head(
    model: TraceFormer,
    config: TraceFormerConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """严格加载 schema-5 基线，只保留事件精修头为可训练。"""

    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != REFINEMENT_SOURCE_SCHEMA_VERSION:
        raise ValueError("refinement initialization requires a schema-5 checkpoint")
    if _fixed_model_config(checkpoint.get("model_config")) != _serializable(
        asdict(config.model)
    ):
        raise ValueError("refinement source model configuration does not match")
    source_mixture = checkpoint.get("sparse_mixture_config")
    expected_mixture = _serializable(asdict(config.sparse_mixture))
    if not isinstance(source_mixture, dict) or _config_subset(
        source_mixture, SPARSE_INFERENCE_FIELDS
    ) != _config_subset(expected_mixture, SPARSE_INFERENCE_FIELDS):
        raise ValueError("refinement source sparse mixture configuration does not match")
    if checkpoint.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError("refinement source resolution does not match")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("refinement source checkpoint has no model state")
    expected_base = {
        name
        for name in model.state_dict()
        if not name.startswith("event_refinement_head.")
    }
    if set(source_state) != expected_base:
        missing = sorted(expected_base - set(source_state))
        extra = sorted(set(source_state) - expected_base)
        raise ValueError(
            f"refinement source state mismatch: missing={missing}, extra={extra}"
        )
    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("event_refinement_head.")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError("refinement source checkpoint could not be loaded strictly")
    model.configure_training_stage("refinement")
    return {
        "kind": "event_refinement",
        "path": str(path),
        "sha256": _file_sha256(path),
        "schema_version": checkpoint.get("schema_version"),
        "epoch": checkpoint.get("epoch"),
    }


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """优化器只接收真正允许更新的参数。"""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def _initialize_ablation_stage(
    model: TraceFormer,
    config: TraceFormerConfig,
    ablation: AblationSpec,
    checkpoint_path: str | Path | None,
    *,
    manifest_digest: str,
    device: torch.device,
) -> dict[str, Any] | None:
    """消融阶段只接受同种子、同数据、同结构的直接上游权重。"""
    stage = config.training.stage
    if stage == "full":
        if checkpoint_path is not None:
            raise ValueError("ablation base must start from random initialization")
        return None
    if checkpoint_path is None:
        raise ValueError("ablation stage requires its upstream checkpoint")
    path = Path(checkpoint_path).resolve()
    source = torch.load(path, map_location=device, weights_only=True)
    expected_source = AblationSpec(
        backbone=ablation.backbone,
        sparse=ablation.sparse if stage == "refinement" else False,
        refinement="none",
    )
    if source.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported ablation source schema")
    if source.get("ablation") != expected_source.to_dict():
        raise ValueError("ablation upstream structure mismatch")
    expected_stage = "sparse_mixture" if expected_source.sparse else "full"
    if source.get("training_stage") != expected_stage:
        raise ValueError("ablation upstream training stage mismatch")
    if source.get("training_seed") != config.training.seed:
        raise ValueError("ablation upstream seed mismatch")
    if source.get("dataset_manifest_sha256") != manifest_digest:
        raise ValueError("ablation upstream dataset mismatch")
    if source.get("model_config") != _serializable(asdict(config.model)):
        raise ValueError("ablation upstream model configuration mismatch")
    if source.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError("ablation upstream resolution mismatch")
    if source.get("duration_ms") != config.dataset.duration_ms:
        raise ValueError("ablation upstream duration mismatch")
    if _config_subset(source["sparse_mixture_config"], SPARSE_INFERENCE_FIELDS) != _config_subset(
        _serializable(asdict(config.sparse_mixture)), SPARSE_INFERENCE_FIELDS
    ):
        raise ValueError("ablation upstream sparse inference mismatch")
    prefix = "event_refinement_head." if stage == "refinement" else "sparse_mixture."
    expected = {name for name in model.state_dict() if not name.startswith(prefix)}
    if set(source["model_state"]) != expected:
        raise ValueError("ablation upstream state keys mismatch")
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    missing = {name for name in model.state_dict() if name.startswith(prefix)}
    if set(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise ValueError("ablation upstream could not be loaded strictly")
    return {
        "kind": "ablation_upstream", "path": str(path), "sha256": _file_sha256(path),
        "epoch": source["epoch"], "ablation": source["ablation"],
    }


def _dataset(config: TraceFormerConfig, split: str) -> NPZEventDataset:
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


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    config: TraceFormerConfig,
    *,
    ablation: AblationSpec | None = None,
) -> None:
    """统一校验 schema-7 推理与恢复契约。"""

    # 即使张量形状相同，也不能把窗口/单向实验权重当作正式模型加载。
    expected_ablation = ablation.to_dict() if ablation is not None else None
    if checkpoint.get("ablation") != expected_ablation:
        raise ValueError("checkpoint ablation contract does not match")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if _fixed_model_config(checkpoint.get("model_config")) != _serializable(
        asdict(config.model)
    ):
        raise ValueError("checkpoint model configuration does not match the supplied config")
    if checkpoint.get("resolution") != _serializable(asdict(config.resolution)):
        raise ValueError("checkpoint resolution does not match the supplied config")

    checkpoint_mixture = checkpoint.get("sparse_mixture_config")
    expected_mixture = _serializable(asdict(config.sparse_mixture))
    if not isinstance(checkpoint_mixture, dict) or _config_subset(
        checkpoint_mixture, SPARSE_INFERENCE_FIELDS
    ) != _config_subset(expected_mixture, SPARSE_INFERENCE_FIELDS):
        raise ValueError(
            "checkpoint sparse mixture inference configuration does not match"
        )

    checkpoint_refinement = checkpoint.get("frame_refinement_config")
    expected_refinement = _serializable(asdict(config.frame_refinement))
    if not isinstance(checkpoint_refinement, dict) or _config_subset(
        checkpoint_refinement, REFINEMENT_MODEL_FIELDS
    ) != _config_subset(expected_refinement, REFINEMENT_MODEL_FIELDS):
        raise ValueError("checkpoint refinement model configuration does not match")


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
    config: TraceFormerConfig,
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
    config: TraceFormerConfig,
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
    config: TraceFormerConfig,
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


def _sparse_mixture_weight(config: TraceFormerConfig, epoch: int) -> float:
    """Return the residual-gate ramp while auxiliary experts train from epoch one."""

    mixture = config.sparse_mixture
    if config.training.stage == "refinement":
        return 1.0
    if epoch <= mixture.warmup_epochs:
        return 0.0
    ramp_position = epoch - mixture.warmup_epochs
    return float(min(1.0, ramp_position / mixture.ramp_epochs))


def _forward_with_sparse_mixture(
    model: TraceFormer,
    sequence: LoadedEventSequence,
    events: torch.Tensor,
    batch_indices: torch.Tensor,
    config: TraceFormerConfig,
    *,
    mixture_weight: float,
    compute_sparse_logits: bool = True,
) -> tuple[dict[str, Any], HighThresholdView | None]:
    """Create the deterministic paired view and preserve full-event decoding."""

    spec = getattr(model, "ablation_spec", None)
    if spec is not None and not spec["sparse"]:
        return model(events, batch_indices), None
    mixture = config.sparse_mixture
    if not is_dense_balanced_event_stream(
        sequence.events,
        min_events=mixture.min_events,
        max_polarity_fraction=mixture.max_polarity_fraction,
    ):
        return model(events, batch_indices), None
    view = simulate_high_threshold_events(
        sequence.events,
        width=config.resolution.width,
        height=config.resolution.height,
        threshold_ratio=mixture.threshold_ratio,
    )
    sparse_events = view.events.to(events.device)
    sparse_source_indices = view.source_indices.to(events.device)
    sparse_batch_indices = torch.zeros(
        len(sparse_events), dtype=torch.long, device=events.device
    )
    output = model(
        events,
        batch_indices,
        sparse_events=sparse_events,
        sparse_batch_indices=sparse_batch_indices,
        sparse_source_indices=sparse_source_indices,
        mixture_weight=mixture_weight,
        compute_sparse_logits=compute_sparse_logits,
    )
    return output, view


def forward_with_sparse_mixture(
    model: TraceFormer,
    sequence: LoadedEventSequence,
    events: torch.Tensor,
    batch_indices: torch.Tensor,
    config: TraceFormerConfig,
    *,
    mixture_weight: float = 1.0,
    compute_sparse_logits: bool = True,
) -> tuple[dict[str, Any], HighThresholdView | None]:
    """复用训练和评估中的稀疏混合推理路径。"""

    return _forward_with_sparse_mixture(
        model,
        sequence,
        events,
        batch_indices,
        config,
        mixture_weight=mixture_weight,
        compute_sparse_logits=compute_sparse_logits,
    )


def load_inference_model(
    config: TraceFormerConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    neighbor_backend: str = "cpu",
) -> tuple[TraceFormer, dict[str, Any]]:
    """严格校验检查点契约并构建只读推理模型。"""

    config.validate()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    _validate_checkpoint_contract(checkpoint, config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    from traceformer.models.gpu_neighbors import configure_neighbor_backend

    configure_neighbor_backend(model, neighbor_backend)
    return model, checkpoint

def _sparse_mixture_auxiliary_losses(
    output: dict[str, Any],
    labels: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    events: torch.Tensor,
    batch_indices: torch.Tensor,
    query_indices: torch.Tensor,
    criterion: EventFocalDiceLoss,
    config: TraceFormerConfig,
    *,
    sampling_generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Compute conservative expert losses without changing the main event universe."""

    zero = output["event_logits"].sum() * 0.0
    result = {
        "full_aux_total": zero,
        "sparse_total": zero,
        "object_frame_consistency": zero,
        "positive_preservation": zero,
        "background_consistency": zero,
        "sparse_false_peak": zero,
        "gate_regularization": zero,
    }
    if config.training.stage == "refinement":
        return result
    source_indices = output["sparse_source_indices"]
    if len(source_indices) == 0:
        return result

    mixture = config.sparse_mixture
    if mixture.full_aux_loss_weight:
        full_aux = criterion(
            output["full_event_logits"][query_indices],
            labels[query_indices],
            batch_indices[query_indices],
            object_ids=object_ids[query_indices],
            timestamps_ms=timestamps_ms[query_indices],
            frame_ms=config.evaluation.frame_ms,
            full_logits=output["full_event_logits"],
            full_targets=labels,
            full_batch_indices=batch_indices,
            full_timestamps_ms=timestamps_ms,
            full_coordinates=events[:, :2],
        )
        result["full_aux_total"] = full_aux["total"]
    fused_logits = output["event_logits"]
    sparse_logits = fused_logits[source_indices]
    sparse_labels = labels[source_indices]
    sparse_object_ids = object_ids[source_indices]
    sparse_timestamps_ms = timestamps_ms[source_indices]
    sparse_events = events[source_indices]
    sparse_batch_indices = batch_indices[source_indices]
    sparse_query_indices = sample_event_indices(
        sparse_labels,
        logits=sparse_logits,
        background_ratio=config.training.background_ratio,
        hard_negative_fraction=config.training.hard_negative_fraction,
        generator=sampling_generator,
        max_background_without_positive=(
            config.training.max_background_without_positive
        ),
    )
    sparse_aux = criterion(
        sparse_logits[sparse_query_indices],
        sparse_labels[sparse_query_indices],
        sparse_batch_indices[sparse_query_indices],
        object_ids=sparse_object_ids[sparse_query_indices],
        timestamps_ms=sparse_timestamps_ms[sparse_query_indices],
        frame_ms=config.evaluation.frame_ms,
        full_logits=sparse_logits,
        full_targets=sparse_labels,
        full_batch_indices=sparse_batch_indices,
        full_timestamps_ms=sparse_timestamps_ms,
        full_coordinates=sparse_events[:, :2],
    )
    result["sparse_total"] = sparse_aux["total"]
    if mixture.object_frame_consistency_weight:
        result["object_frame_consistency"] = object_frame_threshold_consistency_loss(
            fused_logits[source_indices],
            output["full_event_logits"],
            source_indices,
            labels,
            object_ids,
            timestamps_ms,
            frame_ms=config.evaluation.frame_ms,
        )
    all_indices = torch.arange(len(labels), dtype=torch.long, device=labels.device)
    result["background_consistency"] = background_threshold_consistency_loss(
        fused_logits,
        output["full_event_logits"],
        all_indices,
        labels,
        logit_margin=mixture.background_consistency_margin,
    )
    result["sparse_false_peak"] = sparse_block_false_peak_loss(
        fused_logits,
        labels,
        batch_indices,
        timestamps_ms,
        events[:, :2],
        frame_ms=config.evaluation.frame_ms,
        block_size_x=mixture.sparse_false_peak_block_size_x,
        block_size_y=mixture.sparse_false_peak_block_size_y,
        logit_margin=mixture.sparse_false_peak_logit_margin,
        topk_per_frame=mixture.sparse_false_peak_topk_per_frame,
    )
    result["positive_preservation"] = positive_logit_preservation_loss(
        fused_logits,
        output["full_event_logits"],
        labels,
        logit_margin=mixture.positive_preservation_margin,
    )
    gate = output["mixture_gate"]
    result["gate_regularization"] = gate.mean() if len(gate) else zero
    return result


def _add_sparse_mixture_losses(
    main_total: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
    config: TraceFormerConfig,
) -> torch.Tensor:
    mixture = config.sparse_mixture
    return (
        main_total
        + mixture.full_aux_loss_weight * auxiliary["full_aux_total"]
        + mixture.sparse_loss_weight * auxiliary["sparse_total"]
        + mixture.object_frame_consistency_weight
        * auxiliary["object_frame_consistency"]
        + mixture.positive_preservation_weight
        * auxiliary["positive_preservation"]
        + mixture.background_consistency_weight
        * auxiliary["background_consistency"]
        + mixture.sparse_false_peak_weight
        * auxiliary["sparse_false_peak"]
        + mixture.gate_regularization_weight
        * auxiliary["gate_regularization"]
    )


def _event_refinement_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    criterion: EventFocalDiceLoss,
    config: TraceFormerConfig,
) -> torch.Tensor:
    """对精修头的 8x8x50 ms 主尺度计算 cell 对象损失。"""

    if output["event_refinement_cell_logits"].numel() == 0:
        return output["event_refinement_cell_logits"].sum() * 0.0
    return frame_cell_objectness_loss(
        output["event_refinement_cell_logits"],
        output["event_refinement_event_cell_indices"],
        labels,
        alpha=criterion.focal_alpha,
        gamma=criterion.focal_gamma,
    )


def _sparse_view_statistics(
    sequence: LoadedEventSequence,
    view: HighThresholdView,
    *,
    frame_ms: float,
) -> dict[str, int | float]:
    indices = view.source_indices.detach().cpu()
    labels = sequence.labels > 0.5
    retained_labels = labels[indices]
    integer_time = torch.floor(sequence.timestamps_ms.to(dtype=torch.float64))
    within_frame = torch.remainder(integer_time, frame_ms)
    positive = labels & (sequence.object_ids > 0) & (within_frame > 0)
    retained_positive = positive[indices]

    def group_count(
        object_ids: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
    ) -> int:
        if not bool(mask.any()):
            return 0
        frames = torch.div(
            torch.floor(timestamps[mask].to(dtype=torch.float64)),
            frame_ms,
            rounding_mode="floor",
        ).long()
        groups = torch.stack((object_ids[mask].long(), frames), dim=1)
        return int(torch.unique(groups, dim=0).shape[0])

    source_groups = group_count(
        sequence.object_ids,
        sequence.timestamps_ms,
        positive,
    )
    retained_groups = group_count(
        sequence.object_ids[indices],
        sequence.timestamps_ms[indices],
        retained_positive,
    )
    return {
        "source_events": sequence.length,
        "retained_events": int(len(indices)),
        "source_positive_events": int(labels.sum().item()),
        "retained_positive_events": int(retained_labels.sum().item()),
        "source_object_frames": source_groups,
        "retained_object_frames": retained_groups,
    }


def _new_sparse_epoch_report(weight: float) -> dict[str, Any]:
    return {
        "weight": float(weight),
        "eligible_files": 0,
        "augmented_files": 0,
        "ratio_usage": {},
        "source_events": 0,
        "retained_events": 0,
        "source_positive_events": 0,
        "retained_positive_events": 0,
        "source_object_frames": 0,
        "retained_object_frames": 0,
    }


def _record_sparse_view(
    report: dict[str, Any],
    view: HighThresholdView,
    statistics: dict[str, int | float],
) -> None:
    report["augmented_files"] += 1
    ratio_key = f"{view.threshold_ratio:g}"
    ratio_usage = report["ratio_usage"]
    ratio_usage[ratio_key] = int(ratio_usage.get(ratio_key, 0)) + 1
    for name, value in statistics.items():
        report[name] += value


def _finalize_sparse_epoch_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    pairs = (
        ("event_retention", "retained_events", "source_events"),
        (
            "positive_event_retention",
            "retained_positive_events",
            "source_positive_events",
        ),
        (
            "object_frame_retention",
            "retained_object_frames",
            "source_object_frames",
        ),
    )
    for output_name, numerator_name, denominator_name in pairs:
        denominator = int(result[denominator_name])
        result[output_name] = (
            float(result[numerator_name] / denominator) if denominator else 0.0
        )
    return result


@torch.no_grad()
def evaluate_model(
    model: TraceFormer,
    dataset: NPZEventDataset,
    config: TraceFormerConfig,
    *,
    device: torch.device,
    progress: bool = False,
    threshold_override: float | None = None,
    criterion: EventFocalDiceLoss | None = None,
) -> dict[str, Any]:
    model.eval()
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]] = []
    loss_totals = (
        {
            "total": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "object_frame": 0.0,
            "false_alarm": 0.0,
            "event_refinement": 0.0,
        }
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
        output, _ = _forward_with_sparse_mixture(
            model,
            sequence,
            events,
            batch_indices,
            config,
            mixture_weight=1.0,
        )
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
            )
            refinement_loss = _event_refinement_loss(
                output,
                labels,
                criterion,
                config,
            )
            losses["event_refinement"] = refinement_loss
            losses["total"] = (
                losses["total"]
                + config.frame_refinement.loss_weight * refinement_loss
            ) * _dense_sequence_loss_weight(
                events,
                labels,
                config,
            )
            for name in loss_totals:
                loss_totals[name] += float(losses[name].detach().item())
        logits = output["event_logits"].detach().cpu()
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
    if threshold_override is None and config.evaluation.optimize_threshold:
        result = _select_competition_threshold(
            sequence_results,
            config,
            steps=config.evaluation.threshold_steps,
            fallback=threshold,
        )
    else:
        result = _metrics_at_threshold(sequence_results, config, threshold)
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
    return "\n".join(
        (
            "# TraceFormer Spatiotemporal-Window Validation Metrics",
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
    config: TraceFormerConfig, output_dir: str | Path | None, *, device_name: str
) -> Path:
    return create_run_directory(
        config.training.output_root,
        device_name,
        output_dir=output_dir,
    )


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
    # 数据协议标识参与续训摘要；品牌更名不改变历史数据身份。
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


def _branch_contract(configuration: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must remain identical across an exact-state branch."""
    contract = json.loads(json.dumps(configuration))
    training = contract["training"]
    for name in (
        "device",
        "epochs",
        "output_root",
        "lr_schedule",
        "fast_learning_epochs",
        "fixed_learning_rate_ratio",
        "min_learning_rate_ratio",
        "plateau_factor",
        "plateau_patience",
        "plateau_min_delta",
        "save_training_state_interval",
    ):
        training.pop(name, None)
    return contract


def _restore_rng_state(
    state: dict[str, Any],
    *,
    torch_device: torch.device,
    order_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
    if torch_device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(
            [
                value.to(device="cpu", dtype=torch.uint8)
                for value in state["cuda_rng_state"]
            ]
        )
    order_generator.set_state(
        state["order_generator_state"].to(device="cpu", dtype=torch.uint8)
    )
    sampling_generator.set_state(
        state["sampling_generator_state"].to(device="cpu", dtype=torch.uint8)
    )


def _training_state_payload(
    *,
    config: dict[str, Any],
    manifest_digest: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    controller: LearningRateController,
    epoch: int,
    history: list[dict[str, Any]],
    sparse_mixture_history: list[dict[str, Any]],
    best_score: float,
    best_metrics: dict[str, Any],
    best_checkpoint: dict[str, Any],
    order_generator: torch.Generator,
    sampling_generator: torch.Generator,
    torch_device: torch.device,
) -> dict[str, Any]:
    return {
        "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "branch_contract": _branch_contract(config),
        "dataset_manifest_sha256": manifest_digest,
        "source_schedule": config["training"]["lr_schedule"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "controller_state": controller.state_dict(),
        "epoch": epoch,
        "history": history,
        "sparse_mixture_history": sparse_mixture_history,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_checkpoint": best_checkpoint,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch_device.type == "cuda" else None
        ),
        "order_generator_state": order_generator.get_state(),
        "sampling_generator_state": sampling_generator.get_state(),
    }


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _relative_parameter_displacement(
    model: torch.nn.Module, before: list[torch.Tensor]
) -> float:
    difference = 0.0
    baseline = 0.0
    for parameter, original in zip(model.parameters(), before, strict=True):
        current = parameter.detach().float()
        original_float = original.float()
        difference += float(torch.sum((current - original_float) ** 2).item())
        baseline += float(torch.sum(original_float**2).item())
    return math.sqrt(difference) / max(math.sqrt(baseline), 1e-12)


def _run_training_impl(
    config: TraceFormerConfig,
    *,
    output_dir: str | Path,
    device: str | None = None,
    progress: bool = False,
    resume_state_path: str | Path | None = None,
    initial_full_checkpoint_path: str | Path | None = None,
    initial_refinement_checkpoint_path: str | Path | None = None,
    ablation: AblationSpec | None = None,
) -> dict[str, Any]:
    config.validate()
    if ablation is not None:
        ablation.validate_stage(config.training.stage)
    device_name = device or config.training.device
    runtime = prepare_device(device_name, num_threads=config.training.num_threads)
    torch_device = runtime.device
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    run_dir = Path(output_dir)
    configuration = _serializable(asdict(config))
    _atomic_write_text(
        run_dir / "config.yaml",
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
    )
    train_dataset = _dataset(config, "train")
    validation_dataset = _dataset(config, "val")
    manifest = _dataset_manifest(train_dataset, validation_dataset)
    manifest_digest = _manifest_identity_digest(manifest)
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
    if ablation is not None:
        _write_json(run_dir / "ablation.json", ablation.to_dict())
    model = build_model(config, ablation=ablation).to(torch_device)
    initialization: dict[str, Any] | None = None
    initialization_paths = (
        initial_full_checkpoint_path,
        initial_refinement_checkpoint_path,
    )
    if resume_state_path is not None and any(
        path is not None for path in initialization_paths
    ):
        raise ValueError("initial checkpoints and resume state are mutually exclusive")
    if all(path is not None for path in initialization_paths):
        raise ValueError("Full and refinement initialization are mutually exclusive")
    stage = config.training.stage
    model.configure_training_stage(stage)
    if resume_state_path is None:
        if ablation is not None:
            initialization = _initialize_ablation_stage(
                model, config, ablation,
                initial_refinement_checkpoint_path or initial_full_checkpoint_path,
                manifest_digest=manifest_digest, device=torch_device,
            )
        elif stage == "refinement":
            if initial_refinement_checkpoint_path is None:
                raise ValueError(
                    "refinement training requires --initialize-refinement"
                )
            initialization = _initialize_refinement_head(
                model,
                config,
                initial_refinement_checkpoint_path,
                device=torch_device,
            )
        elif stage == "sparse_mixture":
            if initial_full_checkpoint_path is None:
                raise ValueError(
                    "sparse-mixture training requires --initialize-full"
                )
            initialization = _initialize_full_path(
                model,
                config,
                initial_full_checkpoint_path,
                device=torch_device,
            )
        elif any(path is not None for path in initialization_paths):
            raise ValueError("full training does not accept initialization checkpoints")
        if stage != "refinement" and initial_refinement_checkpoint_path is not None:
            raise ValueError("--initialize-refinement requires training.stage=refinement")
        if stage != "sparse_mixture" and initial_full_checkpoint_path is not None:
            raise ValueError("--initialize-full requires training.stage=sparse_mixture")
    criterion = EventFocalDiceLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        lambda_dice=config.loss.lambda_dice,
        lambda_object_frame=config.loss.lambda_object_frame,
        object_frame_logit_margin=config.loss.object_frame_logit_margin,
        lambda_false_alarm=config.loss.lambda_false_alarm,
        false_alarm_logit_margin=config.loss.false_alarm_logit_margin,
        false_alarm_topk_per_frame=config.loss.false_alarm_topk_per_frame,
    )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    controller = LearningRateController(config.training)
    order_generator = torch.Generator().manual_seed(config.training.seed)
    sampling_generator = torch.Generator(device=torch_device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, torch_device)
    amp_dtype = _autocast_dtype(config.training.mixed_precision_dtype)
    scaler_enabled = _grad_scaler_is_enabled(amp_enabled, amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    history: list[dict[str, Any]] = []
    sparse_mixture_history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    best_checkpoint: dict[str, Any] | None = None
    start_epoch = 0
    if resume_state_path is not None:
        state = torch.load(
            Path(resume_state_path), map_location=torch_device, weights_only=False
        )
        if state.get("training_state_schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported training state schema version")
        if state.get("branch_contract") != _branch_contract(configuration):
            raise ValueError("training state branch contract does not match configuration")
        if state.get("dataset_manifest_sha256") != manifest_digest:
            raise ValueError("training state dataset manifest does not match current data")
        _validate_checkpoint_contract(state["best_checkpoint"], config, ablation=ablation)
        initialization = state["best_checkpoint"].get("full_initialization")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        history = list(state["history"])
        sparse_mixture_history = list(state["sparse_mixture_history"])
        best_score = float(state["best_score"])
        best_metrics = dict(state["best_metrics"])
        best_checkpoint = dict(state["best_checkpoint"])
        start_epoch = int(state["epoch"])
        if start_epoch > config.training.epochs or (
            start_epoch == config.training.epochs and ablation is None
        ):
            raise ValueError("training state epoch must be less than configured epochs")
        if state.get("source_schedule") == config.training.lr_schedule:
            controller.load_state_dict(state["controller_state"])
        _restore_rng_state(
            state,
            torch_device=torch_device,
            order_generator=order_generator,
            sampling_generator=sampling_generator,
        )
        _atomic_torch_save(best_checkpoint, run_dir / "best.pt")
        _write_json(run_dir / "metrics.json", best_metrics)
        _atomic_write_text(run_dir / "metrics.md", _metrics_markdown(best_metrics))
        if ablation is not None:
            _write_json(run_dir / "history.json", history)
            _write_json(run_dir / "sparse_mixture.json", sparse_mixture_history)

    print(
        f"TraceFormer training on {torch_device.type}: {len(train_dataset)} train files, "
        f"{len(validation_dataset)} validation files, {config.training.epochs} epochs",
        flush=True,
    )
    for epoch in range(start_epoch, config.training.epochs):
        epoch_number = epoch + 1
        learning_rate = controller.rate_for_epoch(epoch_number)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        mixture_weight = _sparse_mixture_weight(config, epoch_number)
        mixture_report = _new_sparse_epoch_report(mixture_weight)
        model.train()
        totals = {
            "total": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "object_frame": 0.0,
            "false_alarm": 0.0,
            "event_refinement": 0.0,
            "sparse_total": 0.0,
            "full_aux_total": 0.0,
            "object_frame_consistency": 0.0,
            "background_consistency": 0.0,
            "positive_preservation": 0.0,
            "sparse_false_peak": 0.0,
            "gate_regularization": 0.0,
            "mixture_weight": 0.0,
        }
        epoch_started = time.perf_counter()
        epoch_parameters = _parameter_snapshot(model)
        gradient_norms: list[float] = []
        order = torch.randperm(
            len(train_dataset), generator=order_generator
        ).tolist()
        for step, dataset_index in enumerate(order):
            sequence = train_dataset[dataset_index]
            mixture_eligible = is_dense_balanced_event_stream(
                sequence.events,
                min_events=config.sparse_mixture.min_events,
                max_polarity_fraction=(
                    config.sparse_mixture.max_polarity_fraction
                ),
            )
            if mixture_eligible:
                mixture_report["eligible_files"] += 1
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
                output, mixture_view = _forward_with_sparse_mixture(
                    model,
                    sequence,
                    events,
                    batch_indices,
                    config,
                    mixture_weight=mixture_weight,
                )
                if mixture_view is not None:
                    _record_sparse_view(
                        mixture_report,
                        mixture_view,
                        _sparse_view_statistics(
                            sequence,
                            mixture_view,
                            frame_ms=config.evaluation.frame_ms,
                        ),
                    )
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
                full_losses = criterion(
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
                )
                refinement_loss = _event_refinement_loss(
                    output,
                    labels,
                    criterion,
                    config,
                )
                if config.training.stage == "sparse_mixture":
                    refinement_loss = refinement_loss * 0.0
                sequence_weight = _dense_sequence_loss_weight(
                    events,
                    labels,
                    config,
                )
                combined_total = (
                    full_losses["total"]
                    + config.frame_refinement.loss_weight * refinement_loss
                ) * sequence_weight
                auxiliary = _sparse_mixture_auxiliary_losses(
                    output,
                    labels,
                    object_ids,
                    timestamps_ms,
                    events,
                    batch_indices,
                    query_indices,
                    criterion,
                    config,
                    sampling_generator=sampling_generator,
                )
                if mixture_view is not None:
                    combined_total = _add_sparse_mixture_losses(
                        combined_total,
                        auxiliary,
                        config,
                    )
                losses = dict(full_losses)
                losses["total"] = combined_total
                losses["event_refinement"] = refinement_loss
                losses["sparse_total"] = auxiliary["sparse_total"]
                for name, value in auxiliary.items():
                    if name != "sparse_total":
                        losses[name] = value
                losses["mixture_weight"] = combined_total.new_tensor(
                    mixture_weight if mixture_view is not None else 0.0
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"non-finite training loss at epoch {epoch_number}, step {step + 1}"
                )
            # 非稠密样本不会经过专家；冻结阶段只记录 loss，不执行空反向。
            has_trainable_path = losses["total"].requires_grad
            if amp_enabled and has_trainable_path:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"non-finite gradient norm at epoch {epoch_number}, step {step + 1}"
                    )
                gradient_norms.append(float(gradient_norm.item()))
                scaler.step(optimizer)
                scaler.update()
            elif has_trainable_path:
                losses["total"].backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"non-finite gradient norm at epoch {epoch_number}, step {step + 1}"
                    )
                gradient_norms.append(float(gradient_norm.item()))
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
        finalized_mixture_report = _finalize_sparse_epoch_report(
            mixture_report
        )
        sparse_mixture_history.append(finalized_mixture_report)
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
        if not gradient_norms:
            raise RuntimeError("epoch has no sample connected to trainable parameters")
        sorted_gradient_norms = sorted(gradient_norms)
        p95_index = max(0, math.ceil(0.95 * len(sorted_gradient_norms)) - 1)
        optimization = {
            "gradient_norm_mean": float(sum(gradient_norms) / len(gradient_norms)),
            "gradient_norm_p95": float(sorted_gradient_norms[p95_index]),
            "gradient_norm_max": float(max(gradient_norms)),
            "gradient_clipped_fraction": float(
                sum(
                    value > config.training.gradient_clip_norm
                    for value in gradient_norms
                )
                / len(gradient_norms)
            ),
            "relative_parameter_displacement": _relative_parameter_displacement(
                model, epoch_parameters
            ),
            "optimizer_steps": len(gradient_norms),
            "nonfinite_steps": 0,
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
        }
        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "learning_rate_schedule": config.training.lr_schedule,
            "learning_phase": controller.phase(epoch_number),
            "optimization": optimization,
            "train_loss": totals,
            "validation_loss": validation_loss,
            "validation": metrics,
            "sparse_mixture": finalized_mixture_report,
        }
        history.append(epoch_record)
        _write_json(run_dir / "history.json", history)
        _write_json(run_dir / "sparse_mixture.json", sparse_mixture_history)
        _update_training_curves(run_dir, history)
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
            best_checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state": {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                },
                "epoch": epoch + 1,
                "decision_threshold": float(metrics["threshold"]),
                "model_config": configuration["model"],
                "sparse_mixture_config": configuration["sparse_mixture"],
                "frame_refinement_config": configuration["frame_refinement"],
                "resolution": configuration["resolution"],
                "validation_manifest_sha256": validation_manifest_digest,
                "full_initialization": initialization,
                "ablation": ablation.to_dict() if ablation is not None else None,
                "training_seed": config.training.seed,
                "training_stage": config.training.stage,
                "duration_ms": config.dataset.duration_ms,
                "dataset_manifest_sha256": manifest_digest,
            }
            _atomic_torch_save(best_checkpoint, run_dir / "best.pt")
            _write_json(run_dir / "metrics.json", best_metrics)
            _atomic_write_text(
                run_dir / "metrics.md", _metrics_markdown(best_metrics)
            )
        controller.update(score, epoch=epoch_number)
        state_interval = config.training.save_training_state_interval
        if state_interval:
            assert best_metrics is not None
            assert best_checkpoint is not None
            training_state = _training_state_payload(
                config=configuration,
                manifest_digest=manifest_digest,
                model=model,
                optimizer=optimizer,
                controller=controller,
                epoch=epoch_number,
                history=history,
                sparse_mixture_history=sparse_mixture_history,
                best_score=best_score,
                best_metrics=best_metrics,
                best_checkpoint=best_checkpoint,
                order_generator=order_generator,
                sampling_generator=sampling_generator,
                torch_device=torch_device,
            )
            _atomic_torch_save(training_state, run_dir / "training_state.pt")
            if epoch_number % state_interval == 0:
                state_dir = run_dir / "training_states"
                state_dir.mkdir(exist_ok=True)
                _atomic_torch_save(
                    training_state, state_dir / f"epoch_{epoch_number:03d}.pt"
                )

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
        "training_stage": config.training.stage,
        "runtime": runtime.metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "full_initialization": initialization,
        "sparse_mixture": sparse_mixture_history,
        "best_score": best_score,
        "best_iou": float(best_metrics["competition"]["iou"]),
        "metrics": best_metrics,
        "resumed_from_training_state": (
            str(Path(resume_state_path).resolve())
            if resume_state_path is not None
            else None
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def run_training(
    config: TraceFormerConfig,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    progress: bool = False,
    note: str | None = None,
    config_source: str | Path | None = None,
    resume_state_path: str | Path | None = None,
    initial_full_checkpoint_path: str | Path | None = None,
    initial_refinement_checkpoint_path: str | Path | None = None,
    ablation: AblationSpec | None = None,
) -> dict[str, Any]:
    """Train while maintaining the run experiment record and output index."""
    config.validate()
    device_name = device or config.training.device
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be exactly 'cpu' or 'cuda'")
    stage = (
        "smoke"
        if config.training.max_train_files is not None
        or config.training.max_val_files is not None
        else "full"
    )
    run_dir = _run_directory(config, output_dir, device_name=device_name)
    if resume_state_path is not None and any(
        path is not None
        for path in (
            initial_full_checkpoint_path,
            initial_refinement_checkpoint_path,
        )
    ):
        raise ValueError("initial checkpoints and resume state are mutually exclusive")
    record_run_started(
        run_dir,
        device=device_name,
        mode="resume" if resume_state_path is not None else "fresh",
        stage=stage,
        note=note,
        config_source=config_source,
        resumed_from=resume_state_path,
        source_epoch=(
            int(
                torch.load(
                    Path(resume_state_path), map_location="cpu", weights_only=False
                )["epoch"]
            )
            if resume_state_path is not None
            else None
        ),
    )
    try:
        summary = _run_training_impl(
            config,
            output_dir=run_dir,
            device=device,
            progress=progress,
            resume_state_path=resume_state_path,
            initial_full_checkpoint_path=initial_full_checkpoint_path,
            initial_refinement_checkpoint_path=(
                initial_refinement_checkpoint_path
            ),
            ablation=ablation,
        )
    except BaseException as error:
        try:
            record_run_finished(run_dir, status="失败", failure=str(error))
            rebuild_output_index(run_dir.parent)
        except Exception as bookkeeping_error:
            if hasattr(error, "add_note"):
                error.add_note(f"artifact update also failed: {bookkeeping_error}")
        raise
    try:
        record_run_finished(run_dir, status="成功")
        rebuild_output_index(run_dir.parent)
    except Exception as bookkeeping_error:
        raise RuntimeError(
            "training core operation succeeded but artifact update failed"
        ) from bookkeeping_error
    return summary


def evaluate_checkpoint(
    config: TraceFormerConfig,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    progress: bool = False,
    ablation: AblationSpec | None = None,
    neighbor_backend: str = "cpu",
) -> dict[str, Any]:
    from traceformer.gpu.runtime import validate_backends
    from traceformer.models.gpu_neighbors import configure_neighbor_backend

    validate_backends(device=device, neighbor_backend=neighbor_backend)
    config.validate()
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    torch_device = runtime.device
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    _validate_checkpoint_contract(checkpoint, config, ablation=ablation)
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    if checkpoint.get("validation_manifest_sha256") != actual_validation_digest:
        raise ValueError("checkpoint validation dataset manifest does not match current data")
    model = build_model(config, ablation=ablation).to(torch_device)
    configure_neighbor_backend(model, neighbor_backend)
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
    config: TraceFormerConfig,
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
    config: TraceFormerConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    progress: bool = False,
    neighbor_backend: str = "cpu",
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
    from traceformer.gpu.runtime import validate_backends
    from traceformer.models.gpu_neighbors import configure_neighbor_backend

    validate_backends(device=device, neighbor_backend=neighbor_backend)
    config.validate()
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    torch_device = runtime.device
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    _validate_checkpoint_contract(checkpoint, config)
    validation_dataset = _dataset(config, "val")
    actual_validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    if checkpoint.get("validation_manifest_sha256") != actual_validation_digest:
        raise ValueError("checkpoint validation dataset manifest does not match current data")
    model = build_model(config).to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    configure_neighbor_backend(model, neighbor_backend)
    threshold = float(checkpoint.get("decision_threshold", config.inference.sigmoid_threshold))
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(validation_dataset)):
        sequence = validation_dataset[index]
        events = sequence.events.to(torch_device)
        with torch.no_grad():
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=torch_device
            )
            output, _ = _forward_with_sparse_mixture(
                model,
                sequence,
                events,
                batch_indices,
                config,
                mixture_weight=1.0,
                compute_sparse_logits=False,
            )
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
    "forward_with_sparse_mixture",
    "load_inference_model",
    "run_training",
]
