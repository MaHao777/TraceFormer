"""S1 与论文消融共用的原始联合训练循环、损失和选模逻辑。"""
from __future__ import annotations

import json
import math
import random
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

from traceformer.config import TraceFormerConfig, normalize_model_config
from traceformer.data import LoadedEventSequence, NPZEventDataset
from traceformer.models import TraceFormer
from traceformer.variants import PaperSpec as AblationSpec
from traceformer.variants import build_model, execution_contract

from .losses import EventFocalDiceLoss, frame_cell_objectness_loss
from .lr_schedules import LearningRateController
from .metrics import EventMetricAccumulator, EVUAVMetricAccumulator, competition_metrics
from .runtime import prepare_device
from .sampling import sample_event_indices

CHECKPOINT_SCHEMA_VERSION = 8
TRAINING_STATE_SCHEMA_VERSION = 4
REFINEMENT_MODEL_FIELDS = ("context_dim", "scales")

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
    """校验唯一三锥架构，并规范化已训练权重的结构元数据。"""

    if not isinstance(value, dict):
        return None
    if value.get("architecture") != "prefix_cone":
        return None
    return normalize_model_config(value)


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """优化器只接收真正允许更新的参数。"""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


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
    """统一校验 schema-8 推理与恢复契约。"""

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

    if checkpoint.get("duration_ms") != config.dataset.duration_ms:
        raise ValueError("checkpoint time normalization differs")
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
    contract["model"] = normalize_model_config(contract["model"])
    training = contract["training"]
    for name in ("device", "output_root"):
        training.pop(name, None)
    contract["dataset"].pop("root", None)
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
    best_score: float,
    best_metrics: dict[str, Any],
    best_checkpoint: dict[str, Any],
    order_generator: torch.Generator,
    sampling_generator: torch.Generator,
    torch_device: torch.device,
) -> dict[str, Any]:
    return {
        "execution_chunks": execution_contract(),
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
    model.configure_training_stage("full")
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
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    best_checkpoint: dict[str, Any] | None = None
    start_epoch = 0
    if resume_state_path is not None:
        state = torch.load(
            Path(resume_state_path), map_location=torch_device, weights_only=False
        )
        if state.get("execution_chunks") != execution_contract():
            raise ValueError("training execution chunk settings changed")
        if state.get("training_state_schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported training state schema version")
        if _branch_contract(state["branch_contract"]) != _branch_contract(configuration):
            raise ValueError("training state branch contract does not match configuration")
        if state.get("dataset_manifest_sha256") != manifest_digest:
            raise ValueError("training state dataset manifest does not match current data")
        _validate_checkpoint_contract(state["best_checkpoint"], config, ablation=ablation)
        initialization = state["best_checkpoint"].get("full_initialization")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        history = list(state["history"])
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
        model.train()
        totals = {
            "total": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "object_frame": 0.0,
            "false_alarm": 0.0,
            "event_refinement": 0.0,
        }
        epoch_started = time.perf_counter()
        epoch_parameters = _parameter_snapshot(model)
        gradient_norms: list[float] = []
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
                sequence_weight = _dense_sequence_loss_weight(
                    events,
                    labels,
                    config,
                )
                combined_total = (
                    full_losses["total"]
                    + config.frame_refinement.loss_weight * refinement_loss
                ) * sequence_weight
                losses = dict(full_losses)
                losses["total"] = combined_total
                losses["event_refinement"] = refinement_loss
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"non-finite training loss at epoch {epoch_number}, step {step + 1}"
                )
            # 空事件样本可能没有可训练路径；其余样本正常反向。
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
            best_checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state": {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                },
                "epoch": epoch + 1,
                "decision_threshold": float(metrics["threshold"]),
                "model_config": configuration["model"],
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
