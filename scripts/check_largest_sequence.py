"""Run forward, production loss, and backward on TraceFormer's largest sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.training.engine import (
    _amp_is_enabled,
    _autocast_dtype,
    _dense_sequence_loss_weight,
    _event_refinement_loss,
    _initialize_refinement_head,
    build_model,
)
from traceformer.training.losses import EventFocalDiceLoss
from traceformer.training.runtime import prepare_device
from traceformer.training.sampling import sample_event_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource-check a complete TraceFormer sequence before full training"
    )
    parser.add_argument(
        "--config",
        default="configs/traceformer_score_optimized_autodl.yaml",
        help="training configuration",
    )
    parser.add_argument(
        "--sequence",
        default="train_096.npz",
        help="NPZ filename under the configured train split; also check train_094.npz",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=7,
        help="one-based epoch recorded for the resource check",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initialize-refinement",
        help="当前三锥 best.pt；只训练多尺度事件精修头",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    runtime = prepare_device(
        arguments.device,
        num_threads=config.training.num_threads,
    )
    device = runtime.device
    dataset = NPZEventDataset(
        Path(config.dataset.root) / "train",
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    matching = [
        index
        for index, path in enumerate(dataset.files)
        if path.name == arguments.sequence
    ]
    if len(matching) != 1:
        parser.error(
            f"expected exactly one {arguments.sequence} under {dataset.root}"
        )
    sequence = dataset[matching[0]]
    if arguments.epoch <= 0:
        parser.error("--epoch must be positive")
    model = build_model(config).to(device)
    model.configure_training_stage(config.training.stage)
    if config.training.stage == "refinement":
        if arguments.initialize_refinement is None:
            parser.error(
                "frozen event refinement resource check requires "
                "--initialize-refinement"
            )
        _initialize_refinement_head(
            model,
            config,
            arguments.initialize_refinement,
            device=device,
        )
    model.train()
    pool_input_counts: list[int] = []
    pool_hook = model.tokenizer.key_projection.register_forward_pre_hook(
        lambda _module, inputs: pool_input_counts.append(len(inputs[0]))
    )
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
    events = sequence.events.to(device)
    labels = sequence.labels.to(device)
    object_ids = sequence.object_ids.to(device)
    timestamps_ms = sequence.timestamps_ms.to(device)
    batch_indices = torch.zeros(len(events), dtype=torch.long, device=device)
    sampling_generator = torch.Generator(device=device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, device)
    amp_dtype = _autocast_dtype(config.training.mixed_precision_dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(events, batch_indices)
        query_indices = sample_event_indices(
            labels,
            logits=output["event_logits"],
            background_ratio=config.training.background_ratio,
            hard_negative_fraction=config.training.hard_negative_fraction,
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
        combined_total = (
            full_losses["total"]
            + config.frame_refinement.loss_weight * refinement_loss
        ) * _dense_sequence_loss_weight(events, labels, config)
    combined_total.backward()
    pool_hook.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    result = {
        "sequence": str(sequence.source_path.resolve()),
        "event_count": sequence.length,
        "token_count": int(output["token_features"].shape[0]),
        "pool_input_count": pool_input_counts[0],
        "loss_query_count": int(len(query_indices)),
        "loss": float(combined_total.detach().float().item()),
        "losses": {
            name: float(value.detach().float().item())
            for name, value in full_losses.items()
        },
        "event_refinement": float(
            refinement_loss.detach().float().item()
        ),
        "false_alarm": float(full_losses["false_alarm"].detach().float().item()),
        "focal": float(full_losses["focal"].detach().float().item()),
        "object_frame": float(
            full_losses["object_frame"].detach().float().item()
        ),
        "max_memory_allocated_bytes": peak_memory,
        "runtime": runtime.metadata,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_base_has_gradient": any(
            parameter.grad is not None
            for module in (
                model.tokenizer,
                model.encoder,
                model.decoder,
            )
            if module is not None
            for parameter in module.parameters()
        ),
        "refinement_has_finite_gradient": bool(
            model.event_refinement_head is not None
            and all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.event_refinement_head.parameters()
            )
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
