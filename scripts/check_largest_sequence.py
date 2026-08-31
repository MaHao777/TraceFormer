"""Run forward, production loss, and backward on HUAV's largest sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from huav.config import load_config
from huav.data import NPZEventDataset
from huav.training.engine import (
    _add_sparse_mixture_losses,
    _amp_is_enabled,
    _autocast_dtype,
    _dense_sequence_loss_weight,
    _event_refinement_loss,
    _forward_with_sparse_mixture,
    _initialize_full_path,
    _initialize_refinement_head,
    _sparse_mixture_auxiliary_losses,
    _sparse_mixture_weight,
    build_model,
)
from huav.training.losses import EventFocalDiceLoss
from huav.training.runtime import prepare_device
from huav.training.sampling import sample_event_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource-check a complete HUAV sequence before full training"
    )
    parser.add_argument(
        "--config",
        default="configs/huav_score_optimized_autodl.yaml",
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
        help="one-based epoch used for sparse-mixture scheduling",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initialize-full",
        help="历史最佳 best.pt；冻结 Full 配置必须提供",
    )
    initialization.add_argument(
        "--initialize-refinement",
        help="schema 5 best.pt；只训练多尺度事件精修头",
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
    elif config.training.stage == "sparse_mixture":
        if arguments.initialize_full is None:
            parser.error("frozen Full resource check requires --initialize-full")
        _initialize_full_path(
            model,
            config,
            arguments.initialize_full,
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
        mixture_weight = _sparse_mixture_weight(config, arguments.epoch)
        output, mixture_view = _forward_with_sparse_mixture(
            model,
            sequence,
            events,
            batch_indices,
            config,
            mixture_weight=mixture_weight,
        )
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
        if config.training.stage == "sparse_mixture":
            refinement_loss = refinement_loss * 0.0
        combined_total = (
            full_losses["total"]
            + config.frame_refinement.loss_weight * refinement_loss
        ) * _dense_sequence_loss_weight(events, labels, config)
        mixture_losses = _sparse_mixture_auxiliary_losses(
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
                mixture_losses,
                config,
            )
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
        "sparse_mixture": {
            "weight": mixture_weight,
            "ratio": (
                mixture_view.threshold_ratio if mixture_view is not None else None
            ),
            "retained_events": (
                int(len(mixture_view.events)) if mixture_view is not None else 0
            ),
            "sparse_tokens": int(output["sparse_token_present"].sum().item()),
            "gate_mean": (
                float(output["mixture_gate"].mean().detach().float().item())
                if len(output["mixture_gate"])
                else 0.0
            ),
            "losses": {
                name: float(value.detach().float().item())
                for name, value in mixture_losses.items()
            },
        },
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
                model.sparse_mixture,
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
