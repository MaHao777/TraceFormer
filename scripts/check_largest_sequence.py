"""Run forward, production loss, and backward on HUAV's largest sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from huav.config import load_config
from huav.data import NPZEventDataset
from huav.training.engine import _amp_is_enabled, _autocast_dtype, build_model
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
        default="train_094.npz",
        help="NPZ filename under the configured train split",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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
    model = build_model(config).to(device).train()
    criterion = EventFocalDiceLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        lambda_dice=config.loss.lambda_dice,
        lambda_object_frame=config.loss.lambda_object_frame,
        object_frame_logit_margin=config.loss.object_frame_logit_margin,
        lambda_false_alarm=config.loss.lambda_false_alarm,
        false_alarm_logit_margin=config.loss.false_alarm_logit_margin,
        false_alarm_topk_per_frame=config.loss.false_alarm_topk_per_frame,
        lambda_stc=config.loss.lambda_stc,
        gamma_stc=config.loss.gamma_stc,
        stc_tau=config.loss.stc_tau,
        stc_r0=config.loss.stc_r0,
        stc_vmax=config.loss.stc_vmax,
        stc_sigma_xy=config.loss.stc_sigma_xy,
        stc_sigma_t=config.loss.stc_sigma_t,
        stc_max_neighbors=config.loss.stc_max_neighbors,
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
        losses = criterion(
            output["event_logits"][query_indices],
            labels[query_indices],
            batch_indices[query_indices],
            coordinates=events[query_indices, :2],
            object_ids=object_ids[query_indices],
            timestamps_ms=timestamps_ms[query_indices],
            frame_ms=config.evaluation.frame_ms,
            full_logits=output["event_logits"],
            full_targets=labels,
            full_batch_indices=batch_indices,
            full_timestamps_ms=timestamps_ms,
            full_coordinates=events[:, :2],
        )
    losses["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    result = {
        "sequence": str(sequence.source_path.resolve()),
        "event_count": sequence.length,
        "token_count": int(output["token_features"].shape[0]),
        "loss_query_count": int(len(query_indices)),
        "loss": float(losses["total"].detach().float().item()),
        "losses": {
            name: float(value.detach().float().item())
            for name, value in losses.items()
        },
        "false_alarm": float(losses["false_alarm"].detach().float().item()),
        "focal": float(losses["focal"].detach().float().item()),
        "stc_weight_mean": float(
            losses["stc_weight_mean"].detach().float().item()
        ),
        "max_memory_allocated_bytes": peak_memory,
        "runtime": runtime.metadata,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
