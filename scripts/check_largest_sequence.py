"""Run forward, production loss, and backward on HUAV's largest sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from huav.config import load_config
from huav.data import (
    NPZEventDataset,
    is_dense_balanced_event_stream,
    simulate_high_threshold_events,
)
from huav.training.engine import (
    _amp_is_enabled,
    _autocast_dtype,
    _dense_sequence_loss_weight,
    _threshold_augmentation_weight,
    _threshold_ratio_for_epoch,
    build_model,
)
from huav.training.losses import (
    EventFocalDiceLoss,
    object_frame_threshold_consistency_loss,
)
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
        help="one-based epoch used for threshold augmentation scheduling",
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
    if arguments.epoch <= 0:
        parser.error("--epoch must be positive")
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
    )
    events = sequence.events.to(device)
    labels = sequence.labels.to(device)
    object_ids = sequence.object_ids.to(device)
    timestamps_ms = sequence.timestamps_ms.to(device)
    batch_indices = torch.zeros(len(events), dtype=torch.long, device=device)
    sampling_generator = torch.Generator(device=device.type).manual_seed(
        config.training.seed
    )
    augmentation_weight = _threshold_augmentation_weight(config, arguments.epoch)
    threshold_view = None
    if augmentation_weight > 0 and is_dense_balanced_event_stream(
        sequence.events,
        min_events=config.threshold_augmentation.min_events,
        max_polarity_fraction=config.threshold_augmentation.max_polarity_fraction,
    ):
        threshold_ratio = _threshold_ratio_for_epoch(
            config,
            epoch=arguments.epoch,
            dataset_index=matching[0],
        )
        threshold_view = simulate_high_threshold_events(
            sequence.events,
            width=config.resolution.width,
            height=config.resolution.height,
            threshold_ratio=threshold_ratio,
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
        sparse_total = full_losses["total"] * 0.0
        consistency = full_losses["total"] * 0.0
        sparse_output = None
        sparse_query_indices = torch.empty(0, dtype=torch.long, device=device)
        if threshold_view is not None:
            source_indices = threshold_view.source_indices.to(device)
            sparse_events = threshold_view.events.to(device)
            sparse_labels = labels[source_indices]
            sparse_object_ids = object_ids[source_indices]
            sparse_timestamps = timestamps_ms[source_indices]
            sparse_batches = torch.zeros(
                len(sparse_events), dtype=torch.long, device=device
            )
            sparse_output = model(sparse_events, sparse_batches)
            sparse_query_indices = sample_event_indices(
                sparse_labels,
                logits=sparse_output["event_logits"],
                background_ratio=config.training.background_ratio,
                hard_negative_fraction=config.training.hard_negative_fraction,
                generator=sampling_generator,
                max_background_without_positive=(
                    config.training.max_background_without_positive
                ),
            )
            sparse_losses = criterion(
                sparse_output["event_logits"][sparse_query_indices],
                sparse_labels[sparse_query_indices],
                sparse_batches[sparse_query_indices],
                object_ids=sparse_object_ids[sparse_query_indices],
                timestamps_ms=sparse_timestamps[sparse_query_indices],
                frame_ms=config.evaluation.frame_ms,
                full_logits=sparse_output["event_logits"],
                full_targets=sparse_labels,
                full_batch_indices=sparse_batches,
                full_timestamps_ms=sparse_timestamps,
                full_coordinates=sparse_events[:, :2],
            )
            sparse_total = sparse_losses["total"]
            consistency = object_frame_threshold_consistency_loss(
                sparse_output["event_logits"],
                output["event_logits"],
                source_indices,
                labels,
                object_ids,
                timestamps_ms,
                frame_ms=config.evaluation.frame_ms,
            )
        combined_total = full_losses["total"] * _dense_sequence_loss_weight(
            events, labels, config
        )
        combined_total = combined_total + augmentation_weight * (
            config.threshold_augmentation.sparse_loss_weight * sparse_total
            + config.threshold_augmentation.consistency_loss_weight * consistency
        )
    combined_total.backward()
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
        "loss": float(combined_total.detach().float().item()),
        "losses": {
            name: float(value.detach().float().item())
            for name, value in full_losses.items()
        },
        "false_alarm": float(full_losses["false_alarm"].detach().float().item()),
        "focal": float(full_losses["focal"].detach().float().item()),
        "object_frame": float(
            full_losses["object_frame"].detach().float().item()
        ),
        "threshold_augmentation": {
            "weight": augmentation_weight,
            "ratio": (
                threshold_view.threshold_ratio if threshold_view is not None else None
            ),
            "retained_events": (
                int(len(threshold_view.events)) if threshold_view is not None else 0
            ),
            "sparse_tokens": (
                int(sparse_output["token_features"].shape[0])
                if sparse_output is not None
                else 0
            ),
            "sparse_queries": int(len(sparse_query_indices)),
            "sparse_loss": float(sparse_total.detach().float().item()),
            "consistency_loss": float(consistency.detach().float().item()),
        },
        "max_memory_allocated_bytes": peak_memory,
        "runtime": runtime.metadata,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
