"""Numerically stable event-level focal and per-sample Dice losses."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits and targets must have matching shape [N]")
    return targets.to(dtype=logits.dtype)


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    if len(logits) == 0:
        return logits.sum() * 0.0
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probabilities = torch.sigmoid(logits)
    target_probability = probabilities * targets + (1 - probabilities) * (
        1 - targets
    )
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    return (
        alpha_factor
        * (1 - target_probability).pow(gamma)
        * cross_entropy
    ).mean()


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    if batch_indices is None:
        batch_indices = torch.zeros(
            len(logits), dtype=torch.long, device=logits.device
        )
    if batch_indices.ndim != 1 or len(batch_indices) != len(logits):
        raise ValueError("batch_indices must have shape [N]")
    if len(logits) == 0:
        return logits.sum() * 0.0
    probabilities = torch.sigmoid(logits)
    sample_losses: list[torch.Tensor] = []
    for batch_index in torch.unique(batch_indices):
        mask = batch_indices == batch_index
        sample_probabilities = probabilities[mask]
        sample_targets = targets[mask]
        intersection = torch.sum(sample_probabilities * sample_targets)
        denominator = sample_probabilities.sum() + sample_targets.sum()
        sample_losses.append(
            1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        )
    return torch.stack(sample_losses).mean()


def object_frame_recall_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    *,
    frame_ms: float = 50.0,
    logit_margin: float = 2.0,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    if not (
        object_ids.ndim == timestamps_ms.ndim == 1
        and len(object_ids) == len(timestamps_ms) == len(logits)
    ):
        raise ValueError("object ids and timestamps must align with logits")
    if frame_ms <= 0 or not math.isfinite(frame_ms):
        raise ValueError("frame_ms must be positive and finite")
    if not math.isfinite(logit_margin):
        raise ValueError("logit_margin must be finite")
    if len(logits) == 0:
        return logits.sum() * 0.0
    integer_time = torch.floor(timestamps_ms.to(dtype=torch.float64))
    within_frame = torch.remainder(integer_time, frame_ms)
    positive = (targets > 0.5) & (object_ids > 0) & (within_frame > 0)
    if not bool(positive.any()):
        return logits.sum() * 0.0
    frame_indices = torch.div(
        integer_time[positive],
        frame_ms,
        rounding_mode="floor",
    ).long()
    group_coordinates = torch.stack(
        (object_ids[positive].long(), frame_indices),
        dim=1,
    )
    _, inverse = torch.unique(group_coordinates, dim=0, return_inverse=True)
    group_maximum = torch.full(
        (int(inverse.max().item()) + 1,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    group_maximum.scatter_reduce_(
        0,
        inverse,
        logits[positive],
        reduce="amax",
        include_self=True,
    )
    return F.softplus(float(logit_margin) - group_maximum).mean()


def false_alarm_pixel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_indices: torch.Tensor,
    timestamps_ms: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    frame_ms: float = 50.0,
    logit_margin: float = 0.0,
    topk_per_frame: int = 32,
) -> torch.Tensor:
    """Suppress the strongest background response at each frame/pixel.

    The competition counts connected components of false-positive pixels in
    50 ms frames. Connectivity is discrete, so this differentiable surrogate
    first collapses repeated background events at the same frame/pixel to their
    maximum logit. Reducing those pixel maxima removes the source pixels from
    which false connected components are formed.
    """

    targets = _validate_inputs(logits, targets)
    if not (
        batch_indices.ndim == timestamps_ms.ndim == 1
        and len(batch_indices) == len(timestamps_ms) == len(logits)
    ):
        raise ValueError("batch indices and timestamps must align with logits")
    if coordinates.shape != (len(logits), 2):
        raise ValueError("coordinates must have shape [N,2]")
    if frame_ms <= 0 or not math.isfinite(frame_ms):
        raise ValueError("frame_ms must be positive and finite")
    if not math.isfinite(logit_margin):
        raise ValueError("logit_margin must be finite")
    if (
        isinstance(topk_per_frame, bool)
        or not isinstance(topk_per_frame, int)
        or topk_per_frame <= 0
    ):
        raise ValueError("topk_per_frame must be a positive integer")
    if len(logits) == 0:
        return logits.sum() * 0.0

    integer_time = torch.floor(timestamps_ms.to(dtype=torch.float64))
    within_frame = torch.remainder(integer_time, frame_ms)
    background = (targets <= 0.5) & (within_frame > 0)
    if not bool(background.any()):
        return logits.sum() * 0.0

    frame_indices = torch.div(
        integer_time[background],
        frame_ms,
        rounding_mode="floor",
    ).long()
    group_coordinates = torch.cat(
        (
            batch_indices[background].long()[:, None],
            frame_indices[:, None],
            coordinates[background].long(),
        ),
        dim=1,
    )
    unique_group_coordinates, inverse = torch.unique(
        group_coordinates,
        dim=0,
        return_inverse=True,
    )
    pixel_maximum = torch.full(
        (int(inverse.max().item()) + 1,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    pixel_maximum.scatter_reduce_(
        0,
        inverse,
        logits[background],
        reduce="amax",
        include_self=True,
    )
    frame_coordinates = unique_group_coordinates[:, :2]
    unique_frames, frame_inverse = torch.unique(
        frame_coordinates,
        dim=0,
        return_inverse=True,
    )
    frame_losses: list[torch.Tensor] = []
    for frame_index in range(len(unique_frames)):
        frame_pixels = pixel_maximum[frame_inverse == frame_index]
        hard_pixels = torch.topk(
            frame_pixels,
            k=min(int(topk_per_frame), len(frame_pixels)),
            largest=True,
            sorted=False,
        ).values
        frame_losses.append(
            F.softplus(hard_pixels - float(logit_margin)).mean()
        )
    return torch.stack(frame_losses).mean()


def object_frame_threshold_consistency_loss(
    sparse_logits: torch.Tensor,
    full_logits: torch.Tensor,
    source_indices: torch.Tensor,
    targets: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    *,
    frame_ms: float = 50.0,
) -> torch.Tensor:
    """Match target-frame confidence across full and high-threshold views."""

    if sparse_logits.ndim != 1 or full_logits.ndim != 1:
        raise ValueError("full and sparse logits must be one-dimensional")
    if source_indices.ndim != 1 or source_indices.dtype != torch.long:
        raise TypeError("source_indices must be a one-dimensional int64 tensor")
    if len(sparse_logits) != len(source_indices):
        raise ValueError("sparse logits must align with source_indices")
    if not (
        targets.ndim == object_ids.ndim == timestamps_ms.ndim == 1
        and len(targets) == len(object_ids) == len(timestamps_ms) == len(full_logits)
    ):
        raise ValueError("full-view annotations must align with full logits")
    if len(source_indices) and (
        bool(torch.any(source_indices < 0))
        or bool(torch.any(source_indices >= len(full_logits)))
    ):
        raise ValueError("source_indices are outside full logits")
    if frame_ms <= 0 or not math.isfinite(frame_ms):
        raise ValueError("frame_ms must be positive and finite")
    if len(source_indices) == 0:
        return sparse_logits.sum() * 0.0

    selected_targets = targets[source_indices]
    selected_objects = object_ids[source_indices]
    selected_timestamps = timestamps_ms[source_indices]
    integer_time = torch.floor(selected_timestamps.to(dtype=torch.float64))
    within_frame = torch.remainder(integer_time, frame_ms)
    positive = (
        (selected_targets > 0.5)
        & (selected_objects > 0)
        & (within_frame > 0)
    )
    if not bool(positive.any()):
        return sparse_logits.sum() * 0.0

    frame_indices = torch.div(
        integer_time[positive], frame_ms, rounding_mode="floor"
    ).long()
    groups = torch.stack((selected_objects[positive].long(), frame_indices), dim=1)
    _, inverse = torch.unique(groups, dim=0, return_inverse=True)
    group_count = int(inverse.max().item()) + 1

    def group_maximum(values: torch.Tensor) -> torch.Tensor:
        maximum = torch.full(
            (group_count,),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        )
        maximum.scatter_reduce_(
            0,
            inverse,
            values[positive],
            reduce="amax",
            include_self=True,
        )
        return maximum

    sparse_maximum = group_maximum(sparse_logits)
    full_maximum = group_maximum(full_logits[source_indices].detach())
    return F.smooth_l1_loss(
        torch.sigmoid(sparse_maximum),
        torch.sigmoid(full_maximum),
    )


class EventFocalDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_dice: float = 1.0,
        lambda_object_frame: float = 0.0,
        object_frame_logit_margin: float = 2.0,
        lambda_false_alarm: float = 0.0,
        false_alarm_logit_margin: float = 0.0,
        false_alarm_topk_per_frame: int = 32,
    ) -> None:
        super().__init__()
        if not 0 <= focal_alpha <= 1:
            raise ValueError("focal_alpha must be in [0,1]")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if lambda_dice <= 0:
            raise ValueError("lambda_dice must be positive")
        if lambda_object_frame < 0:
            raise ValueError("lambda_object_frame must be non-negative")
        if lambda_false_alarm < 0:
            raise ValueError("lambda_false_alarm must be non-negative")
        if not math.isfinite(object_frame_logit_margin):
            raise ValueError("object_frame_logit_margin must be finite")
        if not math.isfinite(false_alarm_logit_margin):
            raise ValueError("false_alarm_logit_margin must be finite")
        if (
            isinstance(false_alarm_topk_per_frame, bool)
            or not isinstance(false_alarm_topk_per_frame, int)
            or false_alarm_topk_per_frame <= 0
        ):
            raise ValueError(
                "false_alarm_topk_per_frame must be a positive integer"
            )
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.lambda_dice = float(lambda_dice)
        self.lambda_object_frame = float(lambda_object_frame)
        self.object_frame_logit_margin = float(object_frame_logit_margin)
        self.lambda_false_alarm = float(lambda_false_alarm)
        self.false_alarm_logit_margin = float(false_alarm_logit_margin)
        self.false_alarm_topk_per_frame = int(false_alarm_topk_per_frame)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        *,
        object_ids: torch.Tensor | None = None,
        timestamps_ms: torch.Tensor | None = None,
        frame_ms: float = 50.0,
        full_logits: torch.Tensor | None = None,
        full_targets: torch.Tensor | None = None,
        full_batch_indices: torch.Tensor | None = None,
        full_timestamps_ms: torch.Tensor | None = None,
        full_coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        focal = binary_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        dice = binary_dice_loss(logits, targets, batch_indices)
        if self.lambda_object_frame:
            if object_ids is None or timestamps_ms is None:
                raise ValueError(
                    "object ids and timestamps are required for object-frame loss"
                )
            object_frame = object_frame_recall_loss(
                logits,
                targets,
                object_ids,
                timestamps_ms,
                frame_ms=frame_ms,
                logit_margin=self.object_frame_logit_margin,
            )
        else:
            object_frame = logits.sum() * 0.0
        if self.lambda_false_alarm:
            full_inputs = (
                full_logits,
                full_targets,
                full_batch_indices,
                full_timestamps_ms,
                full_coordinates,
            )
            if any(value is None for value in full_inputs):
                raise ValueError(
                    "full-event logits, targets, batches, timestamps and "
                    "coordinates are required for false-alarm loss"
                )
            false_alarm = false_alarm_pixel_loss(
                full_logits,  # type: ignore[arg-type]
                full_targets,  # type: ignore[arg-type]
                full_batch_indices,  # type: ignore[arg-type]
                full_timestamps_ms,  # type: ignore[arg-type]
                full_coordinates,  # type: ignore[arg-type]
                frame_ms=frame_ms,
                logit_margin=self.false_alarm_logit_margin,
                topk_per_frame=self.false_alarm_topk_per_frame,
            )
        else:
            false_alarm = logits.sum() * 0.0
        return {
            "total": (
                focal
                + self.lambda_dice * dice
                + self.lambda_object_frame * object_frame
                + self.lambda_false_alarm * false_alarm
            ),
            "focal": focal,
            "dice": dice,
            "object_frame": object_frame,
            "false_alarm": false_alarm,
        }


__all__ = [
    "EventFocalDiceLoss",
    "binary_dice_loss",
    "binary_focal_loss",
    "false_alarm_pixel_loss",
    "object_frame_threshold_consistency_loss",
    "object_frame_recall_loss",
]
