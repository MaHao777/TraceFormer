"""Numerically stable event-level focal and per-sample Dice losses."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
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
    targets = targets.to(dtype=logits.dtype)
    if batch_indices is None:
        batch_indices = torch.zeros(
            len(logits), dtype=torch.long, device=logits.device
        )
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


def frame_cell_objectness_loss(
    cell_logits: torch.Tensor,
    event_cell_indices: torch.Tensor,
    event_targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """按单元内事件标签最大值监督对象置信度。"""

    if len(event_targets) == 0:
        return cell_logits.sum() * 0.0
    cell_targets = cell_logits.new_zeros(len(cell_logits))
    cell_targets.scatter_reduce_(
        0,
        event_cell_indices,
        event_targets.to(dtype=cell_logits.dtype),
        reduce="amax",
        include_self=True,
    )
    return binary_focal_loss(
        cell_logits,
        cell_targets,
        alpha=alpha,
        gamma=gamma,
    )


def object_frame_recall_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    *,
    frame_ms: float = 50.0,
    logit_margin: float = 2.0,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
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

    targets = targets.to(dtype=logits.dtype)
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
    "frame_cell_objectness_loss",
    "object_frame_recall_loss",
]
