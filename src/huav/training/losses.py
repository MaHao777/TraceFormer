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


def adaptive_threshold_regularization(
    global_event_offsets: torch.Tensor,
    local_event_offsets: torch.Tensor,
    event_group_indices: torch.Tensor,
    group_batch_indices: torch.Tensor,
    neighbor_edge_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return magnitude, per-sequence centering and neighbor smoothness terms."""

    event_count = len(local_event_offsets)
    if global_event_offsets.shape != (event_count,):
        raise ValueError("global_event_offsets must have shape [N]")
    if event_group_indices.shape != (event_count,):
        raise ValueError("event_group_indices must have shape [N]")
    if group_batch_indices.ndim != 1:
        raise ValueError("group_batch_indices must have shape [G]")
    if neighbor_edge_indices.ndim != 2 or neighbor_edge_indices.shape[0] != 2:
        raise ValueError("neighbor_edge_indices must have shape [2,E]")
    if (
        global_event_offsets.device != local_event_offsets.device
        or event_group_indices.device != local_event_offsets.device
        or group_batch_indices.device != local_event_offsets.device
        or neighbor_edge_indices.device != local_event_offsets.device
    ):
        raise ValueError("adaptive-threshold regularization inputs must share a device")

    differentiable_zero = (
        global_event_offsets.sum() + local_event_offsets.sum()
    ) * 0.0
    if event_count == 0:
        return differentiable_zero, differentiable_zero, differentiable_zero
    if len(group_batch_indices) == 0:
        raise ValueError("group_batch_indices cannot be empty when events exist")
    if torch.any(event_group_indices < 0) or torch.any(
        event_group_indices >= len(group_batch_indices)
    ):
        raise ValueError("event_group_indices contain an out-of-range group")

    magnitude = (
        global_event_offsets.square().mean()
        + local_event_offsets.square().mean()
    )
    event_batches = group_batch_indices[event_group_indices]
    _, event_batch_inverse = torch.unique(
        event_batches,
        sorted=True,
        return_inverse=True,
    )
    batch_count = int(event_batch_inverse.max().item()) + 1
    batch_sums = local_event_offsets.new_zeros(batch_count)
    batch_sums.scatter_add_(0, event_batch_inverse, local_event_offsets)
    batch_counts = torch.bincount(
        event_batch_inverse,
        minlength=batch_count,
    ).to(local_event_offsets.dtype)
    center = (batch_sums / batch_counts).square().mean()

    if neighbor_edge_indices.shape[1] == 0:
        smoothness = local_event_offsets.sum() * 0.0
    else:
        if torch.any(neighbor_edge_indices < 0) or torch.any(
            neighbor_edge_indices >= len(group_batch_indices)
        ):
            raise ValueError("neighbor_edge_indices contain an out-of-range group")
        group_sums = local_event_offsets.new_zeros(len(group_batch_indices))
        group_sums.scatter_add_(0, event_group_indices, local_event_offsets)
        group_counts = torch.bincount(
            event_group_indices,
            minlength=len(group_batch_indices),
        ).to(local_event_offsets.dtype)
        group_offsets = group_sums / group_counts.clamp_min(1)
        targets, sources = neighbor_edge_indices
        smoothness = (
            group_offsets[targets] - group_offsets[sources]
        ).square().mean()
    return magnitude, center, smoothness


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
        lambda_threshold_magnitude: float = 0.0,
        lambda_threshold_smoothness: float = 0.0,
        lambda_threshold_center: float = 0.0,
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
        for value, name in (
            (lambda_threshold_magnitude, "lambda_threshold_magnitude"),
            (lambda_threshold_smoothness, "lambda_threshold_smoothness"),
            (lambda_threshold_center, "lambda_threshold_center"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
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
        self.lambda_threshold_magnitude = float(lambda_threshold_magnitude)
        self.lambda_threshold_smoothness = float(lambda_threshold_smoothness)
        self.lambda_threshold_center = float(lambda_threshold_center)

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
        global_threshold_offsets: torch.Tensor | None = None,
        local_threshold_offsets: torch.Tensor | None = None,
        threshold_group_indices: torch.Tensor | None = None,
        threshold_group_batch_indices: torch.Tensor | None = None,
        threshold_neighbor_edges: torch.Tensor | None = None,
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
        threshold_zero = logits.sum() * 0.0
        threshold_magnitude = threshold_zero
        threshold_center = threshold_zero
        threshold_smoothness = threshold_zero
        if self.lambda_threshold_magnitude:
            if (
                global_threshold_offsets is None
                or local_threshold_offsets is None
            ):
                raise ValueError(
                    "adaptive-threshold global and local offsets are required "
                    "for magnitude regularization"
                )
            if global_threshold_offsets.shape != local_threshold_offsets.shape:
                raise ValueError(
                    "adaptive-threshold global and local offsets must have "
                    "matching shape [N]"
                )
            if len(local_threshold_offsets) == 0:
                threshold_magnitude = (
                    global_threshold_offsets.sum()
                    + local_threshold_offsets.sum()
                ) * 0.0
            else:
                threshold_magnitude = (
                    global_threshold_offsets.square().mean()
                    + local_threshold_offsets.square().mean()
                )
        if self.lambda_threshold_center or self.lambda_threshold_smoothness:
            grouping_inputs = (
                local_threshold_offsets,
                threshold_group_indices,
                threshold_group_batch_indices,
            )
            if any(value is None for value in grouping_inputs):
                raise ValueError(
                    "adaptive-threshold local offsets, group indices and group "
                    "batches are required for center or smoothness regularization"
                )
            if (
                self.lambda_threshold_smoothness
                and threshold_neighbor_edges is None
            ):
                raise ValueError(
                    "adaptive-threshold neighbor edges are required for "
                    "smoothness regularization"
                )
            helper_global_offsets = (
                global_threshold_offsets
                if global_threshold_offsets is not None
                else torch.zeros_like(local_threshold_offsets)
            )
            helper_neighbor_edges = (
                threshold_neighbor_edges
                if threshold_neighbor_edges is not None
                else torch.empty(
                    (2, 0),
                    dtype=torch.long,
                    device=local_threshold_offsets.device,
                )
            )
            _, center, smoothness = adaptive_threshold_regularization(
                helper_global_offsets,
                local_threshold_offsets,  # type: ignore[arg-type]
                threshold_group_indices,  # type: ignore[arg-type]
                threshold_group_batch_indices,  # type: ignore[arg-type]
                helper_neighbor_edges,
            )
            if self.lambda_threshold_center:
                threshold_center = center
            if self.lambda_threshold_smoothness:
                threshold_smoothness = smoothness
        return {
            "total": (
                focal
                + self.lambda_dice * dice
                + self.lambda_object_frame * object_frame
                + self.lambda_false_alarm * false_alarm
                + self.lambda_threshold_magnitude * threshold_magnitude
                + self.lambda_threshold_center * threshold_center
                + self.lambda_threshold_smoothness * threshold_smoothness
            ),
            "focal": focal,
            "dice": dice,
            "object_frame": object_frame,
            "false_alarm": false_alarm,
            "threshold_magnitude": threshold_magnitude,
            "threshold_center": threshold_center,
            "threshold_smoothness": threshold_smoothness,
        }


__all__ = [
    "EventFocalDiceLoss",
    "adaptive_threshold_regularization",
    "binary_dice_loss",
    "binary_focal_loss",
    "false_alarm_pixel_loss",
    "object_frame_recall_loss",
]
