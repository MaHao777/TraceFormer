"""Event focal, Dice, false-alarm, and differentiable trajectory losses."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating dtype")
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits and targets must have matching shape [N]")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    return targets.to(dtype=logits.dtype)


def _validate_event_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    timestamps: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    if coordinates.shape != (len(logits), 2):
        raise ValueError("coordinates must have shape [N,2]")
    if timestamps.ndim != 1 or len(timestamps) != len(logits):
        raise ValueError("timestamps must have shape [N]")
    if batch_indices.ndim != 1 or len(batch_indices) != len(logits):
        raise ValueError("batch_indices must have shape [N]")
    tensors = (coordinates, timestamps, batch_indices)
    if any(value.device != logits.device for value in tensors):
        raise ValueError("all loss inputs must be on the same device")
    if not coordinates.is_floating_point() or not timestamps.is_floating_point():
        raise TypeError("coordinates and timestamps must have floating dtypes")
    if batch_indices.is_floating_point() or batch_indices.dtype == torch.bool:
        raise TypeError("batch_indices must have an integer dtype")
    if len(logits):
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("coordinates must be finite")
        if not bool(torch.isfinite(timestamps).all()):
            raise ValueError("timestamps must be finite")
        if bool((coordinates < 0).any()) or bool((timestamps < 0).any()):
            raise ValueError("coordinates and timestamps must be non-negative")
        if bool((batch_indices < 0).any()):
            raise ValueError("batch_indices must be non-negative")
    return targets


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    alpha = _finite(alpha, "alpha")
    gamma = _finite(gamma, "gamma")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if weights is not None:
        if weights.shape != logits.shape or weights.device != logits.device:
            raise ValueError("weights must align with logits")
        if len(weights) and (
            not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
        ):
            raise ValueError("weights must be finite and non-negative")
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
    terms = alpha_factor * (1 - target_probability).pow(gamma) * cross_entropy
    if weights is not None:
        terms = terms * weights.to(dtype=logits.dtype)
    return terms.mean()


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    targets = _validate_inputs(logits, targets)
    if batch_indices is None:
        batch_indices = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
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


def _trajectory_centers(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    batch_indices: torch.Tensor | None,
    *,
    frame_ms: float = 50.0,
    image_width: int = 346,
    image_height: int = 260,
    roi_radius: float = 16.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return keys, GT centers, and soft predicted centers per object-frame."""
    if batch_indices is None:
        batch_indices = torch.zeros(
            len(logits), dtype=torch.long, device=logits.device
        )
    targets = _validate_event_inputs(
        logits, targets, coordinates, timestamps_ms, batch_indices
    )
    if object_ids.ndim != 1 or len(object_ids) != len(logits):
        raise ValueError("object_ids must have shape [N]")
    if object_ids.device != logits.device:
        raise ValueError("all loss inputs must be on the same device")
    if object_ids.is_floating_point() or object_ids.dtype == torch.bool:
        raise TypeError("object_ids must have an integer dtype")
    if len(object_ids) and bool((object_ids < 0).any()):
        raise ValueError("object_ids must be non-negative")
    frame_ms = _positive(frame_ms, "frame_ms")
    roi_radius = _positive(roi_radius, "roi_radius")
    eps = _positive(eps, "eps")
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 1
    ):
        raise ValueError("image_width must be an integer greater than one")
    if (
        isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 1
    ):
        raise ValueError("image_height must be an integer greater than one")
    if len(coordinates) and (
        bool((coordinates[:, 0] >= image_width).any())
        or bool((coordinates[:, 1] >= image_height).any())
    ):
        raise ValueError("coordinates must be inside the configured image resolution")

    empty_keys = torch.empty((0, 3), dtype=torch.long, device=logits.device)
    empty_centers = torch.empty(
        (0, 2), dtype=torch.float32, device=logits.device
    )
    if len(logits) == 0:
        return empty_keys, empty_centers, empty_centers

    frame_indices = torch.div(
        timestamps_ms.to(dtype=torch.float64), frame_ms, rounding_mode="floor"
    ).long()
    positive = (targets > 0.5) & (object_ids > 0)
    if not bool(positive.any()):
        return empty_keys, empty_centers, empty_centers

    group_keys, positive_inverse = torch.unique(
        torch.stack(
            (
                batch_indices[positive].long(),
                object_ids[positive].long(),
                frame_indices[positive],
            ),
            dim=1,
        ),
        dim=0,
        return_inverse=True,
    )
    coordinate_scale = torch.tensor(
        [image_width - 1, image_height - 1],
        dtype=torch.float32,
        device=coordinates.device,
    )
    normalized_coordinates = coordinates.float() / coordinate_scale
    gt_center_sums = torch.zeros(
        (len(group_keys), 2), dtype=torch.float32, device=logits.device
    )
    gt_center_sums.index_add_(
        0, positive_inverse, normalized_coordinates[positive]
    )
    gt_counts = torch.bincount(
        positive_inverse, minlength=len(group_keys)
    ).to(dtype=torch.float32)
    gt_centers = gt_center_sums / gt_counts[:, None]

    event_frame_keys = torch.stack(
        (batch_indices.long(), frame_indices), dim=1
    )
    group_frame_keys = group_keys[:, (0, 2)]
    _, combined_inverse = torch.unique(
        torch.cat((event_frame_keys, group_frame_keys), dim=0),
        dim=0,
        return_inverse=True,
    )
    event_frame_inverse = combined_inverse[: len(logits)]
    group_frame_inverse = combined_inverse[len(logits) :]
    event_order = torch.argsort(event_frame_inverse, stable=True)
    frame_counts = torch.bincount(event_frame_inverse)
    frame_starts = torch.cumsum(frame_counts, dim=0) - frame_counts
    group_order = torch.argsort(group_frame_inverse, stable=True)
    group_counts = torch.bincount(
        group_frame_inverse, minlength=len(frame_counts)
    )
    group_starts = torch.cumsum(group_counts, dim=0) - group_counts
    probabilities = torch.sigmoid(logits.float())
    occupied_frames = torch.nonzero(group_counts > 0, as_tuple=False).flatten()
    frame_predictions: list[torch.Tensor] = []
    for event_frame_index in occupied_frames.detach().cpu().tolist():
        event_start = int(frame_starts[event_frame_index])
        event_count = int(frame_counts[event_frame_index])
        frame_events = event_order[event_start : event_start + event_count]
        group_start = int(group_starts[event_frame_index])
        group_count = int(group_counts[event_frame_index])
        frame_groups = group_order[group_start : group_start + group_count]
        gt_center_pixels = gt_centers[frame_groups] * coordinate_scale
        in_roi = (
            (
                coordinates[frame_events].float()[None, :, :]
                - gt_center_pixels[:, None, :]
            )
            .abs()
            .le(roi_radius)
            .all(dim=2)
        )
        roi_probabilities = probabilities[frame_events][None, :] * in_roi.to(
            dtype=probabilities.dtype
        )
        frame_predictions.append(
            roi_probabilities @ normalized_coordinates[frame_events]
            / (roi_probabilities.sum(dim=1, keepdim=True) + eps)
        )
    predictions_in_group_order = torch.cat(frame_predictions, dim=0)
    predicted_centers = torch.empty_like(predictions_in_group_order).index_copy(
        0, group_order, predictions_in_group_order
    )
    return group_keys, gt_centers, predicted_centers


def _trajectory_position_from_centers(
    logits: torch.Tensor,
    gt_centers: torch.Tensor,
    predicted_centers: torch.Tensor,
) -> torch.Tensor:
    if len(gt_centers) == 0:
        return logits.sum() * 0.0
    return F.smooth_l1_loss(predicted_centers, gt_centers, reduction="mean")


def _trajectory_velocity_from_centers(
    logits: torch.Tensor,
    group_keys: torch.Tensor,
    gt_centers: torch.Tensor,
    predicted_centers: torch.Tensor,
) -> torch.Tensor:
    if len(group_keys) < 2:
        return logits.sum() * 0.0
    adjacent = (
        (group_keys[1:, 0] == group_keys[:-1, 0])
        & (group_keys[1:, 1] == group_keys[:-1, 1])
        & (group_keys[1:, 2] == group_keys[:-1, 2] + 1)
    )
    if not bool(adjacent.any()):
        return logits.sum() * 0.0
    gt_deltas = gt_centers[1:] - gt_centers[:-1]
    predicted_deltas = predicted_centers[1:] - predicted_centers[:-1]
    return F.smooth_l1_loss(
        predicted_deltas[adjacent], gt_deltas[adjacent], reduction="mean"
    )


def trajectory_position_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
    *,
    frame_ms: float = 50.0,
    image_width: int = 346,
    image_height: int = 260,
    roi_radius: float = 16.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize differentiable soft-centroid error in each GT-centered ROI."""
    _, gt_centers, predicted_centers = _trajectory_centers(
        logits,
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        batch_indices,
        frame_ms=frame_ms,
        image_width=image_width,
        image_height=image_height,
        roi_radius=roi_radius,
        eps=eps,
    )
    return _trajectory_position_from_centers(logits, gt_centers, predicted_centers)


def trajectory_velocity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    object_ids: torch.Tensor,
    timestamps_ms: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
    *,
    frame_ms: float = 50.0,
    image_width: int = 346,
    image_height: int = 260,
    roi_radius: float = 16.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize first-order centroid differences over adjacent valid frames."""
    group_keys, gt_centers, predicted_centers = _trajectory_centers(
        logits,
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        batch_indices,
        frame_ms=frame_ms,
        image_width=image_width,
        image_height=image_height,
        roi_radius=roi_radius,
        eps=eps,
    )
    return _trajectory_velocity_from_centers(
        logits, group_keys, gt_centers, predicted_centers
    )


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
    """Suppress the strongest background response at each frame/pixel."""
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
    topk_per_frame = _positive_integer(topk_per_frame, "topk_per_frame")
    if len(logits) == 0:
        return logits.sum() * 0.0

    integer_time = torch.floor(timestamps_ms.to(dtype=torch.float64))
    within_frame = torch.remainder(integer_time, frame_ms)
    background = (targets <= 0.5) & (within_frame > 0)
    if not bool(background.any()):
        return logits.sum() * 0.0
    frame_indices = torch.div(
        integer_time[background], frame_ms, rounding_mode="floor"
    ).long()
    group_coordinates = torch.cat(
        (
            batch_indices[background].long()[:, None],
            frame_indices[:, None],
            coordinates[background].long(),
        ),
        dim=1,
    )
    unique_groups, inverse = torch.unique(group_coordinates, dim=0, return_inverse=True)
    pixel_maximum = torch.full(
        (int(inverse.max().item()) + 1,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    pixel_maximum.scatter_reduce_(
        0, inverse, logits[background], reduce="amax", include_self=True
    )
    unique_frames, frame_inverse = torch.unique(
        unique_groups[:, :2], dim=0, return_inverse=True
    )
    frame_losses: list[torch.Tensor] = []
    for frame_index in range(len(unique_frames)):
        frame_pixels = pixel_maximum[frame_inverse == frame_index]
        hard_pixels = torch.topk(
            frame_pixels,
            k=min(topk_per_frame, len(frame_pixels)),
            largest=True,
            sorted=False,
        ).values
        frame_losses.append(F.softplus(hard_pixels - float(logit_margin)).mean())
    return torch.stack(frame_losses).mean()


class EventFocalDiceLoss(nn.Module):
    """Compact event classification, false-alarm, and trajectory objective."""

    def __init__(
        self,
        *,
        image_width: int = 346,
        image_height: int = 260,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_dice: float = 0.5,
        lambda_false_alarm: float = 0.05,
        false_alarm_logit_margin: float = 0.0,
        false_alarm_topk_per_frame: int = 32,
        lambda_trajectory_position: float = 1.0,
        lambda_trajectory_velocity: float = 0.5,
        trajectory_roi_radius: float = 16.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(image_width, bool)
            or not isinstance(image_width, int)
            or image_width <= 1
        ):
            raise ValueError("image_width must be an integer greater than one")
        if (
            isinstance(image_height, bool)
            or not isinstance(image_height, int)
            or image_height <= 1
        ):
            raise ValueError("image_height must be an integer greater than one")
        self.image_width = image_width
        self.image_height = image_height
        self.focal_alpha = _finite(focal_alpha, "focal_alpha")
        self.focal_gamma = _finite(focal_gamma, "focal_gamma")
        self.lambda_dice = _finite(lambda_dice, "lambda_dice")
        self.lambda_false_alarm = _finite(
            lambda_false_alarm, "lambda_false_alarm"
        )
        self.false_alarm_logit_margin = _finite(
            false_alarm_logit_margin, "false_alarm_logit_margin"
        )
        self.false_alarm_topk_per_frame = _positive_integer(
            false_alarm_topk_per_frame, "false_alarm_topk_per_frame"
        )
        self.lambda_trajectory_position = _finite(
            lambda_trajectory_position, "lambda_trajectory_position"
        )
        self.lambda_trajectory_velocity = _finite(
            lambda_trajectory_velocity, "lambda_trajectory_velocity"
        )
        self.trajectory_roi_radius = _positive(
            trajectory_roi_radius, "trajectory_roi_radius"
        )
        if not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must be in [0,1]")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if self.lambda_dice <= 0:
            raise ValueError("lambda_dice must be positive")
        if (
            self.lambda_false_alarm < 0
            or self.lambda_trajectory_position < 0
            or self.lambda_trajectory_velocity < 0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
        object_ids: torch.Tensor | None = None,
        timestamps_ms: torch.Tensor | None = None,
        frame_ms: float = 50.0,
        full_logits: torch.Tensor | None = None,
        full_targets: torch.Tensor | None = None,
        full_object_ids: torch.Tensor | None = None,
        full_batch_indices: torch.Tensor | None = None,
        full_timestamps_ms: torch.Tensor | None = None,
        full_coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_indices is None:
            batch_indices = torch.zeros(
                len(logits), dtype=torch.long, device=logits.device
            )
        focal = binary_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        dice = binary_dice_loss(logits, targets, batch_indices)
        if self.lambda_false_alarm and len(logits):
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

        trajectory_enabled = (
            self.lambda_trajectory_position > 0
            or self.lambda_trajectory_velocity > 0
        )
        if trajectory_enabled and len(logits):
            trajectory_inputs = (
                full_logits,
                full_targets,
                full_coordinates,
                full_object_ids,
                full_timestamps_ms,
                full_batch_indices,
            )
            if any(value is None for value in trajectory_inputs):
                raise ValueError(
                    "full-event logits, targets, coordinates, object ids, "
                    "timestamps and batches are required for trajectory loss"
                )
            group_keys, gt_centers, predicted_centers = _trajectory_centers(
                full_logits,  # type: ignore[arg-type]
                full_targets,  # type: ignore[arg-type]
                full_coordinates,  # type: ignore[arg-type]
                full_object_ids,  # type: ignore[arg-type]
                full_timestamps_ms,  # type: ignore[arg-type]
                full_batch_indices,  # type: ignore[arg-type]
                frame_ms=frame_ms,
                image_width=self.image_width,
                image_height=self.image_height,
                roi_radius=self.trajectory_roi_radius,
            )
            trajectory_position = _trajectory_position_from_centers(
                full_logits,  # type: ignore[arg-type]
                gt_centers,
                predicted_centers,
            )
            trajectory_velocity = _trajectory_velocity_from_centers(
                full_logits,  # type: ignore[arg-type]
                group_keys,
                gt_centers,
                predicted_centers,
            )
        else:
            trajectory_position = logits.sum() * 0.0
            trajectory_velocity = logits.sum() * 0.0
        return {
            "total": (
                focal
                + self.lambda_dice * dice
                + self.lambda_false_alarm * false_alarm
                + self.lambda_trajectory_position * trajectory_position
                + self.lambda_trajectory_velocity * trajectory_velocity
            ),
            "focal": focal,
            "dice": dice,
            "false_alarm": false_alarm,
            "trajectory_position": trajectory_position,
            "trajectory_velocity": trajectory_velocity,
        }


__all__ = [
    "EventFocalDiceLoss",
    "binary_dice_loss",
    "binary_focal_loss",
    "false_alarm_pixel_loss",
    "trajectory_position_loss",
    "trajectory_velocity_loss",
]
