"""Event focal, Dice, object-frame, false-alarm, and residual STC losses."""

from __future__ import annotations

import itertools
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


def _cell_event_slots(
    cell_keys: torch.Tensor,
    *,
    slots_per_cell: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly retain capped representatives from every occupied cell."""
    event_count = len(cell_keys)
    order = torch.argsort(cell_keys, stable=True)
    sorted_keys = cell_keys[order]
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    cell_ids = torch.repeat_interleave(
        torch.arange(len(unique_keys), device=cell_keys.device), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(event_count, device=cell_keys.device) - torch.repeat_interleave(
        starts, counts
    )
    repeated_counts = torch.repeat_interleave(counts, counts)
    slots = torch.div(
        ranks * slots_per_cell,
        repeated_counts,
        rounding_mode="floor",
    ).clamp_max(slots_per_cell - 1)
    flat_slot_indices = cell_ids * slots_per_cell + slots
    flat_events = torch.full(
        (len(unique_keys) * slots_per_cell,),
        event_count,
        dtype=torch.long,
        device=cell_keys.device,
    )
    flat_events.scatter_reduce_(
        0,
        flat_slot_indices,
        order,
        reduce="amin",
        include_self=True,
    )
    return unique_keys, flat_events.view(len(unique_keys), slots_per_cell)


def _sparse_cone_neighbors(
    coordinates: torch.Tensor,
    timestamps: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    tau: float,
    r0: float,
    vmax: float,
    sigma_xy: float,
    sigma_t: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find capped exact-cone neighbors through sparse spacetime cells."""
    event_count = len(coordinates)
    if event_count == 0:
        empty_indices = torch.empty(
            (0, max_neighbors), dtype=torch.long, device=coordinates.device
        )
        empty_distances = torch.empty(
            (0, max_neighbors), dtype=torch.float32, device=coordinates.device
        )
        return empty_indices, empty_distances

    max_radius = max(r0 + vmax * tau, torch.finfo(torch.float32).eps)
    x_cells = torch.div(coordinates[:, 0], max_radius, rounding_mode="floor").long()
    y_cells = torch.div(coordinates[:, 1], max_radius, rounding_mode="floor").long()
    t_cells = torch.div(timestamps, tau, rounding_mode="floor").long()
    x_count = int(x_cells.max().item()) + 1
    y_count = int(y_cells.max().item()) + 1
    t_count = int(t_cells.max().item()) + 1

    def linear_key(
        batches: torch.Tensor,
        time_cells: torch.Tensor,
        y_cell_values: torch.Tensor,
        x_cell_values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            ((batches.long() * t_count + time_cells) * y_count + y_cell_values)
            * x_count
            + x_cell_values
        )

    event_cell_keys = linear_key(batch_indices, t_cells, y_cells, x_cells)
    unique_keys, cell_slots = _cell_event_slots(
        event_cell_keys, slots_per_cell=max_neighbors
    )
    sentinel = event_count
    best_indices = torch.full(
        (event_count, max_neighbors),
        sentinel,
        dtype=torch.long,
        device=coordinates.device,
    )
    best_distances = torch.full(
        (event_count, max_neighbors),
        torch.inf,
        dtype=torch.float32,
        device=coordinates.device,
    )
    row_indices = torch.arange(event_count, device=coordinates.device)[:, None]
    float_coordinates = coordinates.float()
    float_timestamps = timestamps.float()

    for delta_t, delta_y, delta_x in itertools.product((-1, 0, 1), repeat=3):
        target_t = t_cells + delta_t
        target_y = y_cells + delta_y
        target_x = x_cells + delta_x
        valid_cell = (
            (target_t >= 0)
            & (target_t < t_count)
            & (target_y >= 0)
            & (target_y < y_count)
            & (target_x >= 0)
            & (target_x < x_count)
        )
        target_keys = linear_key(batch_indices, target_t, target_y, target_x)
        positions = torch.searchsorted(unique_keys, target_keys)
        safe_positions = positions.clamp_max(len(unique_keys) - 1)
        occupied = (
            valid_cell
            & (positions < len(unique_keys))
            & (unique_keys[safe_positions] == target_keys)
        )
        candidates = cell_slots[safe_positions]
        candidates = torch.where(
            occupied[:, None], candidates, torch.full_like(candidates, sentinel)
        )
        safe_candidates = candidates.clamp_max(event_count - 1)
        spatial_delta = (
            float_coordinates[safe_candidates] - float_coordinates[:, None, :]
        )
        spatial_square = spatial_delta.square().sum(dim=-1)
        time_delta = (
            float_timestamps[safe_candidates] - float_timestamps[:, None]
        ).abs()
        cone_radius = r0 + vmax * time_delta
        valid_neighbor = (
            (candidates < sentinel)
            & (candidates != row_indices)
            & (time_delta <= tau)
            & (spatial_square <= cone_radius.square())
        )
        normalized_distance = (
            spatial_square / (sigma_xy * sigma_xy)
            + time_delta.square() / (sigma_t * sigma_t)
        )
        candidate_distances = torch.where(
            valid_neighbor,
            normalized_distance,
            torch.full_like(normalized_distance, torch.inf),
        )
        merged_distances = torch.cat((best_distances, candidate_distances), dim=1)
        merged_indices = torch.cat((best_indices, candidates), dim=1)
        best_distances, selected = torch.topk(
            merged_distances,
            k=max_neighbors,
            dim=1,
            largest=False,
            sorted=False,
        )
        best_indices = torch.gather(merged_indices, 1, selected)

    return best_indices, best_distances


def spatiotemporal_continuity_weight(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    timestamps: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    lambda_stc: float = 0.1,
    gamma_stc: float = 2.0,
    tau: float = 50.0,
    r0: float = 8.0,
    vmax: float = 0.128,
    sigma_xy: float = 8.0,
    sigma_t: float = 25.0,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Return detached residual weights in ``[1, 1 + lambda_stc]``."""
    targets = _validate_event_inputs(
        logits, targets, coordinates, timestamps, batch_indices
    )
    lambda_stc = _finite(lambda_stc, "lambda_stc")
    gamma_stc = _finite(gamma_stc, "gamma_stc")
    tau = _positive(tau, "tau")
    r0 = _finite(r0, "r0")
    vmax = _finite(vmax, "vmax")
    sigma_xy = _positive(sigma_xy, "sigma_xy")
    sigma_t = _positive(sigma_t, "sigma_t")
    max_neighbors = _positive_integer(max_neighbors, "max_neighbors")
    if not 0 <= lambda_stc <= 1:
        raise ValueError("lambda_stc must be in [0,1]")
    if gamma_stc < 0:
        raise ValueError("gamma_stc must be non-negative")
    if r0 < 0 or vmax < 0:
        raise ValueError("r0 and vmax must be non-negative")
    if len(logits) == 0 or lambda_stc == 0:
        return torch.ones_like(logits).detach()

    neighbor_indices, normalized_distances = _sparse_cone_neighbors(
        coordinates,
        timestamps,
        batch_indices,
        tau=tau,
        r0=r0,
        vmax=vmax,
        sigma_xy=sigma_xy,
        sigma_t=sigma_t,
        max_neighbors=max_neighbors,
    )
    valid_neighbors = neighbor_indices < len(logits)
    safe_neighbors = neighbor_indices.clamp_max(len(logits) - 1)
    kernel = torch.where(
        valid_neighbors,
        torch.exp(-0.5 * normalized_distances),
        torch.zeros_like(normalized_distances),
    )
    neighbor_probabilities = torch.sigmoid(logits.detach().float())[safe_neighbors]
    kernel_sum = kernel.sum(dim=1)
    support = (kernel * neighbor_probabilities).sum(dim=1) / kernel_sum.clamp_min(
        torch.finfo(kernel.dtype).eps
    )
    support = torch.where(kernel_sum > 0, support, torch.zeros_like(support))
    disagreement = (
        targets.float() * (1 - support).pow(gamma_stc)
        + (1 - targets.float()) * support.pow(gamma_stc)
    )
    weights = (1 + lambda_stc * disagreement).to(dtype=logits.dtype)
    upper = torch.as_tensor(1 + lambda_stc, dtype=logits.dtype, device=logits.device)
    if float(upper) > 1 + lambda_stc:
        upper = torch.nextafter(upper, torch.full_like(upper, -torch.inf))
    return weights.clamp(min=1.0, max=upper).detach()


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
        integer_time[positive], frame_ms, rounding_mode="floor"
    ).long()
    group_coordinates = torch.stack((object_ids[positive].long(), frame_indices), dim=1)
    _, inverse = torch.unique(group_coordinates, dim=0, return_inverse=True)
    group_maximum = torch.full(
        (int(inverse.max().item()) + 1,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    group_maximum.scatter_reduce_(
        0, inverse, logits[positive], reduce="amax", include_self=True
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
    """Historical best loss with a bounded residual STC focal weight."""

    def __init__(
        self,
        *,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_dice: float = 0.75,
        lambda_object_frame: float = 0.25,
        object_frame_logit_margin: float = 2.0,
        lambda_false_alarm: float = 0.05,
        false_alarm_logit_margin: float = 0.0,
        false_alarm_topk_per_frame: int = 32,
        lambda_stc: float = 0.1,
        gamma_stc: float = 2.0,
        stc_tau: float = 50.0,
        stc_r0: float = 8.0,
        stc_vmax: float = 0.128,
        stc_sigma_xy: float = 8.0,
        stc_sigma_t: float = 25.0,
        stc_max_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.focal_alpha = _finite(focal_alpha, "focal_alpha")
        self.focal_gamma = _finite(focal_gamma, "focal_gamma")
        self.lambda_dice = _finite(lambda_dice, "lambda_dice")
        self.lambda_object_frame = _finite(
            lambda_object_frame, "lambda_object_frame"
        )
        self.object_frame_logit_margin = _finite(
            object_frame_logit_margin, "object_frame_logit_margin"
        )
        self.lambda_false_alarm = _finite(
            lambda_false_alarm, "lambda_false_alarm"
        )
        self.false_alarm_logit_margin = _finite(
            false_alarm_logit_margin, "false_alarm_logit_margin"
        )
        self.false_alarm_topk_per_frame = _positive_integer(
            false_alarm_topk_per_frame, "false_alarm_topk_per_frame"
        )
        self.lambda_stc = _finite(lambda_stc, "lambda_stc")
        self.gamma_stc = _finite(gamma_stc, "gamma_stc")
        self.stc_tau = _positive(stc_tau, "stc_tau")
        self.stc_r0 = _finite(stc_r0, "stc_r0")
        self.stc_vmax = _finite(stc_vmax, "stc_vmax")
        self.stc_sigma_xy = _positive(stc_sigma_xy, "stc_sigma_xy")
        self.stc_sigma_t = _positive(stc_sigma_t, "stc_sigma_t")
        self.stc_max_neighbors = _positive_integer(
            stc_max_neighbors, "stc_max_neighbors"
        )
        if not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must be in [0,1]")
        if self.focal_gamma < 0 or self.gamma_stc < 0:
            raise ValueError("focal and STC gamma values must be non-negative")
        if self.lambda_dice <= 0:
            raise ValueError("lambda_dice must be positive")
        if self.lambda_object_frame < 0 or self.lambda_false_alarm < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if not 0 <= self.lambda_stc <= 1:
            raise ValueError("lambda_stc must be in [0,1]")
        if self.stc_r0 < 0 or self.stc_vmax < 0:
            raise ValueError("stc_r0 and stc_vmax must be non-negative")

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
        full_batch_indices: torch.Tensor | None = None,
        full_timestamps_ms: torch.Tensor | None = None,
        full_coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_indices is None:
            batch_indices = torch.zeros(
                len(logits), dtype=torch.long, device=logits.device
            )
        if self.lambda_stc:
            if coordinates is None or timestamps_ms is None:
                raise ValueError(
                    "coordinates and timestamps are required for STC weighting"
                )
            weights = spatiotemporal_continuity_weight(
                logits,
                targets,
                coordinates,
                timestamps_ms,
                batch_indices,
                lambda_stc=self.lambda_stc,
                gamma_stc=self.gamma_stc,
                tau=self.stc_tau,
                r0=self.stc_r0,
                vmax=self.stc_vmax,
                sigma_xy=self.stc_sigma_xy,
                sigma_t=self.stc_sigma_t,
                max_neighbors=self.stc_max_neighbors,
            )
        else:
            weights = torch.ones_like(logits).detach()
        focal = binary_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            weights=weights,
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
        stc_weight_mean = (
            weights.mean()
            if len(weights)
            else torch.ones((), dtype=logits.dtype, device=logits.device)
        )
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
            "stc_weight_mean": stc_weight_mean,
        }


__all__ = [
    "EventFocalDiceLoss",
    "binary_dice_loss",
    "binary_focal_loss",
    "false_alarm_pixel_loss",
    "object_frame_recall_loss",
    "spatiotemporal_continuity_weight",
]
