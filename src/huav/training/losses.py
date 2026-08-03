"""Spatiotemporally weighted focal and false-alarm margin losses."""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_event_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    timestamps: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating dtype")
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits and targets must have matching shape [N]")
    if coordinates.shape != (len(logits), 2):
        raise ValueError("coordinates must have shape [N,2]")
    if timestamps.ndim != 1 or len(timestamps) != len(logits):
        raise ValueError("timestamps must have shape [N]")
    if batch_indices.ndim != 1 or len(batch_indices) != len(logits):
        raise ValueError("batch_indices must have shape [N]")
    tensors = (targets, coordinates, timestamps, batch_indices)
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
    return targets.to(dtype=logits.dtype)


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


def _unit_interval(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
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
    unique_keys, counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
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
    """Find capped exact-cone neighbors through sparse spacetime cells.

    Cells use the cone's maximum spatial radius and temporal extent. Only the
    27 adjacent cell offsets are examined, and each offset is processed as a
    tensor batch so memory grows as ``O(N * max_neighbors)``, never ``O(N^2)``.
    """
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
    x_cells = torch.div(
        coordinates[:, 0], max_radius, rounding_mode="floor"
    ).long()
    y_cells = torch.div(
        coordinates[:, 1], max_radius, rounding_mode="floor"
    ).long()
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
        event_cell_keys,
        slots_per_cell=max_neighbors,
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
        target_keys = linear_key(
            batch_indices, target_t, target_y, target_x
        )
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
        merged_distances = torch.cat(
            (best_distances, candidate_distances), dim=1
        )
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


def spatiotemporal_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    coordinates: torch.Tensor,
    timestamps: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    alpha: float = 0.55,
    gamma_focal: float = 2.0,
    gamma_stc: float = 2.0,
    eta_pos: float = 0.5,
    eta_neg: float = 0.5,
    tau: float = 50.0,
    r0: float = 8.0,
    vmax: float = 0.128,
    sigma_xy: float = 8.0,
    sigma_t: float = 25.0,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Stable focal loss weighted by detached local cone support.

    Coordinates are source pixels and timestamps are milliseconds. The support
    probabilities are detached so neighboring predictions only control the
    current event's weight and cannot form a shortcut gradient.
    """
    targets = _validate_event_inputs(
        logits, targets, coordinates, timestamps, batch_indices
    )
    alpha = _unit_interval(alpha, "alpha")
    gamma_focal = _finite(gamma_focal, "gamma_focal")
    gamma_stc = _finite(gamma_stc, "gamma_stc")
    eta_pos = _unit_interval(eta_pos, "eta_pos")
    eta_neg = _unit_interval(eta_neg, "eta_neg")
    tau = _positive(tau, "tau")
    r0 = _finite(r0, "r0")
    vmax = _finite(vmax, "vmax")
    sigma_xy = _positive(sigma_xy, "sigma_xy")
    sigma_t = _positive(sigma_t, "sigma_t")
    max_neighbors = _positive_integer(max_neighbors, "max_neighbors")
    if gamma_focal < 0 or gamma_stc < 0:
        raise ValueError("focal and STC gamma values must be non-negative")
    if r0 < 0 or vmax < 0:
        raise ValueError("r0 and vmax must be non-negative")
    if len(logits) == 0:
        return logits.sum() * 0.0

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

    probabilities = torch.sigmoid(logits)
    target_probability = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    positive_weight = eta_pos + (1 - eta_pos) * support.pow(gamma_stc)
    negative_weight = eta_neg + (1 - eta_neg) * (1 - support).pow(gamma_stc)
    continuity_weight = targets.float() * positive_weight + (1 - targets.float()) * negative_weight
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return (
        alpha_factor
        * continuity_weight.to(dtype=logits.dtype)
        * (1 - target_probability).pow(gamma_focal)
        * cross_entropy
    ).mean()


def margin_false_alarm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_indices: torch.Tensor,
    timestamps_ms: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    frame_ms: float = 50.0,
    top_k: int = 32,
    negative_margin: float = -1.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Push each frame's strongest background-pixel logits below a margin."""
    targets = _validate_event_inputs(
        logits, targets, coordinates, timestamps_ms, batch_indices
    )
    frame_ms = _positive(frame_ms, "frame_ms")
    top_k = _positive_integer(top_k, "top_k")
    negative_margin = _finite(negative_margin, "negative_margin")
    temperature = _positive(temperature, "temperature")
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
    pixel_coordinates = torch.cat(
        (
            batch_indices[background].long()[:, None],
            frame_indices[:, None],
            coordinates[background].long(),
        ),
        dim=1,
    )
    unique_pixels, pixel_inverse = torch.unique(
        pixel_coordinates, dim=0, return_inverse=True
    )
    pixel_maximum = torch.full(
        (len(unique_pixels),),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    pixel_maximum.scatter_reduce_(
        0,
        pixel_inverse,
        logits[background],
        reduce="amax",
        include_self=True,
    )
    _, frame_inverse = torch.unique(
        unique_pixels[:, :2], dim=0, return_inverse=True
    )
    frame_count = int(frame_inverse.max().item()) + 1

    # Two stable sorts provide a lexicographic (frame, descending logit) order.
    value_order = torch.argsort(pixel_maximum, descending=True, stable=True)
    grouped_order = value_order[
        torch.argsort(frame_inverse[value_order], stable=True)
    ]
    ordered_frames = frame_inverse[grouped_order]
    counts = torch.bincount(ordered_frames, minlength=frame_count)
    starts = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(len(grouped_order), device=logits.device) - torch.repeat_interleave(
        starts, counts
    )
    selected = ranks < top_k
    selected_frames = ordered_frames[selected]
    selected_logits = pixel_maximum[grouped_order[selected]]
    penalties = temperature * F.softplus(
        (selected_logits - negative_margin) / temperature
    )
    frame_sums = penalties.new_zeros(frame_count)
    frame_sums.scatter_add_(0, selected_frames, penalties)
    selected_counts = torch.bincount(
        selected_frames, minlength=frame_count
    ).to(dtype=penalties.dtype)
    return (frame_sums / selected_counts.clamp_min(1)).mean()


class EventSpatiotemporalLoss(nn.Module):
    """STC-Focal over sampled events plus margin Fa over all events."""

    def __init__(
        self,
        *,
        alpha: float = 0.55,
        gamma_focal: float = 2.0,
        gamma_stc: float = 2.0,
        eta_pos: float = 0.5,
        eta_neg: float = 0.5,
        tau: float = 50.0,
        r0: float = 8.0,
        vmax: float = 0.128,
        sigma_xy: float = 8.0,
        sigma_t: float = 25.0,
        max_neighbors: int = 32,
        lambda_fa: float = 0.5,
        fa_top_k: int = 32,
        fa_negative_margin: float = -1.0,
        fa_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.alpha = _unit_interval(alpha, "alpha")
        self.gamma_focal = _finite(gamma_focal, "gamma_focal")
        self.gamma_stc = _finite(gamma_stc, "gamma_stc")
        self.eta_pos = _unit_interval(eta_pos, "eta_pos")
        self.eta_neg = _unit_interval(eta_neg, "eta_neg")
        self.tau = _positive(tau, "tau")
        self.r0 = _finite(r0, "r0")
        self.vmax = _finite(vmax, "vmax")
        self.sigma_xy = _positive(sigma_xy, "sigma_xy")
        self.sigma_t = _positive(sigma_t, "sigma_t")
        self.max_neighbors = _positive_integer(max_neighbors, "max_neighbors")
        self.lambda_fa = _finite(lambda_fa, "lambda_fa")
        self.fa_top_k = _positive_integer(fa_top_k, "fa_top_k")
        self.fa_negative_margin = _finite(
            fa_negative_margin, "fa_negative_margin"
        )
        self.fa_temperature = _positive(fa_temperature, "fa_temperature")
        if self.gamma_focal < 0 or self.gamma_stc < 0:
            raise ValueError("focal and STC gamma values must be non-negative")
        if self.r0 < 0 or self.vmax < 0 or self.lambda_fa < 0:
            raise ValueError("r0, vmax and lambda_fa must be non-negative")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        coordinates: torch.Tensor,
        timestamps_ms: torch.Tensor,
        batch_indices: torch.Tensor,
        *,
        frame_ms: float = 50.0,
        full_logits: torch.Tensor | None = None,
        full_targets: torch.Tensor | None = None,
        full_batch_indices: torch.Tensor | None = None,
        full_timestamps_ms: torch.Tensor | None = None,
        full_coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stc_focal = spatiotemporal_focal_loss(
            logits,
            targets,
            coordinates,
            timestamps_ms,
            batch_indices,
            alpha=self.alpha,
            gamma_focal=self.gamma_focal,
            gamma_stc=self.gamma_stc,
            eta_pos=self.eta_pos,
            eta_neg=self.eta_neg,
            tau=self.tau,
            r0=self.r0,
            vmax=self.vmax,
            sigma_xy=self.sigma_xy,
            sigma_t=self.sigma_t,
            max_neighbors=self.max_neighbors,
        )
        if self.lambda_fa:
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
            false_alarm = margin_false_alarm_loss(
                full_logits,  # type: ignore[arg-type]
                full_targets,  # type: ignore[arg-type]
                full_batch_indices,  # type: ignore[arg-type]
                full_timestamps_ms,  # type: ignore[arg-type]
                full_coordinates,  # type: ignore[arg-type]
                frame_ms=frame_ms,
                top_k=self.fa_top_k,
                negative_margin=self.fa_negative_margin,
                temperature=self.fa_temperature,
            )
        else:
            false_alarm = logits.sum() * 0.0
        return {
            "total": stc_focal + self.lambda_fa * false_alarm,
            "stc_focal": stc_focal,
            "false_alarm": false_alarm,
        }


__all__ = [
    "EventSpatiotemporalLoss",
    "margin_false_alarm_loss",
    "spatiotemporal_focal_loss",
]
