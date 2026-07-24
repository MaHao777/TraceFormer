"""Sparse-to-dense x-t/y-t projections without constructing an x-y-t volume."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiScaleViews:
    xt: dict[int, torch.Tensor]
    yt: dict[int, torch.Tensor]


def normalized_time_to_grid_coordinate(
    normalized_time: torch.Tensor, time_bins: int
) -> torch.Tensor:
    """Map continuous time to a grid where row k is bin center ``(k+.5)/T``."""
    if time_bins <= 1:
        return torch.zeros_like(normalized_time)
    bin_coordinate = (normalized_time * time_bins - 0.5).clamp(0, time_bins - 1)
    return 2.0 * bin_coordinate / (time_bins - 1) - 1.0


def _validate_events(events: torch.Tensor, batch_indices: torch.Tensor) -> None:
    if events.ndim != 2 or events.shape[1] != 4 or not events.is_floating_point():
        raise ValueError("events must be a floating tensor with shape [N,4]")
    if batch_indices.ndim != 1 or batch_indices.dtype != torch.long:
        raise ValueError("batch_indices must be an int64 tensor with shape [N]")
    if len(events) != len(batch_indices):
        raise ValueError("events and batch_indices must align")
    if len(events) and (
        torch.any(events[:, 2] < 0)
        or torch.any(events[:, 2] > 1)
        or torch.any((events[:, 3] != 0) & (events[:, 3] != 1))
    ):
        raise ValueError("normalized time and polarity must be in [0,1]")


class MultiScaleEventProjector(nn.Module):
    """Create four-channel views at several temporal resolutions.

    Channel order is positive count, negative count, mean normalized time and
    latest normalized time. Counts can be log-compressed for model stability.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        time_bins: Iterable[int],
        normalize_counts: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.time_bins = tuple(int(value) for value in time_bins)
        self.normalize_counts = bool(normalize_counts)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("projection dimensions must be positive")
        if not self.time_bins or any(value <= 0 for value in self.time_bins):
            raise ValueError("time_bins must contain positive values")
        if len(set(self.time_bins)) != len(self.time_bins):
            raise ValueError("time_bins must be unique")

    def _one_view(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        bins: int,
        *,
        coordinate_column: int,
        spatial_size: int,
    ) -> torch.Tensor:
        time_index = torch.clamp((events[:, 2] * bins).long(), max=bins - 1)
        space_index = events[:, coordinate_column].long()
        flat_index = (
            (batch_indices * bins + time_index) * spatial_size + space_index
        )
        cells = batch_size * bins * spatial_size
        dtype = events.dtype
        device = events.device

        positive = torch.zeros(cells, dtype=dtype, device=device)
        negative = torch.zeros_like(positive)
        total = torch.zeros_like(positive)
        time_sum = torch.zeros_like(positive)
        latest = torch.zeros_like(positive)
        polarity = events[:, 3]
        positive.scatter_add_(0, flat_index, polarity)
        negative.scatter_add_(0, flat_index, 1.0 - polarity)
        total.scatter_add_(0, flat_index, torch.ones_like(polarity))
        time_sum.scatter_add_(0, flat_index, events[:, 2])
        latest.scatter_reduce_(0, flat_index, events[:, 2], reduce="amax", include_self=True)

        mean_time = time_sum / total.clamp_min(1.0)
        if self.normalize_counts:
            positive = torch.log1p(positive)
            negative = torch.log1p(negative)
        shape = (batch_size, bins, spatial_size)
        return torch.stack(
            (
                positive.reshape(shape),
                negative.reshape(shape),
                mean_time.reshape(shape),
                latest.reshape(shape),
            ),
            dim=1,
        )

    def forward(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor,
        *,
        batch_size: int,
    ) -> MultiScaleViews:
        _validate_events(events, batch_indices)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(batch_indices) and (
            torch.any(batch_indices < 0) or torch.any(batch_indices >= batch_size)
        ):
            raise ValueError("batch_indices are outside batch_size")
        xt: dict[int, torch.Tensor] = {}
        yt: dict[int, torch.Tensor] = {}
        for bins in self.time_bins:
            xt[bins] = self._one_view(
                events,
                batch_indices,
                batch_size,
                bins,
                coordinate_column=0,
                spatial_size=self.width,
            )
            yt[bins] = self._one_view(
                events,
                batch_indices,
                batch_size,
                bins,
                coordinate_column=1,
                spatial_size=self.height,
            )
        return MultiScaleViews(xt=xt, yt=yt)


def _gaussian_kernel(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    radius = max(1, int(round(3 * sigma)))
    values = [
        [exp(-((row * row + column * column) / (2 * sigma * sigma)))
         for column in range(-radius, radius + 1)]
        for row in range(-radius, radius + 1)
    ]
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)


def _soft_target(
    events: torch.Tensor,
    labels: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    batch_size: int,
    time_bins: int,
    coordinate_column: int,
    spatial_size: int,
    sigma: float,
) -> torch.Tensor:
    target = torch.zeros(
        (batch_size, 1, time_bins, spatial_size),
        dtype=events.dtype,
        device=events.device,
    )
    positive = labels > 0.5
    if torch.any(positive):
        selected = events[positive]
        selected_batch = batch_indices[positive]
        time_index = torch.clamp((selected[:, 2] * time_bins).long(), max=time_bins - 1)
        space_index = selected[:, coordinate_column].long()
        target[selected_batch, 0, time_index, space_index] = 1.0
        kernel = _gaussian_kernel(sigma, device=events.device, dtype=events.dtype)
        padding = kernel.shape[-1] // 2
        target = F.conv2d(target, kernel, padding=padding).clamp_max_(1.0)
    return target


def build_view_targets(
    events: torch.Tensor,
    labels: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    batch_size: int,
    width: int,
    height: int,
    time_bins: int,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_events(events, batch_indices)
    if labels.ndim != 1 or len(labels) != len(events):
        raise ValueError("labels must have shape [N]")
    return (
        _soft_target(
            events,
            labels,
            batch_indices,
            batch_size=batch_size,
            time_bins=time_bins,
            coordinate_column=0,
            spatial_size=width,
            sigma=sigma,
        ),
        _soft_target(
            events,
            labels,
            batch_indices,
            batch_size=batch_size,
            time_bins=time_bins,
            coordinate_column=1,
            spatial_size=height,
            sigma=sigma,
        ),
    )


__all__ = [
    "MultiScaleEventProjector",
    "MultiScaleViews",
    "build_view_targets",
    "normalized_time_to_grid_coordinate",
]
