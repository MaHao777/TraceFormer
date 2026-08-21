"""Physics-inspired higher-contrast-threshold event views."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HighThresholdView:
    """An ordered source-event subset emitted at a higher contrast threshold."""

    events: torch.Tensor
    source_indices: torch.Tensor
    threshold_ratio: float


def _validate_events(events: torch.Tensor, *, width: int, height: int) -> None:
    if not isinstance(events, torch.Tensor):
        raise TypeError("events must be a torch Tensor")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N,4]")
    if not events.is_floating_point():
        raise TypeError("events must have a floating dtype")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if len(events) == 0:
        return
    coordinates = events[:, :2]
    if not bool(torch.isfinite(events).all()):
        raise ValueError("events must be finite")
    if not bool(torch.equal(coordinates, coordinates.round())):
        raise ValueError("event coordinates must be integer-valued")
    if (
        bool(torch.any(coordinates[:, 0] < 0))
        or bool(torch.any(coordinates[:, 0] >= width))
        or bool(torch.any(coordinates[:, 1] < 0))
        or bool(torch.any(coordinates[:, 1] >= height))
    ):
        raise ValueError("event coordinates are outside configured bounds")
    polarity = events[:, 3]
    if bool(torch.any((polarity != 0) & (polarity != 1))):
        raise ValueError("event polarity must be 0 or 1")


def simulate_high_threshold_events(
    events: torch.Tensor,
    *,
    width: int,
    height: int,
    threshold_ratio: float,
) -> HighThresholdView:
    """Re-integrate signed contrast quanta and retain threshold crossings.

    Each source ON/OFF event contributes ``+1``/``-1`` to a persistent,
    per-pixel residual.  A source event is retained when that residual crosses
    ``+threshold_ratio`` or ``-threshold_ratio``.  The crossing event keeps its
    exact source values and index, so annotations can be aligned by the caller
    without entering the simulation itself.
    """

    _validate_events(events, width=width, height=height)
    if (
        isinstance(threshold_ratio, bool)
        or not isinstance(threshold_ratio, (int, float))
        or not math.isfinite(float(threshold_ratio))
        or float(threshold_ratio) <= 1.0
    ):
        raise ValueError("threshold_ratio must be finite and greater than 1")
    ratio = float(threshold_ratio)
    if len(events) == 0:
        source_indices = torch.empty(0, dtype=torch.long, device=events.device)
        return HighThresholdView(events=events.clone(), source_indices=source_indices, threshold_ratio=ratio)

    source = events.detach().cpu().numpy()
    x = source[:, 0].astype(np.int64, copy=False)
    y = source[:, 1].astype(np.int64, copy=False)
    signed_polarity = np.where(source[:, 3] > 0.5, 1.0, -1.0)
    residual = np.zeros((height, width), dtype=np.float64)
    keep = np.zeros(len(source), dtype=np.bool_)

    for index, (event_x, event_y, delta) in enumerate(
        zip(x, y, signed_polarity, strict=True)
    ):
        value = residual[event_y, event_x] + delta
        if value >= ratio:
            keep[index] = True
            value -= ratio
        elif value <= -ratio:
            keep[index] = True
            value += ratio
        residual[event_y, event_x] = value

    source_indices = torch.from_numpy(np.flatnonzero(keep).astype(np.int64, copy=False)).to(
        device=events.device
    )
    retained = events.index_select(0, source_indices)
    return HighThresholdView(
        events=retained,
        source_indices=source_indices,
        threshold_ratio=ratio,
    )


def is_dense_balanced_event_stream(
    events: torch.Tensor,
    *,
    min_events: int,
    max_polarity_fraction: float,
) -> bool:
    """Return whether a stream qualifies without inspecting annotations."""

    if isinstance(min_events, bool) or not isinstance(min_events, int) or min_events <= 0:
        raise ValueError("min_events must be a positive integer")
    if (
        isinstance(max_polarity_fraction, bool)
        or not isinstance(max_polarity_fraction, (int, float))
        or not math.isfinite(float(max_polarity_fraction))
        or not 0.5 <= float(max_polarity_fraction) <= 1.0
    ):
        raise ValueError("max_polarity_fraction must be in [0.5,1]")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N,4]")
    if len(events) < min_events:
        return False
    polarity_fraction = float(events[:, 3].float().mean().item())
    return max(polarity_fraction, 1.0 - polarity_fraction) <= float(
        max_polarity_fraction
    )


__all__ = [
    "HighThresholdView",
    "is_dense_balanced_event_stream",
    "simulate_high_threshold_events",
]
