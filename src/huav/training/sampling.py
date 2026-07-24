"""Positive-preserving event sampling for imbalanced training data."""

from __future__ import annotations

import torch


def sample_event_indices(
    labels: torch.Tensor,
    *,
    background_ratio: int = 10,
    generator: torch.Generator | None = None,
    max_background_without_positive: int = 4096,
) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must have shape [N]")
    if background_ratio < 0 or max_background_without_positive <= 0:
        raise ValueError("sampling limits are invalid")
    positive = torch.where(labels > 0.5)[0]
    negative = torch.where(labels <= 0.5)[0]
    maximum_negative = (
        background_ratio * len(positive)
        if len(positive)
        else min(len(negative), max_background_without_positive)
    )
    if len(negative) > maximum_negative:
        order = torch.randperm(len(negative), generator=generator, device=negative.device)
        negative = negative[order[:maximum_negative]]
    selected = torch.cat((positive, negative))
    if len(selected):
        order = torch.randperm(len(selected), generator=generator, device=selected.device)
        selected = selected[order]
    return selected


__all__ = ["sample_event_indices"]
