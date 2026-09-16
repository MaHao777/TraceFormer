"""Positive-preserving event sampling for imbalanced training data."""

from __future__ import annotations

import torch


def sample_event_indices(
    labels: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    background_ratio: int = 10,
    hard_negative_fraction: float = 0.0,
    generator: torch.Generator | None = None,
    max_background_without_positive: int = 4096,
) -> torch.Tensor:
    positive = torch.where(labels > 0.5)[0]
    negative = torch.where(labels <= 0.5)[0]
    maximum_negative = (
        background_ratio * len(positive)
        if len(positive)
        else min(len(negative), max_background_without_positive)
    )
    if len(negative) > maximum_negative:
        hard_count = min(
            maximum_negative,
            int(round(maximum_negative * hard_negative_fraction)),
        )
        hard_negative = negative.new_empty((0,))
        remaining_negative = negative
        if hard_count:
            negative_logits = logits.detach()[negative]  # type: ignore[union-attr]
            hard_local = torch.topk(
                negative_logits,
                k=hard_count,
                largest=True,
                sorted=False,
            ).indices
            hard_negative = negative[hard_local]
            keep_mask = torch.ones(
                len(negative), dtype=torch.bool, device=negative.device
            )
            keep_mask[hard_local] = False
            remaining_negative = negative[keep_mask]
        random_count = maximum_negative - hard_count
        if random_count:
            order = torch.randperm(
                len(remaining_negative),
                generator=generator,
                device=negative.device,
            )
            random_negative = remaining_negative[order[:random_count]]
            negative = torch.cat((hard_negative, random_negative))
        else:
            negative = hard_negative
    selected = torch.cat((positive, negative))
    if len(selected):
        order = torch.randperm(len(selected), generator=generator, device=selected.device)
        selected = selected[order]
    return selected


__all__ = ["sample_event_indices"]
