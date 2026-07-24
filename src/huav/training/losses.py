"""Loss functions for event and projection-view supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    target_probability = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_factor * (1 - target_probability).pow(gamma) * cross_entropy).mean()


def binary_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, smooth: float = 1.0
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    probabilities = torch.sigmoid(logits).reshape(-1)
    flat_targets = targets.reshape(-1)
    intersection = torch.sum(probabilities * flat_targets)
    denominator = probabilities.sum() + flat_targets.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def focal_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return binary_focal_loss(logits, targets) + binary_dice_loss(logits, targets)


class DualViewLoss(nn.Module):
    def __init__(
        self,
        *,
        view_event_weight: float = 0.25,
        view_weight: float = 0.5,
        consistency_weight: float = 0.1,
        temporal_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.view_event_weight = float(view_event_weight)
        self.view_weight = float(view_weight)
        self.consistency_weight = float(consistency_weight)
        self.temporal_weight = float(temporal_weight)
        if min(
            self.view_event_weight,
            self.view_weight,
            self.consistency_weight,
            self.temporal_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")

    def forward(
        self,
        output: dict[str, torch.Tensor],
        event_targets: torch.Tensor,
        xt_target: torch.Tensor,
        yt_target: torch.Tensor,
        *,
        auxiliary_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        event = focal_dice_loss(output["event_logits"], event_targets)
        view_event = 0.5 * (
            focal_dice_loss(output["xt_event_logits"], event_targets)
            + focal_dice_loss(output["yt_event_logits"], event_targets)
        )
        view = 0.5 * (
            focal_dice_loss(output["xt_logits"], xt_target)
            + focal_dice_loss(output["yt_logits"], yt_target)
        )
        consistency = F.l1_loss(
            torch.sigmoid(output["xt_event_logits"]),
            torch.sigmoid(output["yt_event_logits"]),
        )
        xt_probability = torch.sigmoid(output["xt_logits"])
        yt_probability = torch.sigmoid(output["yt_logits"])
        temporal = 0.5 * (
            torch.abs(xt_probability[:, :, 1:] - xt_probability[:, :, :-1]).mean()
            + torch.abs(yt_probability[:, :, 1:] - yt_probability[:, :, :-1]).mean()
        )
        scale = float(auxiliary_scale)
        total = (
            event
            + self.view_event_weight * view_event
            + self.view_weight * view
            + scale * self.consistency_weight * consistency
            + scale * self.temporal_weight * temporal
        )
        return {
            "total": total,
            "event": event,
            "view_event": view_event,
            "view": view,
            "consistency": consistency,
            "temporal": temporal,
        }


__all__ = [
    "DualViewLoss",
    "binary_dice_loss",
    "binary_focal_loss",
    "focal_dice_loss",
]
