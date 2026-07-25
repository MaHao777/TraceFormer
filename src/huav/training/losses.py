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


class EventFocalDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_dice: float = 1.0,
        lambda_object_frame: float = 0.0,
        object_frame_logit_margin: float = 2.0,
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
        if not math.isfinite(object_frame_logit_margin):
            raise ValueError("object_frame_logit_margin must be finite")
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.lambda_dice = float(lambda_dice)
        self.lambda_object_frame = float(lambda_object_frame)
        self.object_frame_logit_margin = float(object_frame_logit_margin)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        *,
        object_ids: torch.Tensor | None = None,
        timestamps_ms: torch.Tensor | None = None,
        frame_ms: float = 50.0,
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
        return {
            "total": (
                focal
                + self.lambda_dice * dice
                + self.lambda_object_frame * object_frame
            ),
            "focal": focal,
            "dice": dice,
            "object_frame": object_frame,
        }


__all__ = [
    "EventFocalDiceLoss",
    "binary_dice_loss",
    "binary_focal_loss",
    "object_frame_recall_loss",
]
