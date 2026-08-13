"""Public model contracts and the dual-view event classifier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from huav.data import EventBatch

from .spatiotemporal_cone import SpatioTemporalConeDetector
from .token_importance import TokenImportance, TokenImportanceScores


@dataclass(frozen=True)
class ModelOutput:
    """Dense logits indexed as ``[batch, time_bin, y, x]``."""

    voxel_logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.voxel_logits, torch.Tensor):
            raise TypeError("voxel_logits must be a torch Tensor")
        if self.voxel_logits.ndim != 4:
            raise ValueError("voxel_logits must have shape [B,T,H,W]")
        if not self.voxel_logits.is_floating_point():
            raise TypeError("voxel_logits must have a floating dtype")
        if any(size <= 0 for size in self.voxel_logits.shape):
            raise ValueError("voxel_logits dimensions B, T, H and W must all be positive")


class EventSegmenter(ABC):
    """Abstract event segmentation model contract."""

    @abstractmethod
    def forward(self, batch: EventBatch) -> ModelOutput:
        """Return voxel logits aligned to the batch's ``[B,T,H,W]`` grid."""


__all__ = [
    "EventSegmenter",
    "ModelOutput",
    "SpatioTemporalConeDetector",
    "TokenImportance",
    "TokenImportanceScores",
]
