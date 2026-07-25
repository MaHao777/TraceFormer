"""Training losses, sampling, metrics and public interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .losses import (
    EventFocalDiceLoss,
    binary_dice_loss,
    binary_focal_loss,
    object_frame_recall_loss,
)
from .metrics import EventMetricAccumulator, EVUAVMetricAccumulator, competition_metrics
from .sampling import sample_event_indices


@dataclass(frozen=True)
class EvaluationResult:
    """Named target metrics reserved for the later evaluator implementation."""

    target_iou: float
    precision: float
    recall: float
    f1: float
    pd: float
    fa: float


class Trainer(Protocol):
    def fit(self) -> None:
        """Run later training work; intentionally unspecified in Phase 1."""


class Evaluator(Protocol):
    def evaluate(self) -> EvaluationResult:
        """Return target IoU, precision, recall, F1, PD and FA."""


__all__ = [
    "EventFocalDiceLoss",
    "EvaluationResult",
    "Evaluator",
    "EVUAVMetricAccumulator",
    "EventMetricAccumulator",
    "Trainer",
    "binary_dice_loss",
    "binary_focal_loss",
    "competition_metrics",
    "object_frame_recall_loss",
    "sample_event_indices",
]
