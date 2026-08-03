"""Training losses, sampling, metrics and public interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .losses import (
    EventSpatiotemporalLoss,
    margin_false_alarm_loss,
    spatiotemporal_focal_loss,
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
    "EventSpatiotemporalLoss",
    "EvaluationResult",
    "Evaluator",
    "EVUAVMetricAccumulator",
    "EventMetricAccumulator",
    "Trainer",
    "competition_metrics",
    "margin_false_alarm_loss",
    "sample_event_indices",
    "spatiotemporal_focal_loss",
]
