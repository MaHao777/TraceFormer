"""Epoch-level learning-rate policies used by TraceFormer training experiments."""

from __future__ import annotations

import math
from typing import Any

from traceformer.config import TrainingConfig


class LearningRateController:
    """Return deterministic epoch-start rates and maintain plateau state."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.current_rate = float(config.learning_rate)
        self.best_score = -math.inf
        self.bad_epochs = 0

    def phase(self, epoch: int) -> str:
        if self.config.lr_schedule == "constant":
            return "fast"
        return "fast" if epoch <= self.config.fast_learning_epochs else "fine"

    def rate_for_epoch(self, epoch: int) -> float:
        if not 1 <= epoch <= self.config.epochs:
            raise ValueError("epoch must be in [1, training.epochs]")
        schedule = self.config.lr_schedule
        peak = float(self.config.learning_rate)
        minimum = peak * float(self.config.min_learning_rate_ratio)
        if schedule == "constant":
            return peak
        if schedule == "legacy_cosine":
            progress = (epoch - 1) / max(1, self.config.epochs)
            return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
        if epoch <= self.config.fast_learning_epochs:
            return peak
        if schedule == "constant_then_fixed":
            return peak * float(self.config.fixed_learning_rate_ratio)
        if schedule == "constant_then_plateau":
            return max(minimum, self.current_rate)
        fine_epochs = self.config.epochs - self.config.fast_learning_epochs
        fine_index = epoch - self.config.fast_learning_epochs
        progress = fine_index / max(1, fine_epochs)
        if schedule == "constant_then_cosine":
            return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
        if schedule == "constant_then_exponential":
            if minimum <= 0:
                raise ValueError("exponential learning-rate floor must be positive")
            return peak * (minimum / peak) ** progress
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")

    def update(self, score: float, *, epoch: int) -> None:
        if self.config.lr_schedule != "constant_then_plateau":
            return
        if epoch <= self.config.fast_learning_epochs:
            self.best_score = max(self.best_score, float(score))
            return
        if float(score) >= self.best_score + self.config.plateau_min_delta:
            self.best_score = float(score)
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.config.plateau_patience:
            floor = self.config.learning_rate * self.config.min_learning_rate_ratio
            self.current_rate = max(floor, self.current_rate * self.config.plateau_factor)
            self.bad_epochs = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_rate": self.current_rate,
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_rate = float(state["current_rate"])
        self.best_score = float(state["best_score"])
        self.bad_epochs = int(state["bad_epochs"])


__all__ = ["LearningRateController"]
