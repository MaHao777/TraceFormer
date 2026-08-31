"""Validated, model-safe event data contracts for HUAV."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .dataset import LoadedEventSequence, NPZEventDataset
from .trigger_threshold import (
    HighThresholdView,
    is_dense_balanced_event_stream,
    simulate_high_threshold_events,
)


def _array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array or torch Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return value


def _integer_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    result = _array(value, name)
    if not np.issubdtype(result.dtype, np.integer) or result.dtype == np.bool_:
        raise TypeError(f"{name} must have an integer dtype")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return int(value)



@dataclass(frozen=True)
class EventSequence:
    """One ordered source sequence; :attr:`inputs` deliberately omits annotations."""

    x: np.ndarray | torch.Tensor
    y: np.ndarray | torch.Tensor
    t: np.ndarray | torch.Tensor
    p: np.ndarray | torch.Tensor
    width: int
    height: int
    label: np.ndarray | torch.Tensor | None = None
    object_id: np.ndarray | torch.Tensor | None = None
    source_path: Path | str | None = None
    source_name: str | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.width, "width")
        _positive_integer(self.height, "height")
        x, y, p = (_integer_array(self.x, "x"), _integer_array(self.y, "y"), _integer_array(self.p, "p"))
        t = _array(self.t, "t")
        if not (np.issubdtype(t.dtype, np.integer) or np.issubdtype(t.dtype, np.floating)) or t.dtype == np.bool_:
            raise TypeError("t must have a real integer or floating dtype")
        count = len(x)
        if any(len(item) != count for item in (y, t, p)):
            raise ValueError("x, y, t and p must have the same length")
        if np.any(x < 0) or np.any(x >= self.width) or np.any(y < 0) or np.any(y >= self.height):
            raise ValueError("event coordinates are outside the configured resolution")
        if np.any((p != 0) & (p != 1)):
            raise ValueError("p values must be 0 or 1")
        if not np.all(np.isfinite(t)) or np.any(t[1:] < t[:-1]):
            raise ValueError("t must be finite and monotonically non-decreasing")
        for name in ("label", "object_id"):
            optional = getattr(self, name)
            if optional is not None:
                checked = _integer_array(optional, name)
                if len(checked) != count:
                    raise ValueError(f"{name} must have the same length as x")
                if name == "label" and np.any((checked != 0) & (checked != 1)):
                    raise ValueError("label values must be 0 or 1")
                object.__setattr__(self, name, checked)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "p", p)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def length(self) -> int:
        return int(len(self.x))

    @property
    def inputs(self) -> Mapping[str, np.ndarray]:
        """Coordinates, timestamps and polarity only; never labels or object IDs."""
        return MappingProxyType({"x": self.x, "y": self.y, "t": self.t, "p": self.p})

    @property
    def source_stem(self) -> str:
        name = self.source_name or (self.source_path.name if self.source_path else None)
        if not name:
            raise ValueError("a source_name or source_path is required for a submission")
        return Path(name).stem



__all__ = [
    "EventSequence",
    "HighThresholdView",
    "LoadedEventSequence",
    "NPZEventDataset",
    "is_dense_balanced_event_stream",
    "simulate_high_threshold_events",
]
