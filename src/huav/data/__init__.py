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


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative non-boolean integer")
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


def _tensor(value: torch.Tensor, name: str, dimensions: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return value


def _binary_labels(labels: torch.Tensor | None, count: int) -> torch.Tensor | None:
    if labels is None:
        return None
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.dtype not in (torch.uint8, torch.int64):
        raise TypeError("labels must be a one-dimensional uint8 or int64 tensor")
    if len(labels) != count or torch.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must align with events and contain only 0 or 1")
    return labels


@dataclass(frozen=True)
class EventWindow:
    """A single voxelized window and its event-to-output-voxel mapping."""

    volume: torch.Tensor
    event_indices: torch.Tensor
    voxel_indices: torch.Tensor
    time_bins: int
    height: int
    width: int
    sequence_length: int | None = None
    labels: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.time_bins, "time_bins")
        _positive_integer(self.height, "height")
        _positive_integer(self.width, "width")
        if self.sequence_length is not None:
            _nonnegative_integer(self.sequence_length, "sequence_length")
        _tensor(self.volume, "volume", 3, torch.float32)
        if tuple(self.volume.shape) != (2 * self.time_bins, self.height, self.width):
            raise ValueError("volume must have shape [2*T,H,W]")
        _tensor(self.event_indices, "event_indices", 1, torch.int64)
        _tensor(self.voxel_indices, "voxel_indices", 2, torch.int64)
        if self.voxel_indices.shape != (len(self.event_indices), 3):
            raise ValueError("voxel_indices must have shape [N,3] aligned with event_indices")
        if torch.any(self.event_indices < 0) or (self.sequence_length is not None and torch.any(self.event_indices >= self.sequence_length)):
            raise ValueError("event_indices are outside the source sequence")
        if torch.unique(self.event_indices).numel() != len(self.event_indices):
            raise ValueError("event_indices must be unique within an EventWindow")
        limits = torch.tensor([self.time_bins, self.height, self.width], device=self.voxel_indices.device)
        if torch.any(self.voxel_indices < 0) or torch.any(self.voxel_indices >= limits):
            raise ValueError("voxel_indices are outside [T,H,W]")
        _binary_labels(self.labels, len(self.event_indices))


@dataclass(frozen=True)
class EventBatch:
    """Windows from one source sequence; event indices may overlap across batches, never within a pair."""

    volumes: torch.Tensor
    batch_indices: torch.Tensor
    event_indices: torch.Tensor
    voxel_indices: torch.Tensor
    time_bins: int
    height: int
    width: int
    sequence_length: int | None = None
    labels: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _tensor(self.volumes, "volumes", 4, torch.float32)
        _positive_integer(self.time_bins, "time_bins")
        _positive_integer(self.height, "height")
        _positive_integer(self.width, "width")
        if self.sequence_length is not None:
            _nonnegative_integer(self.sequence_length, "sequence_length")
        if self.volumes.shape[0] <= 0:
            raise ValueError("batch dimensions must be positive")
        if tuple(self.volumes.shape[1:]) != (2 * self.time_bins, self.height, self.width):
            raise ValueError("volumes must have shape [B,2*T,H,W]")
        for name in ("batch_indices", "event_indices"):
            _tensor(getattr(self, name), name, 1, torch.int64)
        _tensor(self.voxel_indices, "voxel_indices", 2, torch.int64)
        count = len(self.event_indices)
        if len(self.batch_indices) != count or self.voxel_indices.shape != (count, 3):
            raise ValueError("all per-event mappings must have the same length")
        if torch.any(self.batch_indices < 0) or torch.any(self.batch_indices >= self.volumes.shape[0]):
            raise ValueError("batch_indices are outside the batch")
        if torch.any(self.event_indices < 0) or (self.sequence_length is not None and torch.any(self.event_indices >= self.sequence_length)):
            raise ValueError("event_indices are outside the source sequence")
        pairs = torch.stack((self.batch_indices, self.event_indices), dim=1)
        if torch.unique(pairs, dim=0).shape[0] != count:
            raise ValueError("(batch_index, event_index) pairs must be unique")
        limits = torch.tensor([self.time_bins, self.height, self.width], device=self.voxel_indices.device)
        if torch.any(self.voxel_indices < 0) or torch.any(self.voxel_indices >= limits):
            raise ValueError("voxel_indices are outside [T,H,W]")
        _binary_labels(self.labels, count)


__all__ = [
    "EventBatch",
    "EventSequence",
    "EventWindow",
    "LoadedEventSequence",
    "NPZEventDataset",
]
