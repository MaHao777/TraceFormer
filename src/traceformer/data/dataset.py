"""NPZ-backed event sequences used by the train and validation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_EVENT_FIELDS = ("x", "y", "t", "p", "label", "name")


@dataclass(frozen=True)
class LoadedEventSequence:
    """One source file in original event order.

    ``events`` stores ``x, y, normalized_time, polarity``. Raw millisecond
    timestamps and object IDs are retained for EV-UAV-compatible metrics.
    """

    events: torch.Tensor
    labels: torch.Tensor
    object_ids: torch.Tensor
    timestamps_ms: torch.Tensor
    source_path: Path

    @property
    def length(self) -> int:
        return int(self.events.shape[0])


class NPZEventDataset(Dataset[LoadedEventSequence]):
    """Read TraceFormer's structured ``ev`` arrays without reordering events."""

    def __init__(
        self,
        root: str | Path,
        *,
        width: int = 346,
        height: int = 260,
        duration_ms: float = 8000.0,
        limit_files: int | None = None,
        require_annotations: bool = True,
    ) -> None:
        self.root = Path(root)
        self.require_annotations = require_annotations
        self.width = int(width)
        self.height = int(height)
        self.duration_ms = float(duration_ms)
        if self.width <= 0 or self.height <= 0 or self.duration_ms <= 0:
            raise ValueError("width, height and duration_ms must be positive")
        self.files = sorted(self.root.glob("*.npz"))
        if limit_files is not None:
            if limit_files <= 0:
                raise ValueError("limit_files must be positive when provided")
            self.files = self.files[:limit_files]
        if not self.files:
            raise FileNotFoundError(f"no NPZ files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> LoadedEventSequence:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as archive:
            if "ev" not in archive:
                raise ValueError(f"{path.name}: missing structured 'ev' array")
            source = archive["ev"]
        fields = source.dtype.names
        required = REQUIRED_EVENT_FIELDS if self.require_annotations else ("x", "y", "t", "p")
        if fields is None or any(name not in fields for name in required):
            raise ValueError(f"{path.name}: ev must contain fields {required}")

        x = source["x"]
        y = source["y"]
        timestamp = source["t"].astype(np.float64, copy=False)
        polarity = source["p"]
        # 无标注推理不读取标签；占位数组只满足共享序列容器。
        label = source["label"] if self.require_annotations else np.zeros(len(source), dtype=np.int64)
        object_id = source["name"] if self.require_annotations else np.zeros(len(source), dtype=np.int64)

        if len(source) == 0:
            raise ValueError(f"{path.name}: empty event sequence")
        for name, values in (
            ("x", x),
            ("y", y),
            ("p", polarity),
            ("label", label),
            ("name", object_id),
        ):
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{path.name}: {name} must have an integer dtype")
        if not (
            np.issubdtype(timestamp.dtype, np.integer)
            or np.issubdtype(timestamp.dtype, np.floating)
        ):
            raise TypeError(f"{path.name}: t must have a real numeric dtype")
        if (
            np.any(x < 0)
            or np.any(x >= self.width)
            or np.any(y < 0)
            or np.any(y >= self.height)
        ):
            raise ValueError(f"{path.name}: event coordinates are outside the configured resolution")
        if np.any((polarity != 0) & (polarity != 1)):
            raise ValueError(f"{path.name}: event polarity must be 0 or 1")
        if np.any((label != 0) & (label != 1)):
            raise ValueError(f"{path.name}: event labels must be 0 or 1")
        if (
            not np.all(np.isfinite(timestamp))
            or np.any(timestamp[1:] < timestamp[:-1])
            or timestamp[0] < 0
            or timestamp[-1] > self.duration_ms + 1e-6
        ):
            raise ValueError(
                f"{path.name}: timestamps must be ordered within [0, {self.duration_ms}] ms"
            )

        events = np.column_stack(
            (
                x.astype(np.float32, copy=False),
                y.astype(np.float32, copy=False),
                (timestamp / self.duration_ms).astype(np.float32, copy=False),
                polarity.astype(np.float32, copy=False),
            )
        )
        return LoadedEventSequence(
            events=torch.from_numpy(np.ascontiguousarray(events)),
            labels=torch.from_numpy(label.astype(np.float32, copy=True)),
            object_ids=torch.from_numpy(object_id.astype(np.int64, copy=True)),
            timestamps_ms=torch.from_numpy(timestamp.copy()),
            source_path=path,
        )


__all__ = ["LoadedEventSequence", "NPZEventDataset", "REQUIRED_EVENT_FIELDS"]
