"""后处理缓存、数据划分和哈希契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CACHE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """分块计算文件哈希，避免把大检查点一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_sequence_token(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def make_internal_split(
    names: Iterable[str], *, seed: int, holdout_files: int
) -> tuple[list[str], list[str]]:
    """按队友协议用序列名哈希构造固定79/20划分。"""

    ordered = sorted(set(names), key=lambda name: stable_sequence_token(seed, name))
    if holdout_files <= 0 or holdout_files >= len(ordered):
        raise ValueError("holdout_files must leave non-empty training and holdout sets")
    return sorted(ordered[holdout_files:]), sorted(ordered[:holdout_files])


def fold_of(name: str, *, seed: int, folds: int) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    return int(stable_sequence_token(seed, name)[:16], 16) % folds


def density_group(locations: np.ndarray) -> tuple[str, float]:
    """按每50 ms帧的事件密度划分稀疏、普通和密集序列。"""

    values = np.asarray(locations, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("locations must have shape [N,3] and be non-empty")
    frames = values[:, 2] // 50
    density = len(frames) / max(len(np.unique(frames)), 1)
    if density < 600:
        return "sparse", float(density)
    if density < 2000:
        return "ordinary", float(density)
    return "dense", float(density)


@dataclass(frozen=True)
class PostprocessProtocol:
    seed: int = 37
    holdout_files: int = 20
    folds: int = 5
    hard_negative_threshold: float = 0.03
    negative_to_positive_ratio: int = 4
    minimum_negative_samples: int = 2500
    small_rounds: int = 997
    large_rounds: int = 1998
    blend_step: float = 0.05

    def validate(self) -> None:
        if self.holdout_files <= 0 or self.folds < 2:
            raise ValueError("invalid split settings")
        if not 0 < self.blend_step <= 1:
            raise ValueError("blend_step must be in (0,1]")
        if self.small_rounds <= 0 or self.large_rounds <= 0:
            raise ValueError("boosting rounds must be positive")


@dataclass(frozen=True)
class CacheManifest:
    split: str
    cache_schema_version: int
    feature_schema_version: int
    checkpoint_sha256: str
    config_sha256: str
    files: tuple[dict[str, Any], ...]
    labels_included: bool

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> CacheManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["files"] = tuple(raw["files"])
        return cls(**raw)


@dataclass(frozen=True)
class EventCache:
    name: str
    locations: np.ndarray
    raw_features: np.ndarray
    base_probability: np.ndarray
    token_features: np.ndarray
    event_token_indices: np.ndarray
    prediction: np.ndarray
    labels: np.ndarray | None = None
    target_ids: np.ndarray | None = None

    def validate(self, *, allow_labels: bool) -> None:
        count = len(self.locations)
        if self.locations.shape != (count, 3):
            raise ValueError(f"{self.name}: locations must have shape [N,3]")
        if self.raw_features.shape != (count, 4):
            raise ValueError(f"{self.name}: raw_features must have shape [N,4]")
        for name, values in (
            ("base_probability", self.base_probability),
            ("event_token_indices", self.event_token_indices),
            ("prediction", self.prediction),
        ):
            if values.shape != (count,):
                raise ValueError(f"{self.name}: {name} must align with events")
        if self.token_features.ndim != 2 or self.token_features.shape[1] != 64:
            raise ValueError(f"{self.name}: token_features must have shape [T,64]")
        if count and (
            int(self.event_token_indices.min()) < 0
            or int(self.event_token_indices.max()) >= len(self.token_features)
        ):
            raise ValueError(f"{self.name}: event token index is outside token_features")
        if not np.all(np.isfinite(self.base_probability)):
            raise ValueError(f"{self.name}: probabilities must be finite")
        if np.any((self.base_probability < 0) | (self.base_probability > 1)):
            raise ValueError(f"{self.name}: probabilities must be in [0,1]")
        if (self.labels is None) != (self.target_ids is None):
            raise ValueError(f"{self.name}: labels and target_ids must appear together")
        if self.labels is not None:
            if not allow_labels:
                raise ValueError(f"{self.name}: inference cache must not contain labels")
            if self.labels.shape != (count,) or self.target_ids.shape != (count,):
                raise ValueError(f"{self.name}: labels must align with events")

    def save(self, path: str | Path, *, include_labels: bool) -> None:
        self.validate(allow_labels=include_labels)
        arrays: dict[str, np.ndarray] = {
            "locations": self.locations.astype(np.int32, copy=False),
            "raw_features": self.raw_features.astype(np.float32, copy=False),
            "base_probability": self.base_probability.astype(np.float32, copy=False),
            "token_features": self.token_features.astype(np.float16, copy=False),
            "event_token_indices": self.event_token_indices.astype(np.int32, copy=False),
            "prediction": self.prediction.astype(np.uint8, copy=False),
        }
        if include_labels:
            if self.labels is None or self.target_ids is None:
                raise ValueError(f"{self.name}: training cache requires labels")
            arrays["labels"] = self.labels.astype(np.uint8, copy=False)
            arrays["target_ids"] = self.target_ids.astype(np.int32, copy=False)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, *, allow_labels: bool) -> EventCache:
        source = Path(path)
        with np.load(source, allow_pickle=False) as saved:
            cache = cls(
                name=source.name,
                locations=saved["locations"].astype(np.int64),
                raw_features=saved["raw_features"].astype(np.float32),
                base_probability=saved["base_probability"].astype(np.float32),
                token_features=saved["token_features"].astype(np.float32),
                event_token_indices=saved["event_token_indices"].astype(np.int64),
                prediction=saved["prediction"].astype(bool),
                labels=(saved["labels"].astype(bool) if "labels" in saved else None),
                target_ids=(
                    saved["target_ids"].astype(np.int64) if "target_ids" in saved else None
                ),
            )
        cache.validate(allow_labels=allow_labels)
        return cache
