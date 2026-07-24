"""Validated YAML configuration for the HUAV dual-view model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResolutionConfig:
    width: int = 346
    height: int = 260


@dataclass(frozen=True)
class ModelConfig:
    time_bins: tuple[int, ...] = (32, 16, 8)
    feature_channels: int = 8
    attention_heads: int = 2
    window_time: int = 4
    window_space: int = 16
    cross_time_radius: int = 1
    dropout: float = 0.1


@dataclass(frozen=True)
class InferenceConfig:
    sigmoid_threshold: float = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    root: str | Path = "../训练集、验证集"
    duration_ms: float = 8000.0


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cpu"
    epochs: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 37
    background_ratio: int = 10
    num_threads: int = 14
    output_root: str | Path = "outputs"
    auxiliary_warmup_epochs: int = 1
    auxiliary_ramp_epochs: int = 2
    target_sigma: float = 1.25
    gradient_clip_norm: float = 1.0
    max_train_files: int | None = None
    max_val_files: int | None = None


@dataclass(frozen=True)
class LossConfig:
    view_event_weight: float = 0.25
    view_weight: float = 0.5
    consistency_weight: float = 0.1
    temporal_weight: float = 0.01


@dataclass(frozen=True)
class EvaluationConfig:
    optimize_threshold: bool = True
    threshold_steps: int = 19
    frame_ms: float = 50.0
    object_correct_fraction: float = 0.0001


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite non-boolean real value")
    return float(value)


def _path_value(value: object, name: str) -> None:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a non-empty string or path")
    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string or path")


@dataclass(frozen=True)
class HUAVConfig:
    resolution: ResolutionConfig = ResolutionConfig()
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def default(cls) -> "HUAVConfig":
        return cls()

    def validate(self) -> "HUAVConfig":
        _positive_integer(self.resolution.width, "resolution width")
        _positive_integer(self.resolution.height, "resolution height")
        if not isinstance(self.model.time_bins, (tuple, list)) or not self.model.time_bins:
            raise ValueError("model time_bins must be a non-empty list or tuple")
        for value in self.model.time_bins:
            _positive_integer(value, "model time_bins entry")
        if len(set(self.model.time_bins)) != len(self.model.time_bins):
            raise ValueError("model time_bins must be unique")
        _positive_integer(self.model.feature_channels, "feature_channels")
        _positive_integer(self.model.attention_heads, "attention_heads")
        _positive_integer(self.model.window_time, "window_time")
        _positive_integer(self.model.window_space, "window_space")
        if (
            isinstance(self.model.cross_time_radius, bool)
            or not isinstance(self.model.cross_time_radius, Integral)
            or self.model.cross_time_radius < 0
        ):
            raise ValueError("cross_time_radius must be a non-negative integer")
        if self.model.feature_channels % self.model.attention_heads:
            raise ValueError("feature_channels must be divisible by attention_heads")
        dropout = _finite_real(self.model.dropout, "dropout")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        threshold = _finite_real(self.inference.sigmoid_threshold, "sigmoid_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("sigmoid_threshold must be in [0, 1]")
        _path_value(self.dataset.root, "dataset root")
        if _finite_real(self.dataset.duration_ms, "duration_ms") <= 0:
            raise ValueError("duration_ms must be positive")
        if self.training.device not in {"cpu", "cuda"}:
            raise ValueError("training device must be 'cpu' or 'cuda'")
        _positive_integer(self.training.epochs, "epochs")
        if _finite_real(self.training.learning_rate, "learning_rate") <= 0:
            raise ValueError("learning_rate must be positive")
        if _finite_real(self.training.weight_decay, "weight_decay") < 0:
            raise ValueError("weight_decay must be non-negative")
        if isinstance(self.training.seed, bool) or not isinstance(self.training.seed, Integral):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.training.background_ratio, bool)
            or not isinstance(self.training.background_ratio, Integral)
            or self.training.background_ratio < 0
        ):
            raise ValueError("background_ratio must be a non-negative integer")
        _positive_integer(self.training.num_threads, "num_threads")
        _path_value(self.training.output_root, "output_root")
        if (
            isinstance(self.training.auxiliary_warmup_epochs, bool)
            or not isinstance(self.training.auxiliary_warmup_epochs, Integral)
            or self.training.auxiliary_warmup_epochs < 0
        ):
            raise ValueError("auxiliary_warmup_epochs must be non-negative")
        _positive_integer(self.training.auxiliary_ramp_epochs, "auxiliary_ramp_epochs")
        if _finite_real(self.training.target_sigma, "target_sigma") < 0:
            raise ValueError("target_sigma must be non-negative")
        if _finite_real(self.training.gradient_clip_norm, "gradient_clip_norm") <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        for name in ("max_train_files", "max_val_files"):
            value = getattr(self.training, name)
            if value is not None:
                _positive_integer(value, name)
        for name in (
            "view_event_weight",
            "view_weight",
            "consistency_weight",
            "temporal_weight",
        ):
            if _finite_real(getattr(self.loss, name), name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.evaluation.optimize_threshold, bool):
            raise ValueError("optimize_threshold must be boolean")
        _positive_integer(self.evaluation.threshold_steps, "threshold_steps")
        if _finite_real(self.evaluation.frame_ms, "frame_ms") <= 0:
            raise ValueError("frame_ms must be positive")
        fraction = _finite_real(
            self.evaluation.object_correct_fraction, "object_correct_fraction"
        )
        if not 0 <= fraction <= 1:
            raise ValueError("object_correct_fraction must be in [0,1]")
        return self

    def with_overrides(self, **values: Any) -> "HUAVConfig":
        supported = {"width", "height", "sigmoid_threshold"}
        unknown = set(values) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown configuration override(s): {names}")
        resolution = replace(self.resolution, **{key: values[key] for key in ("width", "height") if key in values})
        inference = replace(self.inference, **{key: values[key] for key in ("sigmoid_threshold",) if key in values})
        return replace(self, resolution=resolution, inference=inference).validate()


def load_config(path: Path | str) -> HUAVConfig:
    """Load YAML and resolve data/output paths relative to the configuration file."""
    config_path = Path(path).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    allowed_sections = {
        "resolution",
        "model",
        "inference",
        "dataset",
        "training",
        "loss",
        "evaluation",
    }
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        names = ", ".join(sorted(str(name) for name in unknown_sections))
        raise ValueError(f"unknown configuration section(s): {names}")
    try:
        config = HUAVConfig(
            resolution=ResolutionConfig(**raw.get("resolution", {})),
            model=ModelConfig(**raw.get("model", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            training=TrainingConfig(**raw.get("training", {})),
            loss=LossConfig(**raw.get("loss", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
    except TypeError as error:
        raise ValueError(f"invalid configuration keys: {error}") from error
    _path_value(config.dataset.root, "dataset root")
    _path_value(config.training.output_root, "output_root")
    dataset_root = Path(config.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent / dataset_root).resolve()
    output_root = Path(config.training.output_root)
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    config = replace(
        config,
        dataset=replace(config.dataset, root=dataset_root),
        training=replace(config.training, output_root=output_root),
    )
    return config.validate()
