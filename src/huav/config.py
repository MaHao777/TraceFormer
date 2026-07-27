"""Validated YAML configuration for the HUAV spatiotemporal cone model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResolutionConfig:
    width: int = 346
    height: int = 260


@dataclass(frozen=True)
class ConeHeadConfig:
    base_radius: float
    slope: float
    max_time_distance: int


DEFAULT_CONE_HEADS = (
    ConeHeadConfig(base_radius=1.0, slope=0.5, max_time_distance=4),
    ConeHeadConfig(base_radius=1.0, slope=1.0, max_time_distance=4),
    ConeHeadConfig(base_radius=2.0, slope=1.5, max_time_distance=6),
    ConeHeadConfig(base_radius=2.0, slope=2.0, max_time_distance=8),
)


@dataclass(frozen=True)
class ModelConfig:
    spatial_block_size_x: int = 8
    spatial_block_size_y: int = 8
    num_time_blocks: int = 64
    token_dim: int = 32
    event_embed_dim: int = 16
    num_pool_queries: int = 4
    num_layers: int = 3
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    cone_heads: tuple[ConeHeadConfig, ...] = DEFAULT_CONE_HEADS
    layer_time_scales: tuple[float, ...] = (0.5, 0.75, 1.0)
    max_neighbors_per_cone: int = 8
    causal_mode: bool = False
    direction_dropout: float = 0.0
    debug_attention_stats: bool = False


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
    hard_negative_fraction: float = 0.0
    max_background_without_positive: int = 4096
    dense_sequence_event_threshold: int = 300000
    dense_sequence_background_polarity_max: float = 0.65
    dense_sequence_weight: float = 1.0
    min_learning_rate_ratio: float = 1.0
    num_threads: int = 14
    output_root: str | Path = "outputs"
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"
    max_train_files: int | None = None
    max_val_files: int | None = None


@dataclass(frozen=True)
class LossConfig:
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    lambda_dice: float = 1.0
    lambda_object_frame: float = 0.0
    object_frame_logit_margin: float = 2.0


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
        for name in (
            "spatial_block_size_x",
            "spatial_block_size_y",
            "num_time_blocks",
            "token_dim",
            "event_embed_dim",
            "num_pool_queries",
            "num_layers",
            "num_heads",
            "max_neighbors_per_cone",
        ):
            _positive_integer(getattr(self.model, name), f"model {name}")
        if self.model.token_dim % self.model.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if _finite_real(self.model.mlp_ratio, "mlp_ratio") <= 0:
            raise ValueError("mlp_ratio must be positive")
        dropout = _finite_real(self.model.dropout, "dropout")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.model.cone_heads, (tuple, list)):
            raise ValueError("cone_heads must be a tuple or list")
        if len(self.model.cone_heads) != self.model.num_heads:
            raise ValueError("cone_heads must contain exactly one configuration per head")
        for index, cone in enumerate(self.model.cone_heads):
            if not isinstance(cone, ConeHeadConfig):
                raise ValueError(f"cone_heads[{index}] must be a ConeHeadConfig")
            if _finite_real(cone.base_radius, "cone base_radius") < 0:
                raise ValueError("cone base_radius must be non-negative")
            if _finite_real(cone.slope, "cone slope") < 0:
                raise ValueError("cone slope must be non-negative")
            _positive_integer(cone.max_time_distance, "cone max_time_distance")
        if (
            not isinstance(self.model.layer_time_scales, (tuple, list))
            or len(self.model.layer_time_scales) != self.model.num_layers
        ):
            raise ValueError("layer_time_scales must contain one value per layer")
        previous_scale = 0.0
        for scale in self.model.layer_time_scales:
            value = _finite_real(scale, "layer_time_scale")
            if value <= 0 or value > 1:
                raise ValueError("layer_time_scales must be in (0,1]")
            if value < previous_scale:
                raise ValueError("layer_time_scales must be non-decreasing")
            previous_scale = value
        if not isinstance(self.model.causal_mode, bool):
            raise ValueError("causal_mode must be boolean")
        direction_dropout = _finite_real(
            self.model.direction_dropout, "direction_dropout"
        )
        if not 0 <= direction_dropout <= 1:
            raise ValueError("direction_dropout must be in [0,1]")
        if not isinstance(self.model.debug_attention_stats, bool):
            raise ValueError("debug_attention_stats must be boolean")
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
        hard_negative_fraction = _finite_real(
            self.training.hard_negative_fraction,
            "hard_negative_fraction",
        )
        if not 0 <= hard_negative_fraction <= 1:
            raise ValueError("hard_negative_fraction must be in [0,1]")
        _positive_integer(
            self.training.max_background_without_positive,
            "max_background_without_positive",
        )
        _positive_integer(
            self.training.dense_sequence_event_threshold,
            "dense_sequence_event_threshold",
        )
        polarity_max = _finite_real(
            self.training.dense_sequence_background_polarity_max,
            "dense_sequence_background_polarity_max",
        )
        if not 0.5 <= polarity_max <= 1:
            raise ValueError(
                "dense_sequence_background_polarity_max must be in [0.5,1]"
            )
        if (
            _finite_real(
                self.training.dense_sequence_weight,
                "dense_sequence_weight",
            )
            < 1
        ):
            raise ValueError("dense_sequence_weight must be at least 1")
        minimum_lr_ratio = _finite_real(
            self.training.min_learning_rate_ratio,
            "min_learning_rate_ratio",
        )
        if not 0 <= minimum_lr_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in [0,1]")
        _positive_integer(self.training.num_threads, "num_threads")
        _path_value(self.training.output_root, "output_root")
        if _finite_real(self.training.gradient_clip_norm, "gradient_clip_norm") <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not isinstance(self.training.mixed_precision, bool):
            raise ValueError("mixed_precision must be boolean")
        if self.training.mixed_precision_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                "mixed_precision_dtype must be 'float16' or 'bfloat16'"
            )
        for name in ("max_train_files", "max_val_files"):
            value = getattr(self.training, name)
            if value is not None:
                _positive_integer(value, name)
        focal_alpha = _finite_real(self.loss.focal_alpha, "focal_alpha")
        if not 0 <= focal_alpha <= 1:
            raise ValueError("focal_alpha must be in [0,1]")
        if _finite_real(self.loss.focal_gamma, "focal_gamma") < 0:
            raise ValueError("focal_gamma must be non-negative")
        if _finite_real(self.loss.lambda_dice, "lambda_dice") <= 0:
            raise ValueError("lambda_dice must be positive")
        if _finite_real(self.loss.lambda_object_frame, "lambda_object_frame") < 0:
            raise ValueError("lambda_object_frame must be non-negative")
        _finite_real(
            self.loss.object_frame_logit_margin,
            "object_frame_logit_margin",
        )
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
        raw_model = dict(raw.get("model", {}))
        if "cone_heads" in raw_model:
            cone_values = raw_model["cone_heads"]
            if not isinstance(cone_values, list):
                raise ValueError("model cone_heads must be a list")
            raw_model["cone_heads"] = tuple(
                ConeHeadConfig(**value) if isinstance(value, dict) else value
                for value in cone_values
            )
        if "layer_time_scales" in raw_model:
            raw_model["layer_time_scales"] = tuple(raw_model["layer_time_scales"])
        config = HUAVConfig(
            resolution=ResolutionConfig(**raw.get("resolution", {})),
            model=ModelConfig(**raw_model),
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
