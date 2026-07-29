from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.config import ConeHeadConfig, HUAVConfig, load_config


def raises(exc_type: type[Exception], fn: object) -> None:
    try:
        fn()  # type: ignore[operator]
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_base_yaml_loads_with_phase_one_defaults() -> None:
    config = load_config(Path("configs/huav_base.yaml"))
    assert config.resolution.width == 346
    assert config.resolution.height == 260
    assert config.inference.sigmoid_threshold == 0.5
    assert Path(config.dataset.root).is_absolute()
    assert Path(config.dataset.root).name == "训练集、验证集"


def test_base_yaml_exposes_cone_training_and_loss_configuration() -> None:
    config = load_config(Path("configs/huav_base.yaml"))
    assert config.model.spatial_block_size_x == 8
    assert config.model.spatial_block_size_y == 8
    assert config.model.num_time_blocks == 64
    assert config.model.token_dim == 32
    assert config.model.event_embed_dim == 16
    assert config.model.num_pool_queries == 4
    assert config.model.num_layers == 3
    assert config.model.num_heads == 4
    assert tuple(config.model.layer_time_scales) == (0.5, 0.75, 1.0)
    assert [head.slope for head in config.model.cone_heads] == [0.5, 1.0, 1.5, 2.0]
    assert [head.base_radius for head in config.model.cone_heads] == [1.0, 1.0, 2.0, 2.0]
    assert config.model.max_neighbors_per_cone == 8
    assert config.model.causal_mode is False
    assert config.model.direction_dropout == 0.0
    assert config.model.adaptive_threshold_mode == "none"
    assert config.model.threshold_hidden_dim == 32
    assert config.model.threshold_time_bins == 160
    assert config.model.threshold_max_global_offset == 0.5
    assert config.model.threshold_max_local_offset == 0.75
    assert config.dataset.duration_ms == 8000
    assert config.training.device == "cpu"
    assert config.training.background_ratio == 10
    assert config.training.hard_negative_fraction == 0.0
    assert config.training.dense_sequence_weight == 1.0
    assert config.training.min_learning_rate_ratio == 1.0
    assert config.training.mixed_precision is False
    assert config.training.mixed_precision_dtype == "float16"
    assert config.loss.focal_alpha == 0.75
    assert config.loss.focal_gamma == 2.0
    assert config.loss.lambda_dice == 1.0
    assert config.loss.lambda_object_frame == 0.0
    assert config.loss.lambda_false_alarm == 0.0
    assert config.loss.false_alarm_topk_per_frame == 32
    assert config.loss.lambda_threshold_magnitude == 0.0
    assert config.loss.lambda_threshold_smoothness == 0.0
    assert config.loss.lambda_threshold_center == 0.0
    assert config.evaluation.frame_ms == 50


def test_score_optimized_yaml_exposes_dense_domain_training_configuration() -> None:
    config = load_config(Path("configs/huav_score_optimized.yaml"))

    assert config.model.spatial_block_size_x == 8
    assert config.model.spatial_block_size_y == 8
    assert config.model.num_time_blocks == 64
    assert config.model.token_dim == 48
    assert config.model.event_embed_dim == 64
    assert config.model.num_pool_queries == 4
    assert config.model.num_layers == 3
    assert tuple(config.model.layer_time_scales) == (0.5, 0.75, 1.0)
    assert config.model.max_neighbors_per_cone == 8
    assert config.training.epochs == 6
    assert config.training.hard_negative_fraction == 0.5
    assert config.training.dense_sequence_event_threshold == 300000
    assert config.training.dense_sequence_background_polarity_max == 0.65
    assert config.training.dense_sequence_weight == 3.0
    assert config.training.min_learning_rate_ratio == 0.1
    assert config.loss.lambda_dice == 0.75
    assert config.loss.lambda_object_frame == 0.25
    assert config.loss.object_frame_logit_margin == 2.0
    assert config.loss.lambda_false_alarm == 0.05
    assert config.loss.false_alarm_logit_margin == 0.0
    assert config.loss.false_alarm_topk_per_frame == 32


def test_autodl_configs_preserve_score_model_and_select_cuda_bfloat16() -> None:
    smoke = load_config(Path("configs/huav_score_optimized_autodl_smoke.yaml"))
    full = load_config(Path("configs/huav_score_optimized_autodl.yaml"))

    for config in (smoke, full):
        assert config.model.spatial_block_size_x == 8
        assert config.model.spatial_block_size_y == 8
        assert config.model.num_time_blocks == 64
        assert config.model.token_dim == 48
        assert config.model.event_embed_dim == 64
        assert config.model.num_pool_queries == 4
        assert config.model.num_layers == 3
        assert config.training.device == "cuda"
        assert config.training.mixed_precision is True
        assert config.training.mixed_precision_dtype == "bfloat16"
        assert config.loss.lambda_false_alarm == 0.05
        assert config.loss.false_alarm_logit_margin == 0.0
        assert config.loss.false_alarm_topk_per_frame == 32
    assert smoke.training.max_train_files == 4
    assert smoke.training.max_val_files == 2
    assert full.training.epochs == 18
    assert full.training.max_train_files is None
    assert full.training.max_val_files is None


def test_cross_field_config_validation_rejects_inconsistent_values() -> None:
    valid = HUAVConfig.default()
    raises(ValueError, lambda: valid.with_overrides(sigmoid_threshold=1.1))
    raises(ValueError, lambda: valid.with_overrides(width=0))
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(valid.model, token_dim=30, num_heads=4),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(valid.model, num_pool_queries=0),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(valid.model, cone_heads=valid.model.cone_heads[:2]),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(
                valid.model,
                cone_heads=(
                    ConeHeadConfig(base_radius=1.0, slope=-0.1, max_time_distance=2),
                )
                + valid.model.cone_heads[1:],
            ),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(valid.model, direction_dropout=1.1),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            loss=replace(valid.loss, lambda_dice=0.0),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            training=replace(valid.training, hard_negative_fraction=1.1),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            training=replace(valid.training, dense_sequence_weight=0.5),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            training=replace(valid.training, mixed_precision_dtype="float32"),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            loss=replace(valid.loss, lambda_object_frame=-0.1),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            loss=replace(valid.loss, lambda_false_alarm=-0.1),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            loss=replace(valid.loss, false_alarm_topk_per_frame=0),
        ).validate(),
    )


def test_adaptive_threshold_configuration_accepts_aligned_valid_values() -> None:
    valid = HUAVConfig.default()

    configured = replace(
        valid,
        model=replace(
            valid.model,
            adaptive_threshold_mode="hierarchical",
            threshold_hidden_dim=16,
            threshold_time_bins=80,
            threshold_max_global_offset=0.25,
            threshold_max_local_offset=1.0,
        ),
        evaluation=replace(valid.evaluation, frame_ms=100.0),
        loss=replace(
            valid.loss,
            lambda_threshold_magnitude=0.1,
            lambda_threshold_smoothness=0.2,
            lambda_threshold_center=0.3,
        ),
    ).validate()

    assert configured.model.adaptive_threshold_mode == "hierarchical"


def test_adaptive_threshold_configuration_rejects_invalid_fields() -> None:
    valid = HUAVConfig.default()
    invalid_models = (
        replace(valid.model, adaptive_threshold_mode="invalid"),
        replace(valid.model, adaptive_threshold_mode=[]),  # type: ignore[arg-type]
        replace(valid.model, threshold_hidden_dim=0),
        replace(valid.model, threshold_time_bins=True),
        replace(valid.model, threshold_max_global_offset=-0.1),
        replace(valid.model, threshold_max_local_offset=float("nan")),
    )
    for model in invalid_models:
        raises(ValueError, lambda model=model: replace(valid, model=model).validate())

    for field in (
        "lambda_threshold_magnitude",
        "lambda_threshold_smoothness",
        "lambda_threshold_center",
    ):
        for invalid_value in (-0.1, float("nan")):
            invalid_loss = replace(valid.loss, **{field: invalid_value})
            raises(
                ValueError,
                lambda invalid_loss=invalid_loss: replace(
                    valid, loss=invalid_loss
                ).validate(),
            )


def test_adaptive_threshold_requires_evaluation_frame_alignment() -> None:
    valid = HUAVConfig.default()
    adaptive_model = replace(
        valid.model,
        adaptive_threshold_mode="local",
        threshold_time_bins=159,
    )

    raises(
        ValueError,
        lambda: replace(valid, model=adaptive_model).validate(),
    )
    replace(
        valid,
        model=replace(valid.model, threshold_time_bins=159),
    ).validate()


def test_config_overrides_reject_unknown_keys_including_mixed_valid_invalid_values() -> None:
    valid = HUAVConfig.default()
    raises(ValueError, lambda: valid.with_overrides(widht=346))
    raises(ValueError, lambda: valid.with_overrides(width=346, unknown_option=True))


def test_config_validation_rejects_malformed_scalar_and_dataset_types_as_value_errors() -> None:
    valid = HUAVConfig.default()
    malformed_overrides = (
        {"width": 346.0}, {"height": True},
        {"sigmoid_threshold": True}, {"sigmoid_threshold": float("nan")},
        {"sigmoid_threshold": float("inf")},
    )
    for values in malformed_overrides:
        raises(ValueError, lambda values=values: valid.with_overrides(**values))


def test_yaml_loader_rejects_null_and_empty_dataset_root_as_value_errors() -> None:
    cases = (
        "dataset: {root: null}",
        "dataset: {root: ''}",
        "dataset: {root: 123}",
        "resolution: {width: 346.0, height: 260}",
    )
    with TemporaryDirectory() as directory:
        for index, contents in enumerate(cases):
            path = Path(directory) / f"bad-{index}.yaml"
            path.write_text(contents, encoding="utf-8")
            raises(ValueError, lambda path=path: load_config(path))


def test_yaml_loader_rejects_non_mapping_unknown_and_syntactically_invalid_documents() -> None:
    cases = (
        "false", "0", "[]", "plain string", "unknown_section: {enabled: true}", "resolution: [",
    )
    with TemporaryDirectory() as directory:
        for index, contents in enumerate(cases):
            path = Path(directory) / f"document-{index}.yaml"
            path.write_text(contents, encoding="utf-8")
            raises(ValueError, lambda path=path: load_config(path))
