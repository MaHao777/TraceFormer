from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.config import (
    HUAVConfig,
    LocalConeConfig,
    ThresholdAugmentationConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


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
    assert not config.threshold_augmentation.enabled


def test_score_config_enables_physical_threshold_augmentation() -> None:
    config = load_config(Path("configs/huav_score_optimized.yaml"))

    assert config.training.lr_schedule == "legacy_cosine"
    assert config.training.fast_learning_epochs == 0
    assert config.threshold_augmentation == ThresholdAugmentationConfig(
        enabled=True,
        min_events=300000,
        max_polarity_fraction=0.65,
        threshold_ratios=(1.5, 2.0),
        warmup_epochs=3,
        ramp_epochs=3,
        sparse_loss_weight=0.5,
        consistency_loss_weight=0.1,
    )


def test_learning_rate_strategy_validation() -> None:
    base = HUAVConfig.default()
    valid = replace(
        base,
        training=replace(
            base.training,
            epochs=24,
            lr_schedule="constant_then_cosine",
            fast_learning_epochs=6,
            fixed_learning_rate_ratio=0.2,
            plateau_factor=0.5,
            plateau_patience=2,
            plateau_min_delta=0.001,
            save_training_state_interval=3,
        ),
    )
    valid.validate()

    for field, value in (
        ("lr_schedule", "unknown"),
        ("fast_learning_epochs", 24),
        ("fixed_learning_rate_ratio", 0.0),
        ("plateau_factor", 1.0),
        ("plateau_patience", 0),
        ("plateau_min_delta", -0.1),
        ("save_training_state_interval", -1),
    ):
        raises(
            ValueError,
            lambda field=field, value=value: replace(
                base,
                training=replace(base.training, **{field: value}),
            ).validate(),
        )

def test_threshold_augmentation_rejects_duplicate_or_nonphysical_ratios() -> None:
    base = HUAVConfig.default()
    duplicate = replace(
        base,
        threshold_augmentation=replace(
            base.threshold_augmentation, threshold_ratios=(1.5, 1.5)
        ),
    )
    nonphysical = replace(
        base,
        threshold_augmentation=replace(
            base.threshold_augmentation, threshold_ratios=(1.0,)
        ),
    )

    raises(ValueError, duplicate.validate)
    raises(ValueError, nonphysical.validate)


def test_base_yaml_exposes_cone_training_and_loss_configuration() -> None:
    config = load_config(Path("configs/huav_base.yaml"))
    assert config.model.spatial_block_size_x == 8
    assert config.model.spatial_block_size_y == 8
    assert config.model.num_time_blocks == 64
    assert config.model.token_dim == 32
    assert config.model.event_embed_dim == 16
    assert config.model.num_pool_queries == 4
    assert config.model.embedding_mode == "event_attention"
    assert config.model.voxel_size_x == 2
    assert config.model.voxel_size_y == 2
    assert config.model.voxel_duration_ms == 25.0
    assert config.model.use_block_statistics is True
    assert config.model.num_cone_blocks == 6
    assert config.model.num_heads == 4
    assert config.model.local_cone == LocalConeConfig(
        base_radius=1.0,
        slope=1.0,
        max_neighbors_per_time_slice=8,
    )
    assert not any(
        name.startswith("token_importance")
        for name in config.model.__dataclass_fields__
    )
    assert config.model.causal_mode is False
    assert config.model.direction_dropout == 0.0
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
    assert config.loss.object_frame_logit_margin == 2.0
    assert config.loss.lambda_false_alarm == 0.0
    assert config.loss.false_alarm_logit_margin == 0.0
    assert config.loss.false_alarm_topk_per_frame == 32
    assert config.evaluation.frame_ms == 50


def test_score_optimized_yaml_exposes_dense_domain_training_configuration() -> None:
    config = load_config(Path("configs/huav_score_optimized.yaml"))
    smoke = load_config(Path("configs/huav_score_optimized_smoke.yaml"))

    assert config.model.spatial_block_size_x == 8
    assert config.model.spatial_block_size_y == 8
    assert config.model.num_time_blocks == 64
    assert config.model.token_dim == 64
    assert config.model.event_embed_dim == 64
    assert config.model.num_pool_queries == 4
    assert config.model.num_cone_blocks == 6
    assert config.model.local_cone == LocalConeConfig(
        base_radius=1.0,
        slope=1.0,
        max_neighbors_per_time_slice=8,
    )
    assert config.training.epochs == 24
    assert config.training.hard_negative_fraction == 0.5
    assert config.training.dense_sequence_event_threshold == 300000
    assert config.training.dense_sequence_background_polarity_max == 0.65
    assert config.training.dense_sequence_weight == 3.0
    assert config.training.min_learning_rate_ratio == 0.1
    assert config.loss.focal_alpha == 0.75
    assert config.loss.focal_gamma == 2.0
    assert config.loss.lambda_dice == 0.75
    assert config.loss.lambda_object_frame == 0.25
    assert config.loss.object_frame_logit_margin == 2.0
    assert config.loss.lambda_false_alarm == 0.05
    assert smoke.model == config.model
    assert smoke.loss == config.loss


def test_smoke_configs_use_a_separate_output_root() -> None:
    expected_smoke_root = (ROOT / "smoke_outputs").resolve()
    expected_full_root = (ROOT / "outputs").resolve()

    for path in (
        "configs/huav_cpu_smoke.yaml",
        "configs/huav_score_optimized_smoke.yaml",
        "configs/huav_score_optimized_autodl_smoke.yaml",
        "configs/huav_embedding_a_autodl_smoke.yaml",
        "configs/huav_embedding_b_autodl_smoke.yaml",
        "configs/huav_embedding_c_autodl_smoke.yaml",
    ):
        assert Path(load_config(Path(path)).training.output_root) == expected_smoke_root

    for path in (
        "configs/huav_base.yaml",
        "configs/huav_score_optimized.yaml",
        "configs/huav_score_optimized_autodl.yaml",
        "configs/huav_embedding_a_autodl.yaml",
        "configs/huav_embedding_b_autodl.yaml",
        "configs/huav_embedding_c_autodl.yaml",
    ):
        assert Path(load_config(Path(path)).training.output_root) == expected_full_root


def test_local_cone_depth_supports_requested_ablation_values() -> None:
    valid = HUAVConfig.default()

    for depth in (1, 3, 5, 7):
        configured = replace(
            valid,
            model=replace(valid.model, num_cone_blocks=depth),
        ).validate()

        assert configured.model.num_cone_blocks == depth


def test_embedding_ablation_configs_only_change_the_planned_model_fields() -> None:
    configs = {
        arm: load_config(Path(f"configs/huav_embedding_{arm}_autodl.yaml"))
        for arm in "abc"
    }

    assert configs["a"].model.embedding_mode == "event_attention"
    assert configs["b"].model.embedding_mode == "voxel_attention"
    assert configs["c"].model.embedding_mode == "voxel_attention"
    assert configs["a"].model.use_block_statistics is True
    assert configs["b"].model.use_block_statistics is True
    assert configs["c"].model.use_block_statistics is False
    for config in configs.values():
        assert config.model.voxel_size_x == 2
        assert config.model.voxel_size_y == 2
        assert config.model.voxel_duration_ms == 25.0
        assert config.training.seed == 37
        assert config.training.epochs == 24
        assert config.training.max_train_files is None
        assert config.training.max_val_files is None

    ignored = {"embedding_mode", "use_block_statistics"}
    reference = {
        name: value
        for name, value in configs["a"].model.__dict__.items()
        if name not in ignored
    }
    for arm in "bc":
        comparison = {
            name: value
            for name, value in configs[arm].model.__dict__.items()
            if name not in ignored
        }
        assert comparison == reference


def test_model_config_no_longer_exposes_multi_cone_or_layer_scale_fields() -> None:
    fields = HUAVConfig.default().model.__dataclass_fields__

    assert "cone_heads" not in fields
    assert "layer_time_scales" not in fields
    assert "num_layers" not in fields
    assert "max_neighbors_per_cone" not in fields


def test_autodl_configs_preserve_score_model_and_select_cuda_bfloat16() -> None:
    smoke = load_config(Path("configs/huav_score_optimized_autodl_smoke.yaml"))
    full = load_config(Path("configs/huav_score_optimized_autodl.yaml"))

    for config in (smoke, full):
        assert config.model.spatial_block_size_x == 8
        assert config.model.spatial_block_size_y == 8
        assert config.model.num_time_blocks == 64
        assert config.model.token_dim == 64
        assert config.model.event_embed_dim == 64
        assert config.model.num_pool_queries == 4
        assert config.model.num_cone_blocks == 6
        assert config.model.local_cone == LocalConeConfig(
            base_radius=1.0,
            slope=1.0,
            max_neighbors_per_time_slice=8,
        )
        assert config.training.device == "cuda"
        assert config.training.mixed_precision is True
        assert config.training.mixed_precision_dtype == "bfloat16"
        assert config.loss.focal_alpha == 0.75
        assert config.loss.lambda_dice == 0.75
        assert config.loss.lambda_object_frame == 0.25
        assert config.loss.object_frame_logit_margin == 2.0
        assert config.loss.lambda_false_alarm == 0.05
        assert config.loss.false_alarm_logit_margin == 0.0
        assert config.loss.false_alarm_topk_per_frame == 32
    assert smoke.training.max_train_files == 4
    assert smoke.training.max_val_files == 2
    assert full.training.epochs == 24
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
            model=replace(valid.model, embedding_mode="unknown"),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(
                valid.model,
                embedding_mode="voxel_attention",
                voxel_size_x=3,
            ),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(
                valid.model,
                embedding_mode="voxel_attention",
                voxel_duration_ms=30.0,
            ),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(
                valid.model,
                local_cone=replace(valid.model.local_cone, slope=-0.1),
            ),
        ).validate(),
    )
    raises(
        ValueError,
        lambda: replace(
            valid,
            model=replace(
                valid.model,
                local_cone=replace(
                    valid.model.local_cone,
                    max_neighbors_per_time_slice=0,
                ),
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
