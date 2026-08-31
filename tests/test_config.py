from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from huav.config import (
    FrameRefinementConfig,
    HUAVConfig,
    ModelConfig,
    RefinementScaleConfig,
    SparseMixtureConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    ROOT / "configs/huav_score_optimized.yaml",
    ROOT / "configs/huav_score_optimized_smoke.yaml",
    ROOT / "configs/huav_score_optimized_autodl.yaml",
    ROOT / "configs/huav_score_optimized_autodl_smoke.yaml",
)


def test_only_fixed_schema7_configs_are_shipped() -> None:
    assert tuple(sorted((ROOT / "configs").glob("*.yaml"))) == tuple(
        sorted(CONFIG_PATHS)
    )
    for path in CONFIG_PATHS:
        assert load_config(path).validate()


def test_score_config_fixes_model_modules_and_refinement_stage() -> None:
    config = load_config(ROOT / "configs/huav_score_optimized.yaml")

    assert config.training.stage == "refinement"
    assert config.model.token_dim == 64
    assert config.model.event_embed_dim == 64
    assert config.model.num_pool_queries == 4
    assert config.model.num_layers == 3
    assert config.model.layer_time_scales == (0.5, 0.75, 1.0)
    assert config.model.max_neighbors_per_cone == 8
    assert config.sparse_mixture == SparseMixtureConfig(
        min_events=300000,
        max_polarity_fraction=0.65,
        threshold_ratio=1.5,
        warmup_epochs=3,
        ramp_epochs=3,
        initial_router_bias=-2.0,
        initial_residual_scale=0.01,
        full_aux_loss_weight=0.0,
        sparse_loss_weight=0.05,
        object_frame_consistency_weight=0.0,
        positive_preservation_weight=0.2,
        positive_preservation_margin=0.0,
        background_consistency_weight=0.01,
        background_consistency_margin=0.0,
        sparse_false_peak_weight=0.01,
        sparse_false_peak_logit_margin=0.0,
        sparse_false_peak_topk_per_frame=8,
        sparse_false_peak_block_size_x=8,
        sparse_false_peak_block_size_y=8,
        gate_regularization_weight=0.001,
    )
    assert config.frame_refinement == FrameRefinementConfig(
        context_dim=16,
        scales=(
            RefinementScaleConfig(2, 2, 10.0),
            RefinementScaleConfig(4, 4, 25.0),
            RefinementScaleConfig(8, 8, 50.0),
        ),
        loss_weight=0.2,
    )


def test_runtime_variants_keep_one_model_contract() -> None:
    configs = [load_config(path) for path in CONFIG_PATHS]
    reference = configs[0]

    for config in configs[1:]:
        assert config.model == reference.model
        assert config.loss == reference.loss
        assert config.sparse_mixture == reference.sparse_mixture
        assert config.frame_refinement == reference.frame_refinement
        assert config.training.stage == "refinement"

    cpu, cpu_smoke, cuda, cuda_smoke = configs
    assert cpu.training.device == cpu_smoke.training.device == "cpu"
    assert cuda.training.device == cuda_smoke.training.device == "cuda"
    assert not cpu.training.mixed_precision
    assert cuda.training.mixed_precision
    assert cuda.training.mixed_precision_dtype == "bfloat16"
    assert cpu.training.max_train_files is None
    assert cpu_smoke.training.max_train_files == 4
    assert Path(cpu.training.output_root).name == "outputs"
    assert Path(cpu_smoke.training.output_root).name == "smoke_outputs"


@pytest.mark.parametrize("stage", ["full", "sparse_mixture", "refinement"])
def test_training_stage_accepts_only_explicit_fixed_model_phases(stage: str) -> None:
    HUAVConfig(training=replace(HUAVConfig.default().training, stage=stage)).validate()


def test_training_stage_rejects_boolean_or_unknown_values() -> None:
    for value in (True, "adapter", "disabled"):
        with pytest.raises(ValueError, match="training stage"):
            HUAVConfig(
                training=replace(HUAVConfig.default().training, stage=value)  # type: ignore[arg-type]
            ).validate()


def test_removed_module_switches_and_legacy_augmentation_are_rejected() -> None:
    snippets = (
        "threshold_augmentation:\n  enabled: false\n",
        "model:\n  causal_mode: false\n",
        "model:\n  debug_attention_stats: false\n",
        "sparse_mixture:\n  enabled: true\n",
        "sparse_mixture:\n  freeze_full_model: true\n",
        "frame_refinement:\n  enabled: true\n",
        "frame_refinement:\n  freeze_base_model: true\n",
    )
    with TemporaryDirectory(dir=ROOT) as directory:
        for index, contents in enumerate(snippets):
            path = Path(directory) / f"removed-{index}.yaml"
            path.write_text(contents, encoding="utf-8")
            with pytest.raises(ValueError):
                load_config(path)


def test_refinement_requires_increasing_scales_and_fixed_primary_cell() -> None:
    base = HUAVConfig.default()
    invalid_scales = (
        (RefinementScaleConfig(8, 8, 50.0), RefinementScaleConfig(4, 4, 25.0)),
        (RefinementScaleConfig(4, 4, 25.0),),
    )
    for scales in invalid_scales:
        with pytest.raises(ValueError):
            replace(
                base,
                frame_refinement=replace(base.frame_refinement, scales=scales),
            ).validate()


def test_sparse_mixture_validation_rejects_invalid_inference_gate() -> None:
    base = HUAVConfig.default()
    for changes in (
        {"threshold_ratio": 1.0},
        {"max_polarity_fraction": 0.49},
        {"min_events": 0},
        {"ramp_epochs": 0},
    ):
        with pytest.raises(ValueError):
            replace(
                base,
                sparse_mixture=replace(base.sparse_mixture, **changes),
            ).validate()


def test_model_config_contains_structure_not_optional_module_flags() -> None:
    fields = ModelConfig.__dataclass_fields__

    for name in (
        "causal_mode",
        "debug_attention_stats",
        "sparse_mixture_enabled",
        "frame_refinement_enabled",
    ):
        assert name not in fields
    for name in (
        "token_dim",
        "event_embed_dim",
        "num_pool_queries",
        "num_layers",
        "num_heads",
        "cone_heads",
        "layer_time_scales",
    ):
        assert name in fields


def test_config_paths_resolve_relative_to_yaml() -> None:
    config = load_config(ROOT / "configs/huav_score_optimized.yaml")

    assert Path(config.dataset.root).is_absolute()
    assert Path(config.training.output_root).is_absolute()


def test_config_overrides_are_narrow_and_validated() -> None:
    config = HUAVConfig.default().with_overrides(
        width=640,
        height=480,
        sigmoid_threshold=0.8,
    )

    assert config.resolution.width == 640
    assert config.resolution.height == 480
    assert config.inference.sigmoid_threshold == 0.8
    with pytest.raises(ValueError, match="unknown configuration override"):
        config.with_overrides(token_dim=64)


def test_yaml_loader_reports_invalid_documents_as_value_errors() -> None:
    documents = ("[]\n", "unknown: 1\n", "resolution: [\n")
    with TemporaryDirectory(dir=ROOT) as directory:
        for index, contents in enumerate(documents):
            path = Path(directory) / f"invalid-{index}.yaml"
            path.write_text(contents, encoding="utf-8")
            with pytest.raises(ValueError):
                load_config(path)
