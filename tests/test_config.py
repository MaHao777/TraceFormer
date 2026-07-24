from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.config import HUAVConfig, load_config


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


def test_base_yaml_exposes_dual_view_training_and_loss_configuration() -> None:
    config = load_config(Path("configs/huav_base.yaml"))
    assert tuple(config.model.time_bins) == (32, 16, 8)
    assert config.model.feature_channels == 8
    assert config.dataset.duration_ms == 8000
    assert config.training.device == "cpu"
    assert config.training.background_ratio == 10
    assert config.loss.view_weight == 0.5
    assert config.loss.view_event_weight == 0.25
    assert config.loss.consistency_weight == 0.1
    assert config.loss.temporal_weight == 0.01
    assert config.evaluation.frame_ms == 50


def test_cross_field_config_validation_rejects_inconsistent_values() -> None:
    valid = HUAVConfig.default()
    raises(ValueError, lambda: valid.with_overrides(sigmoid_threshold=1.1))
    raises(ValueError, lambda: valid.with_overrides(width=0))


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
