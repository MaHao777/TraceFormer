from __future__ import annotations

from dataclasses import replace

import pytest

from traceformer.config import TrainingConfig
from traceformer.training.lr_schedules import LearningRateController


def _config(schedule: str, **overrides: object) -> TrainingConfig:
    values = {
        "epochs": 6,
        "learning_rate": 1.0,
        "min_learning_rate_ratio": 0.1,
        "lr_schedule": schedule,
        **overrides,
    }
    return replace(TrainingConfig(), **values)


def test_legacy_cosine_preserves_existing_epoch_start_rates() -> None:
    controller = LearningRateController(_config("legacy_cosine"))

    assert [controller.rate_for_epoch(epoch) for epoch in range(1, 7)] == pytest.approx(
        [1.0, 0.9397114317, 0.775, 0.55, 0.325, 0.1602885683]
    )


def test_constant_then_cosine_keeps_fast_phase_and_reaches_floor() -> None:
    controller = LearningRateController(
        _config("constant_then_cosine", fast_learning_epochs=2)
    )

    assert [controller.rate_for_epoch(epoch) for epoch in range(1, 7)] == pytest.approx(
        [1.0, 1.0, 0.8681980515, 0.55, 0.2318019485, 0.1]
    )
    assert [controller.phase(epoch) for epoch in range(1, 7)] == [
        "fast",
        "fast",
        "fine",
        "fine",
        "fine",
        "fine",
    ]


def test_exponential_and_fixed_fine_schedules() -> None:
    exponential = LearningRateController(
        _config("constant_then_exponential", fast_learning_epochs=2)
    )
    fixed = LearningRateController(
        _config(
            "constant_then_fixed",
            fast_learning_epochs=2,
            fixed_learning_rate_ratio=0.2,
        )
    )

    assert [exponential.rate_for_epoch(epoch) for epoch in range(1, 7)] == pytest.approx(
        [1.0, 1.0, 10**-0.25, 10**-0.5, 10**-0.75, 0.1]
    )
    assert [fixed.rate_for_epoch(epoch) for epoch in range(1, 7)] == pytest.approx(
        [1.0, 1.0, 0.2, 0.2, 0.2, 0.2]
    )


def test_plateau_halves_after_configured_bad_epochs_and_restores_state() -> None:
    config = _config(
        "constant_then_plateau",
        fast_learning_epochs=2,
        min_learning_rate_ratio=0.05,
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_min_delta=0.001,
    )
    controller = LearningRateController(config)

    controller.update(0.5, epoch=1)
    controller.update(0.6, epoch=2)
    controller.update(0.6005, epoch=3)
    assert controller.rate_for_epoch(4) == pytest.approx(1.0)
    controller.update(0.6006, epoch=4)
    assert controller.rate_for_epoch(5) == pytest.approx(0.5)

    restored = LearningRateController(config)
    restored.load_state_dict(controller.state_dict())
    restored.update(0.6007, epoch=5)
    restored.update(0.6008, epoch=6)
    assert restored.rate_for_epoch(6) == pytest.approx(0.25)
