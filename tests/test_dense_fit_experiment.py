from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from huav.config import HUAVConfig
from huav.experiments import dense_fit
from huav.experiments.dense_fit import (
    capacity_config,
    cosine_rate,
    decision_from_evaluation,
    parse_files,
)


def _metrics(event_iou: float, event_pd: float, object_pd: float) -> dict:
    return {
        "event": {"iou": event_iou, "pd": event_pd},
        "ev_uav_compatible": {"pd": object_pd},
    }


def test_capacity_config_preserves_geometry_and_removes_regularization() -> None:
    base = HUAVConfig.default()
    base = replace(
        base,
        model=replace(base.model, dropout=0.2),
        training=replace(
            base.training,
            weight_decay=0.1,
            dense_sequence_weight=3.0,
        ),
        threshold_augmentation=replace(base.threshold_augmentation, enabled=True),
    )

    configured = capacity_config(base, seed=73)

    assert configured.model.spatial_block_size_x == base.model.spatial_block_size_x
    assert configured.model.num_time_blocks == base.model.num_time_blocks
    assert configured.model.dropout == 0.0
    assert configured.training.weight_decay == 0.0
    assert configured.training.background_ratio == 200
    assert configured.training.hard_negative_fraction == 0.0
    assert configured.training.dense_sequence_weight == 1.0
    assert configured.training.gradient_clip_norm == 5.0
    assert configured.training.seed == 73
    assert configured.threshold_augmentation.enabled is False


def test_decision_requires_aggregate_and_worst_file_gates() -> None:
    evaluation = {
        "aggregate": _metrics(0.85, 0.90, 0.95),
        "per_file": {
            "train_088.npz": _metrics(0.70, 0.85, 0.80),
            "train_090.npz": _metrics(0.59, 0.90, 0.90),
        },
    }

    decision = decision_from_evaluation(
        evaluation,
        min_event_iou=0.80,
        min_event_pd=0.85,
        min_object_pd=0.90,
        min_file_event_iou=0.60,
        min_file_object_pd=0.75,
    )

    assert decision["passed"] is False
    assert decision["checks"]["event_iou"] is True
    assert decision["checks"]["worst_file_event_iou"] is False


def test_file_parser_and_cosine_schedule_are_deterministic() -> None:
    assert parse_files("train_090.npz, train_091.npz") == (
        "train_090.npz",
        "train_091.npz",
    )
    with pytest.raises(ValueError, match="unique"):
        parse_files("train_090.npz,train_090.npz")
    assert cosine_rate(1, 400, 7e-4) == pytest.approx(7e-4)
    assert cosine_rate(400, 400, 7e-4) == pytest.approx(7e-5)


def test_reproducible_evaluation_forces_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[bool] = []

    def fake_full_evaluation(*args, amp_enabled: bool, **kwargs):
        observed.append(amp_enabled)
        return {"stable": True}

    monkeypatch.setattr(dense_fit, "_full_evaluation", fake_full_evaluation)

    result = dense_fit._reproducible_evaluation(
        object(),
        object(),
        HUAVConfig.default(),
        device=torch.device("cuda"),
    )

    assert result == {"stable": True}
    assert observed == [False]
