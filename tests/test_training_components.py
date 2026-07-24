from __future__ import annotations

import pytest
import torch

from huav.training.losses import DualViewLoss, binary_dice_loss, binary_focal_loss
from huav.training.engine import _auxiliary_scale
from huav.training.metrics import EVUAVMetricAccumulator, EventMetricAccumulator
from huav.training.sampling import sample_event_indices


def test_sampler_keeps_all_positives_and_caps_background() -> None:
    labels = torch.tensor([1, 0, 0, 1] + [0] * 40, dtype=torch.float32)
    selected = sample_event_indices(
        labels,
        background_ratio=3,
        generator=torch.Generator().manual_seed(7),
    )

    assert {0, 3}.issubset(set(selected.tolist()))
    assert int((labels[selected] == 0).sum()) == 6


def test_focal_and_dice_losses_prefer_correct_logits() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    correct = torch.tensor([5.0, -5.0, 5.0, -5.0])
    wrong = -correct

    assert binary_focal_loss(correct, labels) < binary_focal_loss(wrong, labels)
    assert binary_dice_loss(correct, labels) < binary_dice_loss(wrong, labels)


def test_total_loss_uses_positive_configurable_weights() -> None:
    output = {
        "event_logits": torch.tensor([0.0, 0.0], requires_grad=True),
        "xt_event_logits": torch.tensor([0.1, -0.1], requires_grad=True),
        "yt_event_logits": torch.tensor([-0.1, 0.1], requires_grad=True),
        "xt_logits": torch.zeros((1, 1, 2, 3), requires_grad=True),
        "yt_logits": torch.zeros((1, 1, 2, 2), requires_grad=True),
        "event_features": torch.zeros((2, 6)),
    }
    labels = torch.tensor([1.0, 0.0])
    xt_target = torch.zeros((1, 1, 2, 3))
    yt_target = torch.zeros((1, 1, 2, 2))
    criterion = DualViewLoss(
        view_event_weight=0.25,
        view_weight=0.5,
        consistency_weight=0.1,
        temporal_weight=0.01,
    )

    losses = criterion(output, labels, xt_target, yt_target)
    expected = (
        losses["event"]
        + 0.25 * losses["view_event"]
        + 0.5 * losses["view"]
        + 0.1 * losses["consistency"]
        + 0.01 * losses["temporal"]
    )
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()


def test_event_metrics_report_pd_fa_iou_and_overall_accuracy() -> None:
    accumulator = EventMetricAccumulator(threshold=0.5)
    accumulator.update(
        torch.tensor([5.0, 5.0, -5.0, -5.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    metrics = accumulator.compute()

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["pd"] == pytest.approx(0.5)
    assert metrics["fa"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["acc"] == pytest.approx(0.5)


def test_ev_uav_compatibility_uses_reference_floor_frame_denominator() -> None:
    accumulator = EVUAVMetricAccumulator(
        width=4,
        height=3,
        frame_ms=50,
        object_correct_fraction=0.0001,
        threshold=0.5,
    )
    accumulator.update(
        torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0]),
        torch.zeros(5),
        torch.zeros(5, dtype=torch.long),
        torch.tensor([0.0, 49.0, 50.0, 99.0, 100.0], dtype=torch.float64),
        torch.tensor([[0, 0], [0, 0], [1, 0], [2, 0], [3, 0]]),
    )
    metrics = accumulator.compute()
    # Reference code counts int((100 - 0) / 50) == 2 equivalent frames.
    assert metrics["frame_count"] == 2
    # Strict open intervals retain t=49 and t=99: two false components.
    assert metrics["false_components"] == 2
    assert metrics["fa"] == pytest.approx(2 / (2 * 4 * 3))


def test_metrics_reject_nonfinite_logits() -> None:
    accumulator = EventMetricAccumulator(threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        accumulator.update(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_second_stage_auxiliary_losses_ramp_after_warmup() -> None:
    values = [
        _auxiliary_scale(epoch, warmup_epochs=1, ramp_epochs=2)
        for epoch in range(3)
    ]
    assert values == [0.0, 0.5, 1.0]
