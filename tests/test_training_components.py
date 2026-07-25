from __future__ import annotations

import pytest
import torch

from huav.training.losses import (
    EventFocalDiceLoss,
    binary_dice_loss,
    binary_focal_loss,
    object_frame_recall_loss,
)
from huav.training.metrics import (
    EventMetricAccumulator,
    EVUAVMetricAccumulator,
    competition_metrics,
)
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


def test_sampler_reserves_configured_slots_for_highest_logit_negatives() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    logits = torch.tensor([0.0, -2.0, 4.0, 3.0, -1.0])

    selected = sample_event_indices(
        labels,
        logits=logits,
        background_ratio=2,
        hard_negative_fraction=0.5,
        generator=torch.Generator().manual_seed(7),
    )

    assert 0 in selected.tolist()
    assert 2 in selected.tolist()
    assert int((labels[selected] == 0).sum()) == 2


def test_focal_and_dice_losses_prefer_correct_logits() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    correct = torch.tensor([5.0, -5.0, 5.0, -5.0])
    wrong = -correct

    assert binary_focal_loss(correct, labels) < binary_focal_loss(wrong, labels)
    assert binary_dice_loss(correct, labels) < binary_dice_loss(wrong, labels)


def test_total_loss_is_focal_plus_configured_per_sample_dice() -> None:
    logits = torch.tensor([0.0, 0.0, 0.1, -0.1], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    batches = torch.tensor([0, 0, 1, 1])
    criterion = EventFocalDiceLoss(
        focal_alpha=0.75,
        focal_gamma=2.0,
        lambda_dice=1.5,
    )

    losses = criterion(logits, labels, batches)
    expected = losses["focal"] + 1.5 * losses["dice"]
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_event_loss_rejects_disabled_dice_objective() -> None:
    with pytest.raises(ValueError, match="lambda_dice"):
        EventFocalDiceLoss(lambda_dice=0.0)


def test_object_frame_loss_requires_one_confident_event_per_object_frame() -> None:
    labels = torch.ones(4)
    object_ids = torch.tensor([1, 1, 2, 2])
    timestamps_ms = torch.tensor([1.0, 2.0, 51.0, 52.0])
    confident = torch.tensor([3.0, -3.0, 3.0, -3.0])
    missed = torch.full((4,), -3.0)

    confident_loss = object_frame_recall_loss(
        confident,
        labels,
        object_ids,
        timestamps_ms,
        frame_ms=50.0,
        logit_margin=2.0,
    )
    missed_loss = object_frame_recall_loss(
        missed,
        labels,
        object_ids,
        timestamps_ms,
        frame_ms=50.0,
        logit_margin=2.0,
    )

    assert confident_loss < missed_loss


def test_total_loss_adds_configured_object_frame_objective() -> None:
    logits = torch.tensor([2.0, -1.0, 2.0, -1.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    object_ids = torch.tensor([1, 0, 2, 0])
    timestamps_ms = torch.tensor([1.0, 1.0, 51.0, 51.0])
    criterion = EventFocalDiceLoss(
        lambda_dice=0.5,
        lambda_object_frame=0.25,
        object_frame_logit_margin=2.0,
    )

    losses = criterion(
        logits,
        labels,
        object_ids=object_ids,
        timestamps_ms=timestamps_ms,
        frame_ms=50.0,
    )

    expected = (
        losses["focal"]
        + 0.5 * losses["dice"]
        + 0.25 * losses["object_frame"]
    )
    assert torch.allclose(losses["total"], expected)


def test_per_sample_dice_safely_penalizes_false_positives_without_targets() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    batches = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([5.0, -5.0, -5.0, -5.0])
    bad = torch.tensor([5.0, -5.0, 5.0, 5.0])

    good_loss = binary_dice_loss(good, labels, batches)
    bad_loss = binary_dice_loss(bad, labels, batches)

    assert torch.isfinite(good_loss)
    assert good_loss < bad_loss


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


def test_ev_uav_frames_use_official_integer_millisecond_coordinates() -> None:
    accumulator = EVUAVMetricAccumulator(
        width=4,
        height=3,
        frame_ms=50,
        object_correct_fraction=0.0001,
        threshold=0.5,
    )
    accumulator.update(
        torch.full((3,), 5.0),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([49.9, 50.1, 99.9], dtype=torch.float64),
        torch.tensor([[0, 0], [1, 0], [2, 0]]),
    )

    assert accumulator.compute()["false_components"] == 2


def test_ev_uav_false_mask_matches_official_uint8_wraparound() -> None:
    accumulator = EVUAVMetricAccumulator(
        width=4,
        height=3,
        frame_ms=50,
        object_correct_fraction=0.0001,
        threshold=0.5,
    )
    accumulator.update(
        torch.full((256,), 5.0),
        torch.zeros(256),
        torch.zeros(256, dtype=torch.long),
        torch.full((256,), 1.9, dtype=torch.float64),
        torch.zeros((256, 2), dtype=torch.long),
    )

    assert accumulator.compute()["false_components"] == 0


def test_competition_metrics_reproduce_official_website_score() -> None:
    competition = competition_metrics(
        event={"tp": 57205, "fp": 3332, "fn": 8301},
        object_metrics={
            "detected_objects": 4118,
            "object_instances": 4762,
            "false_components": 2394,
            "frame_count": 3752,
        },
        width=346,
        height=260,
    )

    assert competition["iou"] == pytest.approx(0.831009035707)
    assert competition["seg_acc"] == pytest.approx(0.873278783623)
    assert competition["pd"] == pytest.approx(0.864762704746)
    assert competition["fa"] == pytest.approx(7.09270455194e-06)
    assert competition["score_fa"] == pytest.approx(0.931529848925)
    assert competition["score"] == pytest.approx(0.87889372208)
    assert competition["counts"] == {
        "intersection": 57205,
        "union": 68838,
        "true_positive": 57205,
        "gt_positive": 65506,
        "detected_objects": 4118,
        "objects": 4762,
        "false_alarms": 2394,
        "frames": 3752,
    }


def test_metrics_reject_nonfinite_logits() -> None:
    accumulator = EventMetricAccumulator(threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        accumulator.update(torch.tensor([float("nan")]), torch.tensor([0.0]))
