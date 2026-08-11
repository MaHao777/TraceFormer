from __future__ import annotations

import pytest
import torch

from huav.training.losses import (
    EventFocalDiceLoss,
    binary_dice_loss,
    binary_focal_loss,
    false_alarm_pixel_loss,
    trajectory_position_loss,
    trajectory_velocity_loss,
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


def _loss_coordinates(count: int) -> torch.Tensor:
    return torch.stack(
        (torch.arange(count, dtype=torch.float32), torch.zeros(count)), dim=1
    )


def test_focal_and_dice_losses_prefer_correct_logits() -> None:
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    correct = torch.tensor([5.0, -5.0, 5.0, -5.0])
    wrong = -correct

    assert binary_focal_loss(correct, targets) < binary_focal_loss(wrong, targets)
    assert binary_dice_loss(correct, targets) < binary_dice_loss(wrong, targets)


def test_event_loss_uses_unweighted_focal_value_and_gradient() -> None:
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    batches = torch.zeros(4, dtype=torch.long)
    baseline_logits = torch.tensor([0.2, -0.4, 1.1, -1.3], requires_grad=True)
    criterion_logits = baseline_logits.detach().clone().requires_grad_(True)

    baseline = binary_focal_loss(
        baseline_logits, targets, alpha=0.75, gamma=2.0
    )
    criterion = EventFocalDiceLoss(
        lambda_trajectory_position=0.0,
        lambda_trajectory_velocity=0.0,
        lambda_false_alarm=0.0,
    )
    focal = criterion(
        criterion_logits,
        targets,
        batches,
    )["focal"]
    baseline.backward()
    focal.backward()

    torch.testing.assert_close(focal, baseline)
    torch.testing.assert_close(criterion_logits.grad, baseline_logits.grad)


def test_trajectory_position_loss_prefers_probability_mass_at_gt_center() -> None:
    targets = torch.tensor([1.0, 0.0])
    coordinates = torch.tensor([[2.0, 2.0], [8.0, 2.0]])
    object_ids = torch.tensor([1, 0])
    timestamps_ms = torch.tensor([10.0, 10.0])

    accurate = trajectory_position_loss(
        torch.tensor([5.0, -5.0]),
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        image_width=11,
        image_height=11,
        roi_radius=8.0,
    )
    inaccurate = trajectory_position_loss(
        torch.tensor([-5.0, 5.0]),
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        image_width=11,
        image_height=11,
        roi_radius=8.0,
    )

    assert accurate < inaccurate


def test_trajectory_position_loss_is_differentiable_without_thresholding() -> None:
    logits = torch.tensor([0.2, -0.3], requires_grad=True)
    loss = trajectory_position_loss(
        logits,
        torch.tensor([1.0, 0.0]),
        torch.tensor([[2.0, 2.0], [8.0, 2.0]]),
        torch.tensor([1, 0]),
        torch.tensor([10.0, 10.0]),
        image_width=11,
        image_height=11,
        roi_radius=8.0,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad != 0).any())


def test_trajectory_velocity_loss_prefers_matching_adjacent_frame_delta() -> None:
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    coordinates = torch.tensor(
        [[2.0, 2.0], [6.0, 2.0], [4.0, 2.0], [2.0, 2.0]]
    )
    object_ids = torch.tensor([1, 0, 1, 0])
    timestamps_ms = torch.tensor([10.0, 10.0, 60.0, 60.0])

    accurate = trajectory_velocity_loss(
        torch.tensor([5.0, -5.0, 5.0, -5.0]),
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        image_width=11,
        image_height=11,
        roi_radius=8.0,
    )
    stationary = trajectory_velocity_loss(
        torch.tensor([5.0, -5.0, -5.0, 5.0]),
        targets,
        coordinates,
        object_ids,
        timestamps_ms,
        image_width=11,
        image_height=11,
        roi_radius=8.0,
    )

    assert accurate < stationary


def test_trajectory_velocity_ignores_nonadjacent_frames_and_different_batches() -> None:
    logits = torch.tensor([0.5, 0.5], requires_grad=True)
    common = {
        "targets": torch.ones(2),
        "coordinates": torch.tensor([[2.0, 2.0], [4.0, 2.0]]),
        "object_ids": torch.ones(2, dtype=torch.long),
        "image_width": 11,
        "image_height": 11,
        "roi_radius": 8.0,
    }
    frame_gap = trajectory_velocity_loss(
        logits,
        timestamps_ms=torch.tensor([10.0, 110.0]),
        batch_indices=torch.zeros(2, dtype=torch.long),
        **common,
    )
    split_batch = trajectory_velocity_loss(
        logits,
        timestamps_ms=torch.tensor([10.0, 60.0]),
        batch_indices=torch.tensor([0, 1]),
        **common,
    )

    torch.testing.assert_close(frame_gap, logits.sum() * 0.0)
    torch.testing.assert_close(split_batch, logits.sum() * 0.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame_ms": 0.0}, "frame_ms"),
        ({"roi_radius": 0.0}, "roi_radius"),
        ({"image_width": 1}, "image_width"),
        ({"image_height": 1}, "image_height"),
    ],
)
def test_trajectory_losses_validate_geometry(
    overrides: dict[str, float | int], message: str
) -> None:
    arguments = {
        "image_width": 11,
        "image_height": 11,
        "roi_radius": 8.0,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        trajectory_position_loss(
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([[2.0, 2.0]]),
            torch.tensor([1]),
            torch.tensor([10.0]),
            **arguments,
        )


def test_false_alarm_pixel_loss_penalizes_high_background_responses() -> None:
    labels = torch.zeros(4)
    batches = torch.zeros(4, dtype=torch.long)
    timestamps_ms = torch.tensor([1.0, 2.0, 51.0, 52.0])
    coordinates = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )
    low_background = torch.full((4,), -5.0)
    high_background = torch.tensor([5.0, -5.0, 5.0, -5.0])

    low_loss = false_alarm_pixel_loss(
        low_background,
        labels,
        batches,
        timestamps_ms,
        coordinates,
        frame_ms=50.0,
        logit_margin=0.0,
    )
    high_loss = false_alarm_pixel_loss(
        high_background,
        labels,
        batches,
        timestamps_ms,
        coordinates,
        frame_ms=50.0,
        logit_margin=0.0,
    )

    assert low_loss < high_loss


def test_false_alarm_pixel_loss_counts_repeated_pixel_events_once() -> None:
    single_loss = false_alarm_pixel_loss(
        torch.tensor([2.0]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[4.0, 5.0]]),
    )
    repeated_loss = false_alarm_pixel_loss(
        torch.tensor([2.0, -1.0, -3.0]),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]),
    )

    torch.testing.assert_close(single_loss, repeated_loss)


def test_false_alarm_pixel_loss_focuses_on_hardest_pixels_per_frame() -> None:
    single_hard_loss = false_alarm_pixel_loss(
        torch.tensor([3.0]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[1.0, 1.0]]),
        topk_per_frame=1,
    )
    hard_with_easy_pixels = false_alarm_pixel_loss(
        torch.tensor([3.0, -8.0, -8.0, -8.0]),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        ),
        topk_per_frame=1,
    )

    torch.testing.assert_close(single_hard_loss, hard_with_easy_pixels)


def test_false_alarm_pixel_loss_accepts_top_k_larger_than_pixel_count() -> None:
    loss = false_alarm_pixel_loss(
        torch.tensor([2.0, -2.0]),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        topk_per_frame=32,
    )

    assert torch.isfinite(loss)


def test_total_loss_combines_focal_dice_false_alarm_and_trajectory_terms() -> None:
    sampled_logits = torch.tensor([2.0, -2.0], requires_grad=True)
    sampled_labels = torch.tensor([1.0, 0.0])
    full_logits = torch.tensor([2.0, -2.0, 3.0], requires_grad=True)
    full_labels = torch.tensor([1.0, 0.0, 0.0])
    criterion = EventFocalDiceLoss(
        image_width=11,
        image_height=11,
        lambda_dice=0.5,
        lambda_false_alarm=0.05,
        false_alarm_topk_per_frame=2,
        lambda_trajectory_position=1.0,
        lambda_trajectory_velocity=0.5,
        trajectory_roi_radius=8.0,
    )

    losses = criterion(
        sampled_logits,
        sampled_labels,
        torch.zeros(2, dtype=torch.long),
        coordinates=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        object_ids=torch.tensor([1, 0]),
        timestamps_ms=torch.tensor([1.0, 1.0]),
        full_logits=full_logits,
        full_targets=full_labels,
        full_object_ids=torch.tensor([1, 0, 0]),
        full_batch_indices=torch.zeros(3, dtype=torch.long),
        full_timestamps_ms=torch.tensor([1.0, 1.0, 51.0]),
        full_coordinates=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        ),
        frame_ms=50.0,
    )

    expected = (
        losses["focal"]
        + 0.5 * losses["dice"]
        + 0.05 * losses["false_alarm"]
        + losses["trajectory_position"]
        + 0.5 * losses["trajectory_velocity"]
    )
    assert torch.allclose(losses["total"], expected)
    assert set(losses) == {
        "total",
        "focal",
        "dice",
        "false_alarm",
        "trajectory_position",
        "trajectory_velocity",
    }
    losses["total"].backward()
    assert torch.isfinite(sampled_logits.grad).all()
    assert torch.isfinite(full_logits.grad).all()


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_trajectory_loss_is_stable_for_empty_and_single_class(target: float) -> None:
    empty_logits = torch.empty(0, requires_grad=True)
    criterion = EventFocalDiceLoss(image_width=11, image_height=11)
    empty = criterion(
        empty_logits,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        coordinates=torch.empty((0, 2)),
        object_ids=torch.empty(0, dtype=torch.long),
        timestamps_ms=torch.empty(0),
    )
    assert all(torch.isfinite(value) for value in empty.values())
    empty["total"].backward()

    logits = torch.tensor([2.0, -1.0, 1.5, -3.0], requires_grad=True)
    targets = torch.full((4,), target)
    coordinates = _loss_coordinates(4)
    timestamps = torch.tensor([1.0, 2.0, 3.0, 4.0])
    batches = torch.zeros(4, dtype=torch.long)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = criterion(
            logits,
            targets,
            batches,
            coordinates=coordinates,
            object_ids=torch.zeros(4, dtype=torch.long),
            timestamps_ms=timestamps,
            full_logits=logits,
            full_targets=targets,
            full_object_ids=torch.where(
                targets > 0.5,
                torch.ones(4, dtype=torch.long),
                torch.zeros(4, dtype=torch.long),
            ),
            full_batch_indices=batches,
            full_timestamps_ms=timestamps,
            full_coordinates=coordinates,
        )["total"]

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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
