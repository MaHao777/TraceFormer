from __future__ import annotations

import pytest
import torch

from huav.training.losses import (
    EventFocalDiceLoss,
    background_threshold_consistency_loss,
    binary_dice_loss,
    binary_focal_loss,
    false_alarm_pixel_loss,
    frame_cell_objectness_loss,
    object_frame_recall_loss,
    positive_logit_preservation_loss,
    sparse_block_false_peak_loss,
)
from huav.training.metrics import (
    EventMetricAccumulator,
    EVUAVMetricAccumulator,
    competition_metrics,
)
from huav.training.sampling import sample_event_indices


def test_positive_preservation_uses_all_positive_events_and_detaches_teacher() -> None:
    fused = torch.tensor([-2.0, 1.0, -3.0, 4.0], requires_grad=True)
    full = torch.tensor([0.0, -4.0, -1.0, 2.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])

    loss = positive_logit_preservation_loss(
        fused,
        full,
        labels,
        logit_margin=0.5,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert fused.grad is not None
    assert fused.grad[0] < 0 and fused.grad[2] < 0
    assert fused.grad[1] == 0 and fused.grad[3] == 0
    assert full.grad is None


def test_background_threshold_consistency_is_asymmetric_and_detaches_full() -> None:
    sparse = torch.tensor([-2.0, 2.0, 0.0], requires_grad=True)
    full = torch.tensor([-2.0, -2.0, 3.0], requires_grad=True)
    source_indices = torch.tensor([0, 1, 2])
    targets = torch.tensor([0.0, 0.0, 1.0])

    loss = background_threshold_consistency_loss(
        sparse,
        full,
        source_indices,
        targets,
        logit_margin=0.0,
    )
    loss.backward()

    assert loss.item() > 1.0
    assert sparse.grad is not None and torch.isfinite(sparse.grad).all()
    assert full.grad is None


def test_sparse_block_false_peak_penalizes_highest_background_block() -> None:
    common = {
        "targets": torch.zeros(4),
        "batch_indices": torch.zeros(4, dtype=torch.long),
        "timestamps_ms": torch.tensor([10.0, 20.0, 10.0, 20.0]),
        "coordinates": torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [9.0, 0.0], [10.0, 1.0]]
        ),
        "frame_ms": 50.0,
        "block_size_x": 8,
        "block_size_y": 8,
        "logit_margin": 0.0,
        "topk_per_frame": 1,
    }

    low = sparse_block_false_peak_loss(torch.full((4,), -3.0), **common)
    high = sparse_block_false_peak_loss(
        torch.tensor([-3.0, -3.0, 3.0, -3.0]), **common
    )

    assert high > low


def test_sparse_block_false_peak_empty_background_is_backward_safe() -> None:
    logits = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = sparse_block_false_peak_loss(
        logits,
        torch.ones(2),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([10.0, 20.0]),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        frame_ms=50.0,
        block_size_x=8,
        block_size_y=8,
        logit_margin=0.0,
        topk_per_frame=1,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None


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


def test_frame_cell_objectness_loss_uses_group_max_targets_and_backpropagates() -> None:
    indices = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0.0, 1.0, 0.0, 0.0])
    correct = torch.tensor([4.0, -4.0], requires_grad=True)
    wrong = -correct.detach()

    correct_loss = frame_cell_objectness_loss(
        correct,
        indices,
        labels,
        alpha=0.75,
        gamma=2.0,
    )
    wrong_loss = frame_cell_objectness_loss(
        wrong,
        indices,
        labels,
        alpha=0.75,
        gamma=2.0,
    )
    correct_loss.backward()

    assert correct_loss < wrong_loss
    assert correct.grad is not None and torch.isfinite(correct.grad).all()


def test_frame_cell_objectness_loss_empty_input_is_backward_safe() -> None:
    logits = torch.empty(0, requires_grad=True)
    loss = frame_cell_objectness_loss(
        logits,
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        alpha=0.75,
        gamma=2.0,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None


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


def test_total_loss_adds_full_event_false_alarm_objective() -> None:
    sampled_logits = torch.tensor([2.0, -2.0], requires_grad=True)
    sampled_labels = torch.tensor([1.0, 0.0])
    full_logits = torch.tensor([2.0, -2.0, 3.0], requires_grad=True)
    full_labels = torch.tensor([1.0, 0.0, 0.0])
    criterion = EventFocalDiceLoss(
        lambda_dice=0.5,
        lambda_false_alarm=0.2,
        false_alarm_logit_margin=0.0,
        false_alarm_topk_per_frame=2,
    )

    losses = criterion(
        sampled_logits,
        sampled_labels,
        full_logits=full_logits,
        full_targets=full_labels,
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
        + 0.2 * losses["false_alarm"]
    )
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert torch.isfinite(sampled_logits.grad).all()
    assert torch.isfinite(full_logits.grad).all()


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
