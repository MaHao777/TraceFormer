from __future__ import annotations

import pytest
import torch

from huav.training.losses import (
    EventSpatiotemporalLoss,
    margin_false_alarm_loss,
    spatiotemporal_focal_loss,
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


def test_spatiotemporal_focal_loss_handles_empty_events() -> None:
    logits = torch.empty(0, requires_grad=True)
    loss = spatiotemporal_focal_loss(
        logits,
        torch.empty(0),
        torch.empty((0, 2)),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
    )

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_spatiotemporal_focal_loss_is_finite_for_single_class(target: float) -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    loss = spatiotemporal_focal_loss(
        logits,
        torch.full((3,), target),
        _loss_coordinates(3),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.zeros(3, dtype=torch.long),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_spatiotemporal_focal_loss_uses_no_neighbor_support_of_zero() -> None:
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    loss = spatiotemporal_focal_loss(
        logits,
        targets,
        torch.tensor([[0.0, 0.0], [100.0, 100.0]]),
        torch.tensor([0.0, 1000.0]),
        torch.zeros(2, dtype=torch.long),
        alpha=0.55,
        gamma_focal=2.0,
        gamma_stc=2.0,
        eta_pos=0.5,
        eta_neg=0.5,
        tau=50.0,
        r0=8.0,
        vmax=0.128,
        sigma_xy=8.0,
        sigma_t=25.0,
        max_neighbors=32,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected_weights = torch.tensor([0.55 * 0.5, 0.45])
    expected = (expected_weights * 0.25 * bce).mean()

    torch.testing.assert_close(loss, expected)


def test_spatiotemporal_focal_loss_does_not_link_different_batches() -> None:
    logits = torch.tensor([3.0, 3.0])
    targets = torch.ones(2)
    coordinates = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    timestamps = torch.tensor([10.0, 11.0])
    split_batch_loss = spatiotemporal_focal_loss(
        logits,
        targets,
        coordinates,
        timestamps,
        torch.tensor([0, 1]),
    )
    isolated_loss = spatiotemporal_focal_loss(
        logits,
        targets,
        torch.tensor([[4.0, 4.0], [200.0, 200.0]]),
        torch.tensor([10.0, 500.0]),
        torch.zeros(2, dtype=torch.long),
    )

    torch.testing.assert_close(split_batch_loss, isolated_loss)


def test_confident_continuous_target_track_has_lower_loss_than_isolated_predictions() -> None:
    targets = torch.ones(4)
    batches = torch.zeros(4, dtype=torch.long)
    continuous_loss = spatiotemporal_focal_loss(
        torch.full((4,), 4.0),
        targets,
        torch.tensor([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0], [13.0, 10.0]]),
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
        batches,
    )
    isolated_loss = spatiotemporal_focal_loss(
        torch.full((4,), 0.5),
        targets,
        torch.tensor([[0.0, 0.0], [80.0, 0.0], [160.0, 0.0], [240.0, 0.0]]),
        torch.tensor([0.0, 100.0, 200.0, 300.0]),
        batches,
    )

    assert continuous_loss < isolated_loss


def test_margin_false_alarm_loss_penalizes_isolated_high_background() -> None:
    labels = torch.zeros(4)
    batches = torch.zeros(4, dtype=torch.long)
    timestamps_ms = torch.tensor([1.0, 2.0, 51.0, 52.0])
    coordinates = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )
    low_background = torch.full((4,), -5.0)
    high_background = torch.tensor([5.0, -5.0, 5.0, -5.0])

    low_loss = margin_false_alarm_loss(
        low_background,
        labels,
        batches,
        timestamps_ms,
        coordinates,
        frame_ms=50.0,
        negative_margin=-1.0,
        temperature=0.25,
    )
    high_loss = margin_false_alarm_loss(
        high_background,
        labels,
        batches,
        timestamps_ms,
        coordinates,
        frame_ms=50.0,
        negative_margin=-1.0,
        temperature=0.25,
    )

    assert low_loss < high_loss


def test_margin_false_alarm_loss_counts_repeated_pixel_events_once() -> None:
    single_loss = margin_false_alarm_loss(
        torch.tensor([2.0]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[4.0, 5.0]]),
    )
    repeated_loss = margin_false_alarm_loss(
        torch.tensor([2.0, -1.0, -3.0]),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]),
    )

    torch.testing.assert_close(single_loss, repeated_loss)


def test_margin_false_alarm_loss_suppresses_continuous_background_cluster() -> None:
    low_cluster = margin_false_alarm_loss(
        torch.full((4,), -4.0),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([[5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0]]),
    )
    high_cluster = margin_false_alarm_loss(
        torch.full((4,), 4.0),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([[5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0]]),
    )

    assert low_cluster < high_cluster


def test_margin_false_alarm_loss_focuses_on_hardest_pixels_per_frame() -> None:
    single_hard_loss = margin_false_alarm_loss(
        torch.tensor([3.0]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[1.0, 1.0]]),
        top_k=1,
    )
    hard_with_easy_pixels = margin_false_alarm_loss(
        torch.tensor([3.0, -8.0, -8.0, -8.0]),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        ),
        top_k=1,
    )

    torch.testing.assert_close(single_hard_loss, hard_with_easy_pixels)


def test_margin_false_alarm_loss_accepts_top_k_larger_than_pixel_count() -> None:
    loss = margin_false_alarm_loss(
        torch.tensor([2.0, -2.0]),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        top_k=32,
    )

    assert torch.isfinite(loss)


def test_total_loss_is_stc_focal_plus_full_event_margin_fa() -> None:
    sampled_logits = torch.tensor([2.0, -2.0], requires_grad=True)
    sampled_labels = torch.tensor([1.0, 0.0])
    full_logits = torch.tensor([2.0, -2.0, 3.0], requires_grad=True)
    full_labels = torch.tensor([1.0, 0.0, 0.0])
    criterion = EventSpatiotemporalLoss(
        lambda_fa=0.2,
        fa_negative_margin=-1.0,
        fa_temperature=0.25,
        fa_top_k=2,
    )

    losses = criterion(
        sampled_logits,
        sampled_labels,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([1.0, 1.0]),
        torch.zeros(2, dtype=torch.long),
        full_logits=full_logits,
        full_targets=full_labels,
        full_batch_indices=torch.zeros(3, dtype=torch.long),
        full_timestamps_ms=torch.tensor([1.0, 1.0, 51.0]),
        full_coordinates=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        ),
        frame_ms=50.0,
    )

    expected = losses["stc_focal"] + 0.2 * losses["false_alarm"]
    assert torch.allclose(losses["total"], expected)
    assert set(losses) == {"total", "stc_focal", "false_alarm"}
    losses["total"].backward()
    assert torch.isfinite(sampled_logits.grad).all()
    assert torch.isfinite(full_logits.grad).all()
def test_new_losses_remain_finite_under_cpu_bfloat16_autocast() -> None:
    logits = torch.tensor([2.0, -1.0, 1.5, -3.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    coordinates = _loss_coordinates(4)
    timestamps = torch.tensor([1.0, 2.0, 3.0, 4.0])
    batches = torch.zeros(4, dtype=torch.long)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stc = spatiotemporal_focal_loss(
            logits, targets, coordinates, timestamps, batches
        )
        false_alarm = margin_false_alarm_loss(
            logits, targets, batches, timestamps, coordinates
        )
        loss = stc + false_alarm

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
