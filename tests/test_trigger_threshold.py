from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from traceformer.analysis.trigger_thresholds import analyze_trigger_thresholds
from traceformer.data import (
    is_dense_balanced_event_stream,
    simulate_high_threshold_events,
)


def test_same_polarity_crosses_threshold_and_preserves_source_mapping() -> None:
    events = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.1, 1.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 0.3, 1.0],
        ]
    )

    view = simulate_high_threshold_events(
        events, width=2, height=2, threshold_ratio=2.0
    )

    assert view.source_indices.tolist() == [1, 3]
    torch.testing.assert_close(view.events, events[view.source_indices])


def test_opposite_polarities_cancel_with_per_pixel_state_isolation() -> None:
    events = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [0.0, 0.0, 0.2, 0.0],
            [1.0, 0.0, 0.3, 1.0],
            [0.0, 0.0, 0.4, 1.0],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )

    view = simulate_high_threshold_events(
        events, width=2, height=1, threshold_ratio=2.0
    )

    assert view.source_indices.tolist() == [3, 5]


def test_fractional_threshold_carries_residual_between_crossings() -> None:
    events = torch.tensor([[0.0, 0.0, float(index), 1.0] for index in range(4)])

    first = simulate_high_threshold_events(
        events, width=1, height=1, threshold_ratio=1.5
    )
    second = simulate_high_threshold_events(
        events, width=1, height=1, threshold_ratio=1.5
    )

    assert first.source_indices.tolist() == [1, 2]
    torch.testing.assert_close(first.events, second.events)


def test_dense_gate_uses_only_event_count_and_global_polarity() -> None:
    balanced = torch.tensor(
        [[0.0, 0.0, float(index), float(index % 2)] for index in range(6)]
    )
    one_sided = balanced.clone()
    one_sided[:, 3] = 1.0

    assert is_dense_balanced_event_stream(
        balanced, min_events=6, max_polarity_fraction=0.6
    )
    assert not is_dense_balanced_event_stream(
        balanced, min_events=7, max_polarity_fraction=0.6
    )
    assert not is_dense_balanced_event_stream(
        one_sided, min_events=6, max_polarity_fraction=0.6
    )






def test_analysis_reports_eligible_retention_without_using_labels_for_gate(
    tmp_path: Path,
) -> None:
    dtype = np.dtype(
        [
            ("x", "<i4"),
            ("y", "<i4"),
            ("t", "<f8"),
            ("p", "<i4"),
            ("label", "<i4"),
            ("name", "<i4"),
        ]
    )
    events = np.zeros(6, dtype=dtype)
    events["t"] = np.arange(6) * 10.0
    events["p"] = [1, 1, 0, 0, 1, 1]
    events["label"] = [1, 1, 0, 0, 1, 1]
    events["name"] = [3, 3, 0, 0, 3, 3]
    np.savez(tmp_path / "train_000.npz", ev=events)

    report = analyze_trigger_thresholds(
        tmp_path,
        threshold_ratios=(2.0,),
        min_events=6,
        max_polarity_fraction=0.7,
        width=1,
        height=1,
    )

    assert report["dataset"] == {
        "root": str(tmp_path.resolve()),
        "files": 1,
        "eligible_files": 1,
    }
    assert report["method"]["uses_annotations"] is False
    assert report["aggregate"]["2"]["retained_events"] == 3
