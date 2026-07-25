from __future__ import annotations

import numpy as np

from huav.analysis.cone_slopes import analyze_event_array, summarize_measurements

EVENT_DTYPE = [
    ("x", "<i2"),
    ("y", "<i2"),
    ("t", "<f8"),
    ("p", "i1"),
    ("label", "i1"),
    ("name", "i1"),
]


def _moving_track(
    *,
    object_id: int,
    centers: list[tuple[int, int, float]],
) -> list[tuple[int, int, float, int, int, int]]:
    events: list[tuple[int, int, float, int, int, int]] = []
    for center_x, center_y, timestamp in centers:
        events.extend(
            [
                (center_x - 1, center_y, timestamp, 0, 1, object_id),
                (center_x, center_y, timestamp + 1.0, 1, 1, object_id),
                (center_x + 1, center_y, timestamp + 2.0, 1, 1, object_id),
            ]
        )
    return events


def test_known_track_has_one_spatial_block_per_time_block_slope() -> None:
    source = np.array(
        _moving_track(
            object_id=1,
            centers=[(4, 8, 100.0), (8, 8, 1100.0), (12, 8, 2100.0)],
        ),
        dtype=EVENT_DTYPE,
    )

    measured = analyze_event_array(
        source,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        duration_ms=4000.0,
        max_time_distance=2,
    )

    np.testing.assert_allclose(measured.slopes, np.ones(3))
    np.testing.assert_allclose(measured.required_slope_base_1, [0.0, 0.5, 0.0])
    np.testing.assert_allclose(measured.required_slope_base_2, [0.0, 0.0, 0.0])


def test_object_ids_are_never_joined_into_false_high_speed_tracks() -> None:
    rows = _moving_track(
        object_id=1,
        centers=[(4, 8, 100.0), (8, 8, 1100.0)],
    )
    rows += _moving_track(
        object_id=2,
        centers=[(100, 80, 100.0), (104, 80, 1100.0)],
    )
    source = np.array(rows, dtype=EVENT_DTYPE)

    measured = analyze_event_array(
        source,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        duration_ms=4000.0,
        max_time_distance=1,
    )

    np.testing.assert_allclose(measured.slopes, [1.0, 1.0])
    assert measured.track_count == 2


def test_short_groups_and_background_events_do_not_create_measurements() -> None:
    source = np.array(
        [
            (1, 1, 100.0, 0, 0, 0),
            (5, 5, 100.0, 1, 1, 1),
            (6, 5, 101.0, 1, 1, 1),
            (9, 5, 1100.0, 1, 1, 1),
            (10, 5, 1101.0, 1, 1, 1),
        ],
        dtype=EVENT_DTYPE,
    )

    measured = analyze_event_array(
        source,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        duration_ms=4000.0,
        max_time_distance=2,
        min_events_per_group=3,
    )

    assert measured.slopes.size == 0
    assert measured.valid_group_count == 0
    assert measured.track_count == 0


def test_summary_uses_block_per_time_block_units_and_recommends_four_heads() -> None:
    summary = summarize_measurements(
        slopes=np.array([0.25, 0.5, 1.0, 1.5, 2.0]),
        required_slope_base_1=np.array([0.0, 0.25, 0.75, 1.25, 1.75]),
        required_slope_base_2=np.array([0.0, 0.0, 0.5, 1.0, 1.5]),
        target_half_extents=np.array([0.5, 1.0, 1.5]),
    )

    assert summary["slope_unit"] == "spatial_blocks_per_time_block"
    assert summary["raw_slope_quantiles"]["q50"] == 1.0
    assert summary["recommended_cone_slopes"] == [0.5, 1.0, 1.5, 2.0]
    assert summary["recommended_base_radii"] == [1, 1, 2, 2]
