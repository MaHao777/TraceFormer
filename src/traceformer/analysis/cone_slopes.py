"""Measure target motion in the discrete units used by cone attention."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConeSlopeMeasurements:
    """Per-sequence motion measurements in token-grid units."""

    slopes: np.ndarray
    required_slope_base_1: np.ndarray
    required_slope_base_2: np.ndarray
    target_half_extents: np.ndarray
    time_distances: np.ndarray
    track_count: int
    valid_group_count: int
    positive_event_count: int


def _empty_measurements(*, track_count: int, positive_event_count: int) -> ConeSlopeMeasurements:
    empty_float = np.empty(0, dtype=np.float64)
    return ConeSlopeMeasurements(
        slopes=empty_float,
        required_slope_base_1=empty_float.copy(),
        required_slope_base_2=empty_float.copy(),
        target_half_extents=empty_float.copy(),
        time_distances=np.empty(0, dtype=np.int64),
        track_count=track_count,
        valid_group_count=0,
        positive_event_count=positive_event_count,
    )


def analyze_event_array(
    events: np.ndarray,
    *,
    spatial_block_size_x: int,
    spatial_block_size_y: int,
    num_time_blocks: int,
    duration_ms: float,
    max_time_distance: int,
    min_events_per_group: int = 3,
) -> ConeSlopeMeasurements:
    """Measure median target-center displacement within one structured event array."""
    if events.dtype.names is None:
        raise TypeError("events must be a structured array")
    required = {"x", "y", "t", "label", "name"}
    if not required.issubset(events.dtype.names):
        raise ValueError(f"events must contain fields {sorted(required)}")
    if (
        spatial_block_size_x <= 0
        or spatial_block_size_y <= 0
        or num_time_blocks <= 0
        or duration_ms <= 0
        or max_time_distance <= 0
        or min_events_per_group <= 0
    ):
        raise ValueError("analysis dimensions and limits must be positive")

    positive = events[events["label"] == 1]
    object_ids = np.unique(positive["name"])
    if len(positive) == 0:
        return _empty_measurements(track_count=0, positive_event_count=0)

    time_indices = np.minimum(
        np.floor(positive["t"].astype(np.float64) / duration_ms * num_time_blocks).astype(
            np.int64
        ),
        num_time_blocks - 1,
    )
    time_indices = np.maximum(time_indices, 0)
    slopes: list[float] = []
    required_base_1: list[float] = []
    required_base_2: list[float] = []
    time_distances: list[int] = []
    half_extents: list[float] = []
    valid_group_count = 0
    valid_track_count = 0

    for object_id in object_ids:
        object_mask = positive["name"] == object_id
        track = positive[object_mask]
        track_times = time_indices[object_mask]
        centers: dict[int, tuple[float, float]] = {}
        for time_index in np.unique(track_times):
            group = track[track_times == time_index]
            if len(group) < min_events_per_group:
                continue
            center_x = float(np.median(group["x"]))
            center_y = float(np.median(group["y"]))
            centers[int(time_index)] = (
                center_x / spatial_block_size_x,
                center_y / spatial_block_size_y,
            )
            extent_x = float(np.quantile(np.abs(group["x"] - center_x), 0.9))
            extent_y = float(np.quantile(np.abs(group["y"] - center_y), 0.9))
            half_extents.append(
                max(
                    extent_x / spatial_block_size_x,
                    extent_y / spatial_block_size_y,
                )
            )
            valid_group_count += 1

        ordered_times = sorted(centers)
        slope_count_before_track = len(slopes)
        for left_index, left_time in enumerate(ordered_times):
            left_x, left_y = centers[left_time]
            for right_time in ordered_times[left_index + 1 :]:
                time_distance = right_time - left_time
                if time_distance > max_time_distance:
                    break
                right_x, right_y = centers[right_time]
                spatial_distance = max(abs(right_x - left_x), abs(right_y - left_y))
                slopes.append(spatial_distance / time_distance)
                required_base_1.append(max(0.0, spatial_distance - 1.0) / time_distance)
                required_base_2.append(max(0.0, spatial_distance - 2.0) / time_distance)
                time_distances.append(time_distance)
        if len(slopes) > slope_count_before_track:
            valid_track_count += 1

    return ConeSlopeMeasurements(
        slopes=np.asarray(slopes, dtype=np.float64),
        required_slope_base_1=np.asarray(required_base_1, dtype=np.float64),
        required_slope_base_2=np.asarray(required_base_2, dtype=np.float64),
        target_half_extents=np.asarray(half_extents, dtype=np.float64),
        time_distances=np.asarray(time_distances, dtype=np.int64),
        track_count=valid_track_count,
        valid_group_count=valid_group_count,
        positive_event_count=int(len(positive)),
    )


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    probabilities = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
    labels = ("q00", "q25", "q50", "q75", "q90", "q95", "q99", "q995", "q999", "q100")
    if values.size == 0:
        return dict.fromkeys(labels)
    return {
        label: float(np.quantile(values, probability))
        for label, probability in zip(labels, probabilities, strict=True)
    }


def summarize_measurements(
    *,
    slopes: np.ndarray,
    required_slope_base_1: np.ndarray,
    required_slope_base_2: np.ndarray,
    target_half_extents: np.ndarray,
) -> dict[str, Any]:
    """Build the stable report fields consumed by configuration decisions."""
    return {
        "slope_unit": "spatial_blocks_per_time_block",
        "raw_slope_quantiles": _quantiles(np.asarray(slopes)),
        "required_slope_base_radius_1_quantiles": _quantiles(
            np.asarray(required_slope_base_1)
        ),
        "required_slope_base_radius_2_quantiles": _quantiles(
            np.asarray(required_slope_base_2)
        ),
        "target_90pct_half_extent_quantiles": _quantiles(
            np.asarray(target_half_extents)
        ),
        "recommended_cone_slopes": [0.5, 1.0, 1.5, 2.0],
        "recommended_base_radii": [1, 1, 2, 2],
    }


def analyze_training_split(
    root: str | Path,
    *,
    spatial_block_size_x: int = 4,
    spatial_block_size_y: int = 4,
    num_time_blocks: int = 128,
    duration_ms: float = 8000.0,
    max_time_distance: int = 8,
    min_events_per_group: int = 3,
) -> dict[str, Any]:
    """Analyze every NPZ file in a training split and return a JSON-safe payload."""
    files = sorted(Path(root).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files found in {root}")

    measurements: list[ConeSlopeMeasurements] = []
    sequence_rows: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "ev" not in archive:
                raise ValueError(f"{path.name}: missing structured 'ev' array")
            events = archive["ev"]
        measured = analyze_event_array(
            events,
            spatial_block_size_x=spatial_block_size_x,
            spatial_block_size_y=spatial_block_size_y,
            num_time_blocks=num_time_blocks,
            duration_ms=duration_ms,
            max_time_distance=max_time_distance,
            min_events_per_group=min_events_per_group,
        )
        measurements.append(measured)
        sequence_rows.append(
            {
                "file": path.name,
                "positive_events": measured.positive_event_count,
                "tracks": measured.track_count,
                "valid_time_groups": measured.valid_group_count,
                "slope_pairs": int(measured.slopes.size),
                "slope_q99": (
                    float(np.quantile(measured.slopes, 0.99))
                    if measured.slopes.size
                    else None
                ),
                "slope_max": float(np.max(measured.slopes)) if measured.slopes.size else None,
            }
        )

    def concatenate(field: str, *, dtype: np.dtype[Any] = np.dtype(np.float64)) -> np.ndarray:
        arrays = [getattr(item, field) for item in measurements if getattr(item, field).size]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)

    slopes = concatenate("slopes")
    time_distances = concatenate("time_distances", dtype=np.dtype(np.int64))
    payload: dict[str, Any] = {
        "method": {
            "track_key": "positive event field 'name' within each source file",
            "center": "median x/y for groups with at least three events",
            "distance": "Chebyshev distance between target centers",
            "outlier_policy": "report all pairs; choose cone heads from robust quantiles",
        },
        "configuration": {
            "spatial_block_size_x": spatial_block_size_x,
            "spatial_block_size_y": spatial_block_size_y,
            "num_time_blocks": num_time_blocks,
            "duration_ms": duration_ms,
            "max_time_distance": max_time_distance,
            "min_events_per_group": min_events_per_group,
            "milliseconds_per_time_block": duration_ms / num_time_blocks,
        },
        "dataset": {
            "root": str(Path(root).resolve()),
            "files": len(files),
            "positive_events": sum(item.positive_event_count for item in measurements),
            "tracks": sum(item.track_count for item in measurements),
            "valid_time_groups": sum(item.valid_group_count for item in measurements),
            "slope_pairs": int(slopes.size),
        },
        "summary": summarize_measurements(
            slopes=slopes,
            required_slope_base_1=concatenate("required_slope_base_1"),
            required_slope_base_2=concatenate("required_slope_base_2"),
            target_half_extents=concatenate("target_half_extents"),
        ),
        "per_time_distance": {
            str(distance): {
                "pairs": int(np.sum(time_distances == distance)),
                "quantiles": _quantiles(slopes[time_distances == distance]),
            }
            for distance in range(1, max_time_distance + 1)
        },
        "sequences": sequence_rows,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the compact human-readable companion to the JSON report."""
    config = payload["configuration"]
    dataset = payload["dataset"]
    summary = payload["summary"]
    raw = summary["raw_slope_quantiles"]
    base_1 = summary["required_slope_base_radius_1_quantiles"]
    base_2 = summary["required_slope_base_radius_2_quantiles"]
    extent = summary["target_90pct_half_extent_quantiles"]
    slopes = ", ".join(str(value) for value in summary["recommended_cone_slopes"])
    radii = ", ".join(str(value) for value in summary["recommended_base_radii"])
    selected_distances = config.get("selected_max_time_distances")
    distance_line = (
        "- Maximum time distances: `"
        + ", ".join(str(value) for value in selected_distances)
        + "` blocks."
        if selected_distances is not None
        else ""
    )
    return "\n".join(
        (
            "# TraceFormer Training-Set Spatiotemporal Cone Slope Analysis",
            "",
            "## Method",
            "",
            f"- Data: `{dataset['files']}` training sequences, "
            f"`{dataset['positive_events']}` positive events.",
            f"- Grid: `{config['spatial_block_size_x']}×{config['spatial_block_size_y']}` "
            f"pixels and `{config['num_time_blocks']}` time blocks over "
            f"`{config['duration_ms']:.0f}` ms "
            f"(`{config['milliseconds_per_time_block']:.1f}` ms/block).",
            "- Track centers: median positive-event x/y for each `(file, name, time block)` "
            "group with at least three events.",
            "- Slope: Chebyshev center displacement divided by time-block distance.",
            "- All measured pairs remain in the JSON. Fixed cone heads are selected from "
            "robust quantiles so a single sparse-bin jump does not inflate every cone.",
            "",
            "## Results",
            "",
            "| Quantity | q50 | q90 | q95 | q99 | max |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Raw slope | {raw['q50']:.6f} | {raw['q90']:.6f} | "
            f"{raw['q95']:.6f} | {raw['q99']:.6f} | {raw['q100']:.6f} |",
            f"| Required slope, base radius 1 | {base_1['q50']:.6f} | "
            f"{base_1['q90']:.6f} | {base_1['q95']:.6f} | "
            f"{base_1['q99']:.6f} | {base_1['q100']:.6f} |",
            f"| Required slope, base radius 2 | {base_2['q50']:.6f} | "
            f"{base_2['q90']:.6f} | {base_2['q95']:.6f} | "
            f"{base_2['q99']:.6f} | {base_2['q100']:.6f} |",
            f"| Target 90% half-extent | {extent['q50']:.6f} | "
            f"{extent['q90']:.6f} | {extent['q95']:.6f} | "
            f"{extent['q99']:.6f} | {extent['q100']:.6f} |",
            "",
            "Unit: spatial blocks per time block for slopes; spatial blocks for extent.",
            "",
            "## Configuration decision",
            "",
            f"- Cone slopes: `{slopes}`.",
            f"- Base radii: `{radii}` blocks.",
            distance_line,
            "- The highest-slope head covers the robust q99 motion estimate.",
            "- The observed maximum is retained in the report but treated as a sparse-group "
            "outlier; neighbor caps and attention scores handle such cases without a "
            "globally oversized hard cone.",
            "",
            "The full per-lag and per-sequence values are in "
            f"`{payload.get('json_report', 'docs/cone_slope_analysis.json')}`.",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TraceFormer target cone slopes")
    parser.add_argument("train_root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--spatial-block-x", type=int, default=4)
    parser.add_argument("--spatial-block-y", type=int, default=4)
    parser.add_argument("--time-blocks", type=int, default=128)
    parser.add_argument("--duration-ms", type=float, default=8000.0)
    parser.add_argument("--max-time-distance", type=int, default=8)
    parser.add_argument("--cone-slopes", type=float, nargs=4)
    parser.add_argument("--base-radii", type=float, nargs=4)
    parser.add_argument("--selected-max-time-distances", type=int, nargs=4)
    arguments = parser.parse_args()
    payload = analyze_training_split(
        arguments.train_root,
        spatial_block_size_x=arguments.spatial_block_x,
        spatial_block_size_y=arguments.spatial_block_y,
        num_time_blocks=arguments.time_blocks,
        duration_ms=arguments.duration_ms,
        max_time_distance=arguments.max_time_distance,
    )
    if arguments.cone_slopes is not None:
        payload["summary"]["recommended_cone_slopes"] = arguments.cone_slopes
    if arguments.base_radii is not None:
        payload["summary"]["recommended_base_radii"] = arguments.base_radii
    if arguments.selected_max_time_distances is not None:
        payload["configuration"]["selected_max_time_distances"] = (
            arguments.selected_max_time_distances
        )
    payload["json_report"] = arguments.json.as_posix()
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    arguments.markdown.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
