"""Analyze physics-inspired high-threshold views on HUAV NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from huav.data import is_dense_balanced_event_stream, simulate_high_threshold_events


def _events_tensor(events: np.ndarray) -> torch.Tensor:
    required = {"x", "y", "t", "p", "label", "name"}
    if events.dtype.names is None or not required.issubset(events.dtype.names):
        raise ValueError(f"events must contain fields {sorted(required)}")
    values = np.column_stack(
        (
            events["x"].astype(np.float32, copy=False),
            events["y"].astype(np.float32, copy=False),
            events["t"].astype(np.float32, copy=False),
            events["p"].astype(np.float32, copy=False),
        )
    )
    return torch.from_numpy(np.ascontiguousarray(values))


def _object_frame_groups(
    events: np.ndarray,
    indices: np.ndarray,
    *,
    frame_ms: float,
) -> int:
    selected = events[indices]
    positive = (
        (selected["label"] == 1)
        & (selected["name"] > 0)
        & (np.remainder(np.floor(selected["t"]), frame_ms) > 0)
    )
    if not np.any(positive):
        return 0
    groups = np.column_stack(
        (
            selected["name"][positive].astype(np.int64, copy=False),
            np.floor(selected["t"][positive] / frame_ms).astype(
                np.int64, copy=False
            ),
        )
    )
    return int(np.unique(groups, axis=0).shape[0])


def analyze_trigger_thresholds(
    root: str | Path,
    *,
    threshold_ratios: tuple[float, ...] = (1.5, 2.0, 3.0),
    min_events: int = 300000,
    max_polarity_fraction: float = 0.65,
    width: int = 346,
    height: int = 260,
    frame_ms: float = 50.0,
) -> dict[str, Any]:
    """Return per-file and aggregate retention statistics."""

    files = sorted(Path(root).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files found in {root}")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    ratios = tuple(float(value) for value in threshold_ratios)
    if not ratios or len(set(ratios)) != len(ratios) or any(value <= 1 for value in ratios):
        raise ValueError("threshold ratios must be unique and greater than 1")

    aggregates = {
        f"{ratio:g}": {
            "source_events": 0,
            "retained_events": 0,
            "source_positive_events": 0,
            "retained_positive_events": 0,
            "source_object_frames": 0,
            "retained_object_frames": 0,
        }
        for ratio in ratios
    }
    rows: list[dict[str, Any]] = []
    eligible_count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "ev" not in archive:
                raise ValueError(f"{path.name}: missing structured 'ev' array")
            events = archive["ev"]
        event_tensor = _events_tensor(events)
        polarity_fraction = float(np.mean(events["p"]))
        eligible = is_dense_balanced_event_stream(
            event_tensor,
            min_events=min_events,
            max_polarity_fraction=max_polarity_fraction,
        )
        row: dict[str, Any] = {
            "file": path.name,
            "events": int(len(events)),
            "positive_events": int(np.sum(events["label"] == 1)),
            "positive_polarity_fraction": polarity_fraction,
            "maximum_polarity_fraction": max(
                polarity_fraction, 1.0 - polarity_fraction
            ),
            "eligible": eligible,
            "thresholds": {},
        }
        if not eligible:
            rows.append(row)
            continue
        eligible_count += 1
        all_indices = np.arange(len(events), dtype=np.int64)
        source_groups = _object_frame_groups(events, all_indices, frame_ms=frame_ms)
        source_positive = int(np.sum(events["label"] == 1))
        for ratio in ratios:
            view = simulate_high_threshold_events(
                event_tensor,
                width=width,
                height=height,
                threshold_ratio=ratio,
            )
            indices = view.source_indices.numpy()
            retained_positive = int(np.sum(events["label"][indices] == 1))
            retained_groups = _object_frame_groups(events, indices, frame_ms=frame_ms)
            metrics = {
                "retained_events": int(len(indices)),
                "event_retention": float(len(indices) / len(events)),
                "retained_positive_events": retained_positive,
                "positive_event_retention": (
                    float(retained_positive / source_positive)
                    if source_positive
                    else 0.0
                ),
                "source_object_frames": source_groups,
                "retained_object_frames": retained_groups,
                "object_frame_retention": (
                    float(retained_groups / source_groups) if source_groups else 0.0
                ),
            }
            row["thresholds"][f"{ratio:g}"] = metrics
            aggregate = aggregates[f"{ratio:g}"]
            aggregate["source_events"] += int(len(events))
            aggregate["retained_events"] += int(len(indices))
            aggregate["source_positive_events"] += source_positive
            aggregate["retained_positive_events"] += retained_positive
            aggregate["source_object_frames"] += source_groups
            aggregate["retained_object_frames"] += retained_groups
        rows.append(row)

    for aggregate in aggregates.values():
        for output, retained, source in (
            ("event_retention", "retained_events", "source_events"),
            (
                "positive_event_retention",
                "retained_positive_events",
                "source_positive_events",
            ),
            (
                "object_frame_retention",
                "retained_object_frames",
                "source_object_frames",
            ),
        ):
            denominator = aggregate[source]
            aggregate[output] = (
                float(aggregate[retained] / denominator) if denominator else 0.0
            )

    return {
        "method": {
            "name": "per-pixel signed contrast integration",
            "polarity_mapping": {"0": -1, "1": 1},
            "initial_residual": 0.0,
            "reset_policy": "subtract crossed threshold and retain residual",
            "uses_annotations": False,
        },
        "configuration": {
            "threshold_ratios": list(ratios),
            "min_events": min_events,
            "max_polarity_fraction": max_polarity_fraction,
            "width": width,
            "height": height,
            "frame_ms": frame_ms,
        },
        "dataset": {
            "root": str(Path(root).resolve()),
            "files": len(files),
            "eligible_files": eligible_count,
        },
        "aggregate": aggregates,
        "sequences": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    config = payload["configuration"]
    lines = [
        "# HUAV High-Trigger-Threshold Event Analysis",
        "",
        "## Method",
        "",
        "- Per-pixel signed contrast integration; ON/OFF increments share one residual.",
        "- A crossing retains the exact source event and subtracts the threshold while carrying residual state.",
        "- Eligibility and event generation are annotation-independent.",
        f"- Dense gate: at least `{config['min_events']}` events and maximum polarity fraction no greater than `{config['max_polarity_fraction']}`.",
        "",
        "## Aggregate results",
        "",
        f"Data: `{dataset['files']}` files; `{dataset['eligible_files']}` eligible dense balanced files.",
        "",
        "| threshold ratio | event retention | positive retention | object-frame retention |",
        "|---:|---:|---:|---:|",
    ]
    for ratio, aggregate in payload["aggregate"].items():
        lines.append(
            f"| {ratio}× | {aggregate['event_retention']:.6f} | "
            f"{aggregate['positive_event_retention']:.6f} | "
            f"{aggregate['object_frame_retention']:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            "- Train with `1.5×` and `2.0×` views.",
            "- Keep `3.0×` as a diagnostic stress level because its target-frame coverage is substantially lower.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze physics-inspired high-threshold HUAV event views"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--threshold-ratios", type=float, nargs="+", default=(1.5, 2.0, 3.0))
    parser.add_argument("--min-events", type=int, default=300000)
    parser.add_argument("--max-polarity-fraction", type=float, default=0.65)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--frame-ms", type=float, default=50.0)
    arguments = parser.parse_args()
    payload = analyze_trigger_thresholds(
        arguments.root,
        threshold_ratios=tuple(arguments.threshold_ratios),
        min_events=arguments.min_events,
        max_polarity_fraction=arguments.max_polarity_fraction,
        width=arguments.width,
        height=arguments.height,
        frame_ms=arguments.frame_ms,
    )
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    arguments.markdown.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
