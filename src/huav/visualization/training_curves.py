"""Render one run's training history as a headless PNG dashboard."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_COMPETITION_KEYS = ("score", "iou", "seg_acc", "pd", "fa", "score_fa")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _normalise_competition(validation: Mapping[str, Any]) -> dict[str, float]:
    """Return current competition fields, reconstructing the legacy schema."""
    competition = validation.get("competition")
    if isinstance(competition, Mapping):
        return {
            key: _finite_number(competition.get(key), f"competition.{key}")
            for key in _COMPETITION_KEYS
        }

    event = validation.get("event")
    compatible = validation.get("ev_uav_compatible")
    if not isinstance(event, Mapping) or not isinstance(compatible, Mapping):
        raise ValueError("validation must contain competition or legacy metrics")
    iou = _finite_number(event.get("iou"), "event.iou")
    seg_acc = _finite_number(event.get("pd"), "event.pd")
    pd = _finite_number(compatible.get("pd"), "ev_uav_compatible.pd")
    fa = _finite_number(compatible.get("fa"), "ev_uav_compatible.fa")
    score_fa = math.exp(-10000.0 * fa)
    return {
        "score": 0.2 * iou + 0.1 * seg_acc + 0.4 * pd + 0.3 * score_fa,
        "iou": iou,
        "seg_acc": seg_acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
    }


def _mapping(record: Mapping[str, Any], key: str, *, required: bool) -> Mapping[str, Any]:
    value = record.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_series(
    records: Sequence[Mapping[str, Any]],
    container: str,
    field: str,
) -> list[float] | None:
    values: list[float] = []
    for index, record in enumerate(records):
        source = _mapping(record, container, required=False)
        if field not in source:
            return None
        values.append(_finite_number(source[field], f"history[{index}].{container}.{field}"))
    return values


def _dynamic_loss_fields(
    records: Sequence[Mapping[str, Any]], container: str
) -> list[str]:
    fields: set[str] = set()
    for record in records:
        source = _mapping(record, container, required=False)
        fields.update(
            key
            for key, value in source.items()
            if key != "total"
            and not key.endswith("_mean")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    return sorted(fields)


def _validate_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[float], list[dict[str, float]]]:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("history must be a non-empty sequence")
    records: list[Mapping[str, Any]] = []
    epochs: list[float] = []
    competitions: list[dict[str, float]] = []
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        train_loss = _mapping(record, "train_loss", required=True)
        _finite_number(train_loss.get("total"), f"history[{index}].train_loss.total")
        validation = _mapping(record, "validation", required=True)
        epochs.append(_finite_number(record.get("epoch"), f"history[{index}].epoch"))
        competitions.append(_normalise_competition(validation))
        records.append(record)
    return records, epochs, competitions


def _style_axis(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)


def _show_unavailable(axis: Any, title: str) -> None:
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        "Not recorded",
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="0.45",
    )
    axis.set_axis_off()


def _plot_loss_components(
    axis: Any,
    records: Sequence[Mapping[str, Any]],
    epochs: Sequence[float],
    container: str,
    title: str,
    marker: str | None,
) -> None:
    fields = _dynamic_loss_fields(records, container)
    plotted = False
    for field in fields:
        values = _optional_series(records, container, field)
        if values is None:
            continue
        axis.plot(epochs, values, label=field.replace("_", " "), marker=marker)
        plotted = True
    if not plotted:
        _show_unavailable(axis, title)
        return
    _style_axis(axis, title, "Loss")
    axis.legend(fontsize="small", ncols=2)


def render_training_curves(
    history: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    run_name: str,
) -> Path:
    """Render a 3x2 training dashboard and atomically replace *output_path*."""
    records, epochs, competitions = _validate_history(history)
    output = Path(output_path)
    marker = "o" if len(records) == 1 else None
    scores = [item["score"] for item in competitions]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_epoch = epochs[best_index]
    best_validation = _mapping(records[best_index], "validation", required=True)
    best_threshold = _finite_number(
        best_validation.get("threshold"),
        f"history[{best_index}].validation.threshold",
    )

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(16, 32 / 3),
        dpi=150,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    try:
        train_total = [
            _finite_number(
                _mapping(record, "train_loss", required=True).get("total"),
                f"history[{index}].train_loss.total",
            )
            for index, record in enumerate(records)
        ]
        flat_axes[0].plot(epochs, train_total, label="train", marker=marker)
        validation_total = _optional_series(records, "validation_loss", "total")
        if validation_total is not None:
            flat_axes[0].plot(epochs, validation_total, label="validation", marker=marker)
        _style_axis(flat_axes[0], "Total loss", "Loss")
        flat_axes[0].axvline(best_epoch, color="0.45", linestyle="--", linewidth=1)
        flat_axes[0].legend(fontsize="small")

        _plot_loss_components(
            flat_axes[1], records, epochs, "train_loss", "Training loss components", marker
        )
        _plot_loss_components(
            flat_axes[2],
            records,
            epochs,
            "validation_loss",
            "Validation loss components",
            marker,
        )

        metric_labels = {
            "score": "Score",
            "iou": "IoU",
            "seg_acc": "SegAcc",
            "pd": "Pd",
            "score_fa": "ScoreFa",
        }
        for key, label in metric_labels.items():
            flat_axes[3].plot(
                epochs,
                [item[key] for item in competitions],
                label=label,
                marker=marker,
            )
        _style_axis(flat_axes[3], "Competition metrics", "Metric")
        flat_axes[3].set_ylim(0.0, 1.0)
        flat_axes[3].axvline(best_epoch, color="0.45", linestyle="--", linewidth=1)
        flat_axes[3].scatter(
            [best_epoch], [scores[best_index]], marker="*", s=130, color="black", zorder=5
        )
        flat_axes[3].legend(fontsize="small", ncols=2)

        fa_values = [item["fa"] for item in competitions]
        flat_axes[4].plot(epochs, fa_values, label="Fa", marker=marker)
        _style_axis(flat_axes[4], "False alarms and threshold", "Fa")
        if all(value > 0.0 for value in fa_values):
            flat_axes[4].set_yscale("log")
        else:
            flat_axes[4].set_yscale("symlog", linthresh=1e-12)
        threshold_values = [
            _finite_number(
                _mapping(record, "validation", required=True).get("threshold"),
                f"history[{index}].validation.threshold",
            )
            for index, record in enumerate(records)
        ]
        threshold_axis = flat_axes[4].twinx()
        threshold_axis.plot(
            epochs,
            threshold_values,
            label="Threshold",
            color="tab:orange",
            marker=marker,
        )
        threshold_axis.set_ylabel("Threshold")
        threshold_axis.set_ylim(0.0, 1.0)
        flat_axes[4].axvline(best_epoch, color="0.45", linestyle="--", linewidth=1)
        lines = flat_axes[4].lines[:1] + threshold_axis.lines
        flat_axes[4].legend(lines, [line.get_label() for line in lines], fontsize="small")

        learning_rates = []
        for index, record in enumerate(records):
            if "learning_rate" not in record:
                learning_rates = []
                break
            learning_rates.append(
                _finite_number(record["learning_rate"], f"history[{index}].learning_rate")
            )
        if not learning_rates:
            _show_unavailable(flat_axes[5], "Learning rate")
        else:
            flat_axes[5].plot(
                epochs,
                learning_rates,
                label="Learning rate",
                marker=marker,
            )
            _style_axis(flat_axes[5], "Learning rate", "Learning rate")
            flat_axes[5].legend(fontsize="small")

        figure.suptitle(
            f"{run_name} | best epoch {best_epoch:g} | "
            f"Score {scores[best_index]:.6f} | threshold {best_threshold:.6f}",
            fontsize=15,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            figure.savefig(temporary, format="png", dpi=150)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        plt.close(figure)
    return output


__all__ = ["render_training_curves"]
