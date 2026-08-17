from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from huav.visualization.training_curves import (
    _normalise_competition,
    render_training_curves,
)


def _current_record(epoch: int, score: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "learning_rate": 7e-4 / epoch,
        "train_loss": {
            "total": 1.0 / epoch,
            "focal": 0.2 / epoch,
            "dice": 0.4 / epoch,
            "object_frame": 0.3 / epoch,
            "false_alarm": 0.1 / epoch,
        },
        "validation_loss": {
            "total": 0.9 / epoch,
            "focal": 0.18 / epoch,
            "dice": 0.36 / epoch,
            "object_frame": 0.27 / epoch,
            "false_alarm": 0.09 / epoch,
        },
        "validation": {
            "threshold": 0.5 + epoch * 0.05,
            "competition": {
                "score": score,
                "iou": 0.7 + epoch * 0.01,
                "seg_acc": 0.8 + epoch * 0.01,
                "pd": 0.85 + epoch * 0.01,
                "fa": 1e-5 / epoch,
                "score_fa": math.exp(-10000 * (1e-5 / epoch)),
            },
        },
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    contents = path.read_bytes()
    assert contents.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", contents[16:24])


def test_render_training_curves_writes_expected_png_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "training_curves.png"

    result = render_training_curves(
        [_current_record(1, 0.7), _current_record(2, 0.8)],
        output,
        run_name="cpu_example",
    )

    assert result == output
    assert _png_dimensions(output) == (2400, 1600)


def test_render_training_curves_accepts_one_epoch_and_missing_optional_fields(
    tmp_path: Path,
) -> None:
    record = _current_record(7, 0.8)
    record.pop("learning_rate")
    record.pop("validation_loss")
    train_loss = record["train_loss"]
    assert isinstance(train_loss, dict)
    train_loss.pop("object_frame")

    output = tmp_path / "training_curves.png"
    render_training_curves([record], output, run_name="resume_example")

    assert _png_dimensions(output) == (2400, 1600)


def test_legacy_metrics_are_reconstructed_without_mutating_history() -> None:
    validation = {
        "event": {"iou": 0.8, "pd": 0.9},
        "ev_uav_compatible": {"pd": 0.85, "fa": 2e-5},
    }
    original = {
        "event": dict(validation["event"]),
        "ev_uav_compatible": dict(validation["ev_uav_compatible"]),
    }

    metrics = _normalise_competition(validation)

    assert metrics["iou"] == pytest.approx(0.8)
    assert metrics["seg_acc"] == pytest.approx(0.9)
    assert metrics["pd"] == pytest.approx(0.85)
    assert metrics["fa"] == pytest.approx(2e-5)
    assert metrics["score_fa"] == pytest.approx(math.exp(-0.2))
    assert metrics["score"] == pytest.approx(
        0.2 * 0.8 + 0.1 * 0.9 + 0.4 * 0.85 + 0.3 * math.exp(-0.2)
    )
    assert validation == original


@pytest.mark.parametrize(
    "history",
    [[], [_current_record(1, float("nan"))]],
)
def test_render_training_curves_rejects_invalid_history_without_overwriting(
    tmp_path: Path,
    history: list[dict[str, object]],
) -> None:
    output = tmp_path / "training_curves.png"
    output.write_bytes(b"existing")

    with pytest.raises(ValueError):
        render_training_curves(history, output, run_name="invalid")

    assert output.read_bytes() == b"existing"
