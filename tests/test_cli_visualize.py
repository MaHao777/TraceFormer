from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from traceformer.cli import visualize


def _history() -> list[dict[str, object]]:
    return [
        {
            "epoch": 1,
            "learning_rate": 0.001,
            "train_loss": {"total": 1.0, "focal": 0.2},
            "validation": {
                "threshold": 0.5,
                "competition": {
                    "score": 0.8,
                    "iou": 0.75,
                    "seg_acc": 0.8,
                    "pd": 0.85,
                    "fa": 1e-5,
                    "score_fa": 0.9,
                },
            },
        }
    ]


def _write_history(run_dir: Path, history: object) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )


def test_visualize_cli_generates_one_run(tmp_path: Path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "cpu_20260808_170201"
    _write_history(run_dir, _history())
    monkeypatch.setattr(sys, "argv", ["traceformer-visualize", "--run-dir", str(run_dir)])

    visualize.main()

    output = run_dir / "training_curves.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert str(output.resolve()) in capsys.readouterr().out


def test_visualize_cli_batch_continues_and_ignores_similar_names(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_root = tmp_path / "outputs"
    good = output_root / "good"
    empty = output_root / "empty"
    broken = output_root / "broken"
    unrelated = output_root / "experiment" / "artifacts"
    _write_history(good, _history())
    _write_history(empty, [])
    broken.mkdir(parents=True)
    (broken / "history.json").write_text("{broken", encoding="utf-8")
    unrelated.mkdir(parents=True)
    (unrelated / "capacity_history.json").write_text(
        json.dumps(_history()), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["traceformer-visualize", "--all", "--output-root", str(output_root)],
    )

    with pytest.raises(SystemExit) as error:
        visualize.main()

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert (good / "training_curves.png").is_file()
    assert not (empty / "training_curves.png").exists()
    assert not (broken / "training_curves.png").exists()
    assert not (unrelated / "training_curves.png").exists()
    assert "generated=1 skipped=1 failed=1" in captured.out
    assert str(broken / "history.json") in captured.err


def test_visualize_cli_rejects_missing_single_run_history(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run_dir = tmp_path / "missing"
    run_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["traceformer-visualize", "--run-dir", str(run_dir)])

    with pytest.raises(SystemExit) as error:
        visualize.main()

    assert error.value.code == 2
    assert "history.json does not exist" in capsys.readouterr().err
