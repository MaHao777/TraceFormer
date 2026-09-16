from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from traceformer.cli import evaluate, predict


def test_evaluate_defaults_to_checkpoint_run_and_records_exact_match(
    tmp_path: Path, monkeypatch,
) -> None:
    run_dir = tmp_path / "outputs" / "cpu_20260803_213015"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    metrics = {"threshold": 0.5, "competition": {"score": 0.9}}
    serialized = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "metrics.json").write_text(serialized, encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_config", lambda _: object())
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args, **kwargs: metrics)
    monkeypatch.setattr(
        sys,
        "argv",
        ["traceformer-evaluate", "--checkpoint", str(checkpoint), "--quiet"],
    )

    evaluate.main()

    assert (run_dir / "reproduced_metrics.json").read_text(encoding="utf-8") == serialized
    assert "完全一致" in (run_dir / "experiment.md").read_text(encoding="utf-8")
    assert (run_dir.parent / "INDEX.md").is_file()


def test_evaluate_explicit_output_remains_exact(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    output = tmp_path / "custom" / "metrics.json"
    monkeypatch.setattr(evaluate, "load_config", lambda _: object())
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        ["traceformer-evaluate", "--checkpoint", str(checkpoint), "--output", str(output)],
    )

    evaluate.main()

    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_predict_defaults_to_checkpoint_submission_and_records_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    run_dir = tmp_path / "outputs" / "cuda_20260803_213015"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    called: dict[str, Path] = {}

    def fake_generate(_config, _checkpoint, output_dir, **_kwargs):
        destination = Path(output_dir)
        called["output"] = destination
        destination.mkdir(parents=True)
        (destination / "val_000.txt").write_text(
            "1 2 0.0 1 1\n2 3 0.1 0 0\n", encoding="ascii"
        )
        zip_path = destination / "submission.zip"
        zip_path.touch()
        return zip_path

    monkeypatch.setattr(predict, "load_config", lambda _: object())
    monkeypatch.setattr(predict, "generate_submission_package", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["traceformer-predict", "--checkpoint", str(checkpoint), "--quiet"],
    )

    predict.main()

    assert called["output"] == run_dir / "submission"
    experiment = (run_dir / "experiment.md").read_text(encoding="utf-8")
    assert "TXT 文件数：1" in experiment
    assert "总行数：2" in experiment
    assert "预测为 1：1" in experiment
    assert (run_dir.parent / "INDEX.md").is_file()


def test_predict_explicit_output_directory_remains_exact(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    explicit = tmp_path / "custom-submission"
    called: dict[str, Path] = {}

    def fake_generate(_config, _checkpoint, output_dir, **_kwargs):
        destination = Path(output_dir)
        called["output"] = destination
        destination.mkdir(parents=True)
        zip_path = destination / "submission.zip"
        zip_path.touch()
        return zip_path

    monkeypatch.setattr(predict, "load_config", lambda _: object())
    monkeypatch.setattr(predict, "generate_submission_package", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "traceformer-predict",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(explicit),
        ],
    )

    predict.main()

    assert called["output"] == explicit


def test_evaluate_artifact_update_failure_is_concise_cli_error(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    monkeypatch.setattr(evaluate, "load_config", lambda _: object())
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        evaluate,
        "record_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("record failed")),
    )
    monkeypatch.setattr(sys, "argv", ["traceformer-evaluate", "--checkpoint", str(checkpoint)])

    with pytest.raises(SystemExit) as caught:
        evaluate.main()

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "evaluation succeeded but artifact update failed" in captured.err
    assert "record failed" in captured.err
    assert "Traceback" not in captured.err


def test_predict_artifact_update_failure_is_concise_cli_error(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    run_dir = tmp_path / "outputs" / "run"
    submission_dir = run_dir / "submission"
    submission_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.touch()
    zip_path = submission_dir / "submission.zip"
    zip_path.touch()
    monkeypatch.setattr(predict, "load_config", lambda _: object())
    monkeypatch.setattr(predict, "generate_submission_package", lambda *args, **kwargs: zip_path)
    monkeypatch.setattr(
        predict,
        "record_submission",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("record failed")),
    )
    monkeypatch.setattr(sys, "argv", ["traceformer-predict", "--checkpoint", str(checkpoint)])

    with pytest.raises(SystemExit) as caught:
        predict.main()

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "submission generation succeeded but artifact update failed" in captured.err
    assert "record failed" in captured.err
    assert "Traceback" not in captured.err
