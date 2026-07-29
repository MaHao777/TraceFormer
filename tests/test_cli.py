from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from huav.cli import train as train_cli
from huav.config import HUAVConfig

COMMANDS = ("inspect", "train", "evaluate", "predict", "package", "check_gpu")
ROOT = Path(__file__).resolve().parents[1]


def run(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", f"huav.cli.{command}", *args], cwd=ROOT,
        text=True, capture_output=True, check=False, env=env,
    )


def test_each_phase_one_command_has_successful_help() -> None:
    for command in COMMANDS:
        result = run(command, "--help")
        assert result.returncode == 0
        if command == "check_gpu":
            assert "--device" in result.stdout
        else:
            assert "--config" in result.stdout


def test_normal_execution_validates_config_then_fails_explicitly() -> None:
    for command in ("inspect", "package"):
        result = run(command, "--config", "configs/huav_base.yaml")
        assert result.returncode != 0
        assert "Phase 1" in result.stderr
        assert "not implemented" in result.stderr


def test_predict_requires_checkpoint_argument() -> None:
    result = run("predict", "--config", "configs/huav_base.yaml")
    assert result.returncode != 0
    assert "--checkpoint" in result.stderr
    assert "required" in result.stderr


def test_train_and_evaluate_expose_runtime_arguments() -> None:
    train_help = run("train", "--help")
    assert train_help.returncode == 0
    assert "--device" in train_help.stdout
    assert "--output-dir" in train_help.stdout
    assert "--initialize-detector" in train_help.stdout

    evaluate_help = run("evaluate", "--help")
    assert evaluate_help.returncode == 0
    assert "--checkpoint" in evaluate_help.stdout
    assert "--reoptimize-threshold" in evaluate_help.stdout


def test_train_rejects_resume_with_detector_initialization() -> None:
    result = run(
        "train",
        "--resume",
        "resume.pt",
        "--initialize-detector",
        "detector.pt",
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_train_rejects_additional_epochs_without_resume() -> None:
    result = run("train", "--additional-epochs", "2")
    assert result.returncode != 0
    assert "--additional-epochs is only valid with --resume" in result.stderr


def test_train_rejects_non_positive_additional_epochs() -> None:
    for value in ("0", "-1"):
        result = run(
            "train",
            "--resume",
            "checkpoint.pt",
            "--additional-epochs",
            value,
        )
        assert result.returncode != 0
        assert "--additional-epochs must be positive" in result.stderr


def test_train_dispatches_detector_initialization_to_new_training(
    monkeypatch,
) -> None:
    received: dict[str, object] = {}

    def fake_run_training(config: HUAVConfig, **kwargs: object) -> dict[str, str]:
        received.update(kwargs)
        return {"run_dir": "run"}

    monkeypatch.setattr(train_cli, "load_config", lambda _: HUAVConfig.default())
    monkeypatch.setattr(train_cli, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        ["huav-train", "--initialize-detector", "detector.pt", "--quiet"],
    )

    train_cli.main()

    assert received["initialize_detector"] == "detector.pt"


def test_train_reports_fresh_joint_lifecycle_error(
    monkeypatch,
    capsys,
) -> None:
    config = HUAVConfig.default()
    config = __import__("dataclasses").replace(
        config,
        model=__import__("dataclasses").replace(
            config.model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=__import__("dataclasses").replace(
            config.training,
            training_stage="joint",
            detector_learning_rate=1e-4,
        ),
    ).validate()
    monkeypatch.setattr(train_cli, "load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["huav-train", "--quiet"])

    with pytest.raises(SystemExit):
        train_cli.main()

    assert "run_training does not accept the joint training stage" in capsys.readouterr().err


def test_train_reports_non_joint_resume_lifecycle_error(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        train_cli,
        "load_config",
        lambda _: HUAVConfig.default(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["huav-train", "--resume", "checkpoint.pt", "--quiet"],
    )

    with pytest.raises(SystemExit):
        train_cli.main()

    assert "resume_training requires the joint training stage" in capsys.readouterr().err


def test_malformed_yaml_is_a_concise_cli_parser_error_without_traceback() -> None:
    with TemporaryDirectory() as directory:
        config = Path(directory) / "bad.yaml"
        config.write_text("resolution: [", encoding="utf-8")
        result = run("inspect", "--config", str(config))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
