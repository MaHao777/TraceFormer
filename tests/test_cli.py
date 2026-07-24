from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


COMMANDS = ("inspect", "train", "evaluate", "predict", "package")
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
        assert "--config" in result.stdout


def test_normal_execution_validates_config_then_fails_explicitly() -> None:
    for command in ("inspect", "predict", "package"):
        result = run(command, "--config", "configs/huav_base.yaml")
        assert result.returncode != 0
        assert "Phase 1" in result.stderr
        assert "not implemented" in result.stderr


def test_train_and_evaluate_expose_runtime_arguments() -> None:
    train_help = run("train", "--help")
    assert train_help.returncode == 0
    assert "--device" in train_help.stdout
    assert "--output-dir" in train_help.stdout

    evaluate_help = run("evaluate", "--help")
    assert evaluate_help.returncode == 0
    assert "--checkpoint" in evaluate_help.stdout


def test_malformed_yaml_is_a_concise_cli_parser_error_without_traceback() -> None:
    with TemporaryDirectory() as directory:
        config = Path(directory) / "bad.yaml"
        config.write_text("resolution: [", encoding="utf-8")
        result = run("inspect", "--config", str(config))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
