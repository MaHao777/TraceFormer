from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from huav.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", (37, 73, 137))
def test_joint_config_generator_writes_absolute_dataset_root_and_seed(
    tmp_path: Path,
    seed: int,
) -> None:
    output = tmp_path / "outputs" / f"joint_seed{seed}.yaml"
    dataset_root = ROOT / "data" / "ev-uav"
    completed = subprocess.run(
        (
            sys.executable,
            "scripts/make_joint_config.py",
            "--template",
            "configs/huav_adaptive_hierarchical_joint_autodl.yaml",
            "--output",
            str(output),
            "--dataset-root",
            str(dataset_root),
            "--seed",
            str(seed),
        ),
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    config = load_config(output)
    assert Path(config.dataset.root) == dataset_root.resolve()
    assert config.training.seed == seed
    assert config.training.training_stage == "joint"
    assert config.model.adaptive_threshold_mode == "hierarchical"
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.training.detector_learning_rate == pytest.approx(3e-5)


def test_joint_training_guide_uses_validated_generator_not_text_substitution() -> None:
    guide = (
        ROOT / "docs" / "adaptive_threshold_training.md"
    ).read_text(encoding="utf-8")

    assert "python scripts/make_joint_config.py" in guide
    assert '--dataset-root "$REPO_ROOT/data/ev-uav"' in guide
    assert 'sed -i "s|root: ../data/ev-uav|' not in guide
    assert 'sed -i "s/^  seed: 37$/' not in guide
