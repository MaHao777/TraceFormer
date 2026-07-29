from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, replace
from importlib import util
from pathlib import Path

import pytest
import torch

from huav.config import load_config
from huav.data import NPZEventDataset
from huav.training.engine import (
    _dataset_manifest,
    _initialize_detector_weights,
    _manifest_identity_digest,
    _serializable,
    build_model,
)

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
    template = load_config(
        ROOT / "configs" / "huav_adaptive_hierarchical_joint_autodl.yaml"
    )
    assert Path(config.training.output_root) == Path(
        template.training.output_root
    ).resolve()
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


def _load_generator_module() -> object:
    script = ROOT / "scripts" / "make_joint_config.py"
    specification = util.spec_from_file_location("make_joint_config", script)
    assert specification is not None and specification.loader is not None
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_joint_config_generator_refuses_template_and_existing_output(
    tmp_path: Path,
) -> None:
    module = _load_generator_module()
    parsed = module.build_parser().parse_args(
        [
            "--template",
            "template.yaml",
            "--output",
            "output.yaml",
            "--dataset-root",
            "data",
            "--seed",
            "73",
            "--force",
        ]
    )
    assert parsed.force is True
    source_template = (
        ROOT / "configs" / "huav_adaptive_hierarchical_joint_autodl.yaml"
    )
    template = tmp_path / "template.yaml"
    template.write_bytes(source_template.read_bytes())
    original_template = template.read_bytes()
    dataset_root = ROOT / "data" / "ev-uav"

    with pytest.raises(ValueError, match="different"):
        module.generate_joint_config(
            template,
            template,
            dataset_root=dataset_root,
            seed=73,
            force=True,
        )
    assert template.read_bytes() == original_template

    output = tmp_path / "joint.yaml"
    module.generate_joint_config(
        source_template,
        output,
        dataset_root=dataset_root,
        seed=37,
    )
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        module.generate_joint_config(
            source_template,
            output,
            dataset_root=dataset_root,
            seed=73,
        )
    assert output.read_bytes() == original_output
    module.generate_joint_config(
        source_template,
        output,
        dataset_root=dataset_root,
        seed=73,
        force=True,
    )
    assert load_config(output).training.seed == 73


def test_joint_config_generator_validation_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generator_module()
    template = (
        ROOT / "configs" / "huav_adaptive_hierarchical_joint_autodl.yaml"
    )
    output = tmp_path / "joint.yaml"
    output.write_text("sentinel\n", encoding="utf-8")
    real_load_config = module.load_config
    calls = 0

    def fail_generated(path: str | Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("generated validation failed")
        return real_load_config(path)

    monkeypatch.setattr(module, "load_config", fail_generated)
    with pytest.raises(ValueError, match="generated validation failed"):
        module.generate_joint_config(
            template,
            output,
            dataset_root=ROOT / "data" / "ev-uav",
            seed=137,
            force=True,
        )

    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.iterdir()) == [output]


def test_smoke_full_validation_manifest_accepts_schema4_detector(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT
        / "configs"
        / "huav_adaptive_hierarchical_autodl_smoke.yaml"
    )
    validation_root = tmp_path / "val"
    validation_root.mkdir()
    (validation_root / "val_000.npz").write_bytes(b"first")
    (validation_root / "val_001.npz").write_bytes(b"second")
    validation_dataset = NPZEventDataset(
        validation_root,
        limit_files=config.training.max_val_files,
    )
    assert len(validation_dataset) == 2
    validation_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    base = replace(
        config,
        model=replace(config.model, adaptive_threshold_mode="none"),
        training=replace(
            config.training,
            training_stage="full",
            detector_learning_rate=None,
        ),
        loss=replace(
            config.loss,
            lambda_threshold_magnitude=0.0,
            lambda_threshold_center=0.0,
            lambda_threshold_smoothness=0.0,
        ),
    ).validate()
    checkpoint_path = tmp_path / "schema4.pt"
    torch.save(
        {
            "schema_version": 4,
            "model_state": build_model(base).state_dict(),
            "model_config": _serializable(asdict(base.model)),
            "resolution": _serializable(asdict(base.resolution)),
            "validation_manifest_sha256": validation_digest,
            "decision_threshold": 0.5,
        },
        checkpoint_path,
    )

    target = build_model(config)
    metadata = _initialize_detector_weights(
        target,
        config,
        checkpoint_path,
        validation_digest,
        device=torch.device("cpu"),
    )
    assert metadata["source_checkpoint"] == str(checkpoint_path.resolve())
