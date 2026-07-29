from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_editable_install_and_console_scripts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pip install -e ." in readme
    assert 'python -m pip install -e ".[dev]"' in readme
    assert "huav-inspect --help" in readme
    assert "python -m unittest discover -s tests -v" in readme
    assert "python -m pytest" in readme


def test_root_dataset_paths_are_ignored_but_source_data_package_is_addable() -> None:
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/example.npz"], cwd=ROOT, check=False)
    assert ignored.returncode == 0
    source = subprocess.run(
        ["git", "check-ignore", "-q", "src/huav/data/__init__.py"], cwd=ROOT, check=False,
    )
    assert source.returncode == 1
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/huav/data/__init__.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert tracked.returncode == 0
    assert tracked.stdout.strip().replace("\\", "/") == "src/huav/data/__init__.py"


def test_autodl_launcher_preflights_gpu_and_uses_unique_run_directories() -> None:
    launcher = ROOT / "scripts" / "autodl_train.sh"
    assert launcher.is_file()
    contents = launcher.read_text(encoding="utf-8")
    assert "huav.cli.check_gpu" in contents
    assert "huav_score_optimized_autodl_smoke.yaml" in contents
    assert "huav_score_optimized_autodl.yaml" in contents
    assert 'date +"%Y%m%d_%H%M%S"' in contents
    assert "--device cuda" in contents
    assert "tee" in contents


def test_largest_sequence_check_runs_forward_loss_backward_and_reports_memory() -> None:
    checker = ROOT / "scripts" / "check_largest_sequence.py"
    assert checker.is_file()
    contents = checker.read_text(encoding="utf-8")
    assert "train_094.npz" in contents
    assert "model(events, batch_indices)" in contents
    assert "_criterion(config)" in contents
    assert 'full_logits=output["event_logits"]' in contents
    assert "full_coordinates=events[:, :2]" in contents
    assert 'losses["total"].backward()' in contents
    assert '"losses"' in contents
    assert "max_memory_allocated_bytes" in contents


def test_adaptive_largest_sequence_gate_uses_production_training_path() -> None:
    checker = ROOT / "scripts" / "check_largest_sequence.py"
    contents = checker.read_text(encoding="utf-8")
    assert "huav_adaptive_hierarchical_autodl.yaml" in contents
    assert "--initialize-detector" in contents
    assert "_initialize_detector_weights" in contents
    assert "_criterion(config)" in contents
    assert "_threshold_loss_inputs(output)" in contents
    assert "_dense_sequence_loss_weight(" in contents
    assert '_configure_training_stage(model, "calibrator")' in contents
    assert '"source_checkpoint_sha256"' in contents
    assert '"neighbor_count"' in contents
    assert '"all_event_logits_count"' in contents
    assert '"max_memory_reserved_bytes"' in contents
    assert '"finite_gradients"' in contents


def test_adaptive_largest_sequence_gate_parser_and_gradient_summary() -> None:
    script_path = ROOT / "scripts" / "check_largest_sequence.py"
    spec = importlib.util.spec_from_file_location(
        "check_largest_sequence", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module.build_parser()
    defaults = parser.parse_args(["--initialize-detector", "base.pt"])
    assert defaults.config.endswith(
        "huav_adaptive_hierarchical_autodl.yaml"
    )
    assert defaults.sequence == "train_094.npz"
    assert defaults.device == "cuda"
    assert defaults.initialize_detector == "base.pt"

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    model[1].weight.grad = torch.ones_like(model[1].weight)
    model[1].bias.grad = torch.full_like(model[1].bias, float("nan"))
    summary = module._gradient_summary(model)
    assert summary["trainable_parameter_count"] == 3
    assert summary["gradient_parameter_count"] == 3
    assert summary["finite_gradients"] is False


def test_adaptive_threshold_docs_define_schema_lifecycle_and_full_experiment() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    guide = (
        ROOT / "docs" / "adaptive_threshold_training.md"
    ).read_text(encoding="utf-8")

    for contents in (readme, agents):
        assert "schema `5`" in contents
        assert "schema `4`" in contents
        assert "schema `3`" in contents
    assert "check_largest_sequence.py" in agents
    assert "train_094.npz" in agents
    assert "--initialize-detector" in agents
    assert "--reoptimize-threshold" in guide
    assert "huav_adaptive_global_autodl.yaml" in guide
    assert "huav_adaptive_local_autodl.yaml" in guide
    assert "huav_adaptive_local_neighbor_autodl.yaml" in guide
    assert "huav_adaptive_hierarchical_autodl.yaml" in guide
    assert "--resume" in guide
    assert "37" in guide and "73" in guide and "137" in guide
    assert "competition.score" in guide
    assert "python -m huav.cli.evaluate" in guide
    assert "python -m huav.cli.predict" in guide
    assert "submission.zip" in guide
    assert "np.array_equal(rows[:, 2]" in guide
    assert "zipfile.ZipFile" in guide
    assert "total_rows == 1_424_330" in guide
