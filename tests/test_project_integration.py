from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_editable_install_and_console_scripts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pip install -e ." in readme
    assert 'python -m pip install -e ".[dev]"' in readme
    assert "traceformer-train --help" in readme
    assert "python -m unittest discover -s tests -v" in readme
    assert "python -m pytest" in readme


def test_root_dataset_paths_are_ignored_but_source_data_package_is_addable() -> None:
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/example.npz"], cwd=ROOT, check=False)
    assert ignored.returncode == 0
    source = subprocess.run(
        ["git", "check-ignore", "-q", "src/traceformer/data/__init__.py"], cwd=ROOT, check=False,
    )
    assert source.returncode == 1
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/traceformer/data/__init__.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert tracked.returncode == 0
    assert tracked.stdout.strip().replace("\\", "/") == "src/traceformer/data/__init__.py"


def test_autodl_launcher_preflights_gpu_and_uses_unique_run_directories() -> None:
    launcher = ROOT / "scripts" / "autodl_train.sh"
    assert launcher.is_file()
    contents = launcher.read_text(encoding="utf-8")
    assert "traceformer.cli.check_gpu" in contents
    assert "traceformer_score_optimized_autodl_smoke.yaml" in contents
    assert "traceformer_score_optimized_autodl.yaml" in contents
    assert 'date +"%Y%m%d_%H%M%S"' in contents
    assert 'RUN_ROOT="$PROJECT_ROOT/smoke_outputs"' in contents
    assert 'RUN_ROOT="$PROJECT_ROOT/outputs"' in contents
    assert "--device cuda" in contents
    assert '--initialize-refinement "$INITIAL_CHECKPOINT"' in contents
    assert "tee" in contents


def test_generated_smoke_outputs_are_ignored() -> None:
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/smoke_outputs/" in ignore_rules


def test_largest_sequence_check_runs_forward_loss_backward_and_reports_memory() -> None:
    checker = ROOT / "scripts" / "check_largest_sequence.py"
    assert checker.is_file()
    contents = checker.read_text(encoding="utf-8")
    assert "train_094.npz" in contents
    assert "output = model(events, batch_indices)" in contents
    assert "lambda_object_frame=config.loss.lambda_object_frame" in contents
    assert "object_frame_logit_margin=config.loss.object_frame_logit_margin" in contents
    assert "lambda_false_alarm=config.loss.lambda_false_alarm" in contents
    assert "object_ids=object_ids[query_indices]" in contents
    assert 'full_logits=output["event_logits"]' in contents
    assert "full_coordinates=events[:, :2]" in contents
    assert "combined_total.backward()" in contents
    assert "config.training.stage" in contents
    assert '"focal"' in contents
    assert '"object_frame"' in contents
    assert '"false_alarm"' in contents
    assert "max_memory_allocated" in contents
    assert "pool_input_count" in contents
