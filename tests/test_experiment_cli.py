"""统一入口的冻结运行、状态读取和历史源码隔离。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_mean_past_remote

from traceformer.cli import experiment
from traceformer.experiments.archive import archive_inventory, run_archive
from traceformer.experiments.runtime import ExperimentRuntime

bundle = test_mean_past_remote.bundle
packager = test_mean_past_remote.packager


def test_list_contains_only_two_maintained_workflows(capsys):
    experiment.main(["list"])
    result = json.loads(capsys.readouterr().out)
    assert set(result["maintained"]) == {"mean_past_gated", "local_pooling_six"}
    assert {"paper_ablation", "component_core", "original_readout"} <= result["archived"].keys()
    for row in result["archived"].values():
        if row["status"] == "runnable":
            assert row["validated"] and not row["missing"] and not row["missing_dependencies"]


def test_repository_run_requires_an_output():
    with pytest.raises(SystemExit) as error:
        experiment.main(["run", "mean_past_gated"])
    assert error.value.code == 2


def test_repository_cli_freezes_preflight_and_resumes(bundle, tmp_path, capsys):
    directory, data, test = bundle
    output = tmp_path / "new repository run"
    flags = ["run", "mean_past_gated", "--output", str(output),
             "--config", str(directory / "config.yaml"), "--data-root", str(data),
             "--test-root", str(test), "--smoke", "--device", "cpu", "--stage", "preflight"]
    experiment.main(flags)
    assert (output / "source/package_manifest.json").is_file()
    assert json.loads((output / "smoke/status.json").read_text())["status"] == "preflight_complete"
    experiment.main(["run", "mean_past_gated", "--output", str(output),
                     "--smoke", "--resume", "--stage", "preflight"])
    assert not list((output / "smoke").rglob("best.pt"))
    experiment.main(["status", "--output", str(output), "--smoke"])
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "preflight_complete" and state["results"] is None


def test_status_reports_variant_checkpoint_and_failure(tmp_path, capsys):
    root = tmp_path / "full"
    (root / "S0").mkdir(parents=True)
    (root / "status.json").write_text('{"status":"failed","phase":"predict_val","error":"unit failure"}')
    (root / "S0/training.json").write_text('{"checkpoint":"original.pt","reused":true}')
    experiment.main(["status", "--output", str(tmp_path)])
    state = json.loads(capsys.readouterr().out)
    assert state["phase"] == "predict_val" and state["error"] == "unit failure"
    assert state["training"]["S0"]["checkpoint"] == "original.pt"


@pytest.mark.parametrize("relative", [True, False])
def test_repository_launch_resolves_data_before_changing_working_directory(tmp_path, monkeypatch, relative):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(output=tmp_path / "new run", resume=False, experiment="mean_past_gated",
                           data_root=Path("data") if relative else None,
                           test_root=Path("test data") if relative else None,
                           stage="preflight", smoke=True, device="cpu", precision=None)
    monkeypatch.setattr(experiment, "build_from_arguments", lambda _args, target: target.mkdir(parents=True))
    commands = []
    monkeypatch.setattr(experiment.subprocess, "run", lambda command, **kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    assert experiment.run_repository(args) == 0
    command = commands[0]
    expected = tmp_path / "data" if relative else experiment.REPO / "data/ev-uav"
    assert command[command.index("--data-root") + 1] == str(expected.resolve())


@pytest.fixture
def archived(tmp_path):
    # 构造独立的历史包，证明子进程不会导入已安装的当前 TraceFormer。
    source = tmp_path / "snapshot"
    package = source / "src/traceformer"
    (package / "training").mkdir(parents=True)
    (package / "__init__.py").write_text("historical = True\n")
    (package / "training/engine.py").write_text("SCHEMA = 7\n")
    script = source / "scripts/probe.py"
    script.parent.mkdir()
    script.write_text("import traceformer, yaml\nassert traceformer.historical\nprint('historical runtime')\n")
    catalog = {"schema": 1, "runtimes": {"old": {"source": "snapshot", "sha256": {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*.py")}}}, "experiments": {"probe": {
            "runtime": "old", "entry": "snapshot/scripts/probe.py", "frozen_entry": "scripts/probe.py",
            "dependencies": [], "validated": True}}}
    directory = tmp_path / "archive/experiments"
    directory.mkdir(parents=True)
    (directory / "catalog.json").write_text(json.dumps(catalog))
    return tmp_path, package


def test_archive_launches_matching_source_and_rejects_changes(archived):
    repo, package = archived
    assert archive_inventory(repo)["probe"]["status"] == "runnable"
    assert run_archive(repo, "probe") == 0
    (package / "training/engine.py").write_text("SCHEMA = 8\n")
    with pytest.raises(ValueError, match="historical source identity differs"):
        run_archive(repo, "probe")


def test_partial_archive_cannot_fall_back_to_installed_package(archived):
    repo, package = archived
    (package / "__init__.py").unlink()
    assert archive_inventory(repo)["probe"]["status"] == "incomplete"
    with pytest.raises(FileNotFoundError, match="matching historical source is missing"):
        run_archive(repo, "probe", arguments=["--help"])


def test_completed_subprocess_does_not_leave_a_live_worker_identity(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    runtime = ExperimentRuntime(tmp_path)
    runtime.execute(root, "probe", [sys.executable, "-c", "print('done')"])
    state = json.loads((root / "status.json").read_text())
    assert state["status"] == "phase_complete" and state["worker_pid"] is None
    runtime.guard_active_workers(root)


def test_reused_pid_is_ignored_before_inspecting_an_unrelated_process(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    command = [sys.executable, str(tmp_path / "run.py")]
    state = {"status": "running", "worker_pid": 123, "worker_created_at": 10, "command": command}
    (root / "status.json").write_text(json.dumps(state))
    def forbidden():
        raise AssertionError("must not inspect an unrelated reused PID")
    process = SimpleNamespace(create_time=lambda: 20, cmdline=forbidden)
    monkeypatch.setattr("traceformer.experiments.runtime.psutil.Process", lambda pid: process)
    ExperimentRuntime(tmp_path).guard_active_workers(root)
    assert json.loads((root / "status.json").read_text()) == state
