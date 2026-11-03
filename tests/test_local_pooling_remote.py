"""独立六组云端包的完整闭环、恢复及失败门槛。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_training_engine import _config, _write_sequence

from traceformer.experiments.ablation import ComponentCoreSpec, LocalPoolingSpec
from traceformer.experiments.workflows import local_pooling
from traceformer.experiments.workflows.local_pooling import LocalPoolingWorkflow
from traceformer.training.engine import _file_sha256, _serializable, run_training

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("S0", "S1", "S2", "L0", "L1", "L2")


@pytest.fixture
def packager():
    from traceformer.experiments import packaging
    return packaging


@pytest.fixture
def bundle(tmp_path, packager, request):
    data = tmp_path / "existing data"
    for split, count in (("train", 99), ("val", 24), ("test", 24)):
        folder = data / split
        folder.mkdir(parents=True)
        for index in range(count):
            _write_sequence(folder / f"{split}_{index:03d}.npz", 0)
    config = _config(data, tmp_path / "unused")
    config = replace(config, resolution=replace(config.resolution, width=16, height=16),
                     model=replace(config.model, spatial_block_size_x=8, spatial_block_size_y=8,
                                   num_time_blocks=8, token_dim=8, event_embed_dim=8,
                                   num_heads=2, num_layers=3, num_pool_queries=4),
                     dataset=replace(config.dataset, root=data),
                     training=replace(config.training, seed=37, save_training_state_interval=1))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(_serializable(asdict(config))), encoding="utf-8")
    experiment = getattr(request, "param", "local_pooling_six")
    destination = packager.build_package(experiment, tmp_path / "upload folder", template, data, data)
    return destination, data, config


def run_bundle(bundle, *flags, check=True, env_override=None, timeout=420):
    directory, data, _ = bundle
    environment = dict(os.environ, PYTHONPATH="", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment.update(env_override or {})
    result = subprocess.run([sys.executable, "-u", str(directory / "run.py"),
                             "--data-root", str(data), "--test-root", str(data), *flags],
                            cwd=directory.parent, env=environment, capture_output=True,
                            encoding="utf-8", timeout=timeout)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_six_group_package_is_self_contained_and_does_not_overwrite(bundle, packager):
    directory, data, _ = bundle
    manifest = read(directory / "package_manifest.json")
    assert manifest["contains_training_data"] is False
    assert manifest["contains_trained_checkpoint"] is False
    assert not list(directory.rglob("*.npz"))
    assert set(read(directory / "suite.json")["variants"][i]["variant"] for i in range(6)) == set(VARIANTS)
    assert "legacy_workflow.py" not in manifest["sha256"]
    assert not (directory / "workflow.py").exists()
    assert not (directory / "scripts").exists()
    assert not (directory / "train.py").exists()
    assert not (directory / "predict.py").exists()
    assert "src/traceformer/experiments/local_pooling_benchmark.py" in manifest["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        packager.build_package("local_pooling_six", directory, directory / "config.yaml", data, data)


def test_isolated_cpu_smoke_completes_all_six_and_resumes_without_retraining(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--smoke", "--device", "cpu")
    root = directory / "output/smoke"
    assert read(root / "status.json")["status"] == "complete"
    audit = read(root / "completion_audit.json")
    assert audit["variants"] == list(VARIANTS)
    assert audit["budget_enforced"] is False
    assert audit["partitions"] == {"val": 2, "test": 2}
    assert read(root / "trained_timing.json")["budget_enforced"] is False
    before = {}
    for name in VARIANTS:
        checkpoint = Path(read(root / name / "training.json")["checkpoint"])
        before[name] = checkpoint.read_bytes()
        for split in ("val", "test"):
            report = read(root / name / split / "independent_report.json")
            assert report["sequences"] == 2
            assert report["test_threshold_selection"] is False
            assert report["inference_annotations"] == "zeroed labels and object IDs"
            assert (root / name / split / "raw_independent_report.json").is_file()
    assert (directory / "output/smoke_results.zip").is_file()
    run_bundle(bundle, "--smoke", "--device", "cpu", "--resume")
    for name in VARIANTS:
        pointer = read(root / name / "training.json")
        assert Path(pointer["checkpoint"]).read_bytes() == before[name]
        assert len(list((root / name / "training").glob("attempt_*"))) == 1
    # 恢复时执行分块也属于冻结协议，不能悄悄改变数值调用顺序。
    refused = run_bundle(bundle, "--smoke", "--device", "cpu", "--resume", check=False,
                         env_override={"TRACEFORMER_LOCAL_QUERY_CHUNK": "17"})
    assert refused.returncode != 0 and "settings/source changed" in refused.stderr


@pytest.mark.parametrize("tamper", ("data", "source"))
def test_six_group_package_refuses_changed_data_or_source_before_training(bundle, tamper):
    directory, data, _ = bundle
    path = data / "train/train_000.npz" if tamper == "data" else directory / "src/traceformer/config.py"
    with path.open("ab") as stream:
        stream.write(b"\n# changed\n")
    result = run_bundle(bundle, "--smoke", "--device", "cpu", check=False)
    assert result.returncode != 0
    assert ("data identity differs" if tamper == "data" else "package file changed") in result.stderr
    assert not list((directory / "output/smoke").glob("*/training/attempt_*"))


def test_formal_workflow_refuses_cpu_and_does_not_fall_back(bundle):
    result = run_bundle(bundle, "--device", "cpu", check=False)
    assert result.returncode != 0 and "require CUDA FP32" in result.stderr
    unavailable = run_bundle(bundle, "--smoke", check=False,
                             env_override={"CUDA_VISIBLE_DEVICES": ""})
    assert unavailable.returncode != 0 and "CUDA was requested but is not available" in unavailable.stderr


@pytest.mark.parametrize("changed", ("resolution", "duration", "geometry"))
def test_formal_protocol_is_fixed_even_without_imported_baseline(bundle, changed):
    directory, data, config = bundle
    workflow = LocalPoolingWorkflow(directory)
    config = replace(config, resolution=replace(config.resolution, width=346, height=260),
                     model=replace(config.model, num_time_blocks=64, token_dim=64, event_embed_dim=64,
                                   num_heads=4, num_layers=3))
    if changed == "resolution":
        config = replace(config, resolution=replace(config.resolution, width=345))
    elif changed == "duration":
        config = replace(config, dataset=replace(config.dataset, duration_ms=7000))
    else:
        windows = config.model.cone_windows
        config = replace(config, model=replace(config.model, cone_windows=(
            replace(windows[0], spatial_radius=7), *windows[1:])))
    (directory / "config.yaml").write_text(yaml.safe_dump(_serializable(asdict(config))), encoding="utf-8")
    root = directory / "output/formal unit"
    root.mkdir(parents=True)
    args = SimpleNamespace(data_root=data, test_root=data, device="cuda", smoke=False, stage="preflight")
    with pytest.raises(ValueError, match="resolution/duration/summary cone geometry differs"):
        workflow.configure(args, root, "unit manifest")


def test_resume_refuses_orphan_worker_without_starting_duplicate(bundle):
    directory, _, _ = bundle
    root = directory / "output/smoke"
    root.mkdir(parents=True)
    # 模拟协调器已退出、原worker仍等待输入；不杀用户进程或睡眠轮询。
    script = "import sys; print('ready', flush=True); sys.stdin.readline()"
    command = [sys.executable, "-u", "-c", script, str(directory / "run.py")]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as holder:
        try:
            assert holder.stdout.readline().strip() == "ready"
            state = {"status": "running", "worker_pid": holder.pid, "command": command}
            (root / "status.json").write_text(json.dumps(state), encoding="utf-8")
            refused = run_bundle(bundle, "--smoke", "--device", "cpu", "--resume", check=False)
            assert refused.returncode != 0 and "active worker PID" in refused.stderr
            assert read(root / "status.json") == state
            assert not list(root.glob("*/training/attempt_*"))
        finally:
            holder.communicate("release\n", timeout=10)


def test_timing_records_do_not_enforce_a_budget(bundle, monkeypatch):
    directory, _, config = bundle
    workflow = LocalPoolingWorkflow(directory)
    root = directory / "output/unit"
    for name in VARIANTS:
        (root / name).mkdir(parents=True)
        (root / name / "config.yaml").write_text("unused", encoding="utf-8")
    def fake_execute(_root, _phase, command):
        destination = Path(command[command.index("--report") + 1])
        destination.write_text('{"mock":"resource report"}', encoding="utf-8")
    monkeypatch.setattr(workflow, "execute", fake_execute)
    monkeypatch.setattr(local_pooling, "load_config", lambda path: config)
    monkeypatch.setattr(local_pooling, "setup_device", lambda cfg: SimpleNamespace(metadata={"unit": True}))
    monkeypatch.setattr(workflow, "validate_benchmark_identity", lambda *args, **kwargs: None)
    workflow.benchmark_suite(root, SimpleNamespace(smoke=False), "preflight")
    assert read(root / "preflight_timing.json")["budget_enforced"] is False
    assert not (root / "preflight_budget.json").exists()
    assert not list(root.glob("*/training/attempt_*"))


@pytest.mark.parametrize("tamper", ("checkpoint", "variant", "runtime", "data"))
def test_cached_benchmark_identity_cannot_survive_drift(bundle, tamper):
    directory, _, _ = bundle
    workflow = LocalPoolingWorkflow(directory)
    spec = LocalPoolingSpec("L0")
    runtime = {"device": "cuda", "gpu_name": "unit GPU"}
    identities = {"train": {"train_094.npz": {"sha256": "a"}},
                  "val": {"val_000.npz": {"sha256": "b"}}}
    report = {"checkpoint_sha256": None, "variant": spec.to_dict(), "runtime": dict(runtime),
              "training": {"sequences": [{"file": "train_094.npz", "source_sha256": "a"}]},
              "inference": {"sequences": [{"file": "val_000.npz", "source_sha256": "b"}]}}
    workflow.validate_benchmark_identity(report, spec, None, runtime, identities)
    if tamper == "checkpoint":
        report["checkpoint_sha256"] = "wrong checkpoint"
    elif tamper == "variant":
        report["variant"]["variant"] = "L1"
    elif tamper == "runtime":
        report["runtime"]["gpu_name"] = "another GPU"
    else:
        report["training"]["sequences"][0]["source_sha256"] = "changed data"
    with pytest.raises(ValueError, match="cached benchmark"):
        workflow.validate_benchmark_identity(report, spec, None, runtime, identities)


def test_s0_reuse_preserves_original_checkpoint_contract_and_bytes(bundle, monkeypatch, tmp_path):
    directory, data, config = bundle
    workflow = LocalPoolingWorkflow(directory)
    source = directory / "baseline"
    source.mkdir()
    # 微型训练产生真实旧契约权重；此单元测试只检查复用机制，不冒充云端正式基线。
    config = replace(config, training=replace(config.training, max_train_files=2, max_val_files=2,
                                               num_threads=1))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_training(config, output_dir=source, ablation=ComponentCoreSpec("M1_C4"))
    finally:
        torch.set_num_threads(previous_threads)
    original = (source / "best.pt").read_bytes()
    identity = {"checkpoint_sha256": _file_sha256(source / "best.pt")}
    (source / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    root = tmp_path / "reuse"
    (root / "S0").mkdir(parents=True)
    called = []
    monkeypatch.setattr(workflow, "predict_variant", lambda *args, **kwargs: called.append(kwargs["splits"]))
    observed = workflow.baseline(root, config, data, SimpleNamespace(smoke=False))
    assert observed == source / "best.pt"
    assert (source / "best.pt").read_bytes() == original
    assert called == [("val",)]
    assert read(root / "S0/training.json")["ablation"] == ComponentCoreSpec("M1_C4").to_dict()
    assert read(root / "baseline_audit.json")["validation_reproduced"] is True


def test_packager_refuses_invalid_baseline_without_silent_fallback(bundle, packager, tmp_path):
    directory, data, _ = bundle
    broken = tmp_path / "broken baseline"
    broken.mkdir()
    with pytest.raises(ValueError, match="no completed checkpoint"):
        packager.build_package("local_pooling_six", tmp_path / "must not exist", directory / "config.yaml", data, data,
                               baseline_root=broken)
    assert not (tmp_path / "must not exist").exists()


def test_package_preserves_baseline_curves_and_resume_state(bundle, packager, monkeypatch, tmp_path):
    directory, data, _ = bundle
    baseline = tmp_path / "baseline artifacts"
    baseline.mkdir()
    names = (*packager.BASELINE_FILES, *packager.OPTIONAL_BASELINE_FILES)
    for name in names:
        (baseline / name).write_bytes(f"original {name}".encode())
    identity = {"source_files": {name: _file_sha256(baseline / name) for name in names}}
    monkeypatch.setattr(packager, "validate_baseline", lambda *args: (baseline, baseline, identity))
    output = packager.build_package("local_pooling_six", tmp_path / "preserved upload", directory / "config.yaml", data, data,
                                    baseline_root=baseline)
    manifest = read(output / "package_manifest.json")["sha256"]
    for name in names:
        assert (output / "baseline" / name).read_bytes() == (baseline / name).read_bytes()
        assert manifest[f"baseline/{name}"] == identity["source_files"][name]


@pytest.mark.parametrize("section", ("loss", "evaluation", "frame_refinement", "inference"))
def test_baseline_reuse_requires_identical_loss_and_selection_protocol(bundle, packager, monkeypatch, section, tmp_path):
    directory, _, _ = bundle
    config = packager.load_config(directory / "config.yaml")
    baseline = tmp_path / "protocol baseline"
    run = baseline / "training/attempt_001"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"unit checkpoint identity")
    (run / "config.yaml").write_text(yaml.safe_dump(_serializable(asdict(config))), encoding="utf-8")
    (run / "summary.json").write_text('{"device":"cuda","mixed_precision":false,"epochs":24}', encoding="utf-8")
    (run / "history.json").write_text(json.dumps([{"epoch": value} for value in range(1, 25)]), encoding="utf-8")
    (run / "metrics.json").write_text('{"threshold":0.5}', encoding="utf-8")
    digest = _file_sha256(run / "best.pt")
    (baseline / "training.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
    identities = read(directory / "dataset_manifest.json")
    (baseline / "dataset_identity.json").write_text(json.dumps(identities), encoding="utf-8")
    checkpoint = {"ablation": ComponentCoreSpec("M1_C4").to_dict(), "training_seed": 37,
                  "decision_threshold": 0.5}
    # 这里只隔离验证损失/选阈值协议；结构契约由其他模型测试覆盖。
    monkeypatch.setattr(packager.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(packager, "_validate_checkpoint_contract", lambda *args, **kwargs: None)
    assert packager.validate_baseline(baseline, config, identities) is not None
    if section == "loss":
        changed = replace(config, loss=replace(config.loss, lambda_dice=config.loss.lambda_dice + 0.1))
    elif section == "evaluation":
        changed = replace(config, evaluation=replace(config.evaluation, threshold_steps=config.evaluation.threshold_steps + 2))
    elif section == "frame_refinement":
        changed = replace(config, frame_refinement=replace(config.frame_refinement,
                                                           loss_weight=config.frame_refinement.loss_weight + 0.1))
    else:
        changed = replace(config, inference=replace(config.inference, sigmoid_threshold=0.6))
    with pytest.raises(ValueError, match=f"baseline {section} protocol differs"):
        packager.validate_baseline(baseline, changed, identities)
