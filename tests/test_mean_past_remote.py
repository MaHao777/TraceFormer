"""联合结构的精确对照及可独立上传包的训练、预测、恢复闭环。"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import test_paper_ablation
import torch
import yaml
from test_training_engine import _config, _write_sequence

from traceformer.experiments.ablation import ComponentCoreSpec, parse_ablation
from traceformer.experiments.component_core import apply_component_core
from traceformer.training.engine import (
    _serializable,
    _validate_checkpoint_contract,
    build_model,
    run_training,
)

REPO = Path(__file__).resolve().parents[1]
small_config = test_paper_ablation.small_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def joint_config(config):
    return replace(config, model=replace(config.model, num_layers=3, num_pool_queries=4),
                   training=replace(config.training, save_training_state_interval=1))


def test_joint_matches_exact_sequential_m1_c4_replacement(small_config):
    config = joint_config(small_config)
    torch.manual_seed(37)
    base = build_model(config)
    rng = torch.get_rng_state()
    expected = copy.deepcopy(base)
    apply_component_core(expected, ComponentCoreSpec("M1"))
    apply_component_core(expected, ComponentCoreSpec("C4"))
    torch.manual_seed(37)
    actual = build_model(config, ablation=ComponentCoreSpec("M1_C4"))
    assert torch.equal(torch.get_rng_state(), rng)
    assert parse_ablation(actual.ablation_spec) == ComponentCoreSpec("M1_C4")
    assert not any("query" in name or "pool_queries" in name for name in actual.state_dict())
    assert actual.tokenizer.token_projection[0].in_features == 2 * config.model.event_embed_dim + 7
    assert len(actual.encoder.layers) == 3
    for layer in actual.encoder.layers:
        assert layer.attention.directions == (-1,)
        assert isinstance(layer.attention.pool_weight, torch.nn.Linear)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key]), key
    events = torch.tensor([[1.0, 1.0, t, p] for t in (0.01, 0.1, 0.3, 0.7)
                           for p in (0.0, 1.0)])
    actual.eval()
    expected.eval()
    observed, reference = actual(events), expected(events)
    for key in observed:
        assert torch.equal(observed[key], reference[key]), key
    observed["event_logits"].square().mean().backward()
    reference["event_logits"].square().mean().backward()
    for (name, left), (_, right) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        assert torch.equal(left.grad, right.grad), name
    with pytest.raises(ValueError, match="three encoder"):
        build_model(replace(config, model=replace(config.model, num_layers=6)),
                    ablation=ComponentCoreSpec("M1_C4"))


def test_joint_trains_resumes_and_rejects_other_contracts(small_config, tmp_path):
    config = joint_config(small_config)
    spec = ComponentCoreSpec("M1_C4")
    run = tmp_path / "first"
    run_training(config, output_dir=run, ablation=spec)
    saved = torch.load(run / "best.pt", weights_only=True)
    for other in (None, ComponentCoreSpec("M1"), ComponentCoreSpec("C4")):
        with pytest.raises(ValueError, match="ablation contract"):
            _validate_checkpoint_contract(saved, config, ablation=other)
    resumed = tmp_path / "resumed"
    run_training(config, output_dir=resumed, ablation=spec, resume_state_path=run / "training_state.pt")
    restored = torch.load(resumed / "best.pt", weights_only=True)
    assert all(torch.equal(value, restored["model_state"][key]) for key, value in saved["model_state"].items())


@pytest.fixture
def packager():
    from traceformer.experiments import packaging
    return packaging


@pytest.fixture
def bundle(tmp_path, packager):
    # 完整文件数量但每文件仅4事件，模拟远端已有同身份NPZ。
    data, test = tmp_path / "existing data", tmp_path / "existing test"
    for split, parent, count in (("train", data, 99), ("val", data, 24), ("test", test, 24)):
        folder = parent / split
        folder.mkdir(parents=True)
        for index in range(count):
            _write_sequence(folder / f"{split}_{index:03}.npz", 0)
    config = joint_config(_config(data, tmp_path / "unused"))
    config = replace(config, dataset=replace(config.dataset, root=data))
    config_file = tmp_path / "template.yaml"
    config_file.write_text(yaml.safe_dump(_serializable(asdict(config))), encoding="utf-8")
    destination = packager.build_package("mean_past_gated", tmp_path / "upload folder", config_file, data, test)
    return destination, data, test


def run_bundle(bundle, *flags, check=True, env_override=None):
    directory, data, test = bundle
    environment = dict(os.environ, PYTHONPATH="", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.update(env_override or {})
    result = subprocess.run([sys.executable, "-u", str(directory / "run.py"),
                             "--data-root", str(data), "--test-root", str(test), *flags],
                            cwd=directory.parent, env=environment, capture_output=True,
                            encoding="utf-8", timeout=180)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_bundle_is_self_contained_and_does_not_overwrite(bundle, packager):
    directory, data, test = bundle
    manifest = json.loads((directory / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_training_data"] is False
    assert manifest["contains_trained_checkpoint"] is False
    assert not list(directory.rglob("*.npz"))
    assert "torch" not in [line.strip().split(">=")[0] for line in (directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
                           if line.strip() and not line.startswith("#")]
    assert "src/traceformer/models/spatiotemporal_cone.py" in manifest["sha256"]
    assert "src/traceformer/training/engine.py" in manifest["sha256"]
    assert str(REPO) not in (directory / "config.yaml").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        packager.build_package("mean_past_gated", directory, directory / "config.yaml", data, test)


def test_isolated_bundle_smoke_runs_all_stages_and_reuses_completed_training(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--smoke", "--device", "cpu")
    root = directory / "output/smoke"
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    assert read("status.json")["status"] == "complete"
    assert read("completion_audit.json")["partitions"] == {"val": 2, "test": 2}
    assert read("preflight.json")["numerical_check"]["real_cuda_checked"] is False
    assert read("preflight.json")["structure"]["queries"] == 0
    checkpoint = Path(read("training.json")["checkpoint"])
    before = checkpoint.read_bytes()
    assert (directory / "output/smoke_results.zip").is_file()
    for split in ("val", "test"):
        assert len(list((root / split / "probabilities").glob("*.pt"))) == 2
        assert len(list((root / split / "predictions").glob("*.npz"))) == 2
        report = read(f"{split}/independent_report.json")
        assert report["inference_annotations"] == "zeroed labels and object IDs"
        assert report["test_threshold_selection"] is False
        assert report["ablation"] == ComponentCoreSpec("M1_C4").to_dict()
    run_bundle(bundle, "--smoke", "--device", "cpu", "--resume")
    assert len(list((root / "training").glob("attempt_*"))) == 1
    assert checkpoint.read_bytes() == before


def test_bundle_recovers_completed_epoch_state_in_new_attempt(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--stage", "train", "--smoke", "--device", "cpu")
    root = directory / "output/smoke"
    prior = root / "training/attempt_001"
    before = (prior / "best.pt").read_bytes()
    # 模拟最后epoch已落盘、summary尚未落盘时中断；仅移除本测试生成的两个标记。
    (root / "training.json").unlink()
    (prior / "summary.json").unlink()
    (root / "status.json").write_text('{"status":"failed","error":"simulated interruption"}')
    refused = run_bundle(bundle, "--smoke", "--device", "cpu", check=False)
    assert refused.returncode != 0 and "--resume" in refused.stderr
    run_bundle(bundle, "--smoke", "--device", "cpu", "--resume")
    current = root / "training/attempt_002"
    assert (prior / "best.pt").read_bytes() == before
    first = torch.load(prior / "best.pt", weights_only=True)
    second = torch.load(current / "best.pt", weights_only=True)
    assert all(torch.equal(value, second["model_state"][key]) for key, value in first["model_state"].items())
    summary = json.loads((current / "summary.json").read_text(encoding="utf-8"))
    assert Path(summary["resumed_from_training_state"]) == prior / "training_state.pt"


@pytest.mark.parametrize("tamper", ("data", "source"))
def test_bundle_refuses_changed_data_or_code_before_training(bundle, tamper):
    directory, data, _ = bundle
    path = data / "train/train_000.npz" if tamper == "data" else directory / "src/traceformer/config.py"
    with path.open("ab") as stream:
        stream.write(b"\n# tampered\n")
    result = run_bundle(bundle, "--stage", "preflight", "--smoke", "--device", "cpu", check=False)
    assert result.returncode != 0
    assert ("data identity differs" if tamper == "data" else "package file changed") in result.stderr
    assert not list((directory / "output/smoke/training").glob("attempt_*"))


def test_default_cuda_never_silently_falls_back_to_cpu(bundle):
    result = run_bundle(bundle, "--stage", "preflight", "--smoke", check=False,
                        env_override={"CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0
    assert "CUDA was requested but is not available" in result.stderr


def test_bundle_refuses_duplicate_coordinator(bundle):
    directory = bundle[0]
    root = directory / "output/smoke"
    root.mkdir(parents=True)
    # 独立进程持有同一文件锁，第二个入口必须在训练前退出。
    script = """import os,sys
from pathlib import Path
p=Path(sys.argv[1]); f=p.open('a+b'); f.write(b'0'); f.flush(); f.seek(0)
if os.name=='nt':
 import msvcrt; msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
else:
 import fcntl; fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
print('ready',flush=True)
sys.stdin.readline()
"""
    with subprocess.Popen([sys.executable, "-u", "-c", script, str(root / "run.lock")],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as holder:
        try:
            assert holder.stdout.readline().strip() == "ready"
            result = run_bundle(bundle, "--smoke", "--device", "cpu", check=False)
            assert result.returncode != 0
            assert not list((root / "training").glob("attempt_*"))
        finally:
            holder.communicate("release\n", timeout=10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_real_cuda_bundle_smoke(bundle):
    run_bundle(bundle, "--smoke")
    directory = bundle[0]
    report = json.loads((directory / "output/smoke/preflight.json").read_text(encoding="utf-8"))
    assert report["numerical_check"]["real_cuda_checked"] is True
    assert report["runtime"]["cuda_kernel_verified"] is True
