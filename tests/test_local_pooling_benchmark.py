"""验证六组GPU门槛依据原始计时，CPU微型检查不能冒充速度验收。"""

from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch
from test_training_engine import _config, _write_sequence

from traceformer.experiments import local_pooling_benchmark as benchmark
from traceformer.experiments.ablation import LocalPoolingSpec
from traceformer.experiments.local_pooling import VARIANTS


def reports_fixture():
    reports = {}
    for name in VARIANTS:
        reports[name] = {
            "variant": {"version": 2, "variant": "M1_C4"} if name == "S0" else LocalPoolingSpec(name).to_dict(),
            "smoke": False, "device": "cuda", "precision": "fp32",
            "runtime": {"device": "cuda", "gpu_name": "test GPU", "cuda_kernel_verified": True,
                        "total_memory_bytes": 1000},
            "numerical_check": {"passed": True, "parameter_gradients_checked": True,
                                "real_cuda_checked": True},
            "training": {"optimizer": "AdamW", "includes_optimizer_step": True, "sequences": [
                {"file": f"train_{index:03}.npz", "events": 100,
                 "source_sha256": sha256(f"train{index}".encode()).hexdigest(),
                 "warmup": 2, "repeats": 5, "seconds": [1.0] * 5, "median_seconds": 1.0,
                 "finite_gradients": True, "peak_memory_bytes": 400} for index in (94, 96)]},
            "inference": {"sequences": [
                {"file": f"val_{index:03}.npz", "events": 50,
                 "source_sha256": sha256(f"val{index}".encode()).hexdigest(),
                 "warmup": 1, "repeats": 3, "seconds": [0.1] * 3, "median_seconds": 0.1}
                for index in range(24)], "mean_seconds": 0.1, "p95_seconds": 0.1,
                "peak_memory_bytes": 300},
            "peak_memory_bytes": 400, "total_memory_bytes": 1000,
        }
    return reports


def test_gate_accepts_exact_limit_and_reports_ratios():
    reports = reports_fixture()
    row = reports["L2"]
    for sequence in row["training"]["sequences"]:
        sequence.update(seconds=[1.25] * 5, median_seconds=1.25, peak_memory_bytes=500)
    for sequence in row["inference"]["sequences"]:
        sequence.update(seconds=[0.125] * 3, median_seconds=0.125)
    row["inference"].update(mean_seconds=0.125, p95_seconds=0.125)
    row["peak_memory_bytes"] = 500
    result = benchmark.assess_budget(reports)
    assert result["passed"] and result["validated_cuda_budget"]
    assert result["rows"][-1]["training_ratios"]["train_094.npz"] == 1.25


@pytest.mark.parametrize("part", ["training", "inference", "memory", "device_memory"])
def test_gate_rejects_real_budget_overruns(part):
    reports = reports_fixture()
    row = reports["L1"]
    if part == "training":
        row["training"]["sequences"][0].update(seconds=[1.251] * 5, median_seconds=1.251)
    elif part == "inference":
        for sequence in row["inference"]["sequences"]:
            sequence.update(seconds=[0.126] * 3, median_seconds=0.126)
        row["inference"].update(mean_seconds=0.126, p95_seconds=0.126)
    else:
        peak = 501 if part == "memory" else 901
        row["training"]["sequences"][0]["peak_memory_bytes"] = peak
        row["peak_memory_bytes"] = peak
    result = benchmark.assess_budget(reports)
    assert not result["passed"] and not result["validated_cuda_budget"]
    assert any(failure.startswith("L1:") for failure in result["failures"])


@pytest.mark.parametrize("change", [
    "missing_variant", "wrong_variant", "wrong_runtime", "bf16", "cpu", "source",
    "missing_sequence", "warmup", "repeats", "false_gradient", "optimizer", "no_optimizer_step",
    "no_real_cuda", "no_gradient_check", "median", "mean", "p95", "peak", "negative_peak",
    "total_memory", "nonfinite", "no_samples", "zero_cuda_peak", "string_smoke", "empty_source",
])
def test_gate_refuses_incomplete_or_inconsistent_evidence(change):
    reports = reports_fixture()
    row = reports["L0"]
    if change == "missing_variant":
        reports.pop("S2")
    elif change == "wrong_variant":
        row["variant"]["variant"] = "S1"
    elif change == "wrong_runtime":
        row["runtime"]["gpu_name"] = "another GPU"
    elif change == "bf16":
        row["precision"] = "bf16"
    elif change == "cpu":
        row["device"] = "cpu"
    elif change == "source":
        row["inference"]["sequences"][0]["source_sha256"] = "different input"
    elif change == "missing_sequence":
        row["training"]["sequences"].pop()
    elif change == "warmup":
        row["training"]["sequences"][0]["warmup"] = 0
    elif change == "repeats":
        row["training"]["sequences"][0]["seconds"].pop()
    elif change == "false_gradient":
        row["training"]["sequences"][0]["finite_gradients"] = False
    elif change == "optimizer":
        row["training"]["optimizer"] = "SGD"
    elif change == "no_optimizer_step":
        row["training"]["includes_optimizer_step"] = False
    elif change == "no_real_cuda":
        row["numerical_check"]["real_cuda_checked"] = False
    elif change == "no_gradient_check":
        row["numerical_check"]["parameter_gradients_checked"] = False
    elif change == "median":
        row["training"]["sequences"][0]["median_seconds"] = 0.1
    elif change in ("mean", "p95"):
        row["inference"][f"{change}_seconds"] = 0.01
    elif change == "peak":
        row["peak_memory_bytes"] = 10
    elif change == "negative_peak":
        row["training"]["sequences"][0]["peak_memory_bytes"] = -1
    elif change == "total_memory":
        row["total_memory_bytes"] = 100000
    elif change == "nonfinite":
        row["training"]["sequences"][0]["seconds"][0] = float("nan")
    elif change == "no_samples":
        row["inference"]["sequences"][0].update(seconds=[], repeats=0)
    elif change == "zero_cuda_peak":
        row["peak_memory_bytes"] = 0
        row["inference"]["peak_memory_bytes"] = 0
        for sequence in row["training"]["sequences"]:
            sequence["peak_memory_bytes"] = 0
    elif change == "string_smoke":
        for item in reports.values():
            item["smoke"] = "false"
    elif change == "empty_source":
        for item in reports.values():
            item["training"]["sequences"][0]["source_sha256"] = None
    with pytest.raises(ValueError):
        benchmark.assess_budget(reports)


def test_cpu_smoke_does_not_claim_gpu_budget_validation():
    reports = reports_fixture()
    for row in reports.values():
        row.update(smoke=True, device="cpu", peak_memory_bytes=0, total_memory_bytes=0)
        row["runtime"] = {"device": "cpu", "cuda_kernel_verified": False}
        row["numerical_check"]["real_cuda_checked"] = False
        row["inference"]["peak_memory_bytes"] = 0
        for sequence in row["training"]["sequences"]:
            sequence["peak_memory_bytes"] = 0
    result = benchmark.assess_budget(reports)
    assert result["passed"] and result["smoke"]
    assert result["validated_cuda_budget"] is False


def test_real_cpu_smoke_times_training_and_restores_weights(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    try:
        data = tmp_path / "data"
        for split in ("train", "val"):
            folder = data / split
            folder.mkdir(parents=True)
            for index in range(2):
                _write_sequence(folder / f"{split}_{index:03}.npz", index)
        base = _config(data, tmp_path / "unused")
        config = replace(base, model=replace(base.model, num_layers=3),
                         training=replace(base.training, max_train_files=2, max_val_files=2))
        captured = {}
        original = benchmark.build_model

        def capture(*args, **kwargs):
            model = original(*args, **kwargs)
            captured["model"] = model
            captured["weights"] = copy.deepcopy(model.state_dict())
            return model

        monkeypatch.setattr(benchmark, "build_model", capture)
        report = benchmark.run_benchmark(config, LocalPoolingSpec("L1"), smoke=True)
        assert report["numerical_check"]["parameter_gradients_checked"]
        assert report["training"]["includes_optimizer_step"]
        assert len(report["training"]["sequences"]) == 2
        assert len(report["inference"]["sequences"]) == 2
        assert report["checkpoint_sha256"] is None
        for name, value in captured["model"].state_dict().items():
            assert torch.equal(value, captured["weights"][name])
        with pytest.raises(ValueError, match="real CUDA"):
            benchmark.run_benchmark(config, LocalPoolingSpec("L1"))
        with pytest.raises(ValueError, match="FP32"):
            benchmark.run_benchmark(replace(config, training=replace(config.training, mixed_precision=True)),
                                    LocalPoolingSpec("L1"), smoke=True)
    finally:
        torch.set_num_threads(previous_threads)
