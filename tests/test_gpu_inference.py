"""GPU 后端的 CPU 兼容性、隔离 simulator 和实际设备验收。"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from traceformer.postprocessing.gpu import pack_forest


def test_backend_validation_and_no_cuda_dependency(tmp_path):
    from traceformer.gpu.runtime import validate_backends
    from traceformer.postprocessing.learning import load_event_models

    validate_backends()
    assert load_event_models(tmp_path, names=(), backend="cuda") == {}
    with pytest.raises(ValueError, match="prediction-backend cuda"):
        validate_backends(feature_backend="cuda")


def test_cli_backend_defaults():
    import argparse

    from traceformer.gpu.runtime import add_backend_arguments

    parser = argparse.ArgumentParser()
    add_backend_arguments(parser, trees=True)
    assert vars(parser.parse_args([])) == {"prediction_backend": "cpu", "feature_backend": "cpu"}
    assert vars(
        parser.parse_args(["--prediction-backend", "cuda", "--feature-backend", "cuda"])
    ) == {"prediction_backend": "cuda", "feature_backend": "cuda"}


def test_cuda_simulator_kernels():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("CUDA simulator requires numba")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/gpu_kernel_cases.py", "-q"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_comparison_rejects_nonfinite():
    import numpy as np

    from scripts.check_gpu_inference import difference

    for value in (np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            difference([value], [value])
    assert (
        difference(np.array([], dtype=np.float32), np.array([], dtype=np.float32))["different"] == 0
    )


def test_reject_unsupported_forest():
    with pytest.raises(ValueError, match="binary"):
        pack_forest({"objective": "regression"})
    with pytest.raises(ValueError, match="average_output"):
        pack_forest({"objective": "binary sigmoid:1", "average_output": True})
    with pytest.raises(ValueError, match="linear leaves"):
        pack_forest({"objective": "binary sigmoid:1", "linear_tree": True})
    with pytest.raises(ValueError, match="sigmoid"):
        pack_forest({"objective": "binary sigmoid:nan"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
def test_native_prefix_attention_cuda_matches_cpu_and_has_finite_gradients():
    from copy import deepcopy

    from traceformer.config import DEFAULT_CONE_WINDOWS
    from traceformer.models.prefix_cone import PrefixConeAttention

    # 新路径没有独立邻居后端，直接比较 PyTorch CPU/CUDA 前向与反向。
    torch.manual_seed(37)
    cpu = PrefixConeAttention(16, 4, DEFAULT_CONE_WINDOWS, 0.0).eval()
    gpu = deepcopy(cpu).cuda()
    values = torch.randn(24, 16, requires_grad=True)
    cuda_values = values.detach().cuda().requires_grad_()
    coords = torch.randint(0, 5, (24, 3))
    batches = torch.arange(24) % 2
    expected = cpu(values, coords, batches).features
    actual = gpu(cuda_values, coords.cuda(), batches.cuda()).features
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(cuda_values.grad.cpu(), values.grad, atol=1e-4, rtol=1e-4)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters())
