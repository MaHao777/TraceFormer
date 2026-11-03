"""用独立逐步公式核对 Windows 推理算子，不依赖 CUDA 扩展。"""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def ops(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "remote/external_qualitative/tools"
    monkeypatch.syspath_prepend(str(tools))
    return importlib.import_module("windows_ops")


def test_voxels_keep_first_occurrence_and_event_inverse(ops):
    coords = np.array([[0, 8, 2, 1], [0, 1, 9, 2], [0, 8, 2, 1],
                       [1, 8, 2, 1], [0, 1, 9, 2]])
    features = torch.tensor([[1., 4.], [2., 6.], [3., 8.], [5., 9.], [4., 2.]])
    unique, inverse, pooled = ops.voxel_mean(coords, features)
    assert unique.tolist() == coords[[0, 1, 3]].tolist()
    assert inverse.tolist() == [0, 1, 0, 2, 1]
    torch.testing.assert_close(pooled, torch.tensor([[2., 6.], [3., 4.], [5., 9.]]))
    assert np.array_equal(unique[inverse].numpy(), coords)


@pytest.mark.parametrize("reduction", ["mean", "max"])
def test_segment_reduction_matches_independent_groups(ops, reduction):
    values = torch.tensor([[1., -1.], [3., -5.], [2., 4.], [-1., 6.], [9., 7.]])
    pointer = torch.tensor([0, 2, 3, 5])
    expected = torch.stack([getattr(part, reduction)(dim=0).values if reduction == "max"
                            else part.mean(dim=0) for part in values.split([2, 1, 2])])
    torch.testing.assert_close(ops.segment_csr(values, pointer, reduce=reduction), expected)


@pytest.mark.parametrize("groups", [1, 2])
def test_parallel_scan_matches_sequential_state_recurrence(ops, groups):
    torch.manual_seed(37)
    batch, dim, state, length = 2, 4, 3, 517
    u, delta = torch.randn(batch, dim, length), torch.randn(batch, dim, length)
    a = -torch.rand(dim, state)
    b, c = torch.randn(batch, groups, state, length), torch.randn(batch, groups, state, length)
    d, z, bias = torch.randn(dim), torch.randn_like(u), torch.randn(dim)
    steps = F.softplus(delta + bias[None, :, None])
    bb, cc = b.repeat_interleave(dim // groups, dim=1), c.repeat_interleave(dim // groups, dim=1)
    hidden = torch.zeros(batch, dim, state)
    expected = []
    # 独立的串行状态递推用于验证分块边界、分组和门控。
    for t in range(length):
        step = steps[:, :, t, None]
        hidden = torch.exp(step * a) * hidden + step * bb[:, :, :, t] * u[:, :, t, None]
        expected.append((hidden * cc[:, :, :, t]).sum(-1) + d * u[:, :, t])
    expected = torch.stack(expected, dim=-1) * F.silu(z)
    actual, last = ops.selective_scan(u, delta, a, b, c, d, z, bias, True, True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(last, hidden, rtol=2e-5, atol=2e-5)


def test_scan_preserves_tiny_states_over_long_context(ops):
    length = 4097
    u, delta = torch.ones(1, 2, length), torch.full((1, 2, length), 0.001)
    a, b, c = -torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3)
    actual, last = ops.selective_scan(u, delta, a, b, c, return_last_state=True)
    q = torch.exp(torch.tensor(-0.001))
    expected = 0.001 * (1 - q ** length) / (1 - q)
    torch.testing.assert_close(last, torch.full((1, 2, 3), expected), rtol=2e-4, atol=2e-4)
    assert torch.isfinite(actual).all()


def test_cuda_preflight_identifies_cpu_wheel(ops):
    checker = importlib.import_module("check_environment")
    runtime = SimpleNamespace(version=SimpleNamespace(cuda=None), __version__="2.8.0+cpu")
    with pytest.raises(RuntimeError, match="CPU-only PyTorch"):
        checker.require_cuda(runtime)


def test_cuda_preflight_preserves_driver_initialization_error(ops):
    checker = importlib.import_module("check_environment")

    def initialize():
        raise RuntimeError("NVIDIA driver is too old")

    runtime = SimpleNamespace(version=SimpleNamespace(cuda="12.8"),
                              cuda=SimpleNamespace(is_available=lambda: False, init=initialize))
    with pytest.raises(RuntimeError, match="driver is too old"):
        checker.require_cuda(runtime)


def test_cuda_preflight_accepts_available_gpu(ops):
    checker = importlib.import_module("check_environment")
    runtime = SimpleNamespace(version=SimpleNamespace(cuda="11.8"),
                              cuda=SimpleNamespace(is_available=lambda: True))
    checker.require_cuda(runtime)
