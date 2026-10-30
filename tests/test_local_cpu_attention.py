"""融合CPU执行与原Torch实现的输出、反向、dropout随机状态等价性。"""

from __future__ import annotations

import copy

import pytest
import torch

from traceformer.config import ConeWindowConfig
from traceformer.experiments.attention import SlidingLocalAttention
from traceformer.models.prefix_cone import PrefixConeAttention

pytest.importorskip("numba")


@pytest.mark.parametrize("dropout", [0.0, 0.25])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("channels, heads", [(8, 2), (64, 4)])
def test_fused_cpu_matches_original_values_gradients_and_rng(dropout, threads, channels, heads):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(19)
        original = SlidingLocalAttention(PrefixConeAttention(channels, heads, (ConeWindowConfig(2, 2),), dropout),
                                         chunk_size=3)
        original.cpu_backend = "torch"
        fused = copy.deepcopy(original)
        fused.cpu_backend = "fused"
        coords = torch.tensor([[index % 3, 0, index // 3] for index in range(12)] + [[50, 50, 50]])
        batches = torch.zeros(len(coords), dtype=torch.long)
        batches[5] = 1
        tokens = torch.randn(len(coords), channels, requires_grad=True)
        other = tokens.detach().clone().requires_grad_()
        probe = torch.randn_like(tokens)
        torch.manual_seed(37)
        expected = original(tokens, coords, batches).features
        expected_rng = torch.get_rng_state()
        (expected * probe).sum().backward()
        torch.manual_seed(37)
        result = fused(other, coords, batches).features
        assert torch.equal(expected_rng, torch.get_rng_state())
        (result * probe).sum().backward()
        assert torch.equal(expected_rng, torch.get_rng_state())
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(tokens.grad, other.grad, atol=3e-6, rtol=3e-5)
        for (name, parameter), (_, second) in zip(original.named_parameters(), fused.named_parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, second.grad, atol=4e-6, rtol=4e-5, msg=name)
        assert result[-1].eq(0).all()
    finally:
        torch.set_num_threads(previous)
