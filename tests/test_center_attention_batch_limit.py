"""复现大Token批量的CUDA上限，并检查分批前向、梯度和CPU随机数。"""

from __future__ import annotations

import copy
from unittest.mock import PropertyMock, patch

import pytest
import torch

from traceformer.config import DEFAULT_CONE_WINDOWS
from traceformer.experiments.component_core import SummaryAttention
from traceformer.models.prefix_cone import PrefixConeAttention


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pooled(values, weights, geometry, windows, **kwargs):
    # 邻域算法另有独立测试；此处保留可微路径，只隔离MHA批量维度。
    result = (values * weights)[:, None].expand(-1, len(windows), -1)
    counts = torch.ones(result.shape[:2], dtype=torch.long, device=values.device)
    counts[::11] = 0
    return result, counts


def module(dropout=0.0):
    return SummaryAttention(PrefixConeAttention(8, 2, DEFAULT_CONE_WINDOWS, dropout), past_only=True)


def test_large_cuda_batch_guard_preserves_all_queries_outputs_and_gradients():
    torch.manual_seed(37)
    actual = module().double().train()
    reference = copy.deepcopy(actual)
    tokens = torch.randn(65537, 8, dtype=torch.float64, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    calls = []
    original = actual.center_attention.forward

    def guarded(query, key, value, **kwargs):
        calls.append(len(query))
        if len(query) > 65535:
            raise RuntimeError("Efficient attention cannot produce valid seed and offset outputs when the batch size exceeds (65535).")
        return original(query, key, value, **kwargs)

    with patch("traceformer.experiments.component_core.prefix_cone_pool", side_effect=pooled):
        expected = reference(other, None, None, geometry=[]).features
        # 只模拟Python层的CUDA分支选择，真实算子仍在CPU运行，防止误称GPU验收。
        with (patch.object(actual.center_attention, "forward", side_effect=guarded),
              patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True)):
            observed = actual(tokens, None, None, geometry=[]).features
    assert calls == [32768, 32768, 1]
    torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)
    probe = torch.randn_like(observed)
    (observed * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, other.grad, atol=1e-10, rtol=1e-10)
    for (name, left), (_, right) in zip(actual.named_parameters(), reference.named_parameters(), strict=True):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, atol=2e-9, rtol=2e-9)


def test_small_cpu_dropout_preserves_single_call_and_rng():
    torch.manual_seed(41)
    model = module(0.1).train()
    tokens = torch.randn(29, 8, requires_grad=True)
    values = model.pool_value(tokens)
    summaries, counts = pooled(values, torch.sigmoid(model.pool_weight(tokens).float()) + 1e-4,
                               [], model.windows)
    summaries = summaries + (model.shape_embedding[:, None] + model.direction_embedding[None]).flatten(0, 1)
    valid = (counts > 0).any(1)
    mask = counts == 0
    mask[~valid, 0] = False
    torch.manual_seed(99)
    expected, _ = model.center_attention(tokens[:, None], summaries, summaries,
                                         key_padding_mask=mask, need_weights=False)
    expected = torch.where(valid[:, None], expected[:, 0], 0.0)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(99)
    with patch("traceformer.experiments.component_core.prefix_cone_pool", side_effect=pooled):
        observed = model(tokens, None, None, geometry=[]).features
    assert torch.equal(observed, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
def test_real_cuda_large_dropout_forward_backward():
    model = module(0.1).cuda().train()
    tokens = torch.randn(65537, 8, device="cuda", requires_grad=True)
    with patch("traceformer.experiments.component_core.prefix_cone_pool", side_effect=pooled):
        output = model(tokens, None, None, geometry=[]).features
        output.square().mean().backward()
    assert output.shape == tokens.shape and torch.isfinite(output).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
