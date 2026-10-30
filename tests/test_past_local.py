"""过去小锥注意力的枚举对照、门控梯度和分块内存合同。"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from traceformer.config import ConeWindowConfig
from traceformer.experiments.past_local import PastLocalAttention, PastLocalEncoder
from traceformer.models.prefix_cone import (
    PrefixConeAttention,
    PrefixConeEncoder,
    prepare_prefix_geometry,
)


def make_attention(*, chunk_size=3, dropout=0.0, dim=8):
    original = PrefixConeAttention(dim, 2, (ConeWindowConfig(8, 2),), dropout)
    return PastLocalAttention(original, chunk_size)


def enumerate_past(module, tokens, coords, batches):
    # 逐 Token 检查锥几何，显式 softmax，不复用网格查找或 SDPA。
    attention = module.attention
    q, k, v = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
    gates = (torch.sigmoid(module.pool_weight(tokens).float()) + 1e-4).flatten()
    heads = attention.num_heads
    dim = tokens.shape[-1] // heads
    output = []
    for index, (center, batch) in enumerate(zip(coords, batches, strict=True)):
        delta = coords - center
        distance = -delta[:, 2]
        mask = ((batches == batch) & (distance >= 1) & (distance <= 2)
                & (delta[:, :2].abs().amax(-1) <= distance))
        if not mask.any():
            output.append(tokens[index] * 0)
            continue
        query = q[index].reshape(heads, dim)
        keys = k[mask].reshape(-1, heads, dim).transpose(0, 1)
        values = v[mask].reshape(-1, heads, dim).transpose(0, 1)
        scores = (query[:, None] * keys).sum(-1) / math.sqrt(dim)
        weights = (scores + gates[mask].log().to(scores.dtype)[None]).softmax(-1)
        result = (weights[..., None] * values).sum(1).flatten()
        output.append(attention.out_proj(result))
    return torch.stack(output)


@pytest.mark.parametrize("chunk_size", [1, 3, 256])
def test_past_local_matches_enumerated_gated_outputs_and_gradients(chunk_size):
    torch.manual_seed(19)
    actual = make_attention(chunk_size=chunk_size)
    reference = copy.deepcopy(actual)
    coords = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 0, 1], [0, 0, 2],
                           [0, 0, 0], [-1, -1, -1], [1, 0, 0], [7, 7, 8]])
    batches = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
    tokens = torch.randn(8, 8, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    result = actual(tokens, coords, batches).features
    expected = enumerate_past(reference, other, coords, batches)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    assert result[[4, 5, 7]].eq(0).all()
    probe = torch.randn_like(result)
    (result * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, other.grad, atol=3e-6, rtol=3e-5)
    for (name, parameter), (_, second) in zip(actual.named_parameters(),
                                             reference.named_parameters(), strict=True):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, second.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_past_cone_reads_all_34_occupied_neighbors():
    torch.manual_seed(7)
    module = make_attention()
    coords = torch.tensor([[x, y, 2 - distance]
                           for distance in (1, 2)
                           for y in range(-distance, distance + 1)
                           for x in range(-distance, distance + 1)] + [[0, 0, 2]])
    assert module.offsets.shape == (34, 3)
    tokens = torch.randn(35, 8, requires_grad=True)
    module(tokens, coords, torch.zeros(35, dtype=torch.long)).features[-1].sum().backward()
    assert (tokens.grad[:-1].abs().sum(-1) > 0).all()


def test_past_attention_excludes_future_same_time_and_other_samples():
    torch.manual_seed(8)
    module = make_attention().eval()
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 1], [0, 0, 0]])
    batches = torch.tensor([0, 0, 0, 0, 1])
    tokens = torch.randn(5, 8, requires_grad=True)
    result = module(tokens, coords, batches).features[1]
    changed = tokens.detach().clone()
    changed[2:] += 100
    expected = module(changed, coords, batches).features[1]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    assert tokens.grad[0].abs().sum() > 0
    assert tokens.grad[2:].eq(0).all()


def test_empty_and_no_neighbor_outputs_remain_zero_after_biased_projection():
    module = make_attention()
    with torch.no_grad():
        module.attention.out_proj.bias.fill_(3)
    assert module(torch.empty(0, 8), torch.empty(0, 3, dtype=torch.long),
                  torch.empty(0, dtype=torch.long)).features.shape == (0, 8)
    tokens = torch.randn(3, 8, requires_grad=True)
    coords = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    result = module(tokens, coords, torch.tensor([0, 0, 1])).features
    assert result.eq(0).all()
    result.sum().backward()
    assert tokens.grad.eq(0).all()


def test_three_layers_share_geometry_indices_in_training(monkeypatch):
    original = PrefixConeEncoder(token_dim=8, num_heads=2, num_layers=3,
                                 windows=(ConeWindowConfig(8, 2),), mlp_ratio=2, dropout=0)
    builds = []
    for index, layer in enumerate(original.layers):
        layer.attention = PastLocalAttention(layer.attention, 2)
        build = layer.attention._build_neighbors

        def record(sample, *, method=build, layer_index=index):
            builds.append(layer_index)
            return method(sample)

        monkeypatch.setattr(layer.attention, "_build_neighbors", record)
    module = PastLocalEncoder(original)
    coords = torch.tensor([[0, 0, index] for index in range(5)])
    batches = torch.zeros(5, dtype=torch.long)
    tokens = torch.randn(5, 8, requires_grad=True)
    result = module(tokens, coords, batches).token_features
    result.square().sum().backward()
    assert builds == [0]
    module(tokens, coords, batches)
    assert builds == [0, 0]
    assert module(tokens[:0], coords[:0], batches[:0]).token_features.shape == (0, 8)


def test_explicit_geometry_cache_is_reused_across_attention_modules(monkeypatch):
    module = make_attention()
    other = copy.deepcopy(module)
    coords = torch.tensor([[0, 0, index] for index in range(4)])
    batches = torch.zeros(4, dtype=torch.long)
    geometry = [replace(sample, corner_cache={})
                for sample in prepare_prefix_geometry(coords, batches)]
    tokens = torch.randn(4, 8)
    module(tokens, coords, batches, geometry=geometry)
    cache = geometry[0].corner_cache
    assert len(cache) == 1
    cached = next(iter(cache.values()))
    assert cached.shape == (4, 34)
    monkeypatch.setattr(other, "_build_neighbors", lambda _: pytest.fail("indices rebuilt"))
    torch.testing.assert_close(other(tokens, coords, batches, geometry=geometry).features,
                               module(tokens, coords, batches, geometry=geometry).features)
    assert next(iter(cache.values())) is cached


def test_checkpoint_recomputation_preserves_dropout_outputs_gradients_and_rng(monkeypatch):
    torch.manual_seed(19)
    actual = make_attention(dropout=0.25)
    reference = copy.deepcopy(actual)
    coords = torch.tensor([[index % 3, 0, index // 3] for index in range(12)])
    batches = torch.zeros(12, dtype=torch.long)
    tokens = torch.randn(12, 8, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    probe = torch.randn_like(tokens)
    torch.manual_seed(37)
    result = actual(tokens, coords, batches).features
    after_forward = torch.get_rng_state()
    (result * probe).sum().backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    monkeypatch.setattr("traceformer.experiments.past_local.checkpoint",
                        lambda function, *args, **kwargs: function(*args))
    torch.manual_seed(37)
    expected = reference(other, coords, batches).features
    (expected * probe).sum().backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(tokens.grad, other.grad, atol=1e-6, rtol=1e-5)
    for (name, parameter), (_, second) in zip(actual.named_parameters(),
                                             reference.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, second.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_training_saves_linear_size_features_instead_of_all_neighbor_features():
    module = make_attention(chunk_size=7)
    coords = torch.tensor([[x, 0, time] for time in range(16) for x in range(6)])
    tokens = torch.randn(len(coords), 8, requires_grad=True)
    saved_storages = {}

    def record(value):
        # 按底层存储去重；重算引用同一份 Q/K/V 不能重复计成多份内存。
        if value.is_floating_point():
            storage = value.untyped_storage()
            saved_storages[storage.data_ptr()] = storage.nbytes()
        return value

    with torch.autograd.graph.saved_tensors_hooks(record, lambda value: value):
        output = module(tokens, coords, torch.zeros(len(coords), dtype=torch.long)).features
    complete_neighbors = len(coords) * 34 * tokens.shape[-1] * tokens.element_size()
    assert sum(saved_storages.values()) < complete_neighbors
    output.sum().backward()
    assert torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize("value", ["", "0", "-3", "2.5", "abc"])
def test_invalid_environment_chunk_is_rejected(monkeypatch, value):
    monkeypatch.setenv("TRACEFORMER_LOCAL_QUERY_CHUNK", value)
    with pytest.raises(ValueError, match="positive integer"):
        make_attention(chunk_size=None)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_invalid_explicit_chunk_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        make_attention(chunk_size=value)


def test_chunk_environment_and_parameter_structure(monkeypatch):
    monkeypatch.setenv("TRACEFORMER_LOCAL_QUERY_CHUNK", "17")
    assert make_attention(chunk_size=None).chunk_size == 17
    original = PrefixConeAttention(64, 4, (ConeWindowConfig(8, 2),), 0)
    module = PastLocalAttention(original, 3)
    assert module.attention is original.center_attention
    assert module.pool_weight is original.pool_weight
    assert sum(parameter.numel() for parameter in module.parameters()) == 16705
    names = " ".join(module.state_dict())
    assert not any(name in names for name in ("center_attention", "pool_value", "shape_embedding",
                                             "direction_embedding"))
