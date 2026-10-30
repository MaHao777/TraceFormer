from __future__ import annotations

import copy
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from traceformer.experiments.component_core import MeanTokenizer
from traceformer.experiments.spatial_pooling import (
    SpatialConvTokenizer,
    _cell_grid,
    weighted_grid_mean,
)
from traceformer.models.event_tokenizer import EventTokenizer, TokenizedEvents


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def mean_tokenizer(dim=8, block=8):
    tokenizer = EventTokenizer(width=18, height=17, spatial_block_size_x=block,
                               spatial_block_size_y=block, num_time_blocks=4,
                               event_embed_dim=dim, token_dim=16)
    tokenizer.__class__ = MeanTokenizer
    del tokenizer.pool_queries, tokenizer.key_projection
    del tokenizer.value_projection, tokenizer.query_fusion
    return tokenizer


def events():
    # 不等子格事件数、多时间块和图像边界，避免只验证均匀填满的网格。
    return torch.tensor([[0., 0., .10, 1.], [1., 1., .12, -1.], [7., 7., .14, 1.],
                         [8., 0., .30, 1.], [9., 1., .31, -1.], [15., 7., .35, 1.],
                         [17., 16., .99, 1.], [0., 0., .10, -1.], [7., 7., .14, 1.]])


@pytest.mark.parametrize("kernels", [1, 2])
def test_construction_preserves_rng_common_weights_and_exact_parameter_count(kernels):
    original = mean_tokenizer(dim=64).eval()
    state = torch.get_rng_state().clone()
    actual = SpatialConvTokenizer.from_mean(original, kernels)
    assert torch.equal(state, torch.get_rng_state())
    assert not actual.training and not actual.spatial_kernels.training
    for name, value in original.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name
    difference = sum(p.numel() for p in actual.parameters()) - sum(
        p.numel() for p in original.parameters())
    assert difference == kernels * 576
    assert all(kernel.groups == 64 and kernel.bias is None for kernel in actual.spatial_kernels)
    assert not any(hasattr(actual, key) for key in
                   ("pool_queries", "key_projection", "value_projection", "query_fusion"))
    if kernels == 2:
        assert not torch.equal(actual.spatial_kernels[0].weight, actual.spatial_kernels[1].weight)
    assert not hasattr(original, "spatial_kernels")


def test_identity_grid_restores_m1_event_mean_despite_unequal_cell_counts():
    original = mean_tokenizer()
    local = torch.randn(7, 6)
    inverse = torch.tensor([0, 0, 1, 0, 1, 1, 0])
    cells = torch.tensor([0, 0, 15, 15, 1, 1, 0])
    values = original.event_encoder(local)
    grid, cell_counts = _cell_grid(values, cells, inverse, 2)
    expected = original._encode_block_features(local, inverse, torch.tensor([4, 3]),
                                               torch.zeros(2, 7))
    torch.testing.assert_close(weighted_grid_mean(grid, cell_counts), expected)
    assert torch.equal(grid[0, :, 0, 1], torch.zeros(8))
    assert not torch.allclose(grid.mean(dim=(-2, -1)), expected)


@pytest.mark.parametrize("kernels", [1, 2])
def test_mapping_permutation_and_batch_isolation(kernels):
    original = mean_tokenizer().eval()
    actual = SpatialConvTokenizer.from_mean(original, kernels)
    inputs = events()
    batches = torch.tensor([0] * 7 + [1, 1])
    reference, observed = original(inputs, batches), actual(inputs, batches)
    for field in fields(TokenizedEvents):
        if field.name != "tokens":
            assert torch.equal(getattr(reference, field.name), getattr(observed, field.name))
    permutation = torch.tensor([8, 2, 6, 1, 7, 4, 0, 3, 5])
    shuffled = actual(inputs[permutation], batches[permutation])
    torch.testing.assert_close(shuffled.tokens, observed.tokens)
    torch.testing.assert_close(shuffled.event_token_indices,
                               observed.event_token_indices[permutation])
    changed = inputs.clone()
    changed[batches == 1, 3] = -1
    other = actual(changed, batches)
    assert torch.equal(observed.tokens[observed.token_batch_indices == 0],
                       other.tokens[other.token_batch_indices == 0])


@pytest.mark.parametrize("block", [4, 8])
@pytest.mark.parametrize("kernels", [1, 2])
def test_empty_single_event_and_edge_blocks_match_explicit_spatial_readout(block, kernels):
    actual = SpatialConvTokenizer.from_mean(mean_tokenizer(block=block), kernels).eval()
    assert actual(torch.empty((0, 4))).tokens.shape == (0, 16)
    for inputs in (events()[-3:-2], events()):
        output = actual(inputs)
        encoded = actual.event_encoder(output.event_local_features)
        expected = []
        for token in range(len(output.tokens)):
            selected = output.event_token_indices == token
            xy = inputs[selected, :2].long().remainder(block) * 4 // block
            grid = encoded.new_zeros((1, 8, 4, 4))
            counts = encoded.new_zeros((1, 1, 4, 4))
            for x in range(4):
                for y in range(4):
                    members = (xy[:, 0] == x) & (xy[:, 1] == y)
                    if members.any():
                        grid[0, :, y, x] = encoded[selected][members].mean(0)
                        counts[0, 0, y, x] = members.sum()
            convolved = sum(F.gelu(F.conv2d(grid, kernel.weight, padding=1, groups=8))
                            for kernel in actual.spatial_kernels) / kernels
            expected.append((convolved * counts).sum(dim=(-2, -1))[0] / selected.sum())
        features = actual._encode_block_features(output.event_local_features,
                                                 output.event_token_indices,
                                                 torch.bincount(output.event_token_indices),
                                                 output.token_statistics)
        torch.testing.assert_close(features, torch.stack(expected))
        assert torch.isfinite(output.tokens).all()


@pytest.mark.parametrize("kernels", [1, 2])
def test_chunk_recomputation_matches_outputs_gradients_and_encodes_events_once(kernels, monkeypatch):
    actual = SpatialConvTokenizer.from_mean(mean_tokenizer(), kernels).train()
    reference = copy.deepcopy(actual)
    inputs = events().requires_grad_()
    other_inputs = inputs.detach().clone().requires_grad_()
    seen = []
    handle = actual.event_encoder.register_forward_hook(lambda *args: seen.append(1))
    monkeypatch.setenv("TRACEFORMER_SPATIAL_TOKEN_CHUNK", "1")
    import traceformer.experiments.spatial_pooling as module
    with patch.object(module, "checkpoint", wraps=module.checkpoint) as recompute, \
            patch.object(module, "_cell_grid", wraps=module._cell_grid) as grids:
        observed = actual(inputs).tokens
        probe = torch.randn_like(observed)
        (observed * probe).sum().backward()
    handle.remove()
    assert seen == [1]
    assert recompute.call_count == len(observed)
    assert all(call.args[3] == 1 for call in grids.call_args_list)
    monkeypatch.setenv("TRACEFORMER_SPATIAL_TOKEN_CHUNK", "4096")
    with patch.object(module, "checkpoint", side_effect=lambda fn, *args, **kwargs: fn(*args)):
        expected = reference(other_inputs).tokens
        (expected * probe).sum().backward()
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(inputs.grad, other_inputs.grad, atol=2e-6, rtol=2e-5)
    for (name, parameter), other in zip(actual.named_parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=2e-5)
        if name.startswith("spatial_kernels"):
            assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("kernels", [0, 3, True, 1.5])
def test_invalid_kernel_count_is_rejected(kernels):
    with pytest.raises(ValueError, match="kernels"):
        SpatialConvTokenizer.from_mean(mean_tokenizer(), kernels)


@pytest.mark.parametrize("value", ["0", "-1", "bad", "1.5", ""])
def test_invalid_chunk_environment_is_rejected(value, monkeypatch):
    monkeypatch.setenv("TRACEFORMER_SPATIAL_TOKEN_CHUNK", value)
    with pytest.raises(ValueError, match="positive integer"):
        SpatialConvTokenizer.from_mean(mean_tokenizer(), 1)


def test_attention_tokenizer_cannot_be_silently_replaced():
    original = EventTokenizer(width=8, height=8, spatial_block_size_x=8,
                               spatial_block_size_y=8, num_time_blocks=4,
                               event_embed_dim=8, token_dim=16)
    with pytest.raises(ValueError, match="mean tokenizer"):
        SpatialConvTokenizer.from_mean(original, 1)
