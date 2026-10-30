"""逐事件时间记忆及等容量神经解码器的独立契约测试。"""

import numpy as np
import pytest
import torch

from traceformer.experiments.event_memory import (
    MemoryConcatDecoder,
    MemoryDecoder,
    SamePixelEventMemory,
    mean_memory_within_tokens,
    shuffle_memory_within_tokens,
)
from traceformer.models.event_decoder import EventDecoder


def brute_force_gaps(xy, timestamps, polarity):
    # 参考实现逐事件筛选，明确排除全部同时间邻居。
    result = np.full((len(timestamps), 4), np.inf, dtype=np.float64)
    for index, (point, timestamp, sign) in enumerate(zip(xy, timestamps, polarity)):
        same_pixel = np.all(xy == point, axis=1)
        for opposite, offset in ((False, 0), (True, 2)):
            matching = same_pixel & ((polarity != sign) if opposite else (polarity == sign))
            past = timestamps[matching & (timestamps < timestamp)]
            future = timestamps[matching & (timestamps > timestamp)]
            if len(past):
                result[index, offset] = timestamp - past.max()
            if len(future):
                result[index, offset + 1] = future.min() - timestamp
    return result


def test_memory_matches_brute_force_with_unordered_timestamps():
    rng = np.random.default_rng(37)
    xy = rng.integers(0, 4, size=(120, 2))
    timestamps = rng.integers(0, 30, size=120).astype(np.float64)
    polarity = rng.integers(0, 2, size=120)
    extractor = SamePixelEventMemory(width=4, height=4)
    expected = brute_force_gaps(xy, timestamps, polarity)
    np.testing.assert_array_equal(extractor.gaps(xy, timestamps, polarity), expected)
    features = extractor(xy, timestamps, polarity)
    expected_features = np.exp(-expected[:, :, None] / np.array([1.0, 10.0, 50.0]))
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, expected_features.reshape(-1, 12), rtol=1e-6)


def test_equal_time_events_are_excluded_and_order_invariant():
    xy = np.tile([2, 3], (7, 1))
    timestamps = np.array([1, 2, 2, 2, 2, 4, 4], dtype=np.float64)
    polarity = np.array([0, 0, 1, 0, 1, 0, 1])
    extractor = SamePixelEventMemory()
    expected = brute_force_gaps(xy, timestamps, polarity)
    np.testing.assert_array_equal(extractor.gaps(xy, timestamps, polarity), expected)
    order = np.array([6, 3, 2, 0, 4, 1, 5])
    shuffled = extractor(xy[order], timestamps[order], polarity[order])
    np.testing.assert_array_equal(shuffled[np.argsort(order)], extractor(xy, timestamps, polarity))
    # 相同时间的另一极性不能成为间隔为零的邻居。
    np.testing.assert_array_equal(expected[1:5, 2], [np.inf, 1, np.inf, 1])
    assert np.all(expected[1:5, 3] == 2)


def test_polarity_exchange_preserves_relative_channels():
    xy = np.array([[1, 1], [1, 1], [1, 1], [2, 1], [1, 1]])
    timestamps = np.array([1, 3, 5, 6, 8], dtype=np.float64)
    polarity = np.array([0, 1, 0, 1, 1])
    extractor = SamePixelEventMemory()
    np.testing.assert_array_equal(
        extractor(xy, timestamps, polarity), extractor(xy, timestamps, 1 - polarity)
    )


@pytest.mark.parametrize("dtype", [np.int16, np.uint16])
def test_pixel_index_is_widened_before_multiplication(dtype):
    # 两个合法像素的展平索引相差65536，窄整数不能把它们混为同一像素。
    xy = np.array([[2, 0], [144, 189]], dtype=dtype)
    gaps = SamePixelEventMemory().gaps(
        xy, np.array([1.0, 3.0], dtype=np.float64), np.array([1, 1])
    )
    assert np.isinf(gaps).all()


def test_single_event_no_self_missing_history_and_empty_input():
    extractor = SamePixelEventMemory(width=8, height=8)
    xy = np.array([[3, 5]])
    timestamps = np.array([1.25], dtype=np.float64)
    polarity = np.array([1])
    assert np.isinf(extractor.gaps(xy, timestamps, polarity)).all()
    assert np.array_equal(extractor(xy, timestamps, polarity), np.zeros((1, 12)))
    xy = np.array([[3, 5], [3, 5], [4, 5]])
    timestamps = np.array([1, 3, 4], dtype=np.float64)
    polarity = np.array([1, 1, 1])
    gaps = extractor.gaps(xy, timestamps, polarity)
    np.testing.assert_array_equal(gaps[:, :2], [[np.inf, 2], [2, np.inf], [np.inf, np.inf]])
    assert extractor(np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)).shape == (0, 12)


def test_float64_timestamps_keep_sub_float32_resolution_and_accept_torch_inputs():
    extractor = SamePixelEventMemory()
    xy = torch.tensor([[1, 1], [1, 1]])
    timestamps = torch.tensor([7000.0, 7000.00001], dtype=torch.float64)
    polarity = torch.tensor([0, 0])
    gaps = extractor.gaps(xy, timestamps, polarity)
    assert 0 < gaps[0, 1] < 0.0001
    assert gaps[0, 1] == gaps[1, 0]
    with pytest.raises(TypeError, match="float64"):
        extractor(xy, timestamps.float(), polarity)


@pytest.mark.parametrize(
    "xy,timestamps,polarity,message",
    [
        ([[346, 0]], [1.0], [0], "coordinates"),
        ([[-1, 0]], [1.0], [0], "coordinates"),
        ([[1.5, 0]], [1.0], [0], "coordinates"),
        ([[1, 0]], [np.nan], [0], "timestamps"),
        ([[1, 0]], [-1.0], [0], "timestamps"),
        ([[1, 0]], [1.0], [2], "polarity"),
        ([[1, 0]], [1.0, 2.0], [0], "align"),
    ],
)
def test_memory_rejects_invalid_input(xy, timestamps, polarity, message):
    with pytest.raises(ValueError, match=message):
        SamePixelEventMemory()(np.asarray(xy), np.asarray(timestamps), np.asarray(polarity))


@pytest.mark.parametrize("fusion", ["additive", "gated"])
def test_decoder_exact_parameter_count_and_all_parameters_have_finite_gradients(fusion):
    torch.manual_seed(37)
    model = MemoryDecoder(fusion=fusion)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6758
    assert sum(parameter.numel() for parameter in EventDecoder(token_dim=64, dropout=0.1).parameters()) == 6785
    tokens = torch.randn(5, 64, requires_grad=True)
    indices = torch.tensor([3, 1, 4, 0, 2, 1, 3, 2, 4, 0, 3, 4])
    local = torch.randn(len(indices), 6)
    memory = torch.rand(len(indices), 12)
    logits = model(tokens, indices, local, memory)
    assert logits.shape == (len(indices),)
    logits.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert model(tokens, indices[:0], local[:0], memory[:0]).shape == (0,)


def test_new_decoder_controls_have_identical_initialization():
    torch.manual_seed(73)
    additive = MemoryDecoder(fusion="additive")
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    gated = MemoryDecoder(fusion="gated")
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in additive.state_dict().items():
        assert torch.equal(value, gated.state_dict()[name]), name
    with pytest.raises(ValueError, match="fusion"):
        MemoryDecoder(fusion="unknown")


def test_concat_baseline_uses_same_inputs_and_all_parameters():
    model = MemoryConcatDecoder()
    assert sum(parameter.numel() for parameter in model.parameters()) == 6734
    tokens = torch.randn(3, 64, requires_grad=True)
    indices = torch.tensor([0, 2, 1, 0, 2, 1])
    local = torch.randn(6, 6)
    memory = torch.rand(6, 12)
    model(tokens, indices, local, memory).square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert model(tokens, indices[:0], local[:0], memory[:0]).shape == (0,)


def test_memory_shuffle_preserves_joint_rows_tokens_and_independent_rng():
    indices = torch.tensor([4, 1, 4, 1, 7, 4, 7, 1])
    memory = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    local = torch.randn(8, 6)
    before_memory, before_local = memory.clone(), local.clone()
    before_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(101)
    shuffled = shuffle_memory_within_tokens(memory, indices, generator=generator)
    assert torch.equal(before_rng, torch.get_rng_state())
    repeat = shuffle_memory_within_tokens(
        memory, indices, generator=torch.Generator().manual_seed(101)
    )
    assert torch.equal(shuffled, repeat)
    assert not torch.equal(shuffled, memory)
    for token in indices.unique():
        rows = indices == token
        assert torch.equal(memory[rows].sort(dim=0).values, shuffled[rows].sort(dim=0).values)
        assert torch.equal(shuffled[rows, 1:] - shuffled[rows, :1], torch.arange(1, 12).expand(int(rows.sum()), -1))
    assert torch.equal(memory, before_memory)
    assert torch.equal(local, before_local)


def test_mean_and_shuffle_are_differentiable_and_handle_empty_inputs():
    memory = torch.randn(6, 12, requires_grad=True)
    indices = torch.tensor([8, 1, 8, 1, 1, 3])
    means = mean_memory_within_tokens(memory, indices)
    for token in indices.unique():
        rows = indices == token
        torch.testing.assert_close(means[rows], memory[rows].mean(dim=0).expand(int(rows.sum()), -1))
    shuffled = shuffle_memory_within_tokens(
        memory, indices, generator=torch.Generator().manual_seed(37)
    )
    (means.square().sum() + shuffled.square().sum()).backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert mean_memory_within_tokens(memory[:0], indices[:0]).shape == (0, 12)
    assert shuffle_memory_within_tokens(
        memory[:0], indices[:0], generator=torch.Generator().manual_seed(37)
    ).shape == (0, 12)
