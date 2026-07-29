from __future__ import annotations

import math

import pytest
import torch

from huav.models import AdaptiveThresholdCalibrator, AdaptiveThresholdResult


def _calibrator(
    mode: str = "hierarchical",
    *,
    token_dim: int = 4,
    width: int = 16,
    height: int = 16,
    spatial_block_size_x: int = 8,
    spatial_block_size_y: int = 8,
    threshold_time_bins: int = 160,
    max_global_offset: float = 0.5,
    max_local_offset: float = 0.75,
) -> AdaptiveThresholdCalibrator:
    return AdaptiveThresholdCalibrator(
        token_dim=token_dim,
        width=width,
        height=height,
        spatial_block_size_x=spatial_block_size_x,
        spatial_block_size_y=spatial_block_size_y,
        threshold_time_bins=threshold_time_bins,
        mode=mode,
        max_global_offset=max_global_offset,
        max_local_offset=max_local_offset,
    )


def _inputs(
    *,
    token_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    events = torch.tensor(
        [
            [1.0, 1.0, 0.10, -1.0],
            [2.0, 2.0, 0.10, 1.0],
            [10.0, 10.0, 0.50, 1.0],
            [1.0, 1.0, 0.10, 0.0],
        ]
    )
    encoded = torch.arange(len(events) * token_dim, dtype=torch.float32).reshape(
        len(events), token_dim
    )
    statistics = torch.arange(len(events) * 7, dtype=torch.float32).reshape(
        len(events), 7
    )
    logits = torch.tensor([-1.0, 0.5, 2.0, 1.5])
    batches = torch.tensor([0, 0, 0, 1])
    return events, encoded, statistics, logits, batches


def _adjacent_inputs(
    *,
    encoded_dtype: torch.dtype = torch.float32,
    statistics_dtype: torch.dtype = torch.float32,
    logit_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    events = torch.tensor(
        [
            [1.0, 1.0, 0.10, -1.0],
            [9.0, 1.0, 0.10, 1.0],
            [17.0, 1.0, 0.10, 1.0],
        ]
    )
    generator = torch.Generator().manual_seed(123)
    encoded = torch.randn(3, 4, generator=generator).to(encoded_dtype)
    statistics = torch.randn(3, 7, generator=generator).to(statistics_dtype)
    logits = torch.tensor([-0.5, 0.25, 1.0], dtype=logit_dtype)
    batches = torch.zeros(3, dtype=torch.long)
    return events, encoded, statistics, logits, batches


def _forward(
    calibrator: AdaptiveThresholdCalibrator,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> AdaptiveThresholdResult:
    return calibrator(*inputs)


@pytest.mark.parametrize(
    "mode",
    ("none", "global", "local", "local_neighbor", "hierarchical"),
)
def test_zero_initialized_modes_exactly_reproduce_base_logits(mode: str) -> None:
    inputs = _inputs()

    result = _forward(_calibrator(mode), inputs)

    assert isinstance(result, AdaptiveThresholdResult)
    assert torch.equal(result.final_logits, inputs[3])
    assert torch.count_nonzero(result.global_event_offsets) == 0
    assert torch.count_nonzero(result.local_event_offsets) == 0


@pytest.mark.parametrize(
    "mode",
    ("none", "global", "local", "local_neighbor", "hierarchical"),
)
def test_large_finite_singleton_logit_keeps_finite_zero_std_and_identity(
    mode: str,
) -> None:
    base_logits = torch.tensor([1.0e20])

    result = _calibrator(mode)(
        torch.tensor([[1.0, 1.0, 0.5, 1.0]]),
        torch.zeros((1, 4)),
        torch.zeros((1, 7)),
        base_logits,
        torch.zeros(1, dtype=torch.long),
    )

    assert torch.isfinite(result.group_descriptors).all()
    assert result.group_descriptors[0, 15].item() == 0.0
    assert torch.equal(result.final_logits, base_logits)


def test_same_sign_extreme_logits_have_finite_exact_group_summaries() -> None:
    base_logits = torch.tensor([3.0e38, 3.0e38])

    result = _calibrator("hierarchical")(
        torch.tensor(
            [
                [1.0, 1.0, 0.5, 1.0],
                [2.0, 2.0, 0.5, -1.0],
            ]
        ),
        torch.zeros((2, 4)),
        torch.zeros((2, 7)),
        base_logits,
        torch.zeros(2, dtype=torch.long),
    )

    summaries = result.group_descriptors[0, 13:16]
    assert torch.isfinite(summaries).all()
    torch.testing.assert_close(summaries[0], torch.tensor(3.0e38))
    torch.testing.assert_close(summaries[1], torch.tensor(3.0e38))
    assert summaries[2].item() == 0.0
    assert torch.equal(result.final_logits, base_logits)


def test_opposite_sign_extreme_logits_have_finite_scaled_standard_deviation() -> None:
    base_logits = torch.tensor([3.0e38, -3.0e38])

    result = _calibrator("hierarchical")(
        torch.tensor(
            [
                [1.0, 1.0, 0.5, 1.0],
                [2.0, 2.0, 0.5, -1.0],
            ]
        ),
        torch.zeros((2, 4)),
        torch.zeros((2, 7)),
        base_logits,
        torch.zeros(2, dtype=torch.long),
    )

    summaries = result.group_descriptors[0, 13:16]
    assert torch.isfinite(summaries).all()
    assert summaries[0].item() == 0.0
    torch.testing.assert_close(summaries[1], torch.tensor(3.0e38))
    torch.testing.assert_close(summaries[2], torch.tensor(3.0e38))
    assert torch.equal(result.final_logits, base_logits)


def test_non_finite_learned_embedding_raises_instead_of_being_sanitized() -> None:
    calibrator = _calibrator("local")
    with torch.no_grad():
        calibrator.descriptor_projection[0].weight[0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="non-finite learned embedding"):
        _forward(calibrator, _inputs())


def test_low_precision_conditioning_reductions_return_finite_fp32_descriptors() -> None:
    event_count = 512
    encoded = torch.full((event_count, 4), 30000.0, dtype=torch.bfloat16)
    statistics = torch.full((event_count, 7), 1000.0, dtype=torch.float16)
    base_logits = torch.full((event_count,), 100.0, dtype=torch.float16)

    result = _calibrator("none")(
        torch.tensor([[1.0, 1.0, 0.1, 1.0]]).expand(event_count, -1).clone(),
        encoded,
        statistics,
        base_logits,
        torch.zeros(event_count, dtype=torch.long),
    )

    assert result.group_descriptors.dtype == torch.float32
    assert torch.isfinite(result.group_descriptors).all()
    torch.testing.assert_close(
        result.group_descriptors[0, :4], encoded.float().mean(dim=0)
    )
    torch.testing.assert_close(
        result.group_descriptors[0, 4:11], statistics.float().mean(dim=0)
    )
    torch.testing.assert_close(result.group_descriptors[0, 13], torch.tensor(100.0))
    assert result.final_logits.dtype == base_logits.dtype


def test_groups_events_in_50ms_8x8_cells_and_clamps_time_one() -> None:
    events = torch.tensor(
        [
            [0.0, 0.0, 0.00000, -1.0],
            [7.9, 7.9, 0.00624, 1.0],
            [8.0, 0.0, 0.00625, 1.0],
            [0.0, 8.0, 1.00000, 0.0],
        ]
    )
    encoded = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    statistics = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    logits = torch.tensor([1.0, 3.0, -2.0, 4.0])
    batches = torch.zeros(4, dtype=torch.long)

    result = _calibrator("none")(
        events, encoded, statistics, logits, batches
    )

    assert result.event_group_indices.tolist() == [0, 0, 1, 2]
    assert result.group_coords.tolist() == [[0, 0, 0], [1, 0, 1], [0, 1, 159]]
    assert result.group_batch_indices.tolist() == [0, 0, 0]
    assert result.group_descriptors.shape == (3, 19)
    descriptor = result.group_descriptors
    torch.testing.assert_close(descriptor[0, :4], encoded[:2].mean(dim=0))
    torch.testing.assert_close(descriptor[0, 4:11], statistics[:2].mean(dim=0))
    torch.testing.assert_close(descriptor[0, 11], torch.tensor(math.log(3.0)))
    torch.testing.assert_close(descriptor[0, 12], torch.tensor(0.5))
    torch.testing.assert_close(descriptor[0, 13:16], torch.tensor([2.0, 3.0, 1.0]))
    torch.testing.assert_close(descriptor[0, 16:], torch.tensor([0.0, 0.0, 0.0]))
    assert descriptor[1, 15].item() == 0.0
    assert torch.isfinite(descriptor).all()
    torch.testing.assert_close(descriptor[1, 16:], torch.tensor([1.0, 0.0, 1 / 159]))
    torch.testing.assert_close(descriptor[2, 16:], torch.tensor([0.0, 1.0, 1.0]))


def test_zero_polarity_is_not_counted_as_positive() -> None:
    result = _calibrator("none")(
        torch.tensor(
            [
                [1.0, 1.0, 0.1, 0.0],
                [2.0, 2.0, 0.1, 0.0],
            ]
        ),
        torch.zeros((2, 4)),
        torch.zeros((2, 7)),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
    )

    assert result.group_descriptors[0, 12].item() == 0.0


def test_events_in_same_group_share_local_offset_and_offsets_are_bounded() -> None:
    calibrator = _calibrator("local", max_local_offset=0.2)
    with torch.no_grad():
        calibrator.local_head[-1].weight.fill_(0.01)
        calibrator.local_head[-1].bias.fill_(0.01)

    result = _forward(calibrator, _inputs())

    torch.testing.assert_close(
        result.local_event_offsets[0], result.local_event_offsets[1]
    )
    assert torch.all(result.local_event_offsets.abs() <= 0.2)
    assert torch.count_nonzero(result.local_event_offsets) > 0
    assert torch.count_nonzero(result.global_event_offsets) == 0


def test_global_mode_broadcasts_one_bounded_offset_per_batch() -> None:
    calibrator = _calibrator("global", max_global_offset=0.15)
    with torch.no_grad():
        calibrator.global_head[-1].weight.fill_(0.01)
        calibrator.global_head[-1].bias.zero_()

    result = _forward(calibrator, _inputs())

    torch.testing.assert_close(
        result.global_event_offsets[:3],
        result.global_event_offsets[0].expand(3),
    )
    assert result.global_event_offsets[0] != result.global_event_offsets[3]
    assert torch.all(result.global_event_offsets.abs() <= 0.15)
    assert torch.count_nonzero(result.local_event_offsets) == 0


def test_neighbor_edges_are_exact_26_cell_neighbors_and_batch_isolated() -> None:
    calibrator = _calibrator(
        "local_neighbor",
        width=5,
        height=5,
        spatial_block_size_x=1,
        spatial_block_size_y=1,
        threshold_time_bins=4,
    )
    events = torch.tensor(
        [
            [1.0, 1.0, 0.275, 1.0],
            [2.0, 2.0, 0.525, 1.0],
            [0.0, 1.0, 0.025, 1.0],
            [3.0, 1.0, 0.275, 1.0],
            [2.0, 2.0, 0.525, 1.0],
        ]
    )
    encoded = torch.randn(5, 4)
    statistics = torch.randn(5, 7)
    logits = torch.randn(5)
    batches = torch.tensor([0, 0, 0, 0, 1])

    result = calibrator(events, encoded, statistics, logits, batches)

    edges = {
        (int(target), int(source))
        for target, source in result.neighbor_edge_indices.T.tolist()
    }
    expected: set[tuple[int, int]] = set()
    for target in range(len(result.group_coords)):
        for source in range(len(result.group_coords)):
            delta = (result.group_coords[source] - result.group_coords[target]).abs()
            if (
                target != source
                and result.group_batch_indices[target]
                == result.group_batch_indices[source]
                and torch.all(delta <= 1)
            ):
                expected.add((target, source))
    assert edges == expected
    assert edges
    for target, source in edges:
        assert result.group_batch_indices[target] == result.group_batch_indices[source]


@pytest.mark.parametrize("mode", ("none", "global", "local"))
def test_non_neighbor_modes_skip_neighbor_work_and_return_no_edges(mode: str) -> None:
    calibrator = _calibrator(
        mode,
        width=24,
        height=8,
    )
    with torch.no_grad():
        for module in (
            calibrator.neighbor_query,
            calibrator.neighbor_key,
            calibrator.neighbor_value,
            calibrator.relative_encoder,
        ):
            for parameter in module.parameters():
                parameter.fill_(torch.nan)
        if mode == "none":
            calibrator.descriptor_projection[0].weight.fill_(torch.nan)

    result = _forward(calibrator, _adjacent_inputs())

    assert torch.isfinite(result.final_logits).all()
    assert result.neighbor_edge_indices.shape == (2, 0)


def test_neighbor_projections_run_once_per_group_and_relative_once_per_direction() -> None:
    calibrator = _calibrator(
        "local_neighbor",
        width=24,
        height=8,
    )
    observed_shapes: dict[str, list[torch.Size]] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            observed_shapes.setdefault(name, []).append(inputs[0].shape)

        return hook

    handles = [
        calibrator.neighbor_query.register_forward_pre_hook(capture("query")),
        calibrator.neighbor_key.register_forward_pre_hook(capture("key")),
        calibrator.neighbor_value.register_forward_pre_hook(capture("value")),
        calibrator.relative_encoder.register_forward_pre_hook(capture("relative")),
    ]
    try:
        result = _forward(calibrator, _adjacent_inputs())
    finally:
        for handle in handles:
            handle.remove()

    group_count = len(result.group_coords)
    assert observed_shapes == {
        "query": [torch.Size((group_count, calibrator.hidden_dim))],
        "key": [torch.Size((group_count, calibrator.hidden_dim))],
        "value": [torch.Size((group_count, calibrator.hidden_dim))],
        "relative": [torch.Size((26, 3))],
    }


def test_neighbor_context_is_exactly_zero_without_edges() -> None:
    calibrator = _calibrator("local_neighbor")
    embeddings = torch.randn(3, calibrator.hidden_dim)

    context = calibrator._neighbor_context(
        embeddings,
        torch.full((3, 26), -1, dtype=torch.long),
    )

    assert torch.equal(context, torch.zeros_like(embeddings))


def test_relative_coordinate_encoding_changes_neighbor_context() -> None:
    calibrator = _calibrator("local_neighbor")
    embeddings = torch.zeros((2, calibrator.hidden_dim))
    neighbor_table = torch.full((2, 26), -1, dtype=torch.long)
    positive_x_direction = torch.nonzero(
        torch.all(
            calibrator.neighbor_deltas == torch.tensor([1, 0, 0]),
            dim=1,
        ),
        as_tuple=False,
    ).item()
    neighbor_table[0, positive_x_direction] = 1
    with torch.no_grad():
        calibrator.neighbor_value.weight.zero_()
        for layer in calibrator.relative_encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
    without_relative_position = calibrator._neighbor_context(
        embeddings, neighbor_table
    )

    with torch.no_grad():
        calibrator.relative_encoder[0].weight[0, 0] = 1.0
        calibrator.relative_encoder[2].weight[0, 0] = 1.0
    with_relative_position = calibrator._neighbor_context(
        embeddings, neighbor_table
    )

    assert torch.equal(without_relative_position, torch.zeros_like(embeddings))
    assert with_relative_position[0, 0] > 0
    assert torch.count_nonzero(with_relative_position) == 1


def test_mixed_dtype_bfloat16_autocast_backward_reaches_all_calibrator_parameters() -> None:
    calibrator = _calibrator(
        "hierarchical",
        width=24,
        height=8,
    )
    with torch.no_grad():
        calibrator.local_head[-1].weight.fill_(0.02)
        calibrator.global_head[-1].weight.fill_(0.02)
    events, encoded, statistics, logits, batches = _adjacent_inputs(
        encoded_dtype=torch.bfloat16,
        statistics_dtype=torch.float32,
        logit_dtype=torch.bfloat16,
    )
    events.requires_grad_()
    encoded.requires_grad_()
    statistics.requires_grad_()
    logits.requires_grad_()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = calibrator(events, encoded, statistics, logits, batches)
        loss = result.final_logits.float().sum()
    loss.backward()

    assert result.group_descriptors.dtype == torch.float32
    assert result.final_logits.dtype == logits.dtype
    assert result.global_event_offsets.dtype == logits.dtype
    assert result.local_event_offsets.dtype == logits.dtype
    assert events.grad is None
    assert encoded.grad is None
    assert statistics.grad is None
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    modules = (
        calibrator.descriptor_projection,
        calibrator.neighbor_query,
        calibrator.neighbor_key,
        calibrator.neighbor_value,
        calibrator.relative_encoder,
        calibrator.local_head,
        calibrator.global_head,
    )
    for module in modules:
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_empty_input_returns_finite_correctly_shaped_tensors() -> None:
    result = _calibrator("hierarchical")(
        torch.empty((0, 4)),
        torch.empty((0, 4)),
        torch.empty((0, 7)),
        torch.empty((0,)),
        torch.empty((0,), dtype=torch.long),
    )

    assert result.final_logits.shape == (0,)
    assert result.global_event_offsets.shape == (0,)
    assert result.local_event_offsets.shape == (0,)
    assert result.event_group_indices.shape == (0,)
    assert result.group_coords.shape == (0, 3)
    assert result.group_batch_indices.shape == (0,)
    assert result.group_descriptors.shape == (0, 19)
    assert result.neighbor_edge_indices.shape == (2, 0)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (
            result.final_logits,
            result.global_event_offsets,
            result.local_event_offsets,
            result.group_descriptors,
        )
    )


def test_conditioning_is_detached_but_base_logits_keep_direct_gradient_path() -> None:
    calibrator = _calibrator("hierarchical")
    with torch.no_grad():
        calibrator.local_head[-1].weight.fill_(0.25)
        calibrator.global_head[-1].weight.fill_(0.25)
    events, encoded, statistics, logits, batches = _inputs()
    events.requires_grad_()
    encoded.requires_grad_()
    statistics.requires_grad_()
    logits.requires_grad_()

    result = calibrator(events, encoded, statistics, logits, batches)
    result.final_logits.sum().backward()

    torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    assert events.grad is None
    assert encoded.grad is None
    assert statistics.grad is None
    assert calibrator.local_head[-1].weight.grad is not None
    assert calibrator.global_head[-1].weight.grad is not None


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"token_dim": 0}, ValueError),
        ({"mode": "unknown"}, ValueError),
        ({"max_local_offset": -1.0}, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    defaults: dict[str, object] = {
        "token_dim": 4,
        "width": 16,
        "height": 16,
        "spatial_block_size_x": 8,
        "spatial_block_size_y": 8,
        "threshold_time_bins": 160,
        "mode": "local",
    }
    defaults.update(kwargs)

    with pytest.raises(error_type):
        AdaptiveThresholdCalibrator(**defaults)


def test_forward_rejects_invalid_shapes_dtypes_and_coordinate_bounds() -> None:
    calibrator = _calibrator("local")
    events, encoded, statistics, logits, batches = _inputs()

    with pytest.raises(ValueError, match=r"\[N,4\]"):
        calibrator(events[:, :3], encoded, statistics, logits, batches)
    with pytest.raises(TypeError, match="floating"):
        calibrator(events.long(), encoded, statistics, logits, batches)
    with pytest.raises(ValueError, match="outside configured bounds"):
        invalid = events.clone()
        invalid[0, 2] = 1.1
        calibrator(invalid, encoded, statistics, logits, batches)
    with pytest.raises(TypeError, match="integer"):
        calibrator(events, encoded, statistics, logits, batches.float())
