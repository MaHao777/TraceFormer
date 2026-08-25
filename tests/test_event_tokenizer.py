from __future__ import annotations

import math

import torch
from torch import nn

from huav.models.event_tokenizer import EventTokenizer


def _tokenizer() -> EventTokenizer:
    return EventTokenizer(
        width=16,
        height=12,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        event_embed_dim=8,
        token_dim=16,
    )


def _voxel_tokenizer(*, use_block_statistics: bool = True) -> EventTokenizer:
    return EventTokenizer(
        width=16,
        height=8,
        spatial_block_size_x=8,
        spatial_block_size_y=8,
        num_time_blocks=2,
        duration_ms=250.0,
        event_embed_dim=8,
        token_dim=16,
        embedding_mode="voxel_attention",
        voxel_size_x=2,
        voxel_size_y=2,
        voxel_duration_ms=25.0,
        use_block_statistics=use_block_statistics,
    )


def test_same_block_events_share_token_and_retain_local_coordinates() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
            [0.0, 6.0, 2.0, 0.20, 1.0],
        ]
    )

    output = _tokenizer()(events)

    assert output.tokens.shape == (2, 16)
    assert output.event_token_indices.tolist() == [0, 0, 1]
    assert output.token_coords.tolist() == [[0, 0, 0], [1, 0, 0]]
    torch.testing.assert_close(
        output.event_local_features[0],
        torch.tensor([0.25, 0.5, 0.4, 1.0, 1.0 / 3.0, 2.0 / 3.0]),
    )
    assert output.event_counts.tolist() == [2, 2, 1]


def test_batch_index_is_part_of_sparse_cell_identity() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [1.0, 1.0, 2.0, 0.10, 1.0],
        ]
    )

    output = _tokenizer()(events)

    assert output.tokens.shape[0] == 2
    assert output.event_token_indices.tolist() == [0, 1]
    assert output.token_batch_indices.tolist() == [0, 1]
    assert output.token_coords.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_160_time_blocks_map_eight_seconds_to_exact_50_ms_bins() -> None:
    tokenizer = EventTokenizer(
        width=16,
        height=12,
        spatial_block_size_x=8,
        spatial_block_size_y=8,
        num_time_blocks=160,
        event_embed_dim=8,
        token_dim=16,
    )
    events = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 49.999 / 8000.0, 1.0],
            [0.0, 1.0, 1.0, 50.0 / 8000.0, 1.0],
            [0.0, 1.0, 1.0, 7999.0 / 8000.0, 1.0],
        ]
    )

    output = tokenizer(events)

    assert output.event_token_indices.tolist() == [0, 0, 1, 2]
    assert output.token_coords[:, 2].tolist() == [0, 1, 159]


def test_token_statistics_include_sequence_relative_polarity_and_entropy() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
        ]
    )

    output = _tokenizer()(events)
    statistics = output.token_statistics[0]

    assert statistics.shape == (7,)
    torch.testing.assert_close(statistics[1], torch.tensor(0.5))
    torch.testing.assert_close(statistics[2], torch.tensor(0.6))
    torch.testing.assert_close(statistics[3], torch.tensor(0.04))
    assert 0.0 < statistics[4] < 1.0
    torch.testing.assert_close(statistics[5], torch.tensor(0.0))
    torch.testing.assert_close(statistics[6], torch.log(torch.tensor(2.0)))


def test_sequence_polarity_features_are_isolated_by_batch() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
            [1.0, 1.0, 2.0, 0.10, 1.0],
            [1.0, 3.0, 1.0, 0.20, 1.0],
        ]
    )

    output = _tokenizer()(events)

    torch.testing.assert_close(
        output.event_local_features[:, 4:],
        torch.tensor(
            [
                [0.5, 0.5],
                [-0.5, 0.5],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )
    torch.testing.assert_close(
        output.token_statistics[:, 5],
        torch.tensor([0.0, 0.0]),
    )


def test_signed_polarity_is_converted_before_block_statistics() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, -1.0],
            [0.0, 3.0, 1.0, 0.20, 1.0],
        ]
    )

    statistics = _tokenizer()(events).token_statistics[0]

    torch.testing.assert_close(statistics[1], torch.tensor(0.5))
    torch.testing.assert_close(statistics[5], torch.tensor(0.0))
    torch.testing.assert_close(statistics[6], torch.log(torch.tensor(2.0)))


def test_empty_event_tensor_returns_all_empty_outputs() -> None:
    output = _tokenizer()(torch.empty((0, 5)))

    assert output.tokens.shape == (0, 16)
    assert output.token_coords.shape == (0, 3)
    assert output.token_batch_indices.shape == (0,)
    assert output.event_token_indices.shape == (0,)
    assert output.event_local_features.shape == (0, 6)
    assert output.token_statistics.shape == (0, 7)


def test_attention_pool_matches_explicit_variable_length_group_softmax() -> None:
    tokenizer = _tokenizer()
    encoded_events = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        ]
    )
    inverse = torch.tensor([0, 0, 1, 1, 1])
    with torch.no_grad():
        tokenizer.key_projection.weight.copy_(torch.eye(8))
        tokenizer.value_projection.weight.copy_(torch.eye(8))

    actual = tokenizer._attention_pool(encoded_events, inverse, token_count=2)
    expected_groups = []
    for token_index in range(2):
        group = encoded_events[inverse == token_index]
        logits = group @ tokenizer.pool_queries.T / math.sqrt(8)
        weights = torch.softmax(logits, dim=0)
        expected_groups.append(weights.T @ group)
    expected = torch.stack(expected_groups)

    assert actual.shape == (2, 4, 8)
    torch.testing.assert_close(actual, expected)


def test_attention_pool_backpropagates_under_bfloat16_autocast() -> None:
    tokenizer = _tokenizer()
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
            [0.0, 6.0, 2.0, 0.20, 1.0],
        ]
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = tokenizer(events)
        loss = output.tokens.float().square().mean()
    loss.backward()

    assert torch.isfinite(output.tokens).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tokenizer.parameters()
    )


def test_voxel_descriptor_contains_means_variances_and_log_count() -> None:
    tokenizer = _voxel_tokenizer()
    local_features = torch.tensor(
        [
            [0.0, 0.25, 0.2, 1.0, 0.5, 0.5],
            [0.5, 0.75, 0.6, 0.0, -0.5, 0.5],
        ]
    )
    descriptors, parents, counts = tokenizer._voxelize(
        local_features=local_features,
        parent_indices=torch.tensor([0, 0]),
        x=torch.tensor([0.0, 1.0]),
        y=torch.tensor([0.0, 1.0]),
        scaled_time=torch.tensor([0.1, 0.15]),
        time_index=torch.tensor([0, 0]),
    )

    expected_mean = local_features.mean(dim=0)
    expected_variance = local_features[:, :3].var(dim=0, unbiased=False)
    expected = torch.cat(
        (expected_mean, expected_variance, torch.log(torch.tensor([3.0])))
    )
    assert descriptors.shape == (1, 10)
    torch.testing.assert_close(descriptors[0], expected)
    assert parents.tolist() == [0]
    assert counts.tolist() == [2]


def test_voxel_boundaries_and_duration_endpoint_form_expected_groups() -> None:
    tokenizer = _voxel_tokenizer()
    captured: list[torch.Tensor] = []
    assert tokenizer.voxel_encoder is not None
    handle = tokenizer.voxel_encoder[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    events = torch.tensor(
        [
            [0.0, 1.0, 1.0, 24.999 / 250.0, 1.0],
            [0.0, 2.0, 1.0, 25.0 / 250.0, 1.0],
            [0.0, 7.0, 7.0, 124.999 / 250.0, 0.0],
            [0.0, 7.0, 7.0, 125.0 / 250.0, 0.0],
            [0.0, 15.0, 7.0, 1.0, 1.0],
        ]
    )

    output = tokenizer(events)
    handle.remove()

    assert output.event_token_indices.tolist() == [0, 0, 0, 1, 2]
    assert output.token_coords.tolist() == [[0, 0, 0], [0, 0, 1], [1, 0, 1]]
    assert len(captured) == 1
    assert captured[0].shape == (5, 10)


def test_voxel_identity_is_isolated_by_batch_and_parent_block() -> None:
    tokenizer = _voxel_tokenizer()
    captured: list[torch.Tensor] = []
    assert tokenizer.voxel_encoder is not None
    handle = tokenizer.voxel_encoder[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    events = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.01, 1.0],
            [1.0, 1.0, 1.0, 0.01, 1.0],
            [0.0, 9.0, 1.0, 0.01, 1.0],
        ]
    )

    output = tokenizer(events)
    handle.remove()

    assert output.event_token_indices.tolist() == [0, 2, 1]
    assert captured[0].shape[0] == 3


def test_statistics_can_be_removed_only_from_final_voxel_fusion() -> None:
    with_statistics = _voxel_tokenizer(use_block_statistics=True)
    without_statistics = _voxel_tokenizer(use_block_statistics=False)

    assert with_statistics.token_projection[0].in_features == 8 * 2 + 7
    assert without_statistics.token_projection[0].in_features == 8 * 2
    output = without_statistics(
        torch.tensor([[0.0, 1.0, 1.0, 0.1, 1.0]])
    )
    assert output.tokens.shape == (1, 16)
    assert output.token_statistics.shape == (1, 7)


def test_voxel_embedding_backpropagates_under_bfloat16_autocast() -> None:
    tokenizer = _voxel_tokenizer(use_block_statistics=False)
    events = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.10, 1.0],
            [0.0, 1.0, 1.0, 0.12, 0.0],
            [0.0, 6.0, 2.0, 0.20, 1.0],
        ]
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = tokenizer(events)
        loss = output.tokens.float().square().mean()
    loss.backward()

    assert torch.isfinite(output.tokens).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tokenizer.parameters()
    )


def test_pooling_parameters_are_shared_and_do_not_depend_on_event_count() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.pool_queries.shape == (4, 8)
    assert tokenizer.key_projection.weight.shape == (8, 8)
    assert tokenizer.value_projection.weight.shape == (8, 8)
    assert isinstance(tokenizer.query_fusion[0], nn.Linear)
    assert tokenizer.query_fusion[0].in_features == 4 * 8
    assert tokenizer.query_fusion[-1].out_features == 8
    assert not hasattr(tokenizer, "mean_pool")
    assert not hasattr(tokenizer, "max_pool")


def test_token_features_backpropagate_to_event_encoder_and_projection() -> None:
    tokenizer = _tokenizer()
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
            [0.0, 6.0, 2.0, 0.20, 1.0],
        ]
    )

    tokenizer(events).tokens.sum().backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tokenizer.parameters()
    )
