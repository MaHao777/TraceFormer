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
