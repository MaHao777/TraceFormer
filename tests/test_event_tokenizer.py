from __future__ import annotations

import torch

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


def test_token_statistics_include_sequence_relative_polarity_and_entropy() -> None:
    events = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 3.0, 1.0, 0.20, 0.0],
        ]
    )

    output = _tokenizer()(events)
    statistics = output.token_statistics[0]

    assert statistics.shape == (8,)
    torch.testing.assert_close(statistics[1:3], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(statistics[3], torch.tensor(0.6))
    torch.testing.assert_close(statistics[4], torch.tensor(0.04))
    assert 0.0 < statistics[5] < 1.0
    torch.testing.assert_close(statistics[6], torch.tensor(0.0))
    torch.testing.assert_close(statistics[7], torch.log(torch.tensor(2.0)))


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
        output.token_statistics[:, 6],
        torch.tensor([0.0, 0.0]),
    )


def test_empty_event_tensor_returns_all_empty_outputs() -> None:
    output = _tokenizer()(torch.empty((0, 5)))

    assert output.tokens.shape == (0, 16)
    assert output.token_coords.shape == (0, 3)
    assert output.token_batch_indices.shape == (0,)
    assert output.event_token_indices.shape == (0,)
    assert output.event_local_features.shape == (0, 6)
    assert output.token_statistics.shape == (0, 8)


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
