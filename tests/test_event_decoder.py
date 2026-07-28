from __future__ import annotations

import torch

from huav.models.event_decoder import EventDecoder


def test_decoder_uses_parent_tokens_and_event_local_features_only() -> None:
    decoder = EventDecoder(token_dim=16, dropout=0.0)
    token_features = torch.randn(2, 16)
    event_token_indices = torch.tensor([0, 0, 1])
    event_local_features = torch.randn(3, 6)

    logits = decoder(
        token_features,
        event_token_indices,
        event_local_features,
    )

    assert decoder.classifier[0].in_features == 22
    assert logits.shape == (3,)


def test_decoder_returns_empty_logits_without_block_statistics() -> None:
    decoder = EventDecoder(token_dim=16, dropout=0.0)

    logits = decoder(
        torch.empty((0, 16)),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0, 6)),
    )

    assert logits.shape == (0,)
