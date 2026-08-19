"""Complete sparse spatiotemporal-cone event detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from huav.config import LocalConeConfig

from .cone_transformer import LocalConeEncoder
from .event_decoder import EventDecoder
from .event_tokenizer import EventTokenizer


class SpatioTemporalConeDetector(nn.Module):
    """Predict one tiny-target logit for every original event."""

    output_names = (
        "event_logits",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
        "past_attention_stats",
        "future_attention_stats",
    )

    def __init__(
        self,
        *,
        width: int,
        height: int,
        spatial_block_size_x: int,
        spatial_block_size_y: int,
        num_time_blocks: int,
        token_dim: int,
        event_embed_dim: int,
        num_cone_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        local_cone: LocalConeConfig,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
        num_pool_queries: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_cone_blocks = int(num_cone_blocks)
        self.tokenizer = EventTokenizer(
            width=width,
            height=height,
            spatial_block_size_x=spatial_block_size_x,
            spatial_block_size_y=spatial_block_size_y,
            num_time_blocks=num_time_blocks,
            event_embed_dim=event_embed_dim,
            token_dim=token_dim,
            num_pool_queries=num_pool_queries,
        )
        self.encoder = LocalConeEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            num_cone_blocks=num_cone_blocks,
            local_cone=local_cone,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            causal_mode=causal_mode,
            direction_dropout=direction_dropout,
            debug_attention_stats=debug_attention_stats,
        )
        self.decoder = EventDecoder(token_dim=token_dim, dropout=dropout)

    def forward(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        tokenized = self.tokenizer(events, batch_indices)
        encoded = self.encoder(
            tokenized.tokens,
            tokenized.token_coords,
            tokenized.token_batch_indices,
        )
        event_logits = self.decoder(
            encoded.token_features,
            tokenized.event_token_indices,
            tokenized.event_local_features,
        )
        return {
            "event_logits": event_logits,
            "token_features": encoded.token_features,
            "token_coords": tokenized.token_coords,
            "token_batch_indices": tokenized.token_batch_indices,
            "event_token_indices": tokenized.event_token_indices,
            "past_attention_stats": encoded.past_attention_stats,
            "future_attention_stats": encoded.future_attention_stats,
        }


__all__ = ["SpatioTemporalConeDetector"]
