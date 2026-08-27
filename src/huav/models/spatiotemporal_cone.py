"""Complete sparse spatiotemporal-cone event detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from huav.config import ConeHeadConfig

from .cone_transformer import ConeTransformerEncoder
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
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        cone_heads: tuple[ConeHeadConfig, ...],
        layer_time_scales: tuple[float, ...],
        max_neighbors_per_cone: int,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
        num_pool_queries: int = 4,
    ) -> None:
        super().__init__()
        if num_layers != len(layer_time_scales):
            raise ValueError("num_layers must equal len(layer_time_scales)")
        self.token_dim = int(token_dim)
        self.num_layers = int(num_layers)
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
        self.encoder = ConeTransformerEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            cone_heads=cone_heads,
            layer_time_scales=layer_time_scales,
            max_neighbors_per_cone=max_neighbors_per_cone,
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
