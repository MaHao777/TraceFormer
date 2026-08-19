"""Pre-LN residual blocks for iterative fixed-local-cone propagation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from huav.config import LocalConeConfig

from .cone_attention import AttentionStats, LocalConeAttention


@dataclass(frozen=True)
class LocalConeEncoderOutput:
    token_features: torch.Tensor
    past_attention_stats: list[AttentionStats] | None
    future_attention_stats: list[AttentionStats] | None


class LocalConeAttentionBlock(nn.Module):
    """Synchronously update all tokens from one fixed local cone."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        local_cone: LocalConeConfig,
        mlp_ratio: float,
        dropout: float,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(token_dim * mlp_ratio)))
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = LocalConeAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            local_cone=local_cone,
            dropout=dropout,
            causal_mode=causal_mode,
            direction_dropout=direction_dropout,
            debug_attention_stats=debug_attention_stats,
        )
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, AttentionStats | None, AttentionStats | None]:
        attention = self.attention(
            self.attention_norm(tokens),
            token_coords,
            token_batch_indices,
        )
        attended_tokens = tokens + self.attn_scale * self.residual_dropout(
            attention.features
        )
        output_tokens = attended_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(attended_tokens))
        )
        return (
            output_tokens,
            attention.past_attention_stats,
            attention.future_attention_stats,
        )


class LocalConeEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        num_cone_blocks: int,
        local_cone: LocalConeConfig,
        mlp_ratio: float,
        dropout: float,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
    ) -> None:
        super().__init__()
        if num_cone_blocks <= 0:
            raise ValueError("num_cone_blocks must be positive")
        self.debug_attention_stats = bool(debug_attention_stats)
        self.blocks = nn.ModuleList(
            [
                LocalConeAttentionBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    local_cone=local_cone,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    causal_mode=causal_mode,
                    direction_dropout=direction_dropout,
                    debug_attention_stats=debug_attention_stats,
                )
                for _ in range(num_cone_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> LocalConeEncoderOutput:
        past_stats: list[AttentionStats] = []
        future_stats: list[AttentionStats] = []
        for block in self.blocks:
            tokens, block_past, block_future = block(
                tokens, token_coords, token_batch_indices
            )
            if self.debug_attention_stats:
                if block_past is None or block_future is None:
                    raise RuntimeError("debug attention statistics were not returned")
                past_stats.append(block_past)
                future_stats.append(block_future)
        return LocalConeEncoderOutput(
            token_features=self.final_norm(tokens),
            past_attention_stats=past_stats if self.debug_attention_stats else None,
            future_attention_stats=(
                future_stats if self.debug_attention_stats else None
            ),
        )


__all__ = [
    "LocalConeAttentionBlock",
    "LocalConeEncoder",
    "LocalConeEncoderOutput",
]
