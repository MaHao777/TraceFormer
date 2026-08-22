"""Pre-LN local-cone blocks with full attention residuals across depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from huav.config import LocalConeConfig

from .attention_residual import FullAttentionResidual
from .cone_attention import AttentionStats, LocalConeAttention


@dataclass(frozen=True)
class LocalConeEncoderOutput:
    token_features: torch.Tensor
    past_attention_stats: list[AttentionStats] | None
    future_attention_stats: list[AttentionStats] | None


@dataclass(frozen=True)
class LocalConeAttentionBlockOutput:
    attention_delta: torch.Tensor
    ffn_delta: torch.Tensor
    past_attention_stats: AttentionStats | None
    future_attention_stats: AttentionStats | None


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
        self.attention_residual = FullAttentionResidual(token_dim)
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
        self.ffn_residual = FullAttentionResidual(token_dim)
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
        sources: tuple[torch.Tensor, ...],
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> LocalConeAttentionBlockOutput:
        residual_dtype = sources[0].dtype
        attention_input = self.attention_residual(sources)
        attention = self.attention(
            self.attention_norm(attention_input),
            token_coords,
            token_batch_indices,
        )
        attention_delta = self.attn_scale * self.residual_dropout(
            attention.features
        )
        attention_delta = attention_delta.to(dtype=residual_dtype)
        ffn_input = self.ffn_residual((*sources, attention_delta))
        ffn_delta = self.residual_dropout(self.ffn(self.ffn_norm(ffn_input)))
        ffn_delta = ffn_delta.to(dtype=residual_dtype)
        return LocalConeAttentionBlockOutput(
            attention_delta=attention_delta,
            ffn_delta=ffn_delta,
            past_attention_stats=attention.past_attention_stats,
            future_attention_stats=attention.future_attention_stats,
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
        self.output_residual = FullAttentionResidual(token_dim)
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> LocalConeEncoderOutput:
        past_stats: list[AttentionStats] = []
        future_stats: list[AttentionStats] = []
        sources = (tokens,)
        for block in self.blocks:
            block_output = block(sources, token_coords, token_batch_indices)
            sources = (
                *sources,
                block_output.attention_delta,
                block_output.ffn_delta,
            )
            if self.debug_attention_stats:
                if (
                    block_output.past_attention_stats is None
                    or block_output.future_attention_stats is None
                ):
                    raise RuntimeError("debug attention statistics were not returned")
                past_stats.append(block_output.past_attention_stats)
                future_stats.append(block_output.future_attention_stats)
        output_tokens = self.output_residual(sources)
        return LocalConeEncoderOutput(
            token_features=self.final_norm(output_tokens),
            past_attention_stats=past_stats if self.debug_attention_stats else None,
            future_attention_stats=(
                future_stats if self.debug_attention_stats else None
            ),
        )


__all__ = [
    "LocalConeAttentionBlock",
    "LocalConeAttentionBlockOutput",
    "LocalConeEncoder",
    "LocalConeEncoderOutput",
]
