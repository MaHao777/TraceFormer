"""Complete sparse spatiotemporal-cone event detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from traceformer.config import (
    DEFAULT_CONE_WINDOWS,
    DEFAULT_REFINEMENT_SCALES,
    ConeWindowConfig,
    RefinementScaleConfig,
)

from .event_decoder import EventDecoder
from .event_refinement import MultiScaleEventRefinementHead
from .event_tokenizer import EventTokenizer
from .prefix_cone import PrefixConeEncoder


class TraceFormer(nn.Module):
    """TraceFormer 为每个原始事件预测目标 logit，保留既有权重键。"""

    output_names = (
        "event_logits",
        "base_event_logits",
        "event_refinement_cell_logits",
        "event_refinement_event_cell_indices",
        "event_refinement_event_logits",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
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
        num_pool_queries: int = 4,
        frame_refinement_context_dim: int = 16,
        frame_refinement_scales: tuple[RefinementScaleConfig, ...] = DEFAULT_REFINEMENT_SCALES,
        duration_ms: float = 8000.0,
        cone_windows: tuple[ConeWindowConfig, ...] = DEFAULT_CONE_WINDOWS,
    ) -> None:
        super().__init__()
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
        self.encoder = PrefixConeEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            windows=cone_windows,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.decoder = EventDecoder(token_dim=token_dim, dropout=dropout)
        self.event_refinement_head = MultiScaleEventRefinementHead(
            token_dim=token_dim,
            event_feature_dim=6,
            context_dim=frame_refinement_context_dim,
            width=width,
            height=height,
            duration_ms=duration_ms,
            scales=frame_refinement_scales,
            dropout=dropout,
        )
        self._frozen_modules: tuple[nn.Module, ...] = ()

    def configure_training_stage(self, stage: str) -> None:
        """按唯一训练阶段开放参数，模型结构始终保持不变。"""

        self.requires_grad_(True)
        if stage == "full":
            frozen: tuple[nn.Module, ...] = ()
        elif stage == "refinement":
            frozen = (
                self.tokenizer,
                self.encoder,
                self.decoder,
            )
        else:
            raise ValueError(f"unsupported training stage: {stage}")
        for module in frozen:
            module.requires_grad_(False)
            module.eval()
        self._frozen_modules = frozen

    def train(self, mode: bool = True) -> TraceFormer:
        """训练时保持冻结主干为评估态，关闭其中的 dropout。"""

        super().train(mode)
        for module in self._frozen_modules:
            module.eval()
        return self

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
        base_event_logits = self.decoder(
            encoded.token_features,
            tokenized.event_token_indices,
            tokenized.event_local_features,
        )
        # 单视图直接解码全部原始事件，精修仍读取完整 Token。
        token_features = encoded.token_features
        refinement = self.event_refinement_head(
            token_features=token_features,
            event_token_indices=tokenized.event_token_indices,
            event_local_features=tokenized.event_local_features,
            event_xy=tokenized.event_xy,
            normalized_time=tokenized.normalized_time,
            event_batch_indices=tokenized.event_batch_indices,
            base_event_logits=base_event_logits,
        )
        refinement_cell_logits = refinement.cell_logits
        refinement_event_cell_indices = refinement.event_cell_indices
        refinement_event_logits = refinement.event_logits
        event_logits = base_event_logits + refinement_event_logits
        return {
            "event_logits": event_logits,
            "base_event_logits": base_event_logits,
            "event_refinement_cell_logits": refinement_cell_logits,
            "event_refinement_event_cell_indices": refinement_event_cell_indices,
            "event_refinement_event_logits": refinement_event_logits,
            "token_features": token_features,
            "token_coords": tokenized.token_coords,
            "token_batch_indices": tokenized.token_batch_indices,
            "event_token_indices": tokenized.event_token_indices,
        }


__all__ = ["TraceFormer"]
