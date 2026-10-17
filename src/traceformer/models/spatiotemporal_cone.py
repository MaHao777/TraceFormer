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
from .sparse_mixture import SparseTokenMixture


class TraceFormer(nn.Module):
    """TraceFormer 为每个原始事件预测目标 logit，保留既有权重键。"""

    output_names = (
        "event_logits",
        "base_event_logits",
        "event_refinement_cell_logits",
        "event_refinement_event_cell_indices",
        "event_refinement_event_logits",
        "full_event_logits",
        "sparse_event_logits",
        "sparse_source_indices",
        "mixture_gate",
        "sparse_token_present",
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
        sparse_mixture_router_bias: float = -2.0,
        sparse_mixture_residual_scale: float = 0.01,
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
        self.sparse_mixture = SparseTokenMixture(
            token_dim=token_dim,
            num_x_blocks=self.tokenizer.num_x_blocks,
            num_y_blocks=self.tokenizer.num_y_blocks,
            num_time_blocks=num_time_blocks,
            initial_router_bias=sparse_mixture_router_bias,
            initial_residual_scale=sparse_mixture_residual_scale,
        )
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
        elif stage == "sparse_mixture":
            frozen = (
                self.tokenizer,
                self.encoder,
                self.decoder,
                self.event_refinement_head,
            )
        elif stage == "refinement":
            frozen = (
                self.tokenizer,
                self.encoder,
                self.decoder,
                self.sparse_mixture,
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
        *,
        sparse_events: torch.Tensor | None = None,
        sparse_batch_indices: torch.Tensor | None = None,
        sparse_source_indices: torch.Tensor | None = None,
        mixture_weight: float = 1.0,
        compute_sparse_logits: bool = True,
    ) -> dict[str, Any]:
        tokenized = self.tokenizer(events, batch_indices)
        encoded = self.encoder(
            tokenized.tokens,
            tokenized.token_coords,
            tokenized.token_batch_indices,
        )
        full_event_logits = self.decoder(
            encoded.token_features,
            tokenized.event_token_indices,
            tokenized.event_local_features,
        )
        event_logits = full_event_logits
        token_features = encoded.token_features
        sparse_event_logits = full_event_logits.new_empty((0,))
        returned_source_indices = torch.empty(0, dtype=torch.long, device=events.device)
        mixture_gate = full_event_logits.new_zeros(len(encoded.token_features))
        sparse_token_present = torch.zeros(
            len(encoded.token_features), dtype=torch.bool, device=events.device
        )
        paired_arguments = (
            sparse_events is not None,
            sparse_source_indices is not None,
        )
        if any(paired_arguments) and not all(paired_arguments):
            raise ValueError("sparse_events and sparse_source_indices must be provided together")
        if sparse_events is not None and sparse_source_indices is not None:
            if (
                sparse_source_indices.ndim != 1
                or sparse_source_indices.dtype != torch.long
                or len(sparse_source_indices) != len(sparse_events)
            ):
                raise ValueError("sparse_source_indices must be int64 and align with sparse_events")
            if len(sparse_source_indices) and (
                bool(torch.any(sparse_source_indices < 0))
                or bool(torch.any(sparse_source_indices >= len(events)))
            ):
                raise ValueError("sparse_source_indices are outside the full event view")
            sparse_tokenized = self.tokenizer(sparse_events, sparse_batch_indices)
            sparse_encoded = self.encoder(
                sparse_tokenized.tokens,
                sparse_tokenized.token_coords,
                sparse_tokenized.token_batch_indices,
            )
            mixed = self.sparse_mixture(
                encoded.token_features,
                tokenized.token_coords,
                tokenized.token_batch_indices,
                tokenized.token_statistics,
                sparse_encoded.token_features,
                sparse_tokenized.token_coords,
                sparse_tokenized.token_batch_indices,
                sparse_tokenized.token_statistics,
                mixture_weight=mixture_weight,
            )
            token_features = mixed.token_features
            mixture_gate = mixed.gate
            sparse_token_present = mixed.sparse_present
            event_logits = self.decoder(
                token_features,
                tokenized.event_token_indices,
                tokenized.event_local_features,
            )
            # 稀疏事件 logits 仅用于辅助监督，推理可保留空输出并跳过 Decoder。
            if compute_sparse_logits:
                sparse_event_logits = self.decoder(
                    sparse_encoded.token_features,
                    sparse_tokenized.event_token_indices,
                    sparse_tokenized.event_local_features,
                )
            returned_source_indices = sparse_source_indices
        base_event_logits = event_logits
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
            "full_event_logits": full_event_logits,
            "sparse_event_logits": sparse_event_logits,
            "sparse_source_indices": returned_source_indices,
            "mixture_gate": mixture_gate,
            "sparse_token_present": sparse_token_present,
            "token_features": token_features,
            "token_coords": tokenized.token_coords,
            "token_batch_indices": tokenized.token_batch_indices,
            "event_token_indices": tokenized.event_token_indices,
        }


__all__ = ["TraceFormer"]
