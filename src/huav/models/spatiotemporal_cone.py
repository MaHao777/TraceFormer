"""Complete sparse spatiotemporal-cone event detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from huav.config import ConeHeadConfig

from .cone_transformer import ConeTransformerEncoder
from .event_decoder import EventDecoder
from .event_tokenizer import EventTokenizer
from .frame_objectness import FrameCellObjectnessHead
from .sparse_mixture import SparseTokenMixture


class SpatioTemporalConeDetector(nn.Module):
    """Predict one tiny-target logit for every original event."""

    output_names = (
        "event_logits",
        "base_event_logits",
        "frame_objectness_cell_logits",
        "frame_objectness_event_cell_indices",
        "frame_objectness_event_logits",
        "full_event_logits",
        "sparse_event_logits",
        "sparse_source_indices",
        "mixture_gate",
        "sparse_token_present",
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
        sparse_mixture_enabled: bool = False,
        sparse_mixture_router_bias: float = -2.0,
        sparse_mixture_residual_scale: float = 0.01,
        frame_objectness_enabled: bool = False,
        frame_objectness_frame_ms: float = 50.0,
        duration_ms: float = 8000.0,
        frame_objectness_block_size_x: int = 8,
        frame_objectness_block_size_y: int = 8,
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
        self.sparse_mixture = (
            SparseTokenMixture(
                token_dim=token_dim,
                num_x_blocks=self.tokenizer.num_x_blocks,
                num_y_blocks=self.tokenizer.num_y_blocks,
                num_time_blocks=num_time_blocks,
                initial_router_bias=sparse_mixture_router_bias,
                initial_residual_scale=sparse_mixture_residual_scale,
            )
            if sparse_mixture_enabled
            else None
        )
        self.frame_objectness_head = (
            FrameCellObjectnessHead(
                token_dim=token_dim,
                event_feature_dim=6,
                width=width,
                height=height,
                spatial_block_size_x=frame_objectness_block_size_x,
                spatial_block_size_y=frame_objectness_block_size_y,
                duration_ms=duration_ms,
                frame_ms=frame_objectness_frame_ms,
                dropout=dropout,
            )
            if frame_objectness_enabled
            else None
        )
        self._full_path_frozen = False
        self._base_for_objectness_frozen = False

    def freeze_full_path(self) -> None:
        """冻结历史 Full 主干，只让稀疏混合模块学习修正。"""

        if self.sparse_mixture is None:
            raise ValueError("freezing the full path requires sparse mixture")
        for module in (self.tokenizer, self.encoder, self.decoder):
            module.requires_grad_(False)
            module.eval()
        self._full_path_frozen = True

    def train(self, mode: bool = True) -> SpatioTemporalConeDetector:
        """训练时保持冻结主干为评估态，关闭其中的 dropout。"""

        super().train(mode)
        if self._full_path_frozen:
            for module in (self.tokenizer, self.encoder, self.decoder):
                module.eval()
        if self._base_for_objectness_frozen:
            for module in (
                self.tokenizer,
                self.encoder,
                self.decoder,
                self.sparse_mixture,
            ):
                if module is not None:
                    module.eval()
        return self

    def freeze_base_for_objectness(self) -> None:
        """冻结现有检测器，只训练帧单元对象头。"""

        if self.frame_objectness_head is None:
            raise ValueError("freezing the base requires frame objectness")
        for module in (
            self.tokenizer,
            self.encoder,
            self.decoder,
            self.sparse_mixture,
        ):
            if module is not None:
                module.requires_grad_(False)
                module.eval()
        self.frame_objectness_head.requires_grad_(True)
        self._base_for_objectness_frozen = True

    def forward(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        *,
        sparse_events: torch.Tensor | None = None,
        sparse_batch_indices: torch.Tensor | None = None,
        sparse_source_indices: torch.Tensor | None = None,
        mixture_weight: float = 1.0,
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
        returned_source_indices = torch.empty(
            0, dtype=torch.long, device=events.device
        )
        mixture_gate = full_event_logits.new_zeros(len(encoded.token_features))
        sparse_token_present = torch.zeros(
            len(encoded.token_features), dtype=torch.bool, device=events.device
        )
        paired_arguments = (
            sparse_events is not None,
            sparse_source_indices is not None,
        )
        if any(paired_arguments) and not all(paired_arguments):
            raise ValueError(
                "sparse_events and sparse_source_indices must be provided together"
            )
        if sparse_events is not None and sparse_source_indices is not None:
            if self.sparse_mixture is None:
                raise ValueError("sparse mixture inputs require an enabled mixture module")
            if (
                sparse_source_indices.ndim != 1
                or sparse_source_indices.dtype != torch.long
                or len(sparse_source_indices) != len(sparse_events)
            ):
                raise ValueError(
                    "sparse_source_indices must be int64 and align with sparse_events"
                )
            if len(sparse_source_indices) and (
                bool(torch.any(sparse_source_indices < 0))
                or bool(torch.any(sparse_source_indices >= len(events)))
            ):
                raise ValueError("sparse_source_indices are outside the full event view")
            sparse_tokenized = self.tokenizer(
                sparse_events, sparse_batch_indices
            )
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
            sparse_event_logits = self.decoder(
                sparse_encoded.token_features,
                sparse_tokenized.event_token_indices,
                sparse_tokenized.event_local_features,
            )
            returned_source_indices = sparse_source_indices
        base_event_logits = event_logits
        if self.frame_objectness_head is not None:
            objectness = self.frame_objectness_head(
                token_features,
                tokenized.event_token_indices,
                tokenized.event_local_features,
                tokenized.event_xy,
                tokenized.normalized_time,
                tokenized.event_batch_indices,
            )
            objectness_cell_logits = objectness.cell_logits
            objectness_event_cell_indices = objectness.event_cell_indices
            objectness_event_logits = objectness.event_logits
            event_logits = base_event_logits + objectness_event_logits
        else:
            objectness_cell_logits = base_event_logits.new_empty((0,))
            objectness_event_cell_indices = torch.empty(
                0, dtype=torch.long, device=events.device
            )
            objectness_event_logits = base_event_logits.new_zeros(
                len(base_event_logits)
            )
        return {
            "event_logits": event_logits,
            "base_event_logits": base_event_logits,
            "frame_objectness_cell_logits": objectness_cell_logits,
            "frame_objectness_event_cell_indices": objectness_event_cell_indices,
            "frame_objectness_event_logits": objectness_event_logits,
            "full_event_logits": full_event_logits,
            "sparse_event_logits": sparse_event_logits,
            "sparse_source_indices": returned_source_indices,
            "mixture_gate": mixture_gate,
            "sparse_token_present": sparse_token_present,
            "token_features": token_features,
            "token_coords": tokenized.token_coords,
            "token_batch_indices": tokenized.token_batch_indices,
            "event_token_indices": tokenized.event_token_indices,
            "past_attention_stats": encoded.past_attention_stats,
            "future_attention_stats": encoded.future_attention_stats,
        }


__all__ = ["SpatioTemporalConeDetector"]
