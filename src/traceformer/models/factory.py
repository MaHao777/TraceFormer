"""按论文顺序初始化基础层，随后由九组公共工厂选择结构。"""
from __future__ import annotations

from traceformer.config import TraceFormerConfig

from .spatiotemporal_cone import TraceFormer


def build_base(
    config: TraceFormerConfig
) -> TraceFormer:
    model = TraceFormer(
        cone_windows=tuple(config.model.cone_windows),
        width=config.resolution.width,
        height=config.resolution.height,
        spatial_block_size_x=config.model.spatial_block_size_x,
        spatial_block_size_y=config.model.spatial_block_size_y,
        num_time_blocks=config.model.num_time_blocks,
        token_dim=config.model.token_dim,
        event_embed_dim=config.model.event_embed_dim,
        num_pool_queries=config.model.num_pool_queries,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        frame_refinement_context_dim=config.frame_refinement.context_dim,
        frame_refinement_scales=tuple(config.frame_refinement.scales),
        duration_ms=config.dataset.duration_ms,
    )
    return model
