"""Multi-scale sparse context for event-level residual refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from huav.config import RefinementScaleConfig


@dataclass(frozen=True)
class EventRefinementOutput:
    cell_logits: torch.Tensor
    event_cell_indices: torch.Tensor
    event_logits: torch.Tensor


class MultiScaleEventRefinementHead(nn.Module):
    """Aggregate sparse cells at several scales and refine every event separately."""

    def __init__(
        self,
        *,
        token_dim: int,
        event_feature_dim: int,
        context_dim: int,
        width: int,
        height: int,
        duration_ms: float,
        scales: tuple[RefinementScaleConfig, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.event_feature_dim = int(event_feature_dim)
        self.context_dim = int(context_dim)
        self.width = int(width)
        self.height = int(height)
        self.duration_ms = float(duration_ms)
        self.scales = tuple(scales)

        self.event_projection = nn.Sequential(
            nn.Linear(token_dim + event_feature_dim + 1, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        pooled_dim = context_dim * 2 + 1
        self.cell_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pooled_dim, context_dim),
                    nn.LayerNorm(context_dim),
                    nn.GELU(),
                )
                for _ in self.scales
            ]
        )
        self.temporal_fusers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim * 3, context_dim),
                    nn.LayerNorm(context_dim),
                    nn.GELU(),
                )
                for _ in self.scales
            ]
        )
        hidden_dim = max(8, context_dim * 2)
        self.cell_classifier = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        event_input_dim = context_dim * (len(self.scales) + 1) + 2
        self.event_classifier = nn.Sequential(
            nn.Linear(event_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 零初始化两个残差出口，确保加载 schema-5 后首个前向完全不变。
        nn.init.zeros_(self.cell_classifier[-1].weight)
        nn.init.zeros_(self.cell_classifier[-1].bias)
        nn.init.zeros_(self.event_classifier[-1].weight)
        nn.init.zeros_(self.event_classifier[-1].bias)

    @staticmethod
    def _gather_temporal_neighbor(
        unique_keys: torch.Tensor,
        cell_features: torch.Tensor,
        frames: torch.Tensor,
        *,
        cells_per_frame: int,
        frame_count: int,
        direction: int,
    ) -> torch.Tensor:
        target_keys = unique_keys + direction * cells_per_frame
        valid_frame = frames > 0 if direction < 0 else frames + 1 < frame_count
        positions = torch.searchsorted(unique_keys, target_keys)
        safe_positions = positions.clamp(max=len(unique_keys) - 1)
        matched = (
            valid_frame
            & (positions < len(unique_keys))
            & (unique_keys[safe_positions] == target_keys)
        )
        result = torch.zeros_like(cell_features)
        if bool(matched.any()):
            result[matched] = cell_features[safe_positions[matched]]
        return result

    def _scale_context(
        self,
        event_features: torch.Tensor,
        event_xy: torch.Tensor,
        normalized_time: torch.Tensor,
        event_batch_indices: torch.Tensor,
        scale: RefinementScaleConfig,
        scale_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_x_blocks = math.ceil(self.width / scale.spatial_block_size_x)
        num_y_blocks = math.ceil(self.height / scale.spatial_block_size_y)
        frame_count = math.ceil(self.duration_ms / scale.frame_ms)
        x_block = torch.div(
            event_xy[:, 0].long(),
            scale.spatial_block_size_x,
            rounding_mode="floor",
        )
        y_block = torch.div(
            event_xy[:, 1].long(),
            scale.spatial_block_size_y,
            rounding_mode="floor",
        )
        frame = torch.floor(
            normalized_time * (self.duration_ms / scale.frame_ms)
        ).long().clamp(min=0, max=frame_count - 1)
        cells_per_frame = num_x_blocks * num_y_blocks
        cells_per_sample = frame_count * cells_per_frame
        spatial_key = (frame * num_y_blocks + y_block) * num_x_blocks + x_block
        keys = event_batch_indices.long() * cells_per_sample + spatial_key
        unique_keys, inverse, counts = torch.unique(
            keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        cell_count = len(unique_keys)
        feature_indices = inverse[:, None].expand(-1, self.context_dim)
        sums = event_features.new_zeros((cell_count, self.context_dim))
        sums.scatter_add_(0, feature_indices, event_features)
        means = sums / counts.to(event_features.dtype)[:, None]
        maxima = torch.full_like(sums, -torch.inf)
        maxima.scatter_reduce_(
            0,
            feature_indices,
            event_features,
            reduce="amax",
            include_self=True,
        )
        pooled = torch.cat(
            (means, maxima, torch.log1p(counts.to(event_features.dtype))[:, None]),
            dim=1,
        )
        current = self.cell_encoders[scale_index](pooled)
        local_keys = unique_keys.remainder(cells_per_sample)
        cell_frames = torch.div(
            local_keys,
            cells_per_frame,
            rounding_mode="floor",
        )
        previous = self._gather_temporal_neighbor(
            unique_keys,
            current,
            cell_frames,
            cells_per_frame=cells_per_frame,
            frame_count=frame_count,
            direction=-1,
        )
        following = self._gather_temporal_neighbor(
            unique_keys,
            current,
            cell_frames,
            cells_per_frame=cells_per_frame,
            frame_count=frame_count,
            direction=1,
        )
        temporal = self.temporal_fusers[scale_index](
            torch.cat((previous, current, following), dim=1)
        )
        return temporal, inverse

    def forward(
        self,
        *,
        token_features: torch.Tensor,
        event_token_indices: torch.Tensor,
        event_local_features: torch.Tensor,
        event_xy: torch.Tensor,
        normalized_time: torch.Tensor,
        event_batch_indices: torch.Tensor,
        base_event_logits: torch.Tensor,
    ) -> EventRefinementOutput:
        event_count = len(event_token_indices)
        if event_count == 0:
            empty_logits = token_features.new_empty((0,))
            return EventRefinementOutput(
                cell_logits=empty_logits,
                event_cell_indices=torch.empty(
                    0, dtype=torch.long, device=token_features.device
                ),
                event_logits=empty_logits,
            )
        parent_features = token_features[event_token_indices]
        event_features = self.event_projection(
            torch.cat(
                (parent_features, event_local_features, base_event_logits[:, None]),
                dim=1,
            )
        )
        event_contexts: list[torch.Tensor] = []
        primary_cells = event_features.new_empty((0, self.context_dim))
        primary_inverse = torch.empty(
            0, dtype=torch.long, device=event_features.device
        )
        for scale_index, scale in enumerate(self.scales):
            cell_context, inverse = self._scale_context(
                event_features,
                event_xy,
                normalized_time,
                event_batch_indices,
                scale,
                scale_index,
            )
            event_contexts.append(cell_context[inverse])
            if scale_index == len(self.scales) - 1:
                primary_cells = cell_context
                primary_inverse = inverse
        cell_logits = self.cell_classifier(primary_cells).squeeze(-1)
        event_logits = self.event_classifier(
            torch.cat(
                (
                    event_features,
                    *event_contexts,
                    cell_logits[primary_inverse, None],
                    base_event_logits[:, None],
                ),
                dim=1,
            )
        ).squeeze(-1)
        return EventRefinementOutput(
            cell_logits=cell_logits,
            event_cell_indices=primary_inverse,
            event_logits=event_logits,
        )


__all__ = ["EventRefinementOutput", "MultiScaleEventRefinementHead"]
