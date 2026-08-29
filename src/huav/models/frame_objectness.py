"""按比赛时空单元聚合对象置信度。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FrameObjectnessOutput:
    cell_logits: torch.Tensor
    event_cell_indices: torch.Tensor
    event_logits: torch.Tensor


class FrameCellObjectnessHead(nn.Module):
    """把事件证据聚合为 50 ms 空间 cell 的对象 logit。"""

    def __init__(
        self,
        *,
        token_dim: int,
        event_feature_dim: int,
        width: int,
        height: int,
        spatial_block_size_x: int,
        spatial_block_size_y: int,
        duration_ms: float,
        frame_ms: float,
        dropout: float,
    ) -> None:
        super().__init__()
        integer_values = (
            token_dim,
            event_feature_dim,
            width,
            height,
            spatial_block_size_x,
            spatial_block_size_y,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("objectness dimensions must be positive")
        if duration_ms <= 0 or frame_ms <= 0:
            raise ValueError("duration_ms and frame_ms must be positive")
        self.token_dim = int(token_dim)
        self.event_feature_dim = int(event_feature_dim)
        self.block_size_x = int(spatial_block_size_x)
        self.block_size_y = int(spatial_block_size_y)
        self.num_x_blocks = math.ceil(width / spatial_block_size_x)
        self.num_y_blocks = math.ceil(height / spatial_block_size_y)
        self.num_frames = math.ceil(duration_ms / frame_ms)
        self.duration_ms = float(duration_ms)
        self.frame_ms = float(frame_ms)
        self.event_projection = nn.Sequential(
            nn.Linear(token_dim + event_feature_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        hidden_dim = max(8, token_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 零初始化残差，加载旧最佳模型时预测保持完全一致。
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        token_features: torch.Tensor,
        event_token_indices: torch.Tensor,
        event_local_features: torch.Tensor,
        event_xy: torch.Tensor,
        normalized_time: torch.Tensor,
        event_batch_indices: torch.Tensor,
    ) -> FrameObjectnessOutput:
        event_count = len(event_token_indices)
        if event_count == 0:
            empty_logits = token_features.new_empty((0,))
            return FrameObjectnessOutput(
                cell_logits=empty_logits,
                event_cell_indices=torch.empty(
                    0, dtype=torch.long, device=token_features.device
                ),
                event_logits=empty_logits,
            )
        if event_local_features.shape != (event_count, self.event_feature_dim):
            raise ValueError("event_local_features do not align with events")
        if event_xy.shape != (event_count, 2):
            raise ValueError("event_xy must have shape [N,2]")
        one_dimensional = (
            event_token_indices,
            normalized_time,
            event_batch_indices,
        )
        if any(value.ndim != 1 or len(value) != event_count for value in one_dimensional):
            raise ValueError("event indices, time and batches must align")

        parent_features = token_features[event_token_indices]
        event_features = self.event_projection(
            torch.cat((parent_features, event_local_features), dim=1)
        )
        x_block = torch.div(
            event_xy[:, 0].long(), self.block_size_x, rounding_mode="floor"
        )
        y_block = torch.div(
            event_xy[:, 1].long(), self.block_size_y, rounding_mode="floor"
        )
        frame = torch.floor(
            normalized_time * (self.duration_ms / self.frame_ms)
        ).long().clamp(max=self.num_frames - 1)
        cells_per_sample = self.num_frames * self.num_y_blocks * self.num_x_blocks
        cell_key = event_batch_indices.long() * cells_per_sample + (
            (frame * self.num_y_blocks + y_block) * self.num_x_blocks + x_block
        )
        _, inverse, counts = torch.unique(
            cell_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        cell_count = len(counts)
        feature_indices = inverse[:, None].expand(-1, self.token_dim)
        cell_sum = event_features.new_zeros((cell_count, self.token_dim))
        cell_sum.scatter_add_(0, feature_indices, event_features)
        cell_mean = cell_sum / counts.to(event_features.dtype)[:, None]
        cell_maximum = torch.full_like(cell_sum, -torch.inf)
        cell_maximum.scatter_reduce_(
            0,
            feature_indices,
            event_features,
            reduce="amax",
            include_self=True,
        )
        cell_logits = self.classifier(
            torch.cat((cell_mean, cell_maximum), dim=1)
        ).squeeze(-1)
        return FrameObjectnessOutput(
            cell_logits=cell_logits,
            event_cell_indices=inverse,
            event_logits=cell_logits[inverse],
        )


__all__ = ["FrameCellObjectnessHead", "FrameObjectnessOutput"]
