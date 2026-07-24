"""Multi-scale x-t/y-t event classifier with original-event feature queries."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    BidirectionalTimeLocalCrossAttention,
    HierarchicalViewEncoder,
    MultiScaleViewStem,
)
from .projection import MultiScaleEventProjector, normalized_time_to_grid_coordinate


class DualViewEventClassifier(nn.Module):
    """Classify original events after learning and exchanging two projection views."""

    output_names = (
        "event_logits",
        "xt_logits",
        "yt_logits",
        "xt_event_logits",
        "yt_event_logits",
        "event_features",
    )

    def __init__(
        self,
        *,
        width: int = 346,
        height: int = 260,
        time_bins: tuple[int, ...] = (64, 32, 16),
        feature_channels: int = 16,
        attention_heads: int = 4,
        window_size: tuple[int, int] = (8, 16),
        cross_time_radius: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.time_bins = tuple(sorted((int(value) for value in time_bins), reverse=True))
        self.feature_channels = int(feature_channels)
        if self.feature_channels % attention_heads:
            raise ValueError("feature_channels must be divisible by attention_heads")
        self.projector = MultiScaleEventProjector(
            width=self.width,
            height=self.height,
            time_bins=self.time_bins,
            normalize_counts=True,
        )
        self.xt_stem = MultiScaleViewStem(self.time_bins, self.feature_channels)
        self.yt_stem = MultiScaleViewStem(self.time_bins, self.feature_channels)
        encoder_arguments = {
            "heads": attention_heads,
            "window_size": window_size,
            "dropout": dropout,
        }
        self.xt_encoder = HierarchicalViewEncoder(
            self.feature_channels, **encoder_arguments
        )
        self.yt_encoder = HierarchicalViewEncoder(
            self.feature_channels, **encoder_arguments
        )
        self.cross_view = BidirectionalTimeLocalCrossAttention(
            self.feature_channels,
            heads=attention_heads,
            radius=cross_time_radius,
            dropout=dropout,
        )
        self.xt_auxiliary = nn.Conv2d(self.feature_channels, 1, kernel_size=1)
        self.yt_auxiliary = nn.Conv2d(self.feature_channels, 1, kernel_size=1)

        single_input = self.feature_channels + 2
        self.xt_event_head = nn.Sequential(
            nn.Linear(single_input, self.feature_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels, 1),
        )
        self.yt_event_head = nn.Sequential(
            nn.Linear(single_input, self.feature_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels, 1),
        )
        fused_input = self.feature_channels * 4 + 2
        self.event_head = nn.Sequential(
            nn.Linear(fused_input, self.feature_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels * 2, 1),
        )

    @property
    def fine_time_bins(self) -> int:
        return max(self.time_bins)

    def _query(
        self,
        feature: torch.Tensor,
        events: torch.Tensor,
        batch_indices: torch.Tensor,
        *,
        coordinate_column: int,
        spatial_size: int,
    ) -> torch.Tensor:
        result = torch.empty(
            (len(events), feature.shape[1]), dtype=feature.dtype, device=feature.device
        )
        for batch_index in range(feature.shape[0]):
            mask = batch_indices == batch_index
            if not torch.any(mask):
                continue
            selected = events[mask]
            if spatial_size == 1:
                space_coordinate = torch.zeros_like(selected[:, coordinate_column])
            else:
                space_coordinate = (
                    2.0 * selected[:, coordinate_column] / (spatial_size - 1) - 1.0
                )
            time_coordinate = normalized_time_to_grid_coordinate(
                selected[:, 2], feature.shape[-2]
            )
            grid = torch.stack((space_coordinate, time_coordinate), dim=-1)
            grid = grid.reshape(1, -1, 1, 2)
            sampled = F.grid_sample(
                feature[batch_index : batch_index + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            result[mask] = sampled[0, :, :, 0].transpose(0, 1)
        return result

    def forward(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_indices is None:
            batch_indices = torch.zeros(
                len(events), dtype=torch.long, device=events.device
            )
        if batch_size is None:
            batch_size = (
                int(batch_indices.max().item()) + 1 if len(batch_indices) else 1
            )
        views = self.projector(events, batch_indices, batch_size=batch_size)
        xt_feature = self.xt_encoder(self.xt_stem(views.xt))
        yt_feature = self.yt_encoder(self.yt_stem(views.yt))
        xt_feature, yt_feature = self.cross_view(xt_feature, yt_feature)
        xt_logits = self.xt_auxiliary(xt_feature)
        yt_logits = self.yt_auxiliary(yt_feature)

        if query_indices is None:
            query_indices = torch.arange(len(events), device=events.device)
        query_indices = query_indices.long()
        query_events = events[query_indices]
        query_batches = batch_indices[query_indices]
        xt_event_feature = self._query(
            xt_feature,
            query_events,
            query_batches,
            coordinate_column=0,
            spatial_size=self.width,
        )
        yt_event_feature = self._query(
            yt_feature,
            query_events,
            query_batches,
            coordinate_column=1,
            spatial_size=self.height,
        )
        metadata = query_events[:, (3, 2)]
        xt_event_logits = self.xt_event_head(
            torch.cat((xt_event_feature, metadata), dim=-1)
        ).squeeze(-1)
        yt_event_logits = self.yt_event_head(
            torch.cat((yt_event_feature, metadata), dim=-1)
        ).squeeze(-1)
        event_features = torch.cat(
            (
                xt_event_feature,
                yt_event_feature,
                xt_event_feature * yt_event_feature,
                torch.abs(xt_event_feature - yt_event_feature),
                metadata,
            ),
            dim=-1,
        )
        event_logits = self.event_head(event_features).squeeze(-1)
        return {
            "event_logits": event_logits,
            "xt_logits": xt_logits,
            "yt_logits": yt_logits,
            "xt_event_logits": xt_event_logits,
            "yt_event_logits": yt_event_logits,
            "event_features": event_features,
        }


__all__ = ["DualViewEventClassifier"]
