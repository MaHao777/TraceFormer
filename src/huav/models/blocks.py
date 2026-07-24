"""CNN, local-window attention, FPN and time-local cross-view building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LocalWindowAttention(nn.Module):
    """Self-attention inside non-overlapping, padded 2-D windows."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        window_size: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, time, space = inputs.shape
        window_time, window_space = self.window_size
        pad_time = (-time) % window_time
        pad_space = (-space) % window_space
        padded = F.pad(inputs, (0, pad_space, 0, pad_time))
        padded_time, padded_space = padded.shape[-2:]
        time_windows = padded_time // window_time
        space_windows = padded_space // window_space

        windows = (
            padded.permute(0, 2, 3, 1)
            .reshape(
                batch,
                time_windows,
                window_time,
                space_windows,
                window_space,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window_time * window_space, channels)
        )
        normalized = self.norm1(windows)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        windows = windows + attended
        windows = windows + self.feed_forward(self.norm2(windows))
        restored = (
            windows.reshape(
                batch,
                time_windows,
                space_windows,
                window_time,
                window_space,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, padded_time, padded_space, channels)
            .permute(0, 3, 1, 2)
        )
        return restored[:, :, :time, :space]


class MultiScaleViewStem(nn.Module):
    def __init__(self, scales: tuple[int, ...], channels: int) -> None:
        super().__init__()
        self.scales = scales
        self.adapters = nn.ModuleDict(
            {str(scale): ConvBlock(4, channels) for scale in scales}
        )
        self.fuse = ConvBlock(channels * len(scales), channels)

    def forward(self, views: dict[int, torch.Tensor]) -> torch.Tensor:
        fine_scale = max(self.scales)
        target_size = (fine_scale, views[fine_scale].shape[-1])
        encoded = []
        for scale in self.scales:
            feature = self.adapters[str(scale)](views[scale])
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=True
                )
            encoded.append(feature)
        return self.fuse(torch.cat(encoded, dim=1))


class HierarchicalViewEncoder(nn.Module):
    """Two-level CNN/window-attention encoder with an FPN top-down path."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        window_size: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.high_conv = ConvBlock(channels, channels)
        self.high_attention = LocalWindowAttention(
            channels, heads=heads, window_size=window_size, dropout=dropout
        )
        self.downsample = ConvBlock(channels, channels * 2, stride=2)
        self.low_attention = LocalWindowAttention(
            channels * 2,
            heads=heads,
            window_size=window_size,
            dropout=dropout,
        )
        self.lateral = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.smooth = ConvBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        high = self.high_attention(self.high_conv(inputs))
        low = self.low_attention(self.downsample(high))
        top_down = F.interpolate(
            self.lateral(low),
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return self.smooth(high + top_down)


class BidirectionalTimeLocalCrossAttention(nn.Module):
    """Exchange spatially pooled tokens only at equal or nearby time bins."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        radius: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.xt_norm = nn.LayerNorm(channels)
        self.yt_norm = nn.LayerNorm(channels)
        self.xt_from_yt = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.yt_from_xt = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.xt_gate = nn.Parameter(torch.tensor(0.0))
        self.yt_gate = nn.Parameter(torch.tensor(0.0))

    def _mask(self, time_bins: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(time_bins, device=device)
        return (positions[:, None] - positions[None, :]).abs() > self.radius

    def forward(
        self, xt_feature: torch.Tensor, yt_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xt_token = xt_feature.mean(dim=-1).transpose(1, 2)
        yt_token = yt_feature.mean(dim=-1).transpose(1, 2)
        mask = self._mask(xt_token.shape[1], xt_token.device)
        xt_delta, _ = self.xt_from_yt(
            self.xt_norm(xt_token),
            self.yt_norm(yt_token),
            self.yt_norm(yt_token),
            attn_mask=mask,
            need_weights=False,
        )
        yt_delta, _ = self.yt_from_xt(
            self.yt_norm(yt_token),
            self.xt_norm(xt_token),
            self.xt_norm(xt_token),
            attn_mask=mask,
            need_weights=False,
        )
        xt_feature = xt_feature + torch.tanh(self.xt_gate) * xt_delta.transpose(
            1, 2
        ).unsqueeze(-1)
        yt_feature = yt_feature + torch.tanh(self.yt_gate) * yt_delta.transpose(
            1, 2
        ).unsqueeze(-1)
        return xt_feature, yt_feature


__all__ = [
    "BidirectionalTimeLocalCrossAttention",
    "ConvBlock",
    "HierarchicalViewEncoder",
    "LocalWindowAttention",
    "MultiScaleViewStem",
]
