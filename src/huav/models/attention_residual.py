"""Depth-wise full attention residuals over earlier sublayer increments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FullAttentionResidual(nn.Module):
    """Selectively mix aligned token sources with one learned pseudo-query."""

    def __init__(self, token_dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.token_dim = int(token_dim)
        self.query = nn.Parameter(torch.zeros(self.token_dim))
        self.key_norm = nn.RMSNorm(self.token_dim, eps=eps)

    def _validated_sources(
        self,
        sources: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(sources)
        if not values:
            raise ValueError("attention residual requires at least one source")
        reference = values[0]
        if reference.ndim != 2 or reference.shape[-1] != self.token_dim:
            raise ValueError(
                f"sources must have shape [tokens, {self.token_dim}]"
            )
        if not reference.is_floating_point():
            raise TypeError("attention residual sources must be floating tensors")
        for source in values[1:]:
            if source.shape != reference.shape:
                raise ValueError("all attention residual sources must share one shape")
            if source.dtype != reference.dtype or source.device != reference.device:
                raise ValueError(
                    "all attention residual sources must share dtype and device"
                )
        return values

    def attention_weights(
        self,
        sources: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return FP32 per-token softmax weights with shape ``[sources, tokens]``."""
        values = self._validated_sources(sources)
        query = self.query.float()
        logits = torch.stack(
            [
                torch.sum(query * self.key_norm(source).float(), dim=-1)
                for source in values
            ],
            dim=0,
        )
        return torch.softmax(logits, dim=0)

    def forward(self, sources: Sequence[torch.Tensor]) -> torch.Tensor:
        values = self._validated_sources(sources)
        weights = self.attention_weights(values).to(dtype=values[0].dtype)
        mixed = torch.zeros_like(values[0])
        for weight, source in zip(weights.unbind(dim=0), values, strict=True):
            mixed = mixed + weight.unsqueeze(-1) * source
        return mixed


__all__ = ["FullAttentionResidual"]
