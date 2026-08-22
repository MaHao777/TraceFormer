from __future__ import annotations

import pytest
import torch

from huav.models.attention_residual import FullAttentionResidual


def test_single_source_is_an_exact_identity() -> None:
    mixer = FullAttentionResidual(4)
    source = torch.randn(7, 4)

    output = mixer((source,))

    torch.testing.assert_close(output, source, rtol=0.0, atol=0.0)


def test_zero_query_uniformly_averages_all_sources() -> None:
    mixer = FullAttentionResidual(3)
    sources = tuple(torch.randn(5, 3) for _ in range(4))

    weights = mixer.attention_weights(sources)
    output = mixer(sources)

    torch.testing.assert_close(weights, torch.full((4, 5), 0.25))
    torch.testing.assert_close(output, torch.stack(sources).mean(dim=0))


def test_attention_matches_explicit_rmsnorm_softmax_formula() -> None:
    mixer = FullAttentionResidual(3, eps=1e-6)
    sources = (
        torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]]),
        torch.tensor([[2.0, -1.0, 0.5], [-2.0, 1.0, 0.5]]),
    )
    with torch.no_grad():
        mixer.query.copy_(torch.tensor([0.5, -0.25, 0.75]))
        mixer.key_norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5]))

    normalized = []
    for source in sources:
        inverse_rms = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-6)
        normalized.append(source * inverse_rms * mixer.key_norm.weight)
    expected_logits = torch.stack(
        [(value * mixer.query).sum(dim=-1) for value in normalized]
    )
    expected_weights = expected_logits.softmax(dim=0)
    expected_output = sum(
        weight.unsqueeze(-1) * source
        for weight, source in zip(expected_weights, sources, strict=True)
    )

    actual_weights = mixer.attention_weights(sources)
    actual_output = mixer(sources)

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_weights.sum(dim=0), torch.ones(2))
    torch.testing.assert_close(actual_output, expected_output)


def test_attention_residual_backpropagates_to_every_source_and_parameter() -> None:
    mixer = FullAttentionResidual(4)
    with torch.no_grad():
        mixer.query.copy_(torch.tensor([0.2, -0.1, 0.3, 0.4]))
    sources = tuple(torch.randn(6, 4, requires_grad=True) for _ in range(3))

    mixer(sources).square().mean().backward()

    assert mixer.query.grad is not None and torch.isfinite(mixer.query.grad).all()
    assert (
        mixer.key_norm.weight.grad is not None
        and torch.isfinite(mixer.key_norm.weight.grad).all()
    )
    assert all(
        source.grad is not None and torch.isfinite(source.grad).all()
        for source in sources
    )


def test_cpu_bfloat16_autocast_keeps_output_and_gradients_finite() -> None:
    mixer = FullAttentionResidual(8)
    sources = tuple(torch.randn(5, 8, requires_grad=True) for _ in range(3))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = mixer(sources)
        loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert all(source.grad is not None and torch.isfinite(source.grad).all() for source in sources)


def test_only_source_by_token_logits_are_stacked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixer = FullAttentionResidual(4)
    sources = tuple(torch.randn(9, 4) for _ in range(3))
    original_stack = torch.stack
    stacked_shapes: list[tuple[tuple[int, ...], ...]] = []

    def record_stack(
        tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        values = tuple(tensors)
        stacked_shapes.append(tuple(tuple(value.shape) for value in values))
        return original_stack(values, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", record_stack)

    mixer(sources)

    assert stacked_shapes == [((9,), (9,), (9,))]
