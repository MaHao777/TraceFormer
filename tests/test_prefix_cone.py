"""用显式成员枚举独立验证前缀和池化及模型接口。"""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from traceformer.config import (
    DEFAULT_CONE_WINDOWS,
    ConeWindowConfig,
    TraceFormerConfig,
    load_config,
)
from traceformer.models.prefix_cone import (
    PrefixConeAttention,
    prefix_cone_pool,
    prepare_prefix_geometry,
)
from traceformer.training.engine import _fixed_model_config, _serializable, build_model


def enumerate_pool(values, weights, coords, batches, windows=DEFAULT_CONE_WINDOWS):
    # 测试参考只使用定义中的整数不等式，不复用矩形查询实现。
    summaries, counts = [], []
    for center, batch in zip(coords, batches):
        delta = coords - center
        spatial = delta[:, :2].abs().amax(dim=1)
        row, row_counts = [], []
        for window in windows:
            for direction in (-1, 1):
                time = direction * delta[:, 2]
                mask = (
                    (batches == batch)
                    & (time >= 1)
                    & (time <= window.time_horizon)
                    & (window.time_horizon * spatial <= window.spatial_radius * time)
                )
                w = weights[mask].float()
                row.append((values[mask].float() * w).sum(0) / w.sum().clamp_min(1e-8))
                row_counts.append(mask.sum())
        summaries.append(torch.stack(row))
        counts.append(torch.stack(row_counts))
    return torch.stack(summaries), torch.stack(counts)


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_pool_matches_enumeration_values_and_gradients(chunk_size):
    torch.manual_seed(71)
    coords = torch.randint(0, 10, (60, 3))
    batches = torch.arange(60) % 2
    values = torch.randn(60, 8, requires_grad=True)
    weights = (torch.rand(60, 1) + 0.1).requires_grad_()
    actual, counts = prefix_cone_pool(
        values, weights, prepare_prefix_geometry(coords, batches), DEFAULT_CONE_WINDOWS, chunk_size
    )
    expected, expected_counts = enumerate_pool(values, weights, coords, batches)
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), (values, weights), retain_graph=True)
    expected_grad = torch.autograd.grad((expected * probe).sum(), (values, weights))
    for observed, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, reference, atol=1e-4, rtol=1e-4)


def test_exact_boundaries_directions_batches_and_no_topk():
    # 宽短锥第一切片半径 4；边界点纳入，半径 5 和同刻点排除。
    coords = torch.tensor(
        [
            [0, 0, 0],
            *[[i, j, 1] for i in range(5) for j in range(5)],
            [5, 0, 1],
            [1, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
        ]
    )
    batches = torch.zeros(len(coords), dtype=torch.long)
    batches[-1] = 1
    values = torch.zeros(len(coords), 2, requires_grad=True)
    weights = torch.ones(len(coords), 1)
    actual, counts = prefix_cone_pool(
        values, weights, prepare_prefix_geometry(coords, batches), DEFAULT_CONE_WINDOWS
    )
    assert counts[0, 1] == 25
    assert counts[0, 0] == 1
    (gradient,) = torch.autograd.grad(actual[0, 1].sum(), (values,))
    assert (gradient[1:26] > 0).all()
    assert gradient[26:].eq(0).all()
    changed = values.detach().clone()
    changed[25] = 25
    altered, _ = prefix_cone_pool(
        changed, weights, prepare_prefix_geometry(coords, batches), DEFAULT_CONE_WINDOWS
    )
    torch.testing.assert_close(altered[0, 1], torch.ones(2))


def test_attention_empty_and_single_time_output_zero():
    module = PrefixConeAttention(16, 4, DEFAULT_CONE_WINDOWS, 0.0)
    empty = module(
        torch.empty(0, 16), torch.empty(0, 3, dtype=torch.long), torch.empty(0, dtype=torch.long)
    )
    assert empty.features.shape == (0, 16)
    tokens = torch.randn(3, 16, requires_grad=True)
    result = module(
        tokens, torch.tensor([[0, 0, 0], [1, 1, 0], [2, 2, 0]]), torch.zeros(3, dtype=torch.long)
    )
    assert result.features.eq(0).all()
    result.features.sum().backward()
    assert torch.isfinite(tokens.grad).all()


def test_autocast_finite_and_close():
    torch.manual_seed(9)
    module = PrefixConeAttention(16, 4, DEFAULT_CONE_WINDOWS, 0.0)
    tokens = torch.randn(32, 16, requires_grad=True)
    coords = torch.randint(0, 6, (32, 3))
    batches = torch.zeros(32, dtype=torch.long)
    expected = module(tokens, coords, batches).features
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module(tokens, coords, batches).features
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.03)
    actual.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_fp32_prefix_cancellation_on_full_size_grid():
    # 在实际网格边缘和接近最小门权重时，检查四角相减的数值误差。
    torch.manual_seed(12)
    coords = torch.stack(
        (
            torch.randint(0, 44, (1500,)),
            torch.randint(0, 33, (1500,)),
            torch.randint(0, 64, (1500,)),
        ),
        dim=1,
    )
    coords[-1] = torch.tensor([43, 32, 63])
    batches = torch.zeros(len(coords), dtype=torch.long)
    values = torch.randn(len(coords), 4)
    weights = torch.full((len(coords), 1), 1e-4)
    actual, counts = prefix_cone_pool(
        values, weights, prepare_prefix_geometry(coords, batches), DEFAULT_CONE_WINDOWS
    )
    expected, expected_counts = enumerate_pool(values, weights, coords, batches)
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_full_model_backward_and_checkpoint_separation():
    base = TraceFormerConfig.default()
    config = replace(
        base,
        model=replace(
            base.model,
            architecture="prefix_cone",
            num_layers=1,
            token_dim=16,
            event_embed_dim=8,
            dropout=0.0,
        ),
    )
    model = build_model(config.validate())
    model.configure_training_stage("full")
    events = torch.tensor([[2.0, 2.0, 0.1, 1.0], [2.0, 3.0, 0.12, 0.0], [8.0, 8.0, 0.2, 1.0]])
    result = model(events)
    assert result["token_features"].shape[-1] == 16
    assert result["event_logits"].shape == (3,)
    result["event_logits"].square().sum().backward()
    assert model.encoder.layers[0].attention.pool_weight.weight.grad is not None
    restored = build_model(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(events)["event_logits"], restored(events)["event_logits"])
    old = _serializable(asdict(config.model))
    old["architecture"] = "retired_architecture"
    assert _fixed_model_config(old) is None


def test_only_new_structure_config_and_checkpoint_metadata_are_accepted():
    base = TraceFormerConfig.default()
    old = _serializable(asdict(base.model))
    old.pop("architecture")
    assert _fixed_model_config(old) is None
    tagged = _serializable(asdict(base.model))
    tagged.update(
        cone_heads=[], layer_time_scales=[], max_neighbors_per_cone=8, direction_dropout=0
    )
    assert _fixed_model_config(tagged) == _serializable(asdict(base.model))
    with pytest.raises(ValueError, match="architecture"):
        replace(base, model=replace(base.model, architecture="retired_architecture")).validate()
    with pytest.raises(ValueError, match="spatial_radius"):
        replace(
            base, model=replace(base.model, cone_windows=(ConeWindowConfig(0, 2),) * 3)
        ).validate()


def test_experiment_yaml():
    config = load_config(Path(__file__).parents[1] / "configs/traceformer_prefix_cone.yaml")
    assert config.model.architecture == "prefix_cone"
    assert config.model.cone_windows == DEFAULT_CONE_WINDOWS
    assert config.training.stage == "full"
    assert config.model.token_dim == 64
