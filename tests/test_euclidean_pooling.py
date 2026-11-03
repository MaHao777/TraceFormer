"""用定义枚举独立验证欧氏锥、公共初始化、前向梯度与训练恢复。"""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_local_pooling import assert_nested_equal, local_config, one_thread, small_config

from traceformer.config import DEFAULT_CONE_WINDOWS, ConeWindowConfig, TraceFormerConfig
from traceformer.experiments.ablation import EuclideanPoolingSpec, LocalPoolingSpec, parse_ablation
from traceformer.experiments.component_core import SummaryAttention
from traceformer.experiments.euclidean_pooling import VARIANTS, initialization_audit
from traceformer.experiments.past_local import PastLocalAttention
from traceformer.models.cone_geometry import euclidean_offsets
from traceformer.models.prefix_cone import (
    PrefixConeAttention,
    prefix_cone_pool,
    prepare_prefix_geometry,
)
from traceformer.training.engine import (
    _validate_checkpoint_contract,
    evaluate_checkpoint,
    run_training,
)

# 导入既有微型训练数据与线程 fixture，避免另建一套数据契约。
__all__ = ["local_config", "one_thread", "small_config"]


def member_mask(coords, batches, index, window):
    delta = coords - coords[index]
    distance = -delta[:, 2]
    return ((batches == batches[index]) & (distance >= 1) & (distance <= window.time_horizon)
            & (window.time_horizon**2 * delta[:, :2].square().sum(-1)
               <= window.spatial_radius**2 * distance.square()))


def enumerate_pool(values, weights, coords, batches, windows):
    rows, counts = [], []
    for index in range(len(coords)):
        features, members = [], []
        for window in windows:
            mask = member_mask(coords, batches, index, window)
            selected = weights[mask]
            features.append((values[mask] * selected).sum(0) / selected.sum().clamp_min(1e-8))
            members.append(mask.sum())
        rows.append(torch.stack(features))
        counts.append(torch.stack(members))
    return torch.stack(rows), torch.stack(counts)


@pytest.mark.parametrize("radius,horizon", [(2, 2), (8, 2), (4, 4), (2, 8)])
def test_disk_offsets_exactly_match_squared_definition(radius, horizon):
    expected = {(x, y, -d) for d in range(1, horizon + 1)
                for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)
                if horizon**2 * (x*x + y*y) <= radius**2 * d*d}
    actual = euclidean_offsets(radius, horizon)
    assert set(actual) == expected and len(actual) == len(expected)
    if (radius, horizon) == (2, 2):
        assert [sum(t == -d for _, _, t in actual) for d in (1, 2)] == [5, 13]
    if (radius, horizon) == (2, 8):
        assert (1, 1, -6) in actual  # 连续半径1.5包含对角线，提前取整会误删。
        assert (1, 1, -5) not in actual


@pytest.mark.parametrize("chunk", [1, 7, 1024])
def test_euclidean_prefix_matches_enumerated_values_counts_and_gradients(chunk):
    torch.manual_seed(71)
    coords = torch.randint(-3, 7, (60, 3))
    batches = torch.arange(60) % 2
    values = torch.randn(60, 8, requires_grad=True)
    weights = (torch.rand(60, 1) + 0.2).requires_grad_()
    windows = (ConeWindowConfig(2, 2), *DEFAULT_CONE_WINDOWS)
    actual, counts = prefix_cone_pool(values, weights, prepare_prefix_geometry(coords, batches),
                                      windows, chunk, directions=(-1,), metric="euclidean")
    expected, expected_counts = enumerate_pool(values, weights, coords, batches, windows)
    assert torch.equal(counts, expected_counts)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    probe = torch.randn_like(actual)
    observed = torch.autograd.grad((actual * probe).sum(), (values, weights))
    reference = torch.autograd.grad((expected * probe).sum(), (values, weights))
    for left, right in zip(observed, reference, strict=True):
        torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)


def test_local_and_summary_attention_match_brute_force_outputs_and_gradients():
    torch.manual_seed(37)
    coords = torch.tensor([[x, y, t] for t in range(5) for y in range(3) for x in range(3)])
    batches = torch.arange(len(coords)) % 2
    for kind in ("local", "summary"):
        original = PrefixConeAttention(8, 2, DEFAULT_CONE_WINDOWS, 0.0)
        module = (PastLocalAttention(original, chunk_size=7, metric="euclidean")
                  if kind == "local" else SummaryAttention(original, past_only=True))
        module.metric = "euclidean"
        tokens = torch.randn(len(coords), 8, requires_grad=True)
        actual = module(tokens, coords, batches).features
        if kind == "local":
            rows = []
            gate = (torch.sigmoid(module.pool_weight(tokens)) + 1e-4).log().flatten()
            for index in range(len(coords)):
                mask = member_mask(coords, batches, index, ConeWindowConfig(2, 2))
                if mask.any():
                    row, _ = module.attention(tokens[index:index+1, None], tokens[mask][None],
                                              tokens[mask][None], attn_mask=gate[mask][None],
                                              need_weights=False)
                    rows.append(row[0, 0])
                else:
                    rows.append(tokens[index] * 0)
            expected = torch.stack(rows)
        else:
            summaries, counts = enumerate_pool(module.pool_value(tokens),
                torch.sigmoid(module.pool_weight(tokens)) + 1e-4, coords, batches, module.windows)
            summaries = summaries + module.shape_embedding + module.direction_embedding[0]
            valid = counts.any(1)
            mask = counts == 0
            mask[~valid, 0] = False
            expected, _ = module.center_attention(tokens[:, None], summaries, summaries,
                                                  key_padding_mask=mask, need_weights=False)
            expected = torch.where(valid[:, None], expected[:, 0], 0.0)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        probe = torch.randn_like(actual)
        inputs = (tokens, *module.parameters())
        observed = torch.autograd.grad((actual * probe).sum(), inputs)
        reference = torch.autograd.grad((expected * probe).sum(), inputs)
        for left, right in zip(observed, reference, strict=True):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)
        assert actual[coords[:, 2] == 0].eq(0).all()
        assert module(tokens[:0], coords[:0], batches[:0]).features.shape == (0, 8)


def test_local_dropout_recomputation_preserves_output_gradient_and_rng(monkeypatch):
    torch.manual_seed(37)
    module = PastLocalAttention(PrefixConeAttention(8, 2, DEFAULT_CONE_WINDOWS, 0.2),
                                chunk_size=7, metric="euclidean").train()
    direct = copy.deepcopy(module)
    coords = torch.tensor([[x, y, t] for t in range(4) for x in range(3) for y in range(3)])
    batches = torch.zeros(len(coords), dtype=torch.long)
    tokens = torch.randn(len(coords), 8)
    torch.manual_seed(91)
    actual = module(tokens, coords, batches).features
    actual.square().sum().backward()
    expected_rng = torch.get_rng_state()
    monkeypatch.setattr("traceformer.experiments.past_local.checkpoint",
                        lambda fn, *args, **kwargs: fn(*args))
    torch.manual_seed(91)
    expected = direct(tokens, coords, batches).features
    expected.square().sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for left, right in zip(module.parameters(), direct.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)


def test_four_model_initialization_is_paired_and_actual_counts_are_recorded():
    torch.manual_seed(83)
    rng = torch.get_rng_state()
    audit = initialization_audit(TraceFormerConfig.default())
    assert torch.equal(rng, torch.get_rng_state()) and audit["passed"]
    assert {row["variant"]: row["parameters"] for row in audit["variants"]} == {
        "E0": 129526, "E1": 130102, "E2": 142774, "E3": 143350}
    assert all(row["queries"] == 0 and row["layers"] == 3 for row in audit["variants"])


def test_contract_rejects_foreign_geometry_variants_and_stages():
    for name in VARIANTS:
        spec = EuclideanPoolingSpec(name)
        assert parse_ablation(spec.to_dict()) == spec
        with pytest.raises(ValueError, match="full training"):
            spec.validate_stage("refinement")
    for fields in ({"variant": "S0"}, {"metric": "chebyshev"}, {"version": 4}):
        with pytest.raises(ValueError, match="contract"):
            EuclideanPoolingSpec(**fields).validate()
    with pytest.raises(TypeError):
        parse_ablation({"version": 5, "variant": "E0", "windows": [[3, 3]]})


@pytest.mark.parametrize("variant", VARIANTS)
def test_four_groups_train_independently_reevaluate_resume_and_reject_foreign_weights(
    variant, local_config, tmp_path,
):
    spec = EuclideanPoolingSpec(variant)
    config = replace(local_config, training=replace(local_config.training, seed=37))
    output = tmp_path / variant
    run_training(config, output_dir=output, ablation=spec)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert evaluate_checkpoint(config, output / "best.pt", ablation=spec) == metrics
    resumed = tmp_path / (variant + "_resumed")
    run_training(config, output_dir=resumed, ablation=spec,
                 resume_state_path=output / "training_states/epoch_001.pt")
    complete = torch.load(output / "training_state.pt", weights_only=False)
    restored = torch.load(resumed / "training_state.pt", weights_only=False)
    for field in ("model_state", "optimizer_state", "controller_state", "best_metrics", "best_checkpoint",
                  "torch_rng_state", "order_generator_state", "sampling_generator_state"):
        assert_nested_equal(complete[field], restored[field])
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    for foreign in (LocalPoolingSpec("S0"), *(EuclideanPoolingSpec(v) for v in VARIANTS if v != variant)):
        with pytest.raises(ValueError, match="ablation contract"):
            _validate_checkpoint_contract(checkpoint, config, ablation=foreign)
