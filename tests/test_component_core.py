"""独立验证新消融的真实依赖、邻接定义、训练恢复与前置门槛。"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import test_paper_ablation
import torch

from traceformer.config import DEFAULT_CONE_WINDOWS, TraceFormerConfig
from traceformer.experiments.ablation import ComponentCoreSpec, parse_ablation
from traceformer.experiments.attention import matched_box_windows
from traceformer.experiments.component_core import GridLocalAttention, assert_no_view
from traceformer.experiments.component_suite import (
    VARIANTS,
    audit_baseline,
    choose_local_neighbors,
    read_json,
    window_diagnostics,
)
from traceformer.experiments.report import evaluate_ablation
from traceformer.models.prefix_cone import (
    PrefixConeAttention,
    prefix_cone_pool,
    prepare_prefix_geometry,
)
from traceformer.training.engine import (
    _dataset,
    _validate_checkpoint_contract,
    build_model,
    run_training,
)

small_config = test_paper_ablation.small_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def events():
    # 填充相邻的3x3x3个Token，确保测试真的经过邻域交互路径。
    return torch.tensor([[8.0 * x, 8.0 * y, (t + 0.2) / 64, (-1.0) ** (x + y + t)]
                         for x, y, t in itertools.product(range(3), range(3), range(5, 8))])


def test_contract_has_no_view_switch_and_rejects_foreign_fields():
    for variant in VARIANTS:
        spec = ComponentCoreSpec(variant)
        assert parse_ablation(spec.to_dict()) == spec
        assert "sparse" not in spec.to_dict()
        with pytest.raises(ValueError):
            spec.validate_stage("refinement")
    with pytest.raises(TypeError):
        parse_ablation({"version": 2, "variant": "Full", "sparse": True})
    with pytest.raises(ValueError):
        ComponentCoreSpec("M1", local_neighbors=6).validate()


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_preserve_mapping_gradients_and_common_initialization(variant):
    config = TraceFormerConfig.default()
    torch.manual_seed(37)
    full = build_model(config).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    model = build_model(config, ablation=ComponentCoreSpec(variant)).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_no_view(model)
    assert not hasattr(model, "sparse_mixture")
    reference = full.state_dict()
    for name, value in model.state_dict().items():
        # M5保留原主尺度分支，索引由2变为0，不能误比较最细尺度分支。
        source_name = (name.replace("cell_encoders.0.", "cell_encoders.2.")
                       .replace("temporal_fusers.0.", "temporal_fusers.2.") if variant == "M5" else name)
        if source_name in reference and value.shape == reference[source_name].shape:
            assert torch.equal(value, reference[source_name]), name
    output, baseline = model(events()), full(events())
    assert torch.equal(output["event_token_indices"], baseline["event_token_indices"])
    assert torch.equal(output["token_coords"], baseline["token_coords"])
    assert output["event_logits"].shape == (27,)
    assert model(events()[:0])["event_logits"].shape == (0,)
    if variant == "Full":
        for key in output:
            assert torch.equal(output[key], baseline[key]), key
    output["event_logits"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_component_definitions_are_structural():
    models = {v: build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec(v))
              for v in ("M1", "M2", "M3", "M4", "M5", "C3", "C4", "C5")}
    assert not hasattr(models["M1"].tokenizer, "pool_queries")
    assert not hasattr(models["M1"].tokenizer, "query_fusion")
    assert models["M2"].tokenizer.token_projection[0].linear.in_features == 128
    assert models["M4"].decoder.classifier[0].linear.in_features == 64
    assert models["M4"].event_refinement_head.event_projection[0].linear.in_features == 65
    assert len(models["M5"].event_refinement_head.scales) == 1
    assert models["M5"].event_refinement_head.scales[0].frame_ms == 50.0
    assert models["M5"].event_refinement_head.event_classifier[0].in_features == 34
    assert all(len(layer.attention.windows) == 1 for layer in models["C3"].encoder.layers)
    assert all(layer.attention.directions == (-1,) for layer in models["C4"].encoder.layers)
    assert all(not hasattr(layer.attention, "center_attention") for layer in models["C5"].encoder.layers)
    assert not any("attention" in name for name, _ in models["M3"].encoder.named_parameters())


def test_mean_pooling_is_exact_uncompensated_mean():
    tokenizer = build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec("M1")).tokenizer
    local = torch.randn(7, 6)
    inverse = torch.tensor([0, 1, 0, 2, 2, 0, 1])
    counts = torch.tensor([3, 2, 2])
    encoded = tokenizer.event_encoder(local)
    actual = tokenizer._encode_block_features(local, inverse, counts, torch.randn(3, 7))
    expected = torch.stack([encoded[inverse == i].mean(0) for i in range(3)])
    torch.testing.assert_close(actual, expected)


def test_m3_has_no_cross_token_dependency_or_geometry():
    encoder = build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec("M3")).encoder.eval()
    tokens, coords, batches = torch.randn(27, 64), events()[:, :3].long(), torch.zeros(27).long()
    with patch("traceformer.models.prefix_cone.prepare_prefix_geometry", side_effect=AssertionError):
        before = encoder(tokens, coords, batches).token_features
        tokens[0] += torch.randn(64)
        after = encoder(tokens, coords, batches).token_features
    assert torch.equal(before[1:], after[1:])
    assert not torch.equal(before[0], after[0])


def test_m4_cuts_both_local_feature_inputs_after_nonzero_head():
    model = build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec("M4")).eval()
    # 精修零初始化不能掩盖残留旁路，所以显式令出口非零。
    torch.nn.init.normal_(model.event_refinement_head.event_classifier[-1].weight)
    tokenized = model.tokenizer(events())
    changed = replace(tokenized, event_local_features=tokenized.event_local_features + torch.randn(27, 6))
    with patch.object(model.tokenizer, "forward", return_value=tokenized):
        before = model(events())
    with patch.object(model.tokenizer, "forward", return_value=changed):
        after = model(events())
    assert torch.equal(before["event_logits"], after["event_logits"])
    assert torch.equal(before["base_event_logits"], after["base_event_logits"])


@pytest.mark.parametrize("variant", ["C4", "C5"])
def test_summary_empty_neighbor_is_exact_zero(variant):
    attention = build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec(variant)).encoder.layers[0].attention
    result = attention(torch.randn(1, 64), torch.zeros(1, 3).long(), torch.zeros(1).long()).features
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.isfinite(result).all()


def test_past_pool_values_and_gradients_match_bidirectional_past_slots():
    torch.manual_seed(19)
    coords = torch.tensor(list(itertools.product(range(3), range(2), range(4))))
    batches = torch.zeros(len(coords)).long()
    values = torch.randn(len(coords), 8, requires_grad=True)
    weights = torch.rand(len(coords), 1, requires_grad=True)
    geometry = prepare_prefix_geometry(coords, batches)
    both, counts = prefix_cone_pool(values, weights, geometry, DEFAULT_CONE_WINDOWS)
    past, past_counts = prefix_cone_pool(values, weights, geometry, DEFAULT_CONE_WINDOWS, directions=(-1,))
    assert torch.equal(past_counts, counts[:, ::2])
    torch.testing.assert_close(past, both[:, ::2], rtol=0, atol=0)
    gradients = [torch.autograd.grad(value.sum(), (values, weights), retain_graph=True)
                 for value in (past, both[:, ::2])]
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_c4_encoder_does_not_read_future_tokens():
    encoder = build_model(TraceFormerConfig.default(), ablation=ComponentCoreSpec("C4")).encoder.eval()
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    tokens, batches = torch.randn(4, 64), torch.zeros(4).long()
    before = encoder(tokens, coords, batches).token_features
    tokens[-1] += torch.randn(64)
    after = encoder(tokens, coords, batches).token_features
    assert torch.equal(before[:-1], after[:-1])


@pytest.mark.parametrize("neighbors", [6, 26])
def test_local_attention_matches_independent_mha_values_and_gradients(neighbors):
    torch.manual_seed(44)
    actual = GridLocalAttention(PrefixConeAttention(8, 2, DEFAULT_CONE_WINDOWS, 0), neighbors, 3)
    reference = copy.deepcopy(actual)
    coords = torch.tensor(list(itertools.product(range(3), range(3), range(3))) + [(10, 10, 10), (0, 0, 0)])
    batches = torch.tensor([0] * 28 + [1])
    tokens = torch.randn(len(coords), 8, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    observed = actual(tokens, coords, batches).features
    expected = []
    for i, center in enumerate(coords):
        delta = (coords - center).abs()
        mask = (batches == batches[i]) & (delta.amax(1) <= 1) & (delta.sum(1) > 0)
        if neighbors == 6:
            mask &= delta.sum(1) == 1
        if mask.any():
            result, _ = reference.center_attention(other[i][None, None], other[mask][None],
                                                    other[mask][None], need_weights=False)
            expected.append(result[0, 0])
        else:
            expected.append(other[i] * 0)
    expected = torch.stack(expected)
    torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-6)
    probe = torch.randn_like(observed)
    (observed * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, other.grad, atol=2e-6, rtol=2e-6)
    for left, right in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=4e-6)


def test_c2_diagnostics_match_brute_force_membership():
    coords = torch.tensor(list(itertools.product(range(3), range(2), range(5))))
    batches = torch.arange(len(coords)) % 2
    rows = window_diagnostics(coords, batches, DEFAULT_CONE_WINDOWS)
    boxes = matched_box_windows(DEFAULT_CONE_WINDOWS)
    for index, row in enumerate(rows):
        cone, box = DEFAULT_CONE_WINDOWS[index // 2], boxes[index // 2]
        sums = dict(cone=0, box=0, intersection=0, union=0)
        for center, batch in zip(coords, batches, strict=True):
            delta = coords - center
            dt = delta[:, 2] * (-1 if index % 2 == 0 else 1)
            valid = (batches == batch) & (dt >= 1) & (dt <= cone.time_horizon)
            distance = delta[:, :2].abs().amax(1)
            a = valid & (distance <= cone.spatial_radius * dt // cone.time_horizon)
            b = valid & (distance <= box.spatial_radius)
            for name, mask in (("cone", a), ("box", b), ("intersection", a & b), ("union", a | b)):
                sums[name] += int(mask.sum())
        assert all(row[k] == v for k, v in sums.items())


@pytest.mark.parametrize("neighbors", [6, 26])
def test_local_recomputation_preserves_dropout_rng_and_gradients(neighbors):
    actual = GridLocalAttention(PrefixConeAttention(8, 2, DEFAULT_CONE_WINDOWS, 0.2), neighbors, 3)
    reference = copy.deepcopy(actual)
    coords = torch.tensor(list(itertools.product(range(2), range(2), range(3))))
    batches = torch.zeros(len(coords)).long()
    tokens = torch.randn(len(coords), 8, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    torch.manual_seed(90)
    observed = actual(tokens, coords, batches).features
    observed.square().sum().backward()
    rng = torch.get_rng_state()
    torch.manual_seed(90)
    with patch("traceformer.experiments.component_core.checkpoint", side_effect=lambda fn, *a, **kw: fn(*a)):
        expected = reference(other, coords, batches).features
        expected.square().sum().backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(tokens.grad, other.grad)
    for left, right in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)


def test_baseline_waits_and_refuses_failure_or_incomplete_epochs(tmp_path):
    (tmp_path / "status.json").write_text('{"status":"running"}', encoding="utf-8")
    assert audit_baseline(tmp_path) is None
    (tmp_path / "status.json").write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="baseline failed"):
        audit_baseline(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
    (tmp_path / "training").mkdir()
    (tmp_path / "training/history.json").write_text('[{"epoch":1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="24 completed"):
        audit_baseline(tmp_path)


def test_local_budget_uses_both_files_and_no_accuracy():
    full = [{"sequence": name, "median_seconds": 10} for name in ("094", "096")]
    local = [{"sequence": name, "median_seconds": 19, "memory_passed": True,
              "finite_gradients": True} for name in ("094", "096")]
    assert choose_local_neighbors(full, local)
    local[1]["median_seconds"] = 21
    assert not choose_local_neighbors(full, local)
    local[1]["median_seconds"] = 19
    local[1]["memory_passed"] = False
    assert not choose_local_neighbors(full, local)
    with pytest.raises(ValueError):
        choose_local_neighbors(full[:1], local)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_each_variant_trains_recovers_and_independently_reproduces(small_config, tmp_path, variant):
    spec = ComponentCoreSpec(variant)
    config = replace(small_config, training=replace(small_config.training, save_training_state_interval=1))
    output = tmp_path / variant
    run_training(config, output_dir=output, ablation=spec)
    report = evaluate_ablation(config, output / "best.pt", spec, save_predictions=False)
    assert report["metrics"] == read_json(output / "metrics.json")
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    with pytest.raises(ValueError, match="ablation contract"):
        _validate_checkpoint_contract(checkpoint, config)
    # 恢复已完成的一轮不重训，也必须保持检查点和指标一致。
    resumed = tmp_path / f"{variant}_resumed"
    run_training(config, output_dir=resumed, ablation=spec,
                 resume_state_path=output / "training_state.pt")
    assert read_json(resumed / "metrics.json") == report["metrics"]
    model = build_model(config, ablation=spec)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    assert model(_dataset(config, "val")[0].events)["event_logits"].ndim == 1


def test_dispatch_only_records_waiting_without_starting_training(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "archive/experiments/scripts"
    sys.path.insert(0, str(scripts))
    try:
        loader = importlib.util.spec_from_file_location("component_runner_test", scripts / "run_component_core.py")
        runner = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(runner)
        baseline, root = tmp_path / "baseline", tmp_path / "suite"
        baseline.mkdir()
        (baseline / "status.json").write_text('{"status":"running"}', encoding="utf-8")
        with patch.object(sys, "argv", ["runner", "--root", str(root), "--baseline", str(baseline)]), \
             patch.object(runner.subprocess, "Popen", side_effect=AssertionError("must not train")):
            runner.main()
        assert json.loads((root / "status.json").read_text())["status"] == "waiting_for_baseline"
        assert runner.process_alive(None, root) is False
    finally:
        sys.path.remove(str(scripts))
