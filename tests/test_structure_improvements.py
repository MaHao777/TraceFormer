"""验证真实结构、方向依赖、配对初始化和训练恢复契约。"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import test_paper_ablation
import torch

from traceformer.config import TraceFormerConfig
from traceformer.experiments.ablation import StructureImprovementSpec, parse_ablation
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.structure import describe_structure
from traceformer.models.prefix_cone import prefix_cone_pool, prepare_prefix_geometry
from traceformer.training.engine import _validate_checkpoint_contract, build_model, run_training

small_config = test_paper_ablation.small_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs():
    # 每块至少两个事件，避免均值和单Query在单成员块上退化。
    return torch.tensor([[x + dx, y, (t + dt) / 64, polarity]
                         for x, y, t in ((0, 0, 3), (8, 8, 4), (0, 0, 5), (8, 0, 6))
                         for dx, dt, polarity in ((0.2, 0.1, 1.0), (0.8, 0.8, -1.0))])


def test_contract_rejects_unknown_variants_fields_and_stages():
    for variant in ("00", "10", "01", "11"):
        spec = StructureImprovementSpec(variant)
        assert parse_ablation(spec.to_dict()) == spec
        with pytest.raises(ValueError):
            spec.validate_stage("refinement")
    with pytest.raises(ValueError):
        StructureImprovementSpec("A").validate()
    with pytest.raises(TypeError):
        parse_ablation({"version": 3, "variant": "10", "query_fraction": 0.5})


@pytest.mark.parametrize("variant", ("00", "10", "01", "11"))
def test_structure_common_initialization_rng_mappings_and_gradients(variant):
    config = TraceFormerConfig.default()
    torch.manual_seed(37)
    baseline = build_model(config).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    model = build_model(config, ablation=StructureImprovementSpec(variant)).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    original = baseline.state_dict()
    for name, value in model.state_dict().items():
        if name in original and value.shape == original[name].shape:
            assert torch.equal(value, original[name]), name
    structure = describe_structure(model)
    assert structure["queries"] == (1 if variant[0] == "1" else 4)
    assert structure["layers"] == (6 if variant[1] == "1" else 3)
    assert len(set(tuple(map(tuple, windows)) for windows in structure["windows_per_layer"])) == 1
    if variant[1] == "1":
        assert structure["directions"] == [-1]
        assert not any("pool_weight" in name for name in model.state_dict())
    if variant[0] == "1":
        assert model.tokenizer.token_projection[0].in_features == 128
        assert not hasattr(model.tokenizer, "query_fusion")
    observed, reference = model(inputs()), baseline(inputs())
    for key in ("event_token_indices", "token_coords", "token_batch_indices"):
        assert torch.equal(observed[key], reference[key])
    if variant == "00":
        for key in observed:
            assert torch.equal(observed[key], reference[key])
    assert model(inputs()[:0])["event_logits"].shape == (0,)
    observed["event_logits"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_a_exact_mean_statistics_and_all_event_single_query_formula():
    tokenizer = build_model(TraceFormerConfig.default(),
                            ablation=StructureImprovementSpec("10")).tokenizer
    local = torch.randn(7, 6)
    inverse = torch.tensor([0, 1, 0, 2, 2, 0, 1])
    counts, statistics = torch.tensor([3, 2, 2]), torch.randn(3, 7)
    encoded = tokenizer.event_encoder(local)
    means = torch.stack([encoded[inverse == i].mean(0) for i in range(3)])
    keys, values = tokenizer.key_projection(encoded), tokenizer.value_projection(encoded)
    query = tokenizer.pool_queries[0]
    pooled = torch.stack([
        ((keys[inverse == i] @ query / 8).softmax(0)[:, None] * values[inverse == i]).sum(0)
        for i in range(3)])
    expected = 0.75 * (means + tokenizer.statistics_projection(statistics)) + 0.25 * pooled
    actual = tokenizer._encode_block_features(local, inverse, counts, statistics)
    torch.testing.assert_close(actual, expected)
    positions = torch.randn(3, 64)
    # 最终Token投影忽略统计参数，统计只在主支路出现一次。
    assert torch.equal(tokenizer._project_token(actual, statistics, positions),
                       tokenizer._project_token(actual, statistics + 100, positions))


def test_b_past_uniform_summaries_match_brute_force_and_no_future_dependency():
    model = build_model(TraceFormerConfig.default(), ablation=StructureImprovementSpec("01")).eval()
    coords = torch.tensor([[0, 0, t] for t in range(10)] + [[0, 0, 4]])
    batches = torch.tensor([0] * 10 + [1])
    values = torch.randn(11, 64, requires_grad=True)
    geometry = prepare_prefix_geometry(coords, batches)
    windows = model.encoder.layers[0].attention.windows
    summaries, _ = prefix_cone_pool(values, torch.ones(11, 1), geometry, windows, directions=(-1,))
    for i in range(11):
        for j, window in enumerate(windows):
            age = coords[i, 2] - coords[:, 2]
            selected = (batches == batches[i]) & (age > 0) & (age <= window.time_horizon)
            expected = values[selected].mean(0) if selected.any() else values[i] * 0
            torch.testing.assert_close(summaries[i, j], expected, atol=3e-6, rtol=3e-6)
    before = model.encoder(values, coords, batches).token_features
    changed = values.detach().clone()
    changed[6:10] += torch.randn_like(changed[6:10]) * 100
    after = model.encoder(changed, coords, batches).token_features
    torch.testing.assert_close(before[:6], after[:6], atol=0, rtol=0)
    gradient = torch.autograd.grad(before[5].square().sum(), values)[0]
    assert torch.count_nonzero(gradient[6:]) == 0


def test_a_and_b_replacements_have_identical_initialization_in_combination():
    models = {}
    for variant in ("10", "01", "11"):
        torch.manual_seed(37)
        models[variant] = build_model(TraceFormerConfig.default(),
                                      ablation=StructureImprovementSpec(variant))
    for branch, variant in (("tokenizer", "10"), ("encoder", "01")):
        for key, value in getattr(models[variant], branch).state_dict().items():
            assert torch.equal(value, getattr(models["11"], branch).state_dict()[key]), key


@pytest.mark.parametrize("variant", ("10", "01", "11"))
def test_each_improvement_trains_resumes_and_reproduces(small_config, tmp_path, variant):
    # 测试数据仍用小图，但结构模板必须与真实实验同为3层/4Query。
    config = replace(small_config, model=replace(small_config.model, num_layers=3, num_pool_queries=4),
                     training=replace(small_config.training, save_training_state_interval=1))
    spec = StructureImprovementSpec(variant)
    folder = tmp_path / variant
    run_training(config, output_dir=folder, ablation=spec)
    saved = torch.load(folder / "best.pt", weights_only=True)
    with pytest.raises(ValueError, match="ablation contract"):
        _validate_checkpoint_contract(saved, config, ablation=StructureImprovementSpec("00"))
    report = evaluate_ablation(config, folder / "best.pt", spec, save_predictions=True)
    assert report["metrics"] == json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    resumed = tmp_path / (variant + "_resume")
    run_training(config, output_dir=resumed, ablation=spec,
                 resume_state_path=folder / "training_state.pt")
    again = torch.load(resumed / "best.pt", weights_only=True)
    assert all(torch.equal(value, again["model_state"][key]) for key, value in saved["model_state"].items())


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1] / "archive/experiments/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("structure_runner_test", scripts / "run_structure_improvements.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def test_runner_never_trains_00_and_finishes_both_partitions(runner, tmp_path):
    # 用已有done模拟训练已完成，恢复入口只做预测/审计，不能再次调用训练。
    root = tmp_path / "suite"
    root.mkdir()
    full = {"config": "baseline.yaml", "checkpoint": "baseline.pt"}
    jobs = [{"id": f"seed37/{v}", "name": v, "config": f"{v}.yaml"} for v in ("10", "01", "11")]
    suite = {"jobs": jobs, "baseline": full}
    (root / "suite.json").write_text(json.dumps(suite), encoding="utf-8")
    (root / "source_hashes.json").write_text('{}', encoding="utf-8")
    (root / "resource_gate.json").write_text('{"passed":true}', encoding="utf-8")
    calls = []
    with (patch.object(runner, "audit_baseline", return_value=full),
          patch.object(runner, "verify_checks"), patch.object(runner, "validate_suite"),
          patch.object(runner, "execute", side_effect=AssertionError("must not retrain completed jobs")),
          patch.object(runner, "completed_job", side_effect=lambda r, i: {"checkpoint": str(root / i / "best.pt")}),
          patch.object(runner, "finish", side_effect=lambda r, s, v, *args: calls.append(v))):
        runner.run(root, tmp_path / "baseline")
    assert calls == ["00", "10", "01", "11"]
    assert runner.read_json(root / "completion_audit.json")["baseline_retrained"] is False
    assert runner.read_json(root / "status.json")["status"] == "complete"


def test_runner_failure_preserves_failed_status_and_stops(runner, tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    with patch.object(runner, "audit_baseline", side_effect=ValueError("source identity changed")):
        with pytest.raises(ValueError, match="source identity"):
            runner.run(root, tmp_path / "baseline")
    status = runner.read_json(root / "status.json")
    assert status["status"] == "failed"
    assert "source identity changed" in status["error"]
