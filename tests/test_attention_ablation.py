"""独立验证三组消融的几何、容量、初始化与训练恢复契约。"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import test_paper_ablation
import torch

from traceformer.config import DEFAULT_CONE_WINDOWS, TraceFormerConfig, load_config
from traceformer.experiments.attention import matched_box_windows, neighbor_count_summary
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.suite import attention_stage_jobs, prepare_attention_suite
from traceformer.models.prefix_cone import prefix_cone_pool, prepare_prefix_geometry
from traceformer.training.engine import (
    _dataset,
    _validate_checkpoint_contract,
    build_model,
    run_training,
)

small_config = test_paper_ablation.small_config


def enumerate_box(values, weights, coords, batches, windows):
    # 参考实现只枚举定义中的成员，不调用矩形索引或前缀求和。
    rows, counts = [], []
    for center, batch in zip(coords, batches):
        delta = coords - center
        row, count = [], []
        for window in windows:
            for direction in (-1, 1):
                dt = direction * delta[:, 2]
                mask = ((batches == batch) & (dt >= 1) & (dt <= window.time_horizon)
                        & (delta[:, :2].abs().amax(1) <= window.spatial_radius))
                w = weights[mask]
                row.append((values[mask] * w).sum(0) / w.sum().clamp_min(1e-8))
                count.append(mask.sum())
        rows.append(torch.stack(row))
        counts.append(torch.stack(count))
    return torch.stack(rows), torch.stack(counts)


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_box_matches_independent_values_counts_and_gradients(chunk_size):
    torch.manual_seed(71)
    coords = torch.randint(-5, 10, (60, 3))
    batches = torch.arange(60) % 2
    values = torch.randn(60, 8, requires_grad=True)
    weights = (torch.rand(60, 1) + 0.1).requires_grad_()
    windows = matched_box_windows(DEFAULT_CONE_WINDOWS)
    assert [(w.spatial_radius, w.time_horizon) for w in windows] == [(6, 2), (3, 4), (1, 8)]
    actual, counts = prefix_cone_pool(
        values, weights, prepare_prefix_geometry(coords, batches), windows, chunk_size,
        fixed_radius=True,
    )
    expected, expected_counts = enumerate_box(values, weights, coords, batches, windows)
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    probe = torch.randn_like(actual)
    observed = torch.autograd.grad((actual * probe).sum(), (values, weights), retain_graph=True)
    reference = torch.autograd.grad((expected * probe).sum(), (values, weights))
    for left, right in zip(observed, reference):
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)


def test_box_exact_boundaries_batches_and_same_time_exclusion():
    coords = torch.tensor([[0, 0, 0], [6, 6, 1], [7, 0, 1], [1, 1, 0],
                           [0, 0, -2], [0, 0, 3], [0, 0, 1]])
    batches = torch.tensor([0, 0, 0, 0, 0, 0, 1])
    windows = matched_box_windows(DEFAULT_CONE_WINDOWS)
    values = torch.randn(7, 2, requires_grad=True)
    _, counts = prefix_cone_pool(values, torch.ones(7, 1),
                                prepare_prefix_geometry(coords, batches), windows, fixed_radius=True)
    assert counts[0, :2].tolist() == [1, 1]
    stats = neighbor_count_summary(counts, windows)
    assert len(stats) == 6 and stats[0]["direction"] == "past"


def test_registered_parameters_and_common_initialization():
    config = TraceFormerConfig.default()
    models, rng = [], []
    for job in attention_stage_jobs():
        torch.manual_seed(37)
        models.append(build_model(config, ablation=job.spec).eval())
        rng.append(torch.get_rng_state())
    assert [sum(p.numel() for p in model.parameters()) for model in models] == [164077, 163972, 163972]
    assert all(torch.equal(rng[0], state) for state in rng[1:])
    reference = models[2].state_dict()
    for model in models[:2]:
        for name, value in model.state_dict().items():
            if name in reference:
                assert torch.equal(value, reference[name]), name
    assert not any("pool_weight" in name or "center_attention" in name for name in models[0].state_dict())


def test_pointwise_has_no_cross_token_dependency_or_geometry():
    model = build_model(TraceFormerConfig.default(), ablation=attention_stage_jobs()[0].spec).eval()
    tokens = torch.randn(12, 64)
    coords, batches = torch.randint(0, 5, (12, 3)), torch.arange(12) % 2
    with patch("traceformer.models.prefix_cone.prepare_prefix_geometry", side_effect=AssertionError):
        baseline = model.encoder(tokens, coords, batches).token_features
        changed = tokens.clone()
        changed[0] += torch.randn(64) * 10
        altered = model.encoder(changed, coords, batches).token_features
    assert torch.equal(baseline[1:], altered[1:])
    assert not torch.equal(baseline[0], altered[0])


def test_cone_base_matches_production_full_path_exactly(small_config):
    torch.manual_seed(37)
    production = build_model(small_config).eval()
    torch.manual_seed(37)
    experiment = build_model(small_config, ablation=attention_stage_jobs()[2].spec).eval()
    events = _dataset(small_config, "train")[0].events
    assert torch.equal(production(events)["full_event_logits"], experiment(events)["event_logits"])


@pytest.mark.parametrize("job", attention_stage_jobs(), ids=lambda job: job.name)
def test_event_mapping_empty_outputs_and_gradients(small_config, job):
    model = build_model(small_config, ablation=job.spec)
    events = _dataset(small_config, "train")[0].events
    output = model(events)
    assert output["event_logits"].shape == (len(events),)
    assert torch.equal(output["event_logits"], output["full_event_logits"])
    output["event_logits"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model(events[:0])["event_logits"].shape == (0,)
    torch.manual_seed(37)
    reference = build_model(small_config, ablation=attention_stage_jobs()[2].spec)(events)
    assert torch.equal(output["event_token_indices"], reference["event_token_indices"])


def test_attention_suite_has_three_random_starts_and_24_epochs(small_config, tmp_path):
    config = replace(load_config("configs/traceformer_prefix_cone.yaml"), dataset=small_config.dataset)
    manifest = prepare_attention_suite(config, tmp_path / "suite", [37])
    assert manifest["comparisons"] == [["C1", "C0"], ["C2", "C1"], ["C2", "C0"]]
    assert manifest["dataset_manifest_sha256"]
    for job in manifest["jobs"]:
        saved = load_config(job["config"])
        assert job["parent"] is None and job["seed"] == 37
        assert saved.training.epochs == 24 and saved.training.save_training_state_interval == 1
        assert saved.training.num_threads == 8


@pytest.mark.parametrize("job", attention_stage_jobs(), ids=lambda job: job.name)
def test_train_resume_independent_evaluation_and_contract(small_config, tmp_path, job):
    config = replace(small_config, training=replace(
        small_config.training, stage="full", epochs=2, num_threads=1, save_training_state_interval=1,
    ))
    directory = tmp_path / job.name
    run_training(config, output_dir=directory, ablation=job.spec)
    report = evaluate_ablation(config, directory / "best.pt", job.spec, save_predictions=True,
                               neighbor_diagnostics=True)
    assert report["metrics"] == json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert bool(report["neighbor_counts"]) == (job.spec.backbone != "pointwise")
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    for other in [None, *(j.spec for j in attention_stage_jobs() if j != job)]:
        with pytest.raises(ValueError, match="ablation"):
            _validate_checkpoint_contract(checkpoint, config, ablation=other)
    restored = tmp_path / f"{job.name}_resumed"
    run_training(config, output_dir=restored, ablation=job.spec,
                 resume_state_path=directory / "training_states/epoch_001.pt")
    full = torch.load(directory / "training_state.pt", weights_only=False)
    resumed = torch.load(restored / "training_state.pt", weights_only=False)
    for name in full["model_state"]:
        assert torch.equal(full["model_state"][name], resumed["model_state"][name]), name
