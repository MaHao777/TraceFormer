"""四组整体注意力消融的数学定义、公共初始化与训练合同。"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import test_paper_ablation
import torch

from traceformer.config import ConeWindowConfig, ModelConfig, TraceFormerConfig
from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.attention import SlidingLocalAttention
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.suite import (
    center_attention_stage_jobs,
    prepare_center_attention_suite,
    summarize,
)
from traceformer.models.prefix_cone import PrefixConeAttention
from traceformer.training.engine import (
    _dataset,
    _validate_checkpoint_contract,
    build_model,
    run_training,
)

small_config = test_paper_ablation.small_config


def enumerate_local(module, tokens, coords, batches):
    # 参考实现逐中心枚举真实邻居，再直接调用标准MHA，不复用网格查找。
    output = []
    window = module.windows[0]
    for index, (center, batch) in enumerate(zip(coords, batches, strict=True)):
        delta = coords - center
        mask = ((batches == batch) & (delta[:, 2].abs() >= 1)
                & (delta[:, 2].abs() <= window.time_horizon)
                & (delta[:, :2].abs().amax(-1) <= window.spatial_radius))
        if not mask.any():
            output.append(tokens[index] * 0)
            continue
        directions = (delta[mask, 2] > 0).long()
        neighbors = (tokens[mask] + module.shape_embedding
                     + module.direction_embedding[directions])[None]
        result, _ = module.center_attention(tokens[index][None, None], neighbors, neighbors,
                                             need_weights=False)
        output.append(result[0, 0])
    return torch.stack(output)


@pytest.mark.parametrize("chunk_size", [2, 13])
def test_local_values_and_gradients_match_direct_mha(chunk_size):
    torch.manual_seed(9)
    original = PrefixConeAttention(8, 2, (ConeWindowConfig(2, 2),), 0.0)
    actual = SlidingLocalAttention(original, chunk_size=chunk_size)
    reference = copy.deepcopy(actual)
    coords = torch.tensor([[0, 0, 0], [2, 2, 1], [3, 0, 1], [0, 0, 2], [0, 0, 0],
                           [-1, -1, -1], [1, 0, 0], [30, 30, 30]])
    batches = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
    tokens = torch.randn(8, 8, requires_grad=True)
    other_tokens = tokens.detach().clone().requires_grad_()
    output = actual(tokens, coords, batches).features
    expected = enumerate_local(reference, other_tokens, coords, batches)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    assert output[[4, 7]].eq(0).all()
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, other_tokens.grad, atol=3e-6, rtol=3e-5)
    for (name, parameter), (_, other) in zip(actual.named_parameters(),
                                            reference.named_parameters(), strict=True):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=3e-5)
    actual.eval()
    with torch.no_grad():
        torch.testing.assert_close(actual(tokens, coords, batches).features, expected)


def test_local_reads_all_neighbors_without_a_cap():
    original = PrefixConeAttention(8, 2, (ConeWindowConfig(4, 4),), 0.0)
    module = SlidingLocalAttention(original).eval()
    assert module.offsets.shape == (648, 4)
    coords = torch.tensor([[0, 0, 0]] + [[x, y, 1] for y in range(-4, 5) for x in range(-4, 5)])
    tokens = torch.randn(len(coords), 8, requires_grad=True)
    module(tokens, coords, torch.zeros(len(coords), dtype=torch.long)).features[0].sum().backward()
    assert (tokens.grad[1:].abs().sum(-1) > 0).all()


def test_local_recomputation_preserves_dropout_gradients_and_rng(monkeypatch):
    torch.manual_seed(19)
    original = PrefixConeAttention(8, 2, (ConeWindowConfig(2, 2),), 0.25)
    actual = SlidingLocalAttention(original, chunk_size=3)
    reference = copy.deepcopy(actual)
    coords = torch.tensor([[index % 3, 0, index // 3] for index in range(12)])
    batches = torch.zeros(12, dtype=torch.long)
    tokens = torch.randn(12, 8, requires_grad=True)
    other = tokens.detach().clone().requires_grad_()
    probe = torch.randn_like(tokens)
    torch.manual_seed(37)
    result = actual(tokens, coords, batches).features
    after_forward = torch.get_rng_state()
    (result * probe).sum().backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    monkeypatch.setattr("traceformer.experiments.attention.checkpoint",
                        lambda function, *args, **kwargs: function(*args))
    torch.manual_seed(37)
    expected = reference(other, coords, batches).features
    (expected * probe).sum().backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(tokens.grad, other.grad, atol=1e-6, rtol=1e-5)
    for (name, parameter), (_, second) in zip(actual.named_parameters(),
                                             reference.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, second.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_four_variants_preserve_common_initial_weights_and_rng():
    models, states = [], []
    for job in center_attention_stage_jobs():
        torch.manual_seed(37)
        models.append(build_model(TraceFormerConfig.default(), ablation=job.spec))
        states.append(torch.get_rng_state())
    assert all(torch.equal(states[0], state) for state in states[1:])
    reference = models[-1].state_dict()
    for model in models[:3]:
        for name, value in model.state_dict().items():
            expected = reference[name][1:2] if name.endswith("shape_embedding") else reference[name]
            assert torch.equal(value, expected), name
        assert model.encoder.layers[0].attention.windows == (ConeWindowConfig(4, 4),)
    assert [sum(p.numel() for p in m.parameters()) for m in models] == [150913, 163588, 163588, 163972]


@pytest.mark.parametrize("job", center_attention_stage_jobs(), ids=lambda job: job.result_id)
def test_four_variants_train_resume_and_reject_wrong_contract(small_config, tmp_path, job):
    config = replace(small_config, training=replace(
        small_config.training, stage="full", epochs=2, num_threads=1, save_training_state_interval=1,
    ))
    run = tmp_path / job.result_id
    run_training(config, output_dir=run, ablation=job.spec)
    report = evaluate_ablation(config, run / "best.pt", job.spec, save_predictions=True,
                               neighbor_diagnostics=True)
    assert report["metrics"] == json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    saved = torch.load(run / "best.pt", weights_only=True)
    for wrong in [None, *(other.spec for other in center_attention_stage_jobs() if other != job)]:
        with pytest.raises(ValueError, match="ablation"):
            _validate_checkpoint_contract(saved, config, ablation=wrong)
    resumed = tmp_path / f"{job.result_id}_resumed"
    run_training(config, output_dir=resumed, ablation=job.spec,
                 resume_state_path=run / "training_states/epoch_001.pt")
    left = torch.load(run / "training_state.pt", weights_only=False)
    right = torch.load(resumed / "training_state.pt", weights_only=False)
    assert left["best_metrics"] == right["best_metrics"]
    assert all(torch.equal(value, right["model_state"][key]) for key, value in left["model_state"].items())
    sequence = _dataset(config, "val")[0]
    model = build_model(config, ablation=job.spec)
    assert model(sequence.events[:0])["event_logits"].shape == (0,)


def test_suite_locks_geometry_and_avoids_factorial_interaction(small_config, tmp_path):
    root = tmp_path / "suite"
    suite = prepare_center_attention_suite(replace(small_config, model=ModelConfig()), root, [37])
    assert suite["protocol"] == "center_attention_four_variants"
    assert suite["comparisons"] == [["A3", "A0"], ["A1", "A0"], ["A2", "A1"], ["A3", "A2"]]
    assert suite["attention_protocol"]["neighbor_cap"] is None
    for job in suite["jobs"]:
        folder = root / job["id"]
        folder.mkdir(parents=True)
        report = folder / "report.json"
        report.write_text(json.dumps({"metrics": {"threshold": 0.5, "competition": dict.fromkeys(
            ("score", "iou", "seg_acc", "pd", "fa"), 0.1)}, "parameters": 1,
            "mean_seconds_per_sequence": 1.0, "process_peak_rss_bytes": 1024,
            "fixed_threshold": {"competition": {"score": 0.1}}, "groups": {}}), encoding="utf-8")
        (folder / "done.json").write_text(json.dumps({"report": str(report)}), encoding="utf-8")
    summarize(root)
    summary = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    assert len(summary["paired"]) == 4


def test_nondefault_backbone_cannot_enable_compensating_modules():
    for backbone in ("local", "single_box", "single_cone"):
        with pytest.raises(ValueError, match="without Sparse/refinement"):
            AblationSpec(backbone=backbone).validate()
