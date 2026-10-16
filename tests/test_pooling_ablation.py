from __future__ import annotations

import json
from dataclasses import replace

import pytest
import test_paper_ablation
import torch
from test_event_tokenizer import _tokenizer
from torch import nn

from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.pooling import PoolingTokenizer
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.suite import pooling_stage_jobs
from traceformer.training.engine import (
    _dataset,
    _validate_checkpoint_contract,
    build_model,
    run_training,
)

small_config = test_paper_ablation.small_config


def test_default_contract_is_backward_compatible():
    assert AblationSpec("cone", False, "none").to_dict() == {
        "backbone": "cone", "sparse": False, "refinement": "none", "version": 1,
    }
    with pytest.raises(ValueError, match="without Sparse"):
        AblationSpec(pooling="mean").validate()
    assert {j.result_id for j in pooling_stage_jobs()} == {"T0", "T1", "T2"}


@pytest.mark.parametrize("mode", ["mean", "statistics"])
def test_pooling_preserves_mapping_and_is_permutation_invariant(mode):
    original = _tokenizer().eval()
    state = torch.get_rng_state().clone()
    model = PoolingTokenizer.from_attention(original, mode).eval()
    assert torch.equal(state, torch.get_rng_state())
    events = torch.tensor([[1., 1., .1, 1.], [2., 1., .12, 0.], [9., 2., .5, 1.], [1., 2., .11, 0.]])
    base, result = original(events), model(events)
    for key in ("token_coords", "event_token_indices", "token_statistics", "event_local_features"):
        assert torch.equal(getattr(base, key), getattr(result, key))
    permuted = model(events[torch.tensor([3, 2, 0, 1])])
    torch.testing.assert_close(permuted.tokens, result.tokens)
    assert model(events[:0]).tokens.shape == (0, original.token_dim)
    assert not hasattr(model, "pool_queries")
    assert not hasattr(model, "key_projection")
    if mode == "statistics":
        assert not hasattr(model, "event_encoder")


def test_mean_pool_is_true_arithmetic_mean():
    original = _tokenizer()
    model = PoolingTokenizer.from_attention(original, "mean")
    # 四查询设置下仅剩两倍嵌入维度的参数差，避免用明显更弱的均值基线。
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in original.parameters()) == 2 * original.event_embed_dim
    model.block_encoder = nn.Identity()
    local = torch.randn(5, 6)
    inverse = torch.tensor([0, 1, 0, 1, 1])
    values = model.value_projection(model.event_encoder(local))
    expected = torch.stack([values[inverse == k].mean(0) for k in range(2)])
    actual = model._encode_block_features(local, inverse, torch.tensor([2, 3]), torch.zeros(2, 7))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["mean", "statistics"])
def test_common_initialization_and_gradients(small_config, mode):
    torch.manual_seed(37)
    reference = build_model(small_config, ablation=AblationSpec(sparse=False, refinement="none"))
    state = torch.get_rng_state().clone()
    torch.manual_seed(37)
    model = build_model(small_config, ablation=AblationSpec(sparse=False, refinement="none", pooling=mode))
    assert torch.equal(state, torch.get_rng_state())
    for key, value in model.state_dict().items():
        if key in reference.state_dict():
            assert torch.equal(value, reference.state_dict()[key]), key
    events = _dataset(small_config, "train")[0].events
    logits = model(events)["event_logits"]
    assert logits.shape == (len(events),)
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("mode", ["mean", "statistics"])
def test_train_resume_and_independent_evaluation(small_config, tmp_path, mode):
    config = replace(small_config, training=replace(
        small_config.training, stage="full", epochs=1, num_threads=1, save_training_state_interval=1,
    ))
    spec = AblationSpec(sparse=False, refinement="none", pooling=mode)
    run = tmp_path / mode
    run_training(config, output_dir=run, ablation=spec)
    expected = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    report = evaluate_ablation(config, run / "best.pt", spec, save_predictions=True)
    assert report["metrics"] == expected
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    with pytest.raises(ValueError, match="ablation"):
        _validate_checkpoint_contract(checkpoint, config, ablation=AblationSpec(sparse=False, refinement="none"))
    # 完整状态恢复后继续第二轮，并与未中断的配对训练逐权重比较。
    config2 = replace(config, training=replace(config.training, epochs=2))
    # 日程长度是恢复契约的一部分，先跑两轮并保存第一轮状态。
    run_training(config2, output_dir=tmp_path / f"{mode}_full", ablation=spec)
    run_training(config2, output_dir=tmp_path / f"{mode}_resumed", ablation=spec,
                 resume_state_path=tmp_path / f"{mode}_full/training_states/epoch_001.pt")
    full = torch.load(tmp_path / f"{mode}_full/training_state.pt", weights_only=False)
    resumed = torch.load(tmp_path / f"{mode}_resumed/training_state.pt", weights_only=False)
    for name in full["model_state"]:
        assert torch.equal(full["model_state"][name], resumed["model_state"][name]), name
