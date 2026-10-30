"""验证恢复实验保留原前向、配对训练随机数和标注隔离。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive/experiments/scripts"))
import run_original_readout_recovery as recovery

from traceformer.config import TraceFormerConfig
from traceformer.experiments.cached_loss import CachedSequenceLoss
from traceformer.training.engine import build_model


def sample():
    config = TraceFormerConfig()
    torch.manual_seed(37)
    events = torch.column_stack((torch.randint(0, 12, (24,)), torch.randint(0, 12, (24,)),
                                 torch.linspace(.01, .2, 24), torch.randint(0, 2, (24,)))).float()
    model = build_model(config).eval()
    with torch.no_grad():
        tokenized = model.tokenizer(events)
        full = model(events)
    common = dict(zip(recovery.FEATURES, (tokenized.event_token_indices, tokenized.event_local_features,
        tokenized.event_xy, tokenized.normalized_time, tokenized.event_batch_indices), strict=True))
    return config, model, common, full


def test_original_readout_matches_full_neural_graph_and_rejects_labels():
    _, model, common, full = sample()
    head = recovery.OriginalReadout(model).eval()
    with torch.no_grad():
        actual, _ = head(common, full["token_features"])
    assert torch.equal(actual, full["event_logits"])
    assert sum(p.numel() for p in head.parameters()) == 14963
    with pytest.raises(ValueError, match="neural features only"):
        head({**common, "labels": torch.ones(24)}, full["token_features"])


def test_paired_training_preserves_dropout_losses_gradients_and_initial_weights():
    config, _, common, full = sample()
    heads = recovery.fresh_heads(config)
    initial = {name: recovery.state_digest(head) for name, head in heads.items()}
    assert len(set(initial.values())) == 1
    pointers = [next(head.parameters()).data_ptr() for head in heads.values()]
    assert len(set(pointers)) == 3
    labels = (torch.arange(24) % 3 == 0).float()
    target = {"labels": labels, "object_ids": labels.long(),
              "timestamps_ms": common["normalized_time"] * 8000., "weight": 1.}
    item = {"features": common, "tokens": {name: full["token_features"] for name in recovery.VARIANTS},
            "supervision": target, "criterion": CachedSequenceLoss(labels, target["object_ids"],
                target["timestamps_ms"], common["xy"], config)}
    losses, _ = recovery.training_step(heads, item, config, torch.Generator().manual_seed(37), 71)
    assert len(set(losses.values())) == 1
    expected = list(heads["learned"].parameters())
    assert any(p.grad.abs().sum() > 0 for p in expected if p.grad is not None)
    for name, head in heads.items():
        assert recovery.state_digest(head) == initial[name]
        for actual, reference in zip(head.parameters(), expected, strict=True):
            assert torch.equal(actual.grad, reference.grad)
