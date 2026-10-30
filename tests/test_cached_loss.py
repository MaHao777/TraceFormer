"""缓存分组必须保持原损失值、全事件梯度和正样本保留契约。"""

from dataclasses import replace

import pytest
import torch

from traceformer.config import TraceFormerConfig
from traceformer.experiments.cached_loss import CachedSequenceLoss
from traceformer.training.losses import EventFocalDiceLoss


@pytest.mark.parametrize("case", ["random", "duplicates", "boundaries", "no_positive", "empty"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("auxiliary", [True, False])
def test_cached_values_and_all_logit_gradients_match_original(case, dtype, auxiliary):
    generator = torch.Generator().manual_seed(37)
    count = 150
    labels = (torch.rand(count, generator=generator) > 0.65).float()
    ids = torch.where(labels.bool(), torch.randint(1, 4, (count,), generator=generator), 0)
    times = torch.rand(count, generator=generator, dtype=torch.float64) * 351
    coordinates = torch.randint(0, 25, (count, 2), generator=generator).float()
    values = torch.randn(count, generator=generator, dtype=dtype)
    if case == "duplicates":
        coordinates[::2] = coordinates[1::2]
        times[:] = torch.arange(count) % 3 * 50 + 1.25
        values[:] = 0.0  # 相同最大值验证scatter和topk的并列梯度。
    elif case == "boundaries":
        times[:] = torch.tensor([0, 0.99, 1.0, 49.999, 50, 50.99, 51, 100.01, 150.2, 351.5],
                                dtype=torch.float64).repeat(15)
        coordinates += 0.75  # 原实现用long截断坐标。
    elif case == "no_positive":
        labels.zero_()
        ids.zero_()
    elif case == "empty":
        labels, ids, times, coordinates, values = (
            labels[:0], ids[:0], times[:0], coordinates[:0], values[:0])
    positive = torch.where(labels > 0.5)[0]
    negative = torch.where(labels <= 0.5)[0][::2]
    query = torch.cat((positive, negative))
    query = query[torch.randperm(len(query), generator=generator)]
    criterion = EventFocalDiceLoss(lambda_dice=0.75,
                                  lambda_object_frame=0.25 if auxiliary else 0.0,
                                  lambda_false_alarm=0.05 if auxiliary else 0.0,
                                  false_alarm_topk_per_frame=7)
    cached = CachedSequenceLoss(labels, ids, times, coordinates, criterion)
    original_logits = values.clone().requires_grad_()
    cached_logits = values.clone().requires_grad_()
    batch = torch.zeros(len(values), dtype=torch.long)
    expected = criterion(
        original_logits[query], labels[query], batch[query], object_ids=ids[query],
        timestamps_ms=times[query], full_logits=original_logits, full_targets=labels,
        full_batch_indices=batch, full_timestamps_ms=times, full_coordinates=coordinates)
    actual = cached(cached_logits, query)
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        expected_gradient = torch.autograd.grad(expected[name], original_logits, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(actual[name], cached_logits, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_rejects_missing_positives_and_duplicate_queries():
    cached = CachedSequenceLoss(torch.tensor([1.0, 0.0]), torch.tensor([1, 0]),
                                torch.tensor([1.0, 2.0]), torch.zeros(2, 2),
                                EventFocalDiceLoss())
    with pytest.raises(ValueError, match="all positive"):
        cached(torch.zeros(2), torch.tensor([1]))
    with pytest.raises(ValueError, match="unique"):
        cached(torch.zeros(2), torch.tensor([0, 0, 1]))


def test_accepts_complete_config_and_retains_nondefault_frame_size():
    config = TraceFormerConfig()
    config = replace(config, evaluation=replace(config.evaluation, frame_ms=25.0))
    loss = CachedSequenceLoss(torch.tensor([1.0]), torch.tensor([1]), torch.tensor([26.5]),
                              torch.zeros(1, 2), config)
    assert loss.frame_ms == 25.0
    assert loss.criterion.lambda_dice == config.loss.lambda_dice
