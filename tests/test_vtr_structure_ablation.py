"""实验控制保持事件对应、官方计分和验证集选阈值的边界。"""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from traceformer.postprocessing import tracking
from traceformer.postprocessing.metrics import score_binary

SPEC = importlib.util.spec_from_file_location(
    "vtr_ablation", Path(__file__).resolve().parents[1] / "archive/experiments/scripts/run_vtr_structure_ablation.py")
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_frame_permutation_preserves_event_alignment_and_offsets():
    locations = np.column_stack((np.arange(12), np.arange(12) + 20, np.arange(12) * 51))
    permuted, mapping = ablation.permute_frames(locations)
    second, second_mapping = ablation.permute_frames(locations)
    assert np.array_equal(permuted[:, :2], locations[:, :2])
    assert np.array_equal(permuted[:, 2] % 50, locations[:, 2] % 50)
    assert np.array_equal(np.sort(mapping), np.arange(len(mapping)))
    assert np.array_equal(permuted, second)
    assert np.array_equal(mapping, second_mapping)
    inverse = np.argsort(mapping)
    assert np.array_equal(inverse[permuted[:, 2] // 50], locations[:, 2] // 50)


def test_frame_local_preserves_fallback_and_restores_linker():
    # 单帧一个高种子和两个低分事件，验证minimum_events_per_center补足语义。
    locations = np.array([[20, 20, 1], [21, 20, 2], [22, 20, 3],
                          [20, 20, 101], [21, 20, 102], [22, 20, 103]])
    probability = np.array([0.9, 0.04, 0.03, 0.9, 0.04, 0.03], dtype=np.float32)
    settings = {"seed_threshold": 0.8, "low_threshold": 0.08,
                "fallback_threshold": 0.025, "minimum_events_per_center": 2,
                "minimum_track_frames": 8, "extension_frames": 9}
    original = tracking._link_tracks
    actual = ablation.frame_local_prediction(locations, probability, settings)
    assert tracking._link_tracks is original
    assert actual.tolist() == [True, True, False, True, True, False]
    assert settings["minimum_track_frames"] == 8


def test_prepared_scorer_matches_official_boundaries_and_pixel_wrap():
    rng = np.random.default_rng(37)
    locations = np.column_stack((rng.integers(0, 40, 900), rng.integers(0, 30, 900),
                                  np.sort(rng.integers(0, 401, 900))))
    labels = rng.random(900) > 0.6
    ids = np.where(labels, rng.integers(1, 4, 900), 0)
    # 256个同像素背景事件应在官方uint8计数中回绕为零。
    locations = np.vstack((locations, np.tile([60, 60, 151], (256, 1))))
    labels = np.concatenate((labels, np.zeros(256, dtype=bool)))
    ids = np.concatenate((ids, np.zeros(256, dtype=int)))
    order = np.argsort(locations[:, 2], kind="stable")
    locations, labels, ids = locations[order], labels[order], ids[order]
    scorer = ablation.PreparedScoring(locations, labels, ids)
    for prediction in (np.ones(len(labels), dtype=bool), rng.random(len(labels)) > 0.4):
        assert asdict(scorer.score(prediction)) == asdict(
            score_binary(locations, labels, ids, prediction))


def test_threshold_selection_rejects_test_and_uses_fixed_tie_order():
    rows = [{"family": "track", "threshold": 0.5, "aggregate": {"score": 0.9}},
            {"family": "direct", "threshold": 0.8, "aggregate": {"score": 0.9}}]
    assert ablation.select_threshold(rows, split="val") is rows[0]
    with pytest.raises(ValueError, match="validation-only"):
        ablation.select_threshold(rows, split="test")


def test_union_threshold_retains_frozen_direct_mask():
    tracks = np.array([0.1, 0.6, 0.2])
    direct_score = np.array([0.9, 0.2, 0.1])
    direct = direct_score >= 0.85
    assert ablation.threshold_mask("direct_or_track", 0.5, tracks, direct_score, direct).tolist() == [
        True, True, False]
