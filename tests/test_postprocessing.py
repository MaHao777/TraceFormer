from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from huav.cli.postprocess import build_parser
from huav.postprocessing.contracts import (
    EventCache,
    PostprocessProtocol,
    fold_of,
    make_internal_split,
)
from huav.postprocessing.features import EVENT_FEATURE_DIM, build_event_features
from huav.postprocessing.learning import blend_scores, convex_weights, sample_event_rows
from huav.postprocessing.metrics import score_binary
from huav.postprocessing.refinement import (
    apply_component_model,
    component_features,
    connected_components,
)
from huav.postprocessing.submission import (
    audit_submission,
    evaluate_predictions,
    write_prediction_npz,
)
from huav.postprocessing.tracking import track_prediction


def _cache(*, labels: bool = True) -> EventCache:
    locations = np.asarray(
        [[10, 10, 1], [11, 10, 2], [50, 50, 51], [51, 50, 52]], dtype=np.int64
    )
    return EventCache(
        name="sample.npz",
        locations=locations,
        raw_features=np.zeros((4, 4), dtype=np.float32),
        base_probability=np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
        token_features=np.zeros((2, 64), dtype=np.float32),
        event_token_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        prediction=np.asarray([1, 1, 0, 0], dtype=bool),
        labels=np.asarray([1, 0, 0, 0], dtype=bool) if labels else None,
        target_ids=np.asarray([1, 0, 0, 0], dtype=np.int64) if labels else None,
    )


def test_cache_contract_excludes_validation_labels(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    cache = _cache(labels=False)
    cache.save(path, include_labels=False)
    loaded = EventCache.load(path, allow_labels=False)
    assert loaded.labels is None
    with np.load(path, allow_pickle=False) as saved:
        assert "labels" not in saved.files
        assert saved["token_features"].dtype == np.float16


def test_split_and_oof_fold_are_deterministic_and_disjoint() -> None:
    names = [f"train_{index:03d}.npz" for index in range(99)]
    training, holdout = make_internal_split(names, seed=37, holdout_files=20)
    assert len(training) == 79
    assert len(holdout) == 20
    assert set(training).isdisjoint(holdout)
    folds = {name: fold_of(name, seed=37, folds=5) for name in names}
    assert set(folds.values()) == set(range(5))
    assert folds == {name: fold_of(name, seed=37, folds=5) for name in names}


def test_event_features_expand_parent_token_without_reordering() -> None:
    cache = _cache()
    cache.token_features[0, 0] = 3.0
    cache.token_features[1, 0] = 7.0
    features = build_event_features(cache)
    assert features.shape == (4, EVENT_FEATURE_DIM)
    assert features[:, 3].tolist() == [3.0, 3.0, 7.0, 7.0]


def test_event_sampling_and_blend_grid_are_deterministic() -> None:
    cache = _cache()
    protocol = PostprocessProtocol(
        holdout_files=1,
        minimum_negative_samples=2,
        negative_to_positive_ratio=2,
        small_rounds=1,
        large_rounds=1,
    )
    assert np.array_equal(sample_event_rows(cache, protocol), sample_event_rows(cache, protocol))
    weights = convex_weights(0.05)
    assert len(weights) == 231
    assert all(sum(weight) == pytest.approx(1.0) for weight in weights)
    mixed = blend_scores(
        np.asarray([0.1]), np.asarray([0.5]), np.asarray([0.9]), (0.2, 0.3, 0.5)
    )
    assert mixed.item() == pytest.approx(0.42)


def test_tracking_recovers_only_events_near_a_stable_track() -> None:
    rows = []
    probability = []
    for frame in range(10):
        rows.extend(([20 + frame, 30, frame * 50 + 1], [200, 200, frame * 50 + 2]))
        probability.extend((0.9, 0.02))
    locations = np.asarray(rows, dtype=np.int64)
    prediction = track_prediction(
        locations,
        np.asarray(probability, dtype=np.float32),
        seed_threshold=0.8,
        low_threshold=0.1,
        cluster_radius=0,
        minimum_track_frames=8,
        extension_frames=0,
    )
    assert prediction[::2].all()
    assert not prediction[1::2].any()


class _RejectModel:
    def predict(self, features: np.ndarray, *, num_threads: int) -> np.ndarray:
        assert num_threads == 1
        return np.zeros(len(features), dtype=np.float32)


def test_component_pruning_deletes_whole_components() -> None:
    cache = _cache()
    frames, grouped = connected_components(cache.locations, cache.prediction)
    assert frames.tolist() == [0, 0, 1, 1]
    assert len(grouped[0]) == 1
    batch = component_features(
        cache, cache.base_probability, cache.prediction, include_targets=True
    )
    result = apply_component_model(batch, cache.prediction, _RejectModel(), 0.5)
    assert not result.any()
    assert batch.labels is not None and batch.labels.tolist() == [True]


def _write_source(path: Path) -> None:
    dtype = np.dtype(
        [
            ("x", "i4"),
            ("y", "i4"),
            ("t", "f8"),
            ("p", "i4"),
            ("label", "i4"),
            ("name", "i4"),
        ]
    )
    events = np.asarray(
        [(1, 2, 1.25, 0, 1, 7), (2, 2, 51.5, 1, 0, 0)], dtype=dtype
    )
    np.savez_compressed(path, ev=events)


def test_evaluate_and_submission_audit_preserve_source_rows(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    prediction_root = tmp_path / "prediction"
    source_root.mkdir()
    prediction_root.mkdir()
    _write_source(source_root / "val_000.npz")
    write_prediction_npz(
        prediction_root / "val_000.npz", np.asarray([True, False], dtype=bool)
    )
    metrics = evaluate_predictions(prediction_root, source_root)
    assert metrics["aggregate"]["tp"] == 1
    report = audit_submission(
        prediction_root,
        source_root,
        tmp_path / "artifact" / "submission.zip",
        expected_files=1,
        expected_rows=2,
    )
    assert report["file_count"] == 1
    assert report["row_count"] == 2
    with zipfile.ZipFile(report["zip"]) as archive:
        assert archive.namelist() == ["val_000.txt"]
        assert archive.read("val_000.txt").decode("ascii").splitlines() == [
            "1 2 1.25 0 1",
            "2 2 51.5 1 0",
        ]
    saved = json.loads(
        (tmp_path / "artifact" / "submission_audit.json").read_text(encoding="utf-8")
    )
    assert saved["zip_sha256"] == report["zip_sha256"]


def test_binary_metric_uses_strict_frame_boundaries() -> None:
    locations = np.asarray([[1, 1, 0], [1, 1, 1], [2, 2, 50], [3, 3, 51]])
    labels = np.asarray([0, 1, 0, 0], dtype=bool)
    target_ids = np.asarray([0, 3, 0, 0])
    prediction = np.asarray([1, 1, 1, 1], dtype=bool)
    counts = score_binary(locations, labels, target_ids, prediction)
    assert counts.objects == 1
    assert counts.detected_objects == 1
    assert counts.false_components == 1


def test_postprocess_cli_exposes_all_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("cache", "train", "predict", "evaluate", "audit"):
        assert command in help_text
