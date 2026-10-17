from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from traceformer.cli.postprocess import build_parser
from traceformer.postprocessing.contracts import (
    EventCache,
    PostprocessProtocol,
    fold_of,
    make_internal_split,
)
from traceformer.postprocessing.direct_verifier import (
    apply_direct_verifier,
    direct_candidate_features,
    saturate_probability,
)
from traceformer.postprocessing.features import EVENT_FEATURE_DIM, build_event_features
from traceformer.postprocessing.learning import blend_scores, convex_weights, sample_event_rows
from traceformer.postprocessing.metrics import score_binary
from traceformer.postprocessing.pipeline import release_event_models
from traceformer.postprocessing.refinement import (
    apply_component_model,
    component_features,
    connected_components,
)
from traceformer.postprocessing.submission import (
    audit_submission,
    evaluate_predictions,
    write_prediction_npz,
)
from traceformer.postprocessing.tracking import track_prediction


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


class _KeepFirstCandidate:
    def predict(self, features: np.ndarray, *, num_threads: int) -> np.ndarray:
        assert num_threads == 1
        result = np.zeros(len(features), dtype=np.float32)
        result[0] = 1.0
        return result


def test_direct_verifier_only_gates_untracked_components() -> None:
    locations = np.asarray(
        [
            [10, 10, 1],
            [11, 10, 2],
            [100, 100, 1],
            [200, 200, 1],
        ],
        dtype=np.int64,
    )
    cache = EventCache(
        name="direct.npz",
        locations=locations,
        raw_features=np.column_stack(
            (
                np.zeros((4, 3), dtype=np.float32),
                np.asarray([1, 1, 0, 1], dtype=np.float32),
            )
        ),
        base_probability=np.asarray([0.9, 0.8, 0.99, 0.98], dtype=np.float32),
        token_features=np.zeros((1, 64), dtype=np.float32),
        event_token_indices=np.zeros(4, dtype=np.int64),
        prediction=np.ones(4, dtype=bool),
        labels=np.asarray([1, 1, 0, 0], dtype=bool),
        target_ids=np.asarray([1, 1, 0, 0], dtype=np.int64),
    )
    tracked = np.asarray([1, 1, 0, 0], dtype=bool)
    direct = np.ones(4, dtype=bool)
    batch = direct_candidate_features(
        cache,
        cache.base_probability,
        tracked,
        direct,
        include_targets=True,
    )

    assert len(batch.features) == 2
    assert batch.labels is not None and batch.labels.tolist() == [False, False]
    gated = apply_direct_verifier(
        batch,
        tracked,
        direct,
        _KeepFirstCandidate(),
        0.5,
        density_group="sparse",
    )
    assert gated.tolist() == [True, True, True, False]


def test_direct_verifier_preserves_dense_predictions() -> None:
    cache = _cache()
    tracked = np.asarray([1, 0, 0, 0], dtype=bool)
    direct = np.asarray([1, 1, 1, 0], dtype=bool)
    batch = direct_candidate_features(
        cache,
        cache.base_probability,
        tracked,
        direct,
        include_targets=False,
    )
    gated = apply_direct_verifier(
        batch,
        tracked,
        direct,
        _RejectModel(),
        0.5,
        density_group="dense",
    )
    assert np.array_equal(gated, tracked | direct)


def test_saturation_is_monotonic_and_preserves_probability_order() -> None:
    probability = np.asarray([0.1, 0.4, 0.8, 0.99], dtype=np.float32)
    saturated = saturate_probability(probability, scale=2.0, bias=0.75)
    assert np.all(np.diff(saturated) > 0)
    assert saturated[-1] > probability[-1]


def test_event_release_trains_all_cache_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    names = ["train_000.npz", "train_001.npz", "train_002.npz"]
    for name in names:
        (cache_dir / name).touch()
    training_dir = tmp_path / "development"
    training_dir.mkdir()
    training_report = {
        "protocol": {
            "seed": 37,
            "holdout_files": 1,
            "folds": 2,
            "hard_negative_threshold": 0.03,
            "negative_to_positive_ratio": 4,
            "minimum_negative_samples": 10,
            "small_rounds": 1,
            "large_rounds": 1,
            "blend_step": 0.05,
        },
        "training_names": names[1:],
        "holdout_names": names[:1],
        "status": "rejected_component_stage",
        "event_stage": {
            "gate": {"passed": True},
            "direct_weights": [0.0, 0.9, 0.1],
            "track_weights": [0.0, 0.8, 0.2],
            "direct_threshold": 0.88,
            "tracking": {"sparse": {}, "ordinary": {}, "dense": {}},
        },
    }
    report_path = training_dir / "training_report.json"
    report_path.write_text(json.dumps(training_report), encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_train_event_models(
        cache_root: Path,
        training_names: list[str],
        model_dir: Path,
        protocol: PostprocessProtocol,
    ) -> tuple[dict[str, object], dict[str, object]]:
        observed["cache_root"] = Path(cache_root)
        observed["training_names"] = list(training_names)
        observed["protocol"] = protocol
        Path(model_dir).mkdir(parents=True)
        for model_name in ("small.txt", "large.txt"):
            (Path(model_dir) / model_name).write_text(model_name, encoding="ascii")
        return {}, {"training_sequences": list(training_names)}

    monkeypatch.setattr(
        "traceformer.postprocessing.pipeline.train_event_models",
        fake_train_event_models,
    )
    output_dir = tmp_path / "release"
    release = release_event_models(cache_dir, report_path, output_dir, progress=lambda _: None)

    assert observed["cache_root"] == cache_dir
    assert observed["training_names"] == names
    assert isinstance(observed["protocol"], PostprocessProtocol)
    assert release["training_files"] == 3
    assert release["event_contract"]["direct_threshold"] == pytest.approx(0.88)
    assert (output_dir / "events" / "small.txt").is_file()
    assert json.loads((output_dir / "release_report.json").read_text(encoding="utf-8")) == release


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
    for command in (
        "cache",
        "train",
        "release-event",
        "train-direct-verifier",
        "predict",
        "evaluate",
        "audit",
    ):
        assert command in help_text
