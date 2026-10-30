"""当前网络事件发布的量化、来源和训练边界。"""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from traceformer.postprocessing import cache as cache_module
from traceformer.postprocessing.contracts import EventCache


def test_live_features_equal_saved_cache(tmp_path):
    # 非半精度可表示值必须在实时路径中同样经过量化。
    source = {"locations": np.array([[1, 2, 3]]),
              "raw_features": np.zeros((1, 4), dtype=np.float32),
              "labels": None, "target_ids": None}
    output = {"event_logits": torch.tensor([0.2]),
              "token_features": torch.full((1, 64), 0.1234567),
              "event_token_indices": torch.tensor([0])}
    live = cache_module.event_cache_from_output("sample.npz", source, output, 0.5)
    live.save(tmp_path / live.name, include_labels=False)
    saved = EventCache.load(tmp_path / live.name, allow_labels=False)
    assert np.array_equal(live.token_features, saved.token_features)
    assert live.token_features[0, 0] != np.float32(0.1234567)
    assert live.labels is None
    broken = replace(live, token_features=np.full((1, 64), np.inf))
    with pytest.raises(ValueError, match="finite"):
        broken.validate(allow_labels=False)


def _training_caches(tmp_path):
    from traceformer.postprocessing.contracts import CacheManifest, sha256_file

    directory = tmp_path / "train"
    directory.mkdir()
    records = []
    for index in range(3):
        name = f"train_{index:03d}.npz"
        cache = EventCache(
            name, np.array([[1, 2, 3]]), np.zeros((1, 4), np.float32),
            np.array([0.9], np.float32), np.zeros((1, 64), np.float32),
            np.array([0]), np.array([True]), np.array([True]), np.array([1]),
        )
        cache.save(directory / name, include_labels=True)
        records.append({"name": name, "cache_sha256": sha256_file(directory / name)})
    CacheManifest("train", 1, 1, "checkpoint", "config", tuple(records), True).write(directory / "manifest.json")
    return directory


@pytest.mark.parametrize("passed", [True, False])
def test_event_only_stops_at_event_gate_without_oof(tmp_path, monkeypatch, passed):
    from traceformer.postprocessing import pipeline
    from traceformer.postprocessing.contracts import PostprocessProtocol

    directory = _training_caches(tmp_path)
    monkeypatch.setattr(pipeline, "train_event_models", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(pipeline, "predict_event_scores", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_select_direct", lambda *a: ((1, 0, 0), 0.9, {}))
    monkeypatch.setattr(pipeline, "_select_track_weights", lambda *a: (1, 0, 0))
    monkeypatch.setattr(pipeline, "_tune_tracking", lambda *a, **k: ({}, []))
    monkeypatch.setattr(pipeline, "_stage1_predictions", lambda caches, scores, names, **k:
                        ({name: caches[name].prediction for name in names}, {}))
    monkeypatch.setattr(pipeline, "_gate", lambda *a, **k: {"passed": passed})
    calls = []
    monkeypatch.setattr(pipeline, "release_event_models", lambda *a, **k: calls.append(a) or {})
    monkeypatch.setattr(pipeline, "generate_oof_scores", lambda *a, **k: pytest.fail("OOF executed"))
    result = pipeline.train_pipeline(directory, tmp_path / "training",
                                     PostprocessProtocol(holdout_files=1), event_only=True,
                                     progress=lambda _: None)
    assert result["status"] == ("event_released" if passed else "rejected_event_stage")
    assert len(calls) == int(passed)
    assert not (tmp_path / "training/development/oof_79").exists()


def test_manifest_rejects_modified_cache(tmp_path):
    from traceformer.postprocessing.release import validate_cache_manifest

    directory = _training_caches(tmp_path)
    validate_cache_manifest(directory, split="train")
    with (directory / "train_000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        validate_cache_manifest(directory, split="train")


def test_release_binds_checkpoint_config_and_only_loads_active_trees(tmp_path, monkeypatch):
    from traceformer.postprocessing import release as module
    from traceformer.postprocessing.contracts import CacheManifest, sha256_file

    directory = tmp_path / "release"
    (directory / "events").mkdir(parents=True)
    checkpoint, config = tmp_path / "best.pt", tmp_path / "config.yaml"
    checkpoint.write_bytes(b"network")
    config.write_bytes(b"config")
    contract = {"direct_weights": [0, 1, 0], "track_weights": [0, 1, 0],
                "direct_threshold": 0.9, "tracking": {}}
    training_names = ["train_000.npz", "train_001.npz", "train_002.npz"]
    development = {"event_stage": {"gate": {"passed": True}, **contract},
                   "training_names": training_names[:2], "holdout_names": training_names[2:]}
    (directory / "development_contract.json").write_text(json.dumps(development), encoding="utf-8")
    CacheManifest("train", 1, 1, sha256_file(checkpoint), sha256_file(config),
                  tuple({"name": name} for name in training_names), True).write(directory / "training_manifest.json")
    for name in ("small", "large"):
        (directory / f"events/{name}.txt").write_bytes(name.encode())
    report = {
        "provenance": {"checkpoint_sha256": sha256_file(checkpoint), "config_sha256": sha256_file(config),
                       "training_manifest_sha256": sha256_file(directory / "training_manifest.json"),
                       "cache_schema_version": 1, "feature_schema_version": 1, "token_feature_dtype": "float16"},
        "event_contract": contract,
        "training_files": 3, "model": {"training_sequences": training_names},
        "source_training_report_sha256": sha256_file(directory / "development_contract.json"),
        "event_model_sha256": {f"{name}.txt": sha256_file(directory / f"events/{name}.txt")
                               for name in ("small", "large")},
    }
    (directory / "release_report.json").write_text(json.dumps(report), encoding="utf-8")
    observed = []
    monkeypatch.setattr(module, "load_event_models", lambda path, **kwargs: observed.append(kwargs) or {})
    module.load_event_release(directory, checkpoint=checkpoint, config=config)
    assert observed[0]["names"] == ("large",)
    checkpoint.write_bytes(b"another network")
    with pytest.raises(ValueError, match="checkpoint_sha256 differs"):
        module.load_event_release(directory, checkpoint=checkpoint, config=config)
    config.write_bytes(b"another config")
    with pytest.raises(ValueError, match="config_sha256 differs"):
        module.load_event_release(directory, config=config)


def test_validation_gate_cannot_trade_recall_or_fa_for_score():
    from traceformer.postprocessing.release import event_validation_gate

    base = {"aggregate": {"score": 0.9, "fa": 7e-6}, "files": {"val_023.npz": {"pd": 0.8}}}
    candidate = {"aggregate": {"score": 0.92, "fa": 5e-6}, "files": {"val_023.npz": {"pd": 0.8}}}
    assert event_validation_gate(candidate, base)["passed"]
    candidate["files"]["val_023.npz"]["pd"] = 0.79
    assert not event_validation_gate(candidate, base)["passed"]
    candidate["files"]["val_023.npz"]["pd"] = 0.8
    candidate["aggregate"]["fa"] = 7e-6
    assert not event_validation_gate(candidate, base)["passed"]
