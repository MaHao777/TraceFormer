"""推理优化的轻量契约检查；不调用训练或全集计时入口。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traceformer.cli.postprocess import build_parser
from traceformer.config import TraceFormerConfig
from traceformer.postprocessing import learning
from traceformer.postprocessing.contracts import EventCache
from traceformer.postprocessing.direct_verifier import event_branches


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache():
    return EventCache(
        name="tiny.npz",
        locations=np.array([[1, 1, 1], [1, 1, 2], [2, 1, 3], [2, 1, 4]]),
        raw_features=np.zeros((4, 4), dtype=np.float32),
        base_probability=np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32),
        token_features=np.zeros((1, 64), dtype=np.float32),
        event_token_indices=np.zeros(4, dtype=np.int64),
        prediction=np.array([False, False, True, True]),
    )


def test_weight_selection_loading_prediction_and_threads(monkeypatch, tmp_path):
    calls = []

    class Booster:
        def __init__(self, *, model_file):
            calls.append(Path(model_file).name)

        def predict(self, features, *, num_threads):
            calls.append((len(features), num_threads))
            return np.full(len(features), 0.75)

    monkeypatch.setattr(learning, "require_lightgbm", lambda: SimpleNamespace(Booster=Booster))
    contract = {"direct_weights": [0, 0.9, 0.1], "track_weights": [0, 0.8, 0.2]}
    names = learning.required_event_models(contract)
    assert names == ("large",)
    models = learning.load_event_models(tmp_path, names=names)
    scores = learning.predict_event_scores(_cache(), models, num_threads=8, chunk_size=3)
    assert calls == ["large.txt", (3, 8), (1, 8)]
    assert set(scores) == {"large"}
    assert scores["large"].dtype == np.float32
    calls.clear()
    learning.load_event_models(tmp_path)
    assert calls == ["small.txt", "large.txt"]
    contract["track_weights"] = [0.2, 0, 0.8]
    assert learning.required_event_models(contract) == ("small", "large")


def test_development_prediction_defaults_to_one_thread(monkeypatch):
    calls = []
    booster = SimpleNamespace(predict=lambda features, **kwargs:
                              calls.append(kwargs) or np.zeros(len(features)))
    learning.predict_event_scores(_cache(), {"small": booster})
    assert calls == [{"num_threads": 1}]


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_prediction_rejects_invalid_threads(threads):
    with pytest.raises(ValueError, match="positive integer"):
        learning.predict_event_scores(_cache(), {}, num_threads=threads)


def test_pure_base_does_not_load_trees_or_construct_features(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("pure base must not construct tree features")

    monkeypatch.setattr(learning, "require_lightgbm", forbidden)
    monkeypatch.setattr(learning, "make_context", forbidden)
    monkeypatch.setattr(learning, "build_event_features", forbidden)
    contract = {"direct_weights": [0, 0, 1], "track_weights": [0, 0, 1]}
    models = learning.load_event_models(tmp_path, names=learning.required_event_models(contract))
    assert learning.predict_event_scores(_cache(), models) == {}
    result = learning.blend_scores(_cache().base_probability, None, None, (0, 0, 1))
    assert np.array_equal(result, _cache().base_probability)


@pytest.mark.parametrize("weights", [(0, 0.9, 0.1), (0.2, 0.7, 0.1), (0.5, 0, 0.5)])
def test_optional_blend_is_bitwise_equal_and_rejects_missing_scores(weights):
    base = _cache().base_probability
    small, large = base[::-1].copy(), np.full(4, 0.731, dtype=np.float32)
    expected = (weights[0] * small + weights[1] * large + weights[2] * base).astype(np.float32)
    result = learning.blend_scores(base, small if weights[0] else None,
                                  large if weights[1] else None, weights)
    assert np.array_equal(result, expected)
    assert result.dtype == np.float32
    with pytest.raises(ValueError, match="missing nonzero-weight"):
        learning.blend_scores(base, None, None, weights)


def test_event_branches_accepts_only_active_model(monkeypatch):
    from traceformer.postprocessing import direct_verifier

    monkeypatch.setattr(direct_verifier, "track_prediction", lambda locations, probability, **kwargs:
                        probability >= 0.5)
    contract = {"direct_weights": [0, 0.9, 0.1], "track_weights": [0, 0.8, 0.2],
                "direct_threshold": 0.5, "tracking": {"sparse": {}}}
    scores = {"large": _cache().base_probability}
    result = event_branches(_cache(), scores, contract)
    assert all(array.shape == (4,) for array in result)


@pytest.mark.parametrize("event_only,group", [(True, "ordinary"), (True, "dense"), (False, "dense")])
def test_track2_identity_branches_skip_verifier_features(monkeypatch, event_only, group):
    script = _script("predict_postprocess_track2")

    def forbidden(*args, **kwargs):
        raise AssertionError("identity branch must not construct candidates")

    monkeypatch.setattr(script, "direct_candidate_features", forbidden)
    tracked, direct = np.array([True, False]), np.array([False, True])
    result, candidates = script._apply_optional_direct_verifier(
        None, None, tracked, direct, None, None, group=group, event_only=event_only
    )
    assert np.array_equal(result, tracked | direct)
    assert candidates is None


def test_track2_active_verifier_keeps_original_path(monkeypatch):
    script = _script("predict_postprocess_track2")
    candidates = object()
    calls = []
    monkeypatch.setattr(script, "direct_candidate_features", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(script, "apply_direct_verifier", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or np.array([True, False]))
    model = object()
    result, batch = script._apply_optional_direct_verifier(
        None, None, np.array([True, False]), np.array([False, True]), model, 0.8,
        group="ordinary", event_only=False,
    )
    assert batch is candidates
    assert calls[0][0][0] is candidates and calls[0][0][3] is model
    assert calls[0][1] == {"density_group": "ordinary"}
    assert result.tolist() == [True, False]


@pytest.mark.parametrize("entry", ["postprocess", "predict_postprocess_track2",
                                   "benchmark_frozen_system", "evaluate_frozen_system"])
def test_cli_prediction_threads_default_single_and_invalid(entry):
    parser = build_parser() if entry == "postprocess" else _script(entry).build_parser()
    # 自动补齐必填路径，测试仅解析参数，不触发预测入口。
    if entry == "postprocess":
        command = ["predict", "--cache-dir", "cache", "--model-dir", "models", "--output-dir", "out"]
    else:
        command = [item for action in parser._actions if action.required
                   for item in (action.option_strings[0], "placeholder")]
    assert parser.parse_args(command).prediction_threads == 8
    assert parser.parse_args(command + ["--prediction-threads", "1"]).prediction_threads == 1
    for invalid in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            parser.parse_args(command + ["--prediction-threads", invalid])


@pytest.mark.parametrize("prediction_backend,feature_backend", [("cpu", "cpu"), ("cuda", "cpu"), ("cuda", "cuda")])
def test_prediction_pipeline_selects_models_passes_threads_and_reports(monkeypatch, tmp_path,
                                                                     prediction_backend, feature_backend):
    import json

    from traceformer.postprocessing import pipeline

    cache_dir, model_dir = tmp_path / "cache", tmp_path / "models"
    cache_dir.mkdir()
    model_dir.mkdir()
    cache = _cache()
    cache.save(cache_dir / cache.name, include_labels=False)
    frozen = {"direct_weights": [0, 0.9, 0.1], "track_weights": [0, 0.8, 0.2],
              "direct_threshold": 0.5, "tracking": {"sparse": {}},
              "component_threshold": 0.5, "center_thresholds": {"sparse": 0.5}}
    (model_dir / "frozen.json").write_text(json.dumps(frozen), encoding="utf-8")
    calls = []
    monkeypatch.setattr("traceformer.gpu.runtime.validate_backends", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "load_event_models", lambda path, *, names, backend:
                        calls.append((names, backend)) or {"large": object()})
    monkeypatch.setattr(pipeline, "predict_event_scores", lambda cache, models, *, num_threads, feature_backend:
                        calls.append((num_threads, feature_backend)) or {"large": cache.base_probability})
    monkeypatch.setattr(pipeline, "require_lightgbm", lambda: SimpleNamespace(Booster=lambda **_: None))
    monkeypatch.setattr(pipeline, "track_prediction", lambda locations, probability, **kwargs:
                        np.zeros(len(locations), dtype=bool))
    monkeypatch.setattr(pipeline, "component_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "apply_component_model", lambda batch, prediction, *args: prediction)
    monkeypatch.setattr(pipeline, "center_candidates", lambda *args: ([], np.empty((0, 1)), None))
    monkeypatch.setattr(pipeline, "render_center_candidates", lambda cache, probability, prediction,
                        *args: prediction)
    report = pipeline.predict_pipeline(cache_dir, model_dir, tmp_path / "out",
        prediction_threads=1, prediction_backend=prediction_backend, feature_backend=feature_backend,
        progress=lambda _: None)
    assert calls == [(("large",), prediction_backend), (1, feature_backend)]
    assert report["prediction_threads"] == 1 and report["active_event_models"] == ["large"]
    assert report["prediction_backend"] == prediction_backend and report["feature_backend"] == feature_backend
    with np.load(tmp_path / "out" / cache.name) as saved:
        assert np.array_equal(saved["prediction"], cache.prediction)


def test_cache_entry_skips_sparse_logits_and_preserves_format(monkeypatch, tmp_path):
    from traceformer.postprocessing import cache as cache_module

    calls = []
    events = torch.zeros(4, 4)
    source = tmp_path / "tiny.npz"
    source.write_bytes(b"tiny source")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"tiny checkpoint")
    sequence = SimpleNamespace(events=events, source_path=source, length=4)
    cached = _cache()
    monkeypatch.setattr(cache_module, "load_inference_model", lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(cache_module, "NPZEventDataset", lambda *args, **kwargs: [sequence])
    monkeypatch.setattr(cache_module, "forward_with_sparse_mixture", lambda *args, **kwargs:
                        calls.append(kwargs) or ({
                            "event_logits": torch.logit(torch.from_numpy(cached.base_probability)),
                            "token_features": torch.from_numpy(cached.token_features),
                            "event_token_indices": torch.from_numpy(cached.event_token_indices),
                        }, None))
    monkeypatch.setattr(cache_module, "_source_arrays", lambda *args, **kwargs: {
        "locations": cached.locations, "raw_features": cached.raw_features,
        "labels": None, "target_ids": None,
    })
    cache_module.cache_split(TraceFormerConfig(), checkpoint, tmp_path / "output", split="val")
    assert calls[0]["compute_sparse_logits"] is False
    loaded = EventCache.load(tmp_path / "output/val/tiny.npz", allow_labels=False)
    assert np.array_equal(loaded.locations, cached.locations)
    assert np.array_equal(loaded.event_token_indices, cached.event_token_indices)
    with np.load(tmp_path / "output/val/tiny.npz") as saved:
        assert saved["token_features"].dtype == np.float16
        assert "labels" not in saved.files


@pytest.mark.parametrize("group", ["sparse", "dense"])
def test_complete_mask_uses_only_direct_and_main_tracks(monkeypatch, group):
    from torch import nn

    from traceformer.data import NPZEventDataset

    script = _script("benchmark_frozen_system")
    # 高神经响应不能绕过验证与主轨迹规则，密集组也没有额外恢复分支。
    count = 16
    locations = np.array([[20 + frame, 30, frame * 50 + 1] for frame in range(count)])
    logits = torch.full((count,), 3.0)
    model = nn.Linear(1, 1)
    engine = SimpleNamespace(NPZEventDataset=NPZEventDataset,
        _forward_with_sparse_mixture=lambda *args, **kwargs: ({
            "event_logits": logits, "token_features": torch.zeros(1, 64),
            "event_token_indices": torch.zeros(count, dtype=torch.long),
        }, None))
    direct = np.zeros(count, dtype=bool)
    tracked = direct.copy()
    direct[0], tracked[1] = True, True
    monkeypatch.setattr(script, "density_group", lambda _: (group, 1000))
    monkeypatch.setattr(script, "predict_event_scores", lambda *args, **kwargs: {})
    monkeypatch.setattr(script, "event_branches", lambda *args: (None, None, tracked, direct))
    source = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("t", "f8"), ("p", "f4")])
    masks, _, _, timings = script.infer(
        source, np.zeros((count, 4), dtype=np.float32), locations, Path("tiny.npz"),
        (model, TraceFormerConfig(), engine, {"release_threshold": {"threshold": 0.8}}),
        {}, {"event_contract": {}},
    )
    assert np.array_equal(masks["complete"], direct | tracked)
    assert masks["neural"].all()
    assert set(masks) == {"complete", "neural", "direct", "event_track"}
    assert sum(timings[key] for key in ("input_seconds", "main_seconds",
        "device_to_host_seconds", "event_scores_seconds", "tracking_seconds")) == pytest.approx(timings["total_seconds"])
