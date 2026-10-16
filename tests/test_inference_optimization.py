"""推理优化的轻量契约检查；不调用训练或全集计时入口。"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from traceformer.cli.postprocess import build_parser
from traceformer.config import TraceFormerConfig
from traceformer.data.dataset import NPZEventDataset
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


def _detectors(tmp_path):
    config = TraceFormerConfig()
    reference = nn.Module()
    for name in ("tokenizer", "encoder", "decoder"):
        reference.add_module(name, nn.Linear(1, 1))
    main = copy.deepcopy(reference)
    main.add_module("sparse_mixture", nn.Linear(1, 1))
    engines = []
    for name in ("main", "reference"):
        root = tmp_path / name / "huav"
        (root / "models").mkdir(parents=True)
        for filename in ("event_tokenizer", "cone_transformer", "cone_attention",
                         "cone_neighbor_builder", "event_decoder", "relative_position"):
            (root / "models" / f"{filename}.py").write_text("# 相同的归档主干\n", encoding="utf-8")
        engines.append(SimpleNamespace(__file__=str(root / "training/engine.py")))
    return ((main.eval(), config, engines[0], {"schema": 5}),
            (reference.eval(), copy.deepcopy(config), engines[1], {"schema": 4}))


@pytest.mark.parametrize("mismatch", [None, "weight", "normalization", "config", "source", "schema"])
def test_reference_reuse_requires_matching_backbone_and_normalization(tmp_path, mismatch):
    script = _script("benchmark_frozen_system")
    main, reference = _detectors(tmp_path)
    if mismatch == "weight":
        with torch.no_grad():
            reference[0].decoder.weight.add_(1)
    elif mismatch == "normalization":
        from dataclasses import replace
        reference = (reference[0], replace(reference[1], dataset=replace(
            reference[1].dataset, duration_ms=100)), *reference[2:])
    elif mismatch == "config":
        from dataclasses import replace
        reference = (reference[0], replace(reference[1], resolution=replace(
            reference[1].resolution, width=123)), *reference[2:])
    elif mismatch == "source":
        source = Path(reference[2].__file__).parents[1] / "models/event_decoder.py"
        source.write_text("# 不同的主干\n", encoding="utf-8")
    elif mismatch == "schema":
        main[3]["schema"] = 7
    assert script.can_reuse_reference_logits(main, reference) is (mismatch is None)


@pytest.mark.parametrize("reuse", [True, False])
def test_dense_reference_reuse_skips_forward_and_preserves_recovery(monkeypatch, reuse):
    script = _script("benchmark_frozen_system")
    # 八帧长轨迹应被救回，七帧短轨迹仍应排除。
    locations = np.array([[20 + frame, 30, frame * 50 + 1] for frame in range(8)]
                         + [[120 + frame, 130, frame * 50 + 2] for frame in range(7)])
    count = len(locations)
    logits = torch.full((count,), 2.0)
    calls = []

    class Reference(nn.Module):
        def forward(self, events, batch):
            calls.append("reference")
            return {"event_logits": logits}

    model = nn.Linear(1, 1)
    engine = SimpleNamespace(NPZEventDataset=NPZEventDataset,
                             _forward_with_sparse_mixture=lambda *args, **kwargs: ({
                                 "event_logits": logits + 1,
                                 "full_event_logits": logits,
                                 "token_features": torch.zeros(1, 64),
                                 "event_token_indices": torch.zeros(count, dtype=torch.long),
                             }, None))
    monkeypatch.setattr(script, "density_group", lambda _: ("dense", 1000))
    monkeypatch.setattr(script, "event_branches", lambda *args:
                        (None, None, np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)))
    source = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("t", "f8"), ("p", "f4")])
    masks, group, sparse, timings = script.infer(
        source, np.zeros((count, 4), dtype=np.float32), locations, Path("tiny.npz"),
        (model, TraceFormerConfig(), engine, {"release_threshold": {"threshold": 0.8}}),
        (Reference(), None, None, {"release_threshold": {"threshold": 0.6}}),
        {}, {"event_contract": {}}, reuse_reference_logits=reuse,
    )
    expected = torch.sigmoid(logits).numpy() >= 0.6
    assert calls == ([] if reuse else ["reference"])
    assert np.array_equal(masks["reference"], expected)
    original, _ = script.rescue_reference_tracks(locations, masks["event_track"],
                                                expected, density_group="dense")
    assert np.array_equal(masks["complete"], original)
    assert masks["complete"][:8].all() and not masks["complete"][8:].any()
    assert group == "dense" and not sparse
    assert timings["total_seconds"] >= 0


@pytest.mark.parametrize("compute_sparse_logits", [True, False])
def test_sparse_forward_wrapper_passes_auxiliary_flag(monkeypatch, compute_sparse_logits):
    from traceformer.training import engine

    calls = []
    events = torch.zeros(4, 4)
    view = SimpleNamespace(events=events[:2], source_indices=torch.tensor([0, 1]))
    monkeypatch.setattr(engine, "is_dense_balanced_event_stream", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "simulate_high_threshold_events", lambda *args, **kwargs: view)
    output, returned_view = engine.forward_with_sparse_mixture(
        lambda *args, **kwargs: calls.append(kwargs) or {},
        SimpleNamespace(events=events), events, torch.zeros(4, dtype=torch.long),
        TraceFormerConfig(), compute_sparse_logits=compute_sparse_logits,
    )
    assert calls[0]["compute_sparse_logits"] is compute_sparse_logits
    assert returned_view is view and output == {}


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
