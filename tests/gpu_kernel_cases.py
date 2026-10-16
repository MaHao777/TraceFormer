"""由隔离的 CUDA simulator 进程执行同一份 CUDA 内核。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from traceformer.config import ConeHeadConfig
from traceformer.gpu.kernels import (
    cone_kernel,
    forest_predict,
    group_statistics,
    histogram,
    trajectory_context,
)
from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder, prepare_cone_geometry
from traceformer.models.gpu_neighbors import pack_geometry
from traceformer.postprocessing.features import _trajectory_features
from traceformer.postprocessing.gpu import pack_forest


@pytest.mark.parametrize("scale,limit", [(1.0, 1), (1.0, 8), (2.0, 4)])
def test_exact_kd_query(scale, limit):
    rng = np.random.default_rng(37)
    # 包含密集等距、重复坐标、缺失时间片与不同 batch。
    points = rng.integers(0, 8, size=(84, 3), dtype=np.int64)
    points[:, :2] *= 2
    points[:, 2] *= 2
    points[0:4] = [[0, 0, 0], [2, 0, 2], [0, 2, 2], [2, 0, 2]]
    batches = np.repeat([0, 3], 42).astype(np.int64)
    heads = (ConeHeadConfig(base_radius=2, slope=0, max_time_distance=4),
             ConeHeadConfig(base_radius=3, slope=0.5, max_time_distance=3))
    builder = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=limit, layer_time_scale=scale)
    coordinates, batch = torch.from_numpy(points), torch.from_numpy(batches)
    expected = builder(coordinates, batch)
    packed = pack_geometry(prepare_cone_geometry(coordinates, batch))
    output = np.full((len(points), len(heads), 2, limit), -1, np.int64)
    overflow = np.zeros(len(points) * len(heads) * 2, np.uint8)
    radii, slopes = np.array([2, 3], np.float64), np.array([0, 0.5], np.float64)
    horizons = np.ceil(np.array([4, 3]) * scale).astype(np.int64)
    cone_kernel(limit)[(len(overflow) + 31) // 32, 32](
        *packed, radii, slopes, horizons, output, overflow)
    assert not overflow.any()
    np.testing.assert_array_equal(output[:, :, 0], expected.past_indices.numpy())
    np.testing.assert_array_equal(output[:, :, 1], expected.future_indices.numpy())


def test_empty_geometry():
    arrays = pack_geometry(prepare_cone_geometry(torch.empty((0, 3), dtype=torch.int64),
                                               torch.empty(0, dtype=torch.int64)))
    assert arrays[6].shape == (0, 4)
    assert arrays[7].shape == (0, 5)


def test_internal_kd_nodes_and_tied_frontier():
    rng = np.random.default_rng(93)
    points = np.array([(x, y, time) for time in (0, 2, 5)
                       for x in range(8) for y in range(8)], np.int64)
    rng.shuffle(points)
    coordinates = torch.from_numpy(points)
    batches = torch.zeros(len(points), dtype=torch.int64)
    heads = (ConeHeadConfig(base_radius=2, slope=0.5, max_time_distance=6),)
    packed = pack_geometry(prepare_cone_geometry(coordinates, batches))
    assert np.count_nonzero(packed[7][:, 0] >= 0) >= 9
    output = np.full((len(points), 1, 2, 8), -1, np.int64)
    overflow = np.zeros(len(points) * 2, np.uint8)
    cone_kernel(8)[(len(overflow) + 31) // 32, 32](*packed, np.array([2.0]),
        np.array([0.5]), np.array([6], np.int64), output, overflow)
    expected = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=8)(coordinates, batches)
    assert not overflow.any()
    np.testing.assert_array_equal(output[:, :, 0], expected.past_indices.numpy())
    np.testing.assert_array_equal(output[:, :, 1], expected.future_indices.numpy())


def test_real_published_forest():
    import lightgbm as lgb

    root = Path(__file__).resolve().parents[1]
    path = root / "outputs/paper_cuda_20260928/copyback_20260929/artifacts/event_release/events/large.txt"
    if not path.exists():
        pytest.skip("published tree artifact unavailable")
    model = lgb.Booster(model_file=str(path))
    packed = pack_forest(model.dump_model())
    rng = np.random.default_rng(91)
    features = rng.normal(size=(8, model.num_feature())).astype(np.float32)
    features[0] = 0
    features[1] = np.nan
    features[2, ::2] = np.inf
    features[3, ::3] = -np.inf
    root_node = model.dump_model()["tree_info"][0]["tree_structure"]
    column, threshold = root_node["split_feature"], np.float32(root_node["threshold"])
    features[4:7, column] = [np.nextafter(threshold, np.float32(-np.inf)), threshold,
                              np.nextafter(threshold, np.float32(np.inf))]
    output = np.empty(len(features), np.float64)
    forest_predict[1, 8](features, *packed.arrays, packed.sigmoid, output)
    np.testing.assert_array_equal(output.astype(np.float32), model.predict(features, num_threads=1).astype(np.float32))


@pytest.mark.parametrize("kind", ["None", "Zero", "NaN"])
def test_tree_missing_values(kind):
    model = {"objective": "binary sigmoid:2", "max_feature_idx": 0, "tree_info": [{
        "tree_structure": {"split_feature": 0, "decision_type": "<=", "threshold": 0.5,
            "missing_type": kind, "default_left": False,
            "left_child": {"leaf_value": -1}, "right_child": {"leaf_value": 1}}}]}
    packed = pack_forest(model)
    features = np.array([0, 1e-36, 0.5, np.nextafter(np.float32(0.5), np.float32(1)), np.nan], np.float32)[:, None]
    output = np.empty(5, np.float64)
    forest_predict[1, 8](features, *packed.arrays, packed.sigmoid, output)
    right = np.array([kind == "Zero", kind == "Zero", False, True, kind == "NaN" or kind == "Zero"])
    expected = (1 / (1 + np.exp(-2 * np.where(right, 1.0, -1.0)))).astype(np.float32)
    np.testing.assert_array_equal(output.astype(np.float32), expected)


def test_group_statistics_event_order():
    probability = np.array([1, 1e-20, 0.7, 0.2, 0.9], np.float32)
    inverse = np.array([1, 1, 0, 1, 0])
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    offsets = np.array([0, 2, 5], np.int64)
    means, maxima = np.empty(2, np.float64), np.empty(2, np.float32)
    group_statistics[1, 8](order, offsets, probability, means, maxima)
    np.testing.assert_array_equal(means, np.bincount(inverse, weights=probability) / np.bincount(inverse))
    np.testing.assert_array_equal(maxima, [np.float32(0.9), 1])


@pytest.mark.parametrize("size", [2, 4, 8])
def test_trajectory_context(size):
    locations = np.array([[0, 0, 0], [1, 1, 49], [0, 0, 50], [345, 259, 100],
                          [8, 8, 150], [8, 9, 200]], np.int64)
    probability = np.array([0.2, 0.5, 1, 0.9, 0.1, 0.8], np.float32)
    nx, ny, frames = (346 + size - 1) // size, (260 + size - 1) // size, 5
    flat = ((locations[:, 2] // 50 * ny + locations[:, 1] // size) * nx + locations[:, 0] // size)
    counts, maxima = np.zeros(frames * nx * ny, np.int64), np.zeros(frames * nx * ny, np.float32)
    output = np.empty((len(locations), 6), np.float32)
    log_counts = np.log1p(np.arange(len(locations) + 1, dtype=np.float32))
    histogram[1, 8](flat, probability, counts, maxima)
    trajectory_context[1, 8](locations, size, nx, ny, frames, counts, maxima, log_counts, output, 0)
    np.testing.assert_array_equal(output, _trajectory_features(locations, probability, size))


def _simulated_runtime(monkeypatch):
    from traceformer.postprocessing import gpu

    def launch(kernel, count, device, *arguments, threads=128):
        arrays = [value.numpy() if isinstance(value, torch.Tensor) else value for value in arguments]
        if count:
            kernel[(count + threads - 1) // threads, threads](*arrays)

    # 只替换显存/stream 运输；生产特征代码与 CUDA 内核均直接执行。
    monkeypatch.setattr(gpu, "require_cuda", lambda device: (torch.device("cpu"), None))
    monkeypatch.setattr(gpu, "launch", launch)


def _event_cache():
    from traceformer.postprocessing.contracts import EventCache

    rng = np.random.default_rng(173)
    locations = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 50], [345, 259, 100],
                          [8, 8, 150], [8, 9, 200], [8, 8, 200], [8, 9, 201]], np.int64)
    return EventCache(name="simulated.npz", locations=locations,
        base_probability=np.array([0, 1, 0.1, 0.9, 0.4, 0.6, 0.2, 0.8], np.float32),
        raw_features=rng.normal(size=(8, 4)).astype(np.float32),
        token_features=rng.normal(size=(3, 64)).astype(np.float16).astype(np.float32),
        event_token_indices=np.array([0, 1, 0, 2, 1, 2, 1, 0]), prediction=np.zeros(8, bool))


def test_complete_gpu_features(monkeypatch):
    from traceformer.postprocessing.features import build_event_features, make_context
    from traceformer.postprocessing.gpu import CudaEventFeatures

    _simulated_runtime(monkeypatch)
    cache = _event_cache()
    features = CudaEventFeatures(cache)
    np.testing.assert_array_equal(features.context.numpy(), make_context(cache.locations, cache.base_probability))
    for start, stop in ((0, 8), (1, 4), (7, 8)):
        np.testing.assert_array_equal(features.rows(start, stop).numpy(),
                                     build_event_features(cache, indices=np.arange(start, stop)))


def test_double_precision_split_boundary(monkeypatch):
    from types import SimpleNamespace

    from traceformer.postprocessing.gpu import CudaEventModel

    _simulated_runtime(monkeypatch)
    threshold = 0.50000000001
    model = {"objective": "binary sigmoid:1", "max_feature_idx": 0, "tree_info": [{
        "tree_structure": {"split_feature": 0, "decision_type": "<=", "threshold": threshold,
            "missing_type": "None", "default_left": True,
            "left_child": {"leaf_value": -1}, "right_child": {"leaf_value": 1}}}]}
    gpu = CudaEventModel(SimpleNamespace(dump_model=lambda: model))
    features = np.array([[threshold], [np.nextafter(threshold, np.inf)]], np.float64)
    actual = gpu.predict(features)
    expected = (1 / (1 + np.exp(-np.array([-1, 1])))).astype(np.float32)
    np.testing.assert_array_equal(actual.astype(np.float32), expected)


def test_chunked_gpu_scores(monkeypatch):
    from traceformer.postprocessing.learning import load_event_models, predict_event_scores

    _simulated_runtime(monkeypatch)
    directory = Path(__file__).resolve().parents[1] / "outputs/paper_cuda_20260928/copyback_20260929/artifacts/event_release/events"
    if not directory.exists():
        pytest.skip("published tree artifact unavailable")
    cache = _event_cache()
    cpu = load_event_models(directory, names=("large",))
    gpu = load_event_models(directory, names=("large",), backend="cuda")
    expected = predict_event_scores(cache, cpu, chunk_size=5)
    for feature_backend in ("cpu", "cuda"):
        scores = predict_event_scores(cache, gpu, chunk_size=5, feature_backend=feature_backend)
        np.testing.assert_array_equal(scores["large"], expected["large"])
