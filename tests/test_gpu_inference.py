"""GPU 后端的 CPU 兼容性、隔离 simulator 和实际设备验收。"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from traceformer.postprocessing.gpu import pack_forest


def test_backend_validation_and_no_cuda_dependency(tmp_path):
    from traceformer.gpu.runtime import validate_backends
    from traceformer.postprocessing.learning import load_event_models

    validate_backends()
    assert load_event_models(tmp_path, names=(), backend="cuda") == {}
    with pytest.raises(ValueError, match="device cuda"):
        validate_backends(neighbor_backend="cuda")
    with pytest.raises(ValueError, match="prediction-backend cuda"):
        validate_backends(feature_backend="cuda")


def test_cpu_model_parameters_unchanged():
    from traceformer.config import TraceFormerConfig
    from traceformer.models.gpu_neighbors import configure_neighbor_backend
    from traceformer.training.engine import build_model

    model = build_model(TraceFormerConfig())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    configure_neighbor_backend(model, "cpu")
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_archived_adapter_scope_and_reset(monkeypatch):
    from types import SimpleNamespace

    from traceformer.config import ConeHeadConfig
    from traceformer.models import cone_neighbor_builder, gpu_neighbors

    heads = (ConeHeadConfig(base_radius=2, slope=0, max_time_distance=2),)
    originals = [SimpleNamespace(cone_heads=heads, max_neighbors_per_cone=8,
                                layer_time_scale=1, causal_mode=False) for _ in range(2)]
    geometry = object()
    observed = []
    layers = [SimpleNamespace(attention=SimpleNamespace(neighbor_builder=builder)) for builder in originals]

    def forward(*args):
        observed.append([layer.attention.neighbor_builder.geometry for layer in layers])
        raise RuntimeError("synthetic forward failure")

    encoder = SimpleNamespace(layers=layers, forward=forward)
    model = SimpleNamespace(encoder=encoder, parameters=lambda: iter([torch.zeros(1)]))
    monkeypatch.setattr(gpu_neighbors, "require_cuda", lambda *args: None)
    monkeypatch.setattr(cone_neighbor_builder, "prepare_cone_geometry", lambda *args, **kwargs: geometry)
    gpu_neighbors.configure_neighbor_backend(model, "cuda")
    adapters = [layer.attention.neighbor_builder for layer in layers]
    gpu_neighbors.configure_neighbor_backend(model, "cuda")
    assert adapters == [layer.attention.neighbor_builder for layer in layers]
    with pytest.raises(RuntimeError, match="synthetic"):
        encoder.forward(None, None, None)
    assert observed == [[geometry, geometry]]
    assert all(adapter.geometry is None for adapter in adapters)
    gpu_neighbors.configure_neighbor_backend(model, "cpu")
    assert encoder.forward is forward
    assert originals == [layer.attention.neighbor_builder for layer in layers]


def test_cli_backend_defaults():
    import argparse

    from traceformer.gpu.runtime import add_backend_arguments

    parser = argparse.ArgumentParser()
    add_backend_arguments(parser, neural=True, trees=True)
    assert vars(parser.parse_args([])) == {"neighbor_backend": "cpu", "prediction_backend": "cpu", "feature_backend": "cpu"}
    assert vars(parser.parse_args(["--neighbor-backend", "cuda", "--prediction-backend", "cuda", "--feature-backend", "cuda"])) == {
        "neighbor_backend": "cuda", "prediction_backend": "cuda", "feature_backend": "cuda"}


def test_cuda_simulator_kernels():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("CUDA simulator requires numba")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/gpu_kernel_cases.py", "-q"],
        cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_comparison_rejects_nonfinite():
    import numpy as np

    from scripts.check_gpu_inference import difference

    for value in (np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            difference([value], [value])
    assert difference(np.array([], dtype=np.float32), np.array([], dtype=np.float32))["different"] == 0


def test_reject_unsupported_forest():
    with pytest.raises(ValueError, match="binary"):
        pack_forest({"objective": "regression"})
    with pytest.raises(ValueError, match="average_output"):
        pack_forest({"objective": "binary sigmoid:1", "average_output": True})
    with pytest.raises(ValueError, match="linear leaves"):
        pack_forest({"objective": "binary sigmoid:1", "linear_tree": True})
    with pytest.raises(ValueError, match="sigmoid"):
        pack_forest({"objective": "binary sigmoid:nan"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
def test_real_cuda_neighbors():
    from traceformer.config import ConeHeadConfig
    from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder

    points = torch.tensor([[0, 0, 0], [2, 0, 1], [0, 2, 1], [2, 0, 1], [3, 3, 2]])
    batches = torch.zeros(len(points), dtype=torch.int64)
    heads = (ConeHeadConfig(base_radius=2, slope=0.5, max_time_distance=3),)
    cpu = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=4)(points, batches)
    gpu = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=4, backend="cuda")(
        points.cuda(), batches.cuda())
    assert torch.equal(cpu.past_indices, gpu.past_indices.cpu())
    assert torch.equal(cpu.future_indices, gpu.future_indices.cpu())
