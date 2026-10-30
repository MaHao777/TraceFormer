"""读出runner的标签隔离、记忆干预、末层缓存和warmstart契约。"""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from traceformer.config import DEFAULT_CONE_WINDOWS
from traceformer.models.prefix_cone import PrefixConeEncoder, prepare_prefix_geometry


@pytest.fixture(scope="module")
def runner():
    scripts = Path(__file__).resolve().parents[1] / "archive/experiments/scripts"
    spec = importlib.util.spec_from_file_location("neural_memory_runner_test", scripts / "run_neural_event_memory.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(scripts))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


@pytest.fixture
def encoder_and_item():
    torch.manual_seed(37)
    encoder = PrefixConeEncoder(token_dim=64, num_heads=4, num_layers=3,
                                windows=DEFAULT_CONE_WINDOWS, mlp_ratio=2.0, dropout=0.1).eval()
    coords = torch.tensor([[x, y, t] for t in range(2) for y in range(2) for x in range(3)])
    batches = torch.zeros(len(coords), dtype=torch.long)
    penultimate = torch.randn(len(coords), 64)
    with torch.no_grad():
        geometry = prepare_prefix_geometry(coords, batches)
        tokens = encoder.final_norm(encoder.layers[-1](penultimate, coords, batches, geometry=geometry))
    event_count = len(coords) * 3
    return encoder, {"tokens": tokens, "penultimate": penultimate, "coords": coords,
                     "indices": torch.arange(event_count) % len(coords),
                     "locals": torch.randn(event_count, 6), "memory": torch.rand(event_count, 12),
                     "labels": torch.ones(event_count), "reference_logits": torch.ones(event_count) * 99}


class GuardedItem(dict):
    """禁止字段一旦被读取就立即失败，比只更换标签值更严格。"""

    def __init__(self, values, forbidden):
        super().__init__(values)
        self.forbidden = forbidden

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"readout accessed forbidden field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"readout accessed forbidden field {key}")
        return super().get(key, default)


@pytest.mark.parametrize("variant", ["B", "C", "U", "G"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_all_readouts_do_not_access_labels_or_reference_logits(runner, encoder_and_item,
                                                               variant, adaptive):
    encoder, item = encoder_and_item
    model = runner.Readout(variant, encoder, adaptive=adaptive).eval()
    forbidden = {"labels", "reference_logits"}
    if variant == "B":
        forbidden.add("memory")
    with torch.no_grad():
        expected = model(item)
        actual = model(GuardedItem(item, forbidden))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("variant", ["B", "C", "U", "G"])
def test_real_memory_changes_only_memory_enabled_readouts(runner, encoder_and_item, variant):
    encoder, item = encoder_and_item
    torch.manual_seed(101)
    model = runner.Readout(variant, encoder).eval()
    changed = {**item, "memory": torch.zeros_like(item["memory"])}
    with torch.no_grad():
        original = model(item)
        perturbed = model(changed)
    if variant == "B":
        assert torch.equal(original, perturbed)
    else:
        assert bool((original != perturbed).any())


@pytest.mark.parametrize("variant", ["B", "C", "U", "G"])
def test_adaptive_initial_output_matches_same_tail_frozen_tokens(runner, encoder_and_item, variant):
    encoder, item = encoder_and_item
    torch.manual_seed(73)
    frozen = runner.Readout(variant, encoder).eval()
    torch.manual_seed(73)
    adaptive = runner.Readout(variant, encoder, adaptive=True).eval()
    for name, value in frozen.decoder.state_dict().items():
        assert torch.equal(value, adaptive.decoder.state_dict()[name])
    with torch.no_grad():
        assert torch.equal(frozen(item), adaptive(item))


def test_decoder_prefix_warmstart_keeps_all_adaptive_tails_at_source(runner, encoder_and_item):
    encoder, item = encoder_and_item
    # 执行runner实际的state提取表达式，避免测试自身复制一个正确但失联的实现。
    train_tree = ast.parse(inspect.getsource(runner.train))
    assignment = next(node for node in ast.walk(train_tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "decoder_state"
                              for target in node.targets))
    extractor = compile(ast.Expression(assignment.value), runner.__file__, "eval")
    models = {}
    for variant in ("B", "C", "U", "G"):
        torch.manual_seed(37)
        previous = runner.Readout(variant, encoder)
        # 模拟已有训练checkpoint，同时混入不能加载进新tail的其他字段。
        state = {key: value.clone() + 0.125 for key, value in previous.state_dict().items()}
        state["tail.feed_forward.0.weight"] = torch.full((1,), 999.0)
        state["final_norm.weight"] = torch.full((64,), 999.0)
        torch.manual_seed(101)
        adaptive = runner.Readout(variant, encoder, adaptive=True)
        decoder_state = eval(extractor, {}, {"start": {"model_state": state}})
        adaptive.decoder.load_state_dict(decoder_state, strict=True)
        for key, value in decoder_state.items():
            assert torch.equal(value, adaptive.decoder.state_dict()[key])
        for key, value in encoder.layers[-1].state_dict().items():
            assert torch.equal(value, adaptive.tail.state_dict()[key])
        for key, value in encoder.final_norm.state_dict().items():
            assert torch.equal(value, adaptive.final_norm.state_dict()[key])
        models[variant] = adaptive.eval()
    for variant in ("C", "U", "G"):
        for key, value in models["B"].tail.state_dict().items():
            assert torch.equal(value, models[variant].tail.state_dict()[key])
    with torch.no_grad():
        assert all(torch.isfinite(model(item)).all() for model in models.values())
