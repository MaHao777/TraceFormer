"""六组池化/过去局部注意力的工厂、严格契约与完整训练恢复集成。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import test_paper_ablation
import torch
from test_training_engine import _write_sequence

from traceformer.config import TraceFormerConfig
from traceformer.experiments.ablation import (
    ComponentCoreSpec,
    LocalPoolingSpec,
    parse_ablation,
)
from traceformer.experiments.local_pooling import VARIANTS
from traceformer.experiments.past_local import PastLocalEncoder
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.spatial_pooling import SpatialConvTokenizer
from traceformer.training.engine import (
    _dataset,
    _validate_checkpoint_contract,
    build_model,
    evaluate_checkpoint,
    run_training,
)

small_config = test_paper_ablation.small_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def local_config(small_config):
    # 原微型数据均落在同一时间块；改用跨四块数据，真实经过过去邻域与卷积。
    for split in ("train", "val"):
        _write_sequence(small_config.dataset.root / split / f"{split}_000.npz", 0)
    return replace(small_config, model=replace(small_config.model, num_layers=3, dropout=0.1),
                   training=replace(small_config.training, epochs=2,
                                    save_training_state_interval=1)).validate()


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_nested_equal(first, second)
    else:
        assert left == right


def test_six_factories_have_fixed_parameter_counts_and_common_initial_rng():
    config = TraceFormerConfig.default()
    torch.manual_seed(37)
    original = build_model(config, ablation=ComponentCoreSpec("M1_C4"))
    expected_rng = torch.get_rng_state()
    reference = original.state_dict()
    expected_counts = {"S0": 142774, "S1": 143350, "S2": 143926,
                       "L0": 129526, "L1": 130102, "L2": 130678}
    assert set(VARIANTS) == set(expected_counts)
    models = {}
    for variant in VARIANTS:
        torch.manual_seed(37)
        model = build_model(config, ablation=LocalPoolingSpec(variant))
        models[variant] = model
        assert torch.equal(torch.get_rng_state(), expected_rng), variant
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_counts[variant]
        assert model.ablation_spec == {"variant": variant, "version": 4}
        assert len(model.encoder.layers) == 3
        for name, value in model.state_dict().items():
            if name.startswith("tokenizer.spatial_kernels."):
                continue
            # 局部层仅重命名共享 QKV/输出投影，其余公共参数逐位保留。
            source = name.replace(".attention.attention.", ".attention.center_attention.")
            assert source in reference, name
            assert torch.equal(value, reference[source]), (variant, name)
        if variant.startswith("L"):
            assert isinstance(model.encoder, PastLocalEncoder)
            names = " ".join(model.state_dict())
            assert not any(name in names for name in
                           ("center_attention", "pool_value", "shape_embedding",
                            "direction_embedding"))
            assert all(layer.attention.offsets.shape == (34, 3)
                       for layer in model.encoder.layers)
        assert isinstance(model.tokenizer, SpatialConvTokenizer) == (variant[1] != "0")
    for kernels in (1, 2):
        summary = models[f"S{kernels}"].tokenizer.spatial_kernels
        local = models[f"L{kernels}"].tokenizer.spatial_kernels
        assert_nested_equal(summary.state_dict(), local.state_dict())
    assert torch.equal(models["S1"].tokenizer.spatial_kernels[0].weight,
                       models["S2"].tokenizer.spatial_kernels[0].weight)


def test_local_pooling_contract_rejects_wrong_variants_stages_and_foreign_fields():
    for variant in VARIANTS:
        spec = LocalPoolingSpec(variant)
        assert parse_ablation(spec.to_dict()) == spec
        spec.validate_stage("full")
        with pytest.raises(ValueError, match="full training"):
            spec.validate_stage("refinement")
    for variant in ("S3", "L3", "s0", "M1_C4", "Full", ""):
        with pytest.raises(ValueError, match="contract"):
            parse_ablation({"version": 4, "variant": variant})
    for field, value in (("sparse", False), ("local_neighbors", 26), ("pooling", "mean"),
                         ("radius", 3), ("num_layers", 6)):
        with pytest.raises(TypeError):
            parse_ablation({"version": 4, "variant": "S0", field: value})
    with pytest.raises(ValueError, match="contract"):
        LocalPoolingSpec("S0", version=3).validate()


@pytest.mark.parametrize("variant", VARIANTS)
def test_local_pooling_factory_requires_exactly_three_layers(variant):
    config = TraceFormerConfig.default()
    config = replace(config, model=replace(config.model, num_layers=2))
    with pytest.raises(ValueError, match="three encoder layers"):
        build_model(config, ablation=LocalPoolingSpec(variant))


def test_s0_matches_previous_m1_c4_weights_and_all_forward_outputs(local_config):
    torch.manual_seed(37)
    previous = build_model(local_config, ablation=ComponentCoreSpec("M1_C4")).eval()
    torch.manual_seed(37)
    current = build_model(local_config, ablation=LocalPoolingSpec("S0")).eval()
    assert_nested_equal(previous.state_dict(), current.state_dict())
    sequence = _dataset(local_config, "val")[0]
    with torch.no_grad():
        before = previous(sequence.events)
        after = current(sequence.events)
    assert_nested_equal(before, after)
    assert current(sequence.events[:0])["event_logits"].shape == (0,)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_six_train_evaluate_resume_and_reject_cross_group_states(
    variant, local_config, tmp_path,
):
    spec = LocalPoolingSpec(variant)
    directory = tmp_path / variant
    run_training(local_config, output_dir=directory, ablation=spec)
    checkpoint_path = directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    saved_metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert evaluate_checkpoint(local_config, checkpoint_path, ablation=spec) == saved_metrics
    report = evaluate_ablation(local_config, checkpoint_path, spec, save_predictions=True)
    assert report["metrics"] == saved_metrics
    assert report["ablation"] == spec.to_dict()
    prediction = torch.load(directory / "predictions" / "val_000.pt", weights_only=True)
    assert prediction["event_logits"].shape == (4,)
    assert torch.isfinite(prediction["event_logits"]).all()

    resumed = tmp_path / f"{variant}_resumed"
    first_state = directory / "training_states" / "epoch_001.pt"
    run_training(local_config, output_dir=resumed, ablation=spec, resume_state_path=first_state)
    complete = torch.load(directory / "training_state.pt", weights_only=False)
    restored = torch.load(resumed / "training_state.pt", weights_only=False)
    for field in ("model_state", "optimizer_state", "controller_state", "best_metrics",
                  "best_checkpoint", "torch_rng_state", "order_generator_state",
                  "sampling_generator_state"):
        assert_nested_equal(complete[field], restored[field])
    assert complete["epoch"] == restored["epoch"] == 2
    assert evaluate_checkpoint(local_config, resumed / "best.pt", ablation=spec) == saved_metrics

    # 契约必须先于张量加载拒绝跨组权重，即使部分结构与参数形状恰好相同。
    wrong_specs = [None, ComponentCoreSpec("M1_C4"),
                   *(LocalPoolingSpec(other) for other in VARIANTS if other != variant)]
    for wrong in wrong_specs:
        with pytest.raises(ValueError, match="ablation"):
            _validate_checkpoint_contract(checkpoint, local_config, ablation=wrong)
    wrong = LocalPoolingSpec(("L" if variant.startswith("S") else "S") + variant[1])
    with pytest.raises(ValueError, match="ablation"):
        evaluate_checkpoint(local_config, checkpoint_path, ablation=wrong)
    with pytest.raises(ValueError, match="ablation"):
        run_training(local_config, output_dir=tmp_path / f"{variant}_wrong_resume",
                     ablation=wrong, resume_state_path=first_state)
