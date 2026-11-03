"""S1派生实验的科学定义、数值、梯度、公共初始化与训练恢复。"""

import json
from dataclasses import replace

import pytest
import torch
from test_local_pooling import assert_nested_equal, local_config, one_thread, small_config

from traceformer.config import TraceFormerConfig
from traceformer.experiments.ablation import LocalPoolingSpec, S1PaperSpec, parse_ablation
from traceformer.experiments.past_local import PastLocalAttention
from traceformer.experiments.s1_paper import NEW_VARIANTS, initialization_audit, specification
from traceformer.training.engine import (
    _validate_checkpoint_contract,
    build_model,
    evaluate_checkpoint,
    run_training,
)

__all__ = ["local_config", "one_thread", "small_config"]


def test_s1_initialization_pairing_actual_sizes_and_geometry():
    torch.manual_seed(83)
    rng = torch.get_rng_state()
    audit = initialization_audit(TraceFormerConfig.default())
    assert audit["passed"] and torch.equal(rng, torch.get_rng_state())
    rows = {row["variant"]: row for row in audit["variants"]}
    assert {k: v["parameters"] for k, v in rows.items()} == {
        "S1": 143350, "S0": 142774, "A1": 79603, "A2": 135172, "P1": 142838,
        "G1": 143350, "G2": 142966, "F1": 143158, "T1": 130102}
    assert rows["G1"]["windows"] == [[6, 2], [3, 4], [1, 8]] and rows["G1"]["fixed_radius"]
    assert rows["G2"]["windows"] == [[4, 4]]
    assert rows["T1"]["direct_union_candidates"] == 552
    assert rows["P1"]["pool_kernel_sizes"] == [[1, 1]]
    assert rows["A1"]["cross_token_interaction"] is False


def test_s1_contract_rejects_wrong_variants_stages_and_foreign_fields():
    for variant in NEW_VARIANTS:
        spec = S1PaperSpec(variant)
        assert parse_ablation(spec.to_dict()) == spec
        with pytest.raises(ValueError, match="full training"):
            spec.validate_stage("refinement")
    for variant in ("S1", "E0", "A0", "Full", ""):
        with pytest.raises(ValueError, match="contract"):
            S1PaperSpec(variant).validate()
    with pytest.raises(TypeError):
        parse_ablation({"version": 6, "variant": "T1", "neighbor_cap": 32})


def test_union_is_complete_deduplicated_and_local_output_gradient_matches_enumeration():
    config = TraceFormerConfig.default()
    config = replace(config, model=replace(config.model, dropout=0))
    module = build_model(config, ablation=S1PaperSpec("T1")).encoder.layers[0].attention
    assert isinstance(module, PastLocalAttention)
    windows = ((8, 2), (4, 4), (2, 8))
    expected_offsets = {(x, y, -d) for d in range(1, 9) for x in range(-8, 9) for y in range(-8, 9)
                        if any(d <= h and max(abs(x), abs(y)) <= r*d//h for r, h in windows)}
    assert set(map(tuple, module.offsets.tolist())) == expected_offsets
    assert len(expected_offsets) == len(module.offsets) == 552
    module.chunk_size = 7
    coords = torch.tensor([[x, y, t] for t in (0, 1, 3, 6, 8) for x in range(2) for y in range(2)])
    batches = torch.arange(len(coords)) % 2
    tokens = torch.randn(len(coords), 64, requires_grad=True)
    actual = module(tokens, coords, batches).features
    gate = (torch.sigmoid(module.pool_weight(tokens)) + 1e-4).log().flatten()
    expected = []
    for index in range(len(coords)):
        delta = coords - coords[index]
        d, spatial = -delta[:, 2], delta[:, :2].abs().amax(-1)
        mask = torch.zeros(len(coords), dtype=torch.bool)
        for r, h in windows:
            mask |= (d >= 1) & (d <= h) & (spatial*h <= r*d)
        mask &= batches == batches[index]
        if mask.any():
            result, _ = module.attention(tokens[index:index+1, None], tokens[mask][None],
                tokens[mask][None], attn_mask=gate[mask][None], need_weights=False)
            expected.append(result[0, 0])
        else:
            expected.append(tokens[index] * 0)
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    probe = torch.randn_like(actual)
    inputs = (tokens, *module.parameters())
    observed = torch.autograd.grad((actual * probe).sum(), inputs)
    reference = torch.autograd.grad((expected * probe).sum(), inputs)
    for left, right in zip(observed, reference, strict=True):
        torch.testing.assert_close(left, right, atol=5e-5, rtol=5e-5)


def test_pointwise_pooling_retains_activation_and_has_no_neighbor_cell_mixing():
    model = build_model(TraceFormerConfig.default(), ablation=S1PaperSpec("P1"))
    tokenizer = model.tokenizer
    with torch.no_grad():
        tokenizer.spatial_kernels[0].weight.fill_(1)
    values = torch.tensor([[2.0]*64, [3.0]*64])
    cells = torch.tensor([0, 1])
    actual = tokenizer._pool_chunk(values, cells, torch.zeros(2, dtype=torch.long), 1)
    expected = torch.nn.functional.gelu(values).mean(0, keepdim=True)
    torch.testing.assert_close(actual, expected)
    changed_cells = torch.tensor([0, 15])
    torch.testing.assert_close(actual, tokenizer._pool_chunk(values, changed_cells, torch.zeros(2, dtype=torch.long), 1))


@pytest.mark.parametrize("variant", NEW_VARIANTS)
def test_new_variants_train_reevaluate_resume_and_reject_foreign_checkpoint(variant, local_config, tmp_path):
    spec = S1PaperSpec(variant)
    output = tmp_path / variant
    run_training(local_config, output_dir=output, ablation=spec)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert evaluate_checkpoint(local_config, output / "best.pt", ablation=spec) == metrics
    resumed = tmp_path / (variant + "_resumed")
    run_training(local_config, output_dir=resumed, ablation=spec,
                 resume_state_path=output / "training_states/epoch_001.pt")
    complete = torch.load(output / "training_state.pt", weights_only=False)
    restored = torch.load(resumed / "training_state.pt", weights_only=False)
    for field in ("model_state", "optimizer_state", "controller_state", "best_metrics", "best_checkpoint",
                  "torch_rng_state", "order_generator_state", "sampling_generator_state"):
        assert_nested_equal(complete[field], restored[field])
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    for foreign in (LocalPoolingSpec("S1"), S1PaperSpec("A2" if variant != "A2" else "A1")):
        with pytest.raises(ValueError, match="ablation contract"):
            _validate_checkpoint_contract(checkpoint, local_config, ablation=foreign)
    model = build_model(local_config, ablation=spec)
    assert model(torch.empty(0, 4))["event_logits"].shape == (0,)
    if variant == "A2":
        assert metrics["competition"]


def test_references_use_original_contracts():
    assert specification("S1") == LocalPoolingSpec("S1")
    assert specification("S0") == LocalPoolingSpec("S0")
