# Restore Historical Full-K/V Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline and verify each restoration boundary before continuing.

**Goal:** Restore the runtime model and active configurations to the schema-4 full-K/V cone architecture that produced Score 0.9145447953504824, without modifying or deleting any existing training output.

**Architecture:** Reuse the exact model implementation from Git commit `7782111`, whose state-dict names, tensor shapes, model configuration, and checkpoint schema match the archived `full_q4_token64/best.pt`. Keep current artifact management, resume support, loss implementation, learning-rate infrastructure, and dataset pipeline unless they directly conflict with the historical model contract.

**Tech Stack:** Python 3.11, PyTorch, SciPy cKDTree, pytest, Ruff, YAML configuration.

---

### Task 1: Lock the restoration source and preservation boundary

**Files:**
- Preserve: `outputs/cpu_20260826_204924/**`
- Reference: `outputs/outputs_old/cpu_pool_token_ablation_20260730_172827/full_q4_token64/config.yaml`
- Reference: `outputs/outputs_old/cpu_pool_token_ablation_20260730_172827/full_q4_token64/best.pt`

- [ ] Verify the archived checkpoint reports schema 4, epoch 10, threshold 0.38749998807907104, and the exact historical model configuration.
- [ ] Record `git status --short` and do not run destructive Git commands.
- [ ] Confirm no edit targets any file under `outputs/` or `smoke_outputs/`.

### Task 2: Restore the exact historical model graph

**Files:**
- Restore from `7782111`: `src/huav/models/event_tokenizer.py`
- Restore from `7782111`: `src/huav/models/cone_neighbor_builder.py`
- Restore from `7782111`: `src/huav/models/relative_position.py`
- Restore from `7782111`: `src/huav/models/cone_attention.py`
- Restore from `7782111`: `src/huav/models/cone_transformer.py`
- Restore from `7782111`: `src/huav/models/spatiotemporal_cone.py`
- Delete: `src/huav/models/attention_residual.py`
- Delete: `src/huav/models/window_partition.py`
- Delete: `src/huav/models/window_attention.py`
- Delete: `src/huav/models/window_transformer.py`
- Delete: `src/huav/models/spatiotemporal_window.py`

- [ ] Restore the historical six-feature event embedding, four-query attention pooling, seven block statistics, and original event-to-token mapping.
- [ ] Restore per-head bidirectional cone neighborhoods and full per-time-slice K/V candidate construction.
- [ ] Restore three independent cone Transformer layers with time scales 0.5, 0.75, and 1.0.
- [ ] Restore the `SpatioTemporalConeDetector` output contract.

### Task 3: Restore the schema-4 public/configuration contract

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/models/__init__.py`
- Modify: `src/huav/__init__.py`
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Modify: `src/huav/experiments/dense_fit.py`
- Modify: active YAML files under `configs/`
- Modify: `pyproject.toml`

- [ ] Restore `ConeHeadConfig`, `num_layers=3`, the four historical cone heads, layer time scales, neighbor cap, causal mode, and direction dropout.
- [ ] Restore `SpatioTemporalConeDetector` as the only public detector.
- [ ] Set `CHECKPOINT_SCHEMA_VERSION = 4` and construct the detector with exactly the archived model fields.
- [ ] Restore SciPy as a runtime dependency for `cKDTree`.
- [ ] Make every retained active YAML use the exact historical model block; keep current training/evaluation fields that do not affect checkpoint model compatibility.

### Task 4: Restore regression coverage and checkpoint compatibility

**Files:**
- Restore/adapt: `tests/test_cone_neighbor_builder.py`
- Restore/adapt: `tests/test_cone_attention.py`
- Restore/adapt: `tests/test_spatiotemporal_cone_model.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_training_engine.py`
- Delete: `tests/test_window_partition.py`
- Delete: `tests/test_window_attention.py`
- Delete: `tests/test_spatiotemporal_window_model.py`
- Delete: `tests/test_attention_residual.py`

- [ ] Test the historical public class and reject the removed window detector.
- [ ] Test exact model configuration serialization and schema 4.
- [ ] Test that the archived historical checkpoint loads without missing or unexpected state keys.
- [ ] Run focused model/config/checkpoint tests.

### Task 5: Verify without changing archived results

**Files:**
- Read only: archived `best.pt`, `metrics.json`, and validation data.

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m ruff check src tests`.
- [ ] Run `git diff --check`.
- [ ] Independently evaluate the archived schema-4 checkpoint against all 24 validation files, writing any new diagnostic outside the archived run or comparing in memory so the archived result remains unchanged.
- [ ] Confirm reproduced Score 0.9145447953504824 and verify `outputs/cpu_20260826_204924` remains present and unchanged.
- [ ] Report that no new training was run unless separately requested.
