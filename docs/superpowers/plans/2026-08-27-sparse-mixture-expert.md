# Sparse Mixture Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative token-level sparse residual expert to the restored schema-4 full-K/V detector while preserving one prediction per original event.

**Architecture:** The existing tokenizer and cone encoder process both the full event stream and one deterministic 1.5x high-threshold subset with shared weights. Sparse tokens are aligned to their full-view token by batch and spacetime-block coordinates, a small adapter produces a residual, and a token router gates that residual before the unchanged original-event decoder. Training uses the fused full-event output as the main task and adds full-branch preservation, sparse-view supervision, positive-frame consistency, background upper-bound consistency, sparse block-peak suppression, and gate regularization.

**Tech Stack:** Python 3.11, PyTorch, dataclass/YAML configuration, pytest, Ruff.

---

### Task 1: Lock the configuration and checkpoint contract

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Test: `tests/test_config.py`

- [ ] Add `SparseMixtureConfig` with deterministic 1.5x view eligibility, router initialization, warmup/ramp, and all auxiliary-loss weights.
- [ ] Reject configurations that enable legacy threshold augmentation and sparse mixture simultaneously.
- [ ] Add YAML parsing/serialization tests and enable sparse mixture only in score-oriented production/smoke configs.
- [ ] Bump `CHECKPOINT_SCHEMA_VERSION` from 4 to 5 because the state dict and model contract change.
- [ ] Run `python -m pytest tests/test_config.py -q`; expect all tests to pass.

### Task 2: Implement token alignment and conservative residual routing

**Files:**
- Create: `src/huav/models/sparse_mixture.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/models/__init__.py`
- Test: `tests/test_sparse_mixture.py`
- Test: `tests/test_spatiotemporal_cone_model.py`

- [ ] Write failing tests showing sparse tokens align by `(batch,t,y,x)`, missing sparse tokens have zero gates, and zero mixture weight reproduces full logits exactly.
- [ ] Implement `SparseTokenMixture` with shared-coordinate alignment, `LayerNorm` MLP adapter, router bias `-2`, presence masking, and residual LayerScale initialized to `0.01`.
- [ ] Extend `SpatioTemporalConeDetector.forward` with optional sparse events/source indices and return `event_logits`, `full_event_logits`, `sparse_event_logits`, `sparse_source_indices`, and `mixture_gate` without changing original-event order.
- [ ] Verify paired outputs backpropagate through tokenizer, shared encoder, sparse adapter, router, and decoder.
- [ ] Run `python -m pytest tests/test_sparse_mixture.py tests/test_spatiotemporal_cone_model.py -q`; expect all tests to pass.

### Task 3: Add asymmetric sparse losses

**Files:**
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`
- Test: `tests/test_losses.py`

- [ ] Add a background upper-bound consistency loss that detaches full-view logits and penalizes sparse background logits only when they exceed the full branch plus a margin.
- [ ] Add a sparse `50 ms x 8x8` block-peak loss using per-block maxima and per-frame top-k peaks.
- [ ] Add finite empty-input, positive/background selection, detachment, spatial grouping, and backward tests.
- [ ] Run `python -m pytest tests/test_losses.py -q`; expect all tests to pass.

### Task 4: Train and evaluate the true paired model path

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `scripts/check_largest_sequence.py`
- Test: `tests/test_training_engine.py`

- [ ] Create one helper that builds the deterministic sparse view and calls the paired model for training, validation, independent evaluation, and submission inference.
- [ ] Keep main focal/Dice/object-frame/false-alarm loss on fused logits over all original events.
- [ ] Add `0.25 * full_aux`, ramped `0.15 * sparse_aux`, positive-frame consistency, background consistency, block-peak loss, and gate L1 regularization only for eligible paired streams.
- [ ] Record each component, gate usage, retention, and paired-file counts in `history.json` and `sparse_mixture.json`.
- [ ] Update the largest-sequence resource gate to perform full+sparse mixed forward/loss/backward.
- [ ] Run focused engine tests; expect full-event forwarding, deterministic source alignment, checkpoint resume, evaluation, and submission behavior to pass.

### Task 5: Complete repository and data-path verification

**Files:**
- Modify: `docs/architecture.md`
- Read only: `data/ev-uav/train/train_094.npz`

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m ruff check src tests`.
- [ ] Run `git diff --check`.
- [ ] Run the real `train_094` full+sparse forward/loss/backward resource gate and save its report.
- [ ] Run the configured real-data smoke train and independent smoke reevaluation.
- [ ] If every preflight passes, run the mandatory fresh 99-train/24-validation full training, independent reevaluation, prediction generation, and submission audit in a new output directory.
- [ ] Report the exact model/loss changes and every verification result without describing an incomplete training pipeline as complete.
