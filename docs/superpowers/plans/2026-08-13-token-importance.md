# Token Importance K/V Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sample-normalized density, incoming-attention, and trajectory-continuity importance scores that retain every token as a Query while selecting the highest-scoring fraction of each original cone as K/V.

**Architecture:** A stateless `TokenImportance` module receives token features, event counts, coordinates, batch ids, full-cone neighbor indices, and full-cone pre-dropout attention weights. It computes the three requested components and per-sample min-max normalization, combines them with configurable positive weights, and masks each Query/head/direction neighborhood to its configurable Top-K fraction. `BidirectionalConeAttention` performs a full-cone score pass followed by the selected K/V attention pass; transformer residuals and the event decoder continue to update and consume all Query tokens.

**Tech Stack:** Python 3, PyTorch sparse tensor operations, SciPy-backed existing cone neighborhoods, dataclass/YAML configuration, pytest, Ruff.

---

### Task 1: Specify Token Importance behavior with failing tests

**Files:**
- Create: `tests/test_token_importance.py`
- Modify: `tests/test_cone_attention.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Test component scores and sample isolation**

  Build small two-sample tensors with known event counts, full-cone attention weights, matching/mismatching features, and coordinates. Assert the three component tensors lie in `[0,1]`, each sample is normalized independently, and the final score equals `0.3*density + 0.3*attention + 0.4*track`.

- [ ] **Step 2: Test per-cone Top-K semantics**

  Provide padded neighborhoods with different valid sizes and a retention ratio of `0.5`. Assert each Query/head/direction keeps `ceil(valid_count * 0.5)` entries, keeps at least one valid K/V when the cone is non-empty, retains no padding, and selects the highest token importance.

- [ ] **Step 3: Test Query/output preservation and debug observability**

  Run `BidirectionalConeAttention` on more Query tokens than selected neighbors. Assert output feature count equals input token count, selected neighbor counts do not exceed original counts, and gradients reach every input Query and every attention parameter.

- [ ] **Step 4: Test configuration and checkpoint contract**

  Assert defaults load as enabled with ratio `0.5` and weights `0.3/0.3/0.4`, invalid ratios/weights fail validation, model construction receives the values, and the checkpoint schema expectation advances from 7 to 8.

- [ ] **Step 5: Run focused tests and verify RED**

  Run `python -m pytest -q tests/test_token_importance.py tests/test_cone_attention.py tests/test_config.py tests/test_training_engine.py`. Expected result: collection/import or assertion failures caused by the absent module/configuration, not syntax or fixture errors.

### Task 2: Implement the independent scoring and selection module

**Files:**
- Create: `src/huav/models/token_importance.py`
- Modify: `src/huav/models/__init__.py`

- [ ] **Step 1: Implement sample-wise normalization**

  Min-max normalize each score independently for every `token_batch_indices` value. Return zero for a constant component to avoid NaN and cross-sample leakage.

- [ ] **Step 2: Implement relative-density scoring**

  Deduplicate the union of valid past/future cone neighbors across heads for each Query, compute `log(1 + N_i / (mean_neighbor_count + eps))`, and use zero when a Query has no neighbor.

- [ ] **Step 3: Implement incoming-attention scoring**

  Scatter full-cone pre-dropout weights from Query/head/slot edges to their K/V token ids and divide by the number of valid incoming head edges, thereby averaging across Queries and heads.

- [ ] **Step 4: Implement trajectory-continuity scoring**

  For past and future cones independently, score each candidate by the mean of cosine similarity mapped to `[0,1]` and reciprocal Chebyshev spatial proximity, take the best candidate, and average the available direction scores.

- [ ] **Step 5: Implement per-neighborhood Top-K masking**

  Gather final importance for each valid neighbor, deterministically rank descending, retain `ceil(valid_count * ratio)` candidates (minimum one for non-empty rows), and fill rejected/padded entries with `-1`.

- [ ] **Step 6: Run module tests and verify GREEN**

  Run `python -m pytest -q tests/test_token_importance.py` and expect all tests to pass.

### Task 3: Integrate selection into cone attention without changing outputs

**Files:**
- Modify: `src/huav/models/cone_attention.py`
- Modify: `src/huav/models/cone_transformer.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/training/engine.py`

- [ ] **Step 1: Thread exact token event counts to every transformer layer**

  In the detector, compute `torch.bincount(event_token_indices, minlength=token_count)` and pass it through the encoder/block to attention. Do not alter public output names, event order, or event decoder inputs.

- [ ] **Step 2: Split attention weight computation from aggregation**

  Produce full-cone pre-dropout weights for importance, run Top-K selection, then recompute softmax over selected K/V entries and apply the existing dropout/value aggregation. Preserve past/future gates, causal mode, direction dropout, relative position bias, and shared Q/K/V projections.

- [ ] **Step 3: Add selection diagnostics to existing debug stats**

  Keep existing stats keys and add original/selected neighbor counts plus detached importance components/final scores when debug statistics are enabled.

- [ ] **Step 4: Run attention/model tests and verify GREEN**

  Run `python -m pytest -q tests/test_cone_attention.py tests/test_spatiotemporal_cone_model.py tests/test_training_engine.py` and expect all tests to pass with unchanged event-logit shape and full gradient coverage.

### Task 4: Add ablation-ready configuration and schema 8

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`

- [ ] **Step 1: Add and validate model configuration fields**

  Add `token_importance_enabled`, `token_importance_topk_ratio`, and the three component weights. Require a boolean enable flag, ratio in `(0,1]`, finite non-negative weights, and a positive weight sum.

- [ ] **Step 2: Enable requested defaults in all maintained configs**

  Set enabled true, ratio `0.5`, and weights `0.3/0.3/0.4`. A future ablation can restore all K/V with ratio `1.0` or bypass scoring with enabled false.

- [ ] **Step 3: Advance the checkpoint schema**

  Change `CHECKPOINT_SCHEMA_VERSION` to 8 so pre-importance checkpoints cannot be mistaken for the new model contract.

- [ ] **Step 4: Run configuration tests and verify GREEN**

  Run `python -m pytest -q tests/test_config.py tests/test_training_engine.py` and expect all tests to pass.

### Task 5: Verify code and real-data resource behavior

**Files:**
- Modify only if a failing test identifies a defect in the scoped implementation.

- [ ] **Step 1: Run complete static/regression gates**

  Run `python -m pytest -q`, `python -m ruff check src tests`, and `git diff --check`; all must exit 0.

- [ ] **Step 2: Reproduce largest-sequence forward/loss/backward**

  Run the repository largest-sequence checker on `train_094.npz` with the production `8x8/64` configuration and confirm finite loss/gradients without the known Windows `c10.dll` access violation.

- [ ] **Step 3: Run real-data CPU smoke and independent evaluation**

  Run `python -m huav.cli.train --config configs/huav_score_optimized_smoke.yaml --device cpu --note "Token Importance Top-K K/V smoke"`, then evaluate that run's saved `config.yaml` and `best.pt`. Confirm `metrics.json` and `reproduced_metrics.json` are byte-identical.

- [ ] **Step 4: Re-run final code gates**

  Re-run the complete pytest, Ruff, diff, and status commands after smoke evaluation.

### Task 6: Start the requested full CPU training without babysitting

**Files:**
- Generated only: new `outputs/cpu_YYYYMMDD_HHMMSS[_NN]/` run directory and log artifacts.

- [ ] **Step 1: Launch a long-lived detached process**

  Start `python -m huav.cli.train --config configs/huav_score_optimized.yaml --device cpu --note "Token Importance Top-K K/V: density 0.3, attention 0.3, track 0.4, keep 50%"` through a hidden PowerShell process with stdout/stderr redirected to persistent log files.

- [ ] **Step 2: Verify startup, not completion**

  Confirm the process remains alive after startup, identify the newly allocated run directory, and inspect its `experiment.md`/config plus initial log output for an immediate exception. Report PID, run directory, and log paths. Do not wait for 18-epoch training, independent full reevaluation, or submission packaging because the user explicitly requested launch-only handling.
