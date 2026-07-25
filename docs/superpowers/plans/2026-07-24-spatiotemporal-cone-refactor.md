# Spatiotemporal Cone Event Detector Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual-view HUAV model with a sparse block-token, bidirectional multi-cone attention detector whose cone parameters are justified by the real 99-sequence training set, then train and validate it end to end.

**Architecture:** Full eight-second event sequences are mapped to occupied `(x_block, y_block, t_block)` cells, pooled into learned tokens, updated by sparse past/future cone attention, and decoded back to one logit per original event. The training objective is only event-level focal plus per-sequence Dice loss; background sampling selects loss elements but never removes events from tokenization or context construction.

**Tech Stack:** Python 3.12, PyTorch 2.11, NumPy, SciPy `cKDTree`, PyYAML, pytest.

---

## File structure

- Create `src/huav/analysis/cone_slopes.py`: reproducible target-centroid slope measurement and report payloads.
- Create `src/huav/models/event_tokenizer.py`: sparse block assignment, event MLP pooling, token statistics, and event-to-token mapping.
- Create `src/huav/models/cone_neighbor_builder.py`: batch-isolated, capped past/future cone neighbor lookup.
- Create `src/huav/models/relative_position.py`: six-component relative motion encoding.
- Create `src/huav/models/cone_attention.py`: shared K/V bidirectional multi-head cone attention and gated fusion.
- Create `src/huav/models/cone_transformer.py`: pre-norm residual cone blocks and progressive temporal horizons.
- Create `src/huav/models/event_decoder.py`: parent-token plus local-event decoder.
- Create `src/huav/models/spatiotemporal_cone.py`: public `SpatioTemporalConeDetector`.
- Replace `src/huav/training/losses.py`: event focal/Dice loss only.
- Modify `src/huav/config.py`, `src/huav/training/engine.py`, package exports, YAML files, CLI descriptions, README, and architecture documentation.
- Replace obsolete dual-view tests with focused cone-model tests while retaining dataset, metric, CLI, checkpoint, and submission contract coverage.

### Task 1: Reproducible slope analysis

**Files:**
- Create: `tests/test_cone_slope_analysis.py`
- Create: `src/huav/analysis/__init__.py`
- Create: `src/huav/analysis/cone_slopes.py`
- Create: `docs/cone_slope_analysis.json`
- Create: `docs/cone_slope_analysis.md`

- [ ] **Step 1: Write failing tests**

Test a synthetic moving target with known block displacement, verify grouping by `name`, verify missing/short tracks are skipped, and assert percentile fields use units of spatial blocks per time block.

- [ ] **Step 2: Verify the tests fail for the missing analysis module**

Run: `python -m pytest tests/test_cone_slope_analysis.py -q`

Expected: collection failure naming `huav.analysis.cone_slopes`.

- [ ] **Step 3: Implement centroid-track analysis**

For every positive `(file, name, time_block)` group with at least three events, calculate median `(x, y)`, enumerate same-track pairs up to eight time blocks apart, and compute `max(abs(dx), abs(dy)) / dt` after converting pixels to configured spatial blocks. Also compute required slopes after subtracting base radii one and two.

- [ ] **Step 4: Generate the real-data reports**

Run the analyzer over all `99` configured training NPZ files with `4×4` spatial blocks and `128` time blocks. Record event/file counts, quantiles, per-lag quantiles, target spatial half-extent, method, and recommended cone heads.

- [ ] **Step 5: Verify analysis tests and report schema**

Run: `python -m pytest tests/test_cone_slope_analysis.py -q`

Expected: all tests pass and both report files contain slopes `0.5, 1.0, 1.5, 2.0`.

### Task 2: Configuration contract

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/huav/config.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`

- [ ] **Step 1: Replace dual-view configuration assertions with failing cone assertions**

Assert configurable block sizes, time blocks, token/event dimensions, layer/head counts, four per-head cone configurations, layer horizon scales, neighbor cap, causal/direction-dropout switches, focal/Dice weights, mixed precision, and no auxiliary-view fields.

- [ ] **Step 2: Verify expected configuration failures**

Run: `python -m pytest tests/test_config.py -q`

Expected: failures because cone dataclasses and YAML keys do not exist.

- [ ] **Step 3: Implement strict dataclasses and validation**

Require positive block/dimension/horizon values, `token_dim % num_heads == 0`, one cone per head, finite nonnegative radii/slopes, probabilities in range, and strictly positive neighbor/loss limits.

- [ ] **Step 4: Verify configuration tests**

Run: `python -m pytest tests/test_config.py -q`

Expected: all configuration tests pass.

### Task 3: Sparse event tokenizer

**Files:**
- Create: `tests/test_event_tokenizer.py`
- Create: `src/huav/models/event_tokenizer.py`

- [ ] **Step 1: Write failing tokenizer behavior tests**

Cover same-cell mapping, distinct cells, batch isolation, local coordinates, mean/max learned pooling, token statistics, event count preservation, and an empty `[0,5]` input.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_event_tokenizer.py -q`

Expected: missing `EventTokenizer`.

- [ ] **Step 3: Implement sparse block tokenization**

Use `torch.unique(..., return_inverse=True, return_counts=True)` over integer composite keys, `scatter_add_` for means/statistics, and `scatter_reduce_(reduce="amax")` for maxima. Generate no dense voxel volume.

- [ ] **Step 4: Verify tokenizer tests**

Run: `python -m pytest tests/test_event_tokenizer.py -q`

Expected: all tokenizer tests pass.

### Task 4: Sparse cone neighborhoods

**Files:**
- Create: `tests/test_cone_neighbor_builder.py`
- Create: `src/huav/models/cone_neighbor_builder.py`

- [ ] **Step 1: Write failing neighborhood tests**

Assert past/future temporal direction, Chebyshev radius `base + slope*dt`, different heads producing different sets, batch isolation, nearest-first capping, empty sides, and causal future suppression.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_cone_neighbor_builder.py -q`

Expected: missing `ConeNeighborBuilder`.

- [ ] **Step 3: Implement capped KD-tree lookup**

Build a `cKDTree` for each `(batch, time_block)` occupied spatial slice, query only configured temporal slices with `p=inf`, merge candidates by temporal then spatial distance, and return padded `-1` indices shaped `[tokens, heads, max_neighbors]`.

- [ ] **Step 4: Verify neighborhood tests**

Run: `python -m pytest tests/test_cone_neighbor_builder.py -q`

Expected: all neighborhood tests pass without constructing an `M×M` matrix.

### Task 5: Relative encoding and bidirectional attention

**Files:**
- Create: `tests/test_cone_attention.py`
- Create: `src/huav/models/relative_position.py`
- Create: `src/huav/models/cone_attention.py`

- [ ] **Step 1: Write failing attention tests**

Assert finite outputs for no past/future neighbors, shared K/V projection use, separate directional softmax paths, relative feature shape, normalized gates, causal behavior, gradients, and debug count statistics.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_cone_attention.py -q`

Expected: missing attention classes.

- [ ] **Step 3: Implement attention**

Encode `(dx, dy, signed_dt, Chebyshev distance, abs_dt, distance/abs_dt)` with an MLP score bias, add learned past/future direction embeddings to shared projected keys, mask padded neighbors safely, aggregate each direction separately, and fuse with token-conditioned softmax gates.

- [ ] **Step 4: Verify attention tests**

Run: `python -m pytest tests/test_cone_attention.py -q`

Expected: all attention tests pass and backward gradients are finite.

### Task 6: Encoder, decoder, and complete detector

**Files:**
- Create: `tests/test_spatiotemporal_cone_model.py`
- Create: `src/huav/models/cone_transformer.py`
- Create: `src/huav/models/event_decoder.py`
- Create: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/models/__init__.py`
- Modify: `src/huav/__init__.py`

- [ ] **Step 1: Write failing detector tests**

Cover required output keys, one logit per event, variable batch indices in `[N,5]`, no cross-batch attention, empty inputs, debug statistics, three progressive layers, causal mode, and end-to-end gradients.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py -q`

Expected: missing detector and encoder classes.

- [ ] **Step 3: Implement pre-norm cone encoder and event decoder**

Stack three residual attention/FFN blocks with progressively expanded temporal limits. Gather each event's updated parent token and concatenate its local coordinates, polarity, log block count, and bounded density before the MLP classifier.

- [ ] **Step 4: Verify complete model tests**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py -q`

Expected: all model tests pass with finite outputs and gradients.

### Task 7: Event-only loss and full-context training

**Files:**
- Modify: `tests/test_training_components.py`
- Modify: `tests/test_training_engine.py`
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/training/__init__.py`

- [ ] **Step 1: Write failing loss and engine tests**

Assert numerically stable focal loss, per-sample Dice including no-positive samples, configured weighted total, full-event forward followed by loss-index sampling, CPU training artifacts, checkpoint reproduction, and CUDA-only AMP enablement.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_training_components.py tests/test_training_engine.py -q`

Expected: failures on obsolete dual-view loss/output contracts.

- [ ] **Step 3: Implement focal/Dice and refactor the engine**

Remove all projection targets and auxiliary schedules. Always call the detector on every event, select sampled logits only inside the criterion, use AdamW, gradient clipping, optional CUDA autocast/scaling, unchanged event and EV-UAV-compatible validation metrics, and schema-versioned checkpoints.

- [ ] **Step 4: Verify training tests**

Run: `python -m pytest tests/test_training_components.py tests/test_training_engine.py -q`

Expected: all loss and training integration tests pass.

### Task 8: Remove active dual-view architecture and update documentation

**Files:**
- Delete: `src/huav/models/dual_view.py`
- Delete: `src/huav/models/blocks.py`
- Delete: `src/huav/models/projection.py`
- Delete: `tests/test_dual_view_model.py`
- Delete: `tests/test_projection.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `src/huav/cli/train.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update public/API tests to reject obsolete exports**

Assert `SpatioTemporalConeDetector` is public and `DualViewEventClassifier` is absent.

- [ ] **Step 2: Verify red state**

Run: `python -m pytest tests/test_public_api.py tests/test_project_integration.py -q`

Expected: public API still exposes the old class.

- [ ] **Step 3: Remove obsolete modules and rewrite user documentation**

Document the measured cone slopes, sparse tensor flow, configured CPU compromise of `8×8×64`, loss formula, full-context sampling semantics, training/evaluation commands, and output files.

- [ ] **Step 4: Verify no active dual-view references remain**

Run: `rg -n "DualView|xt_logits|yt_logits|FPN|view_event_weight|consistency_weight|temporal_weight" src configs README.md docs/architecture.md tests`

Expected: no matches outside historical plan/comparison documents.

### Task 9: Verification, smoke run, full training, and independent reevaluation

**Files:**
- Create through runtime: `outputs/cone_cpu_smoke_20260724/`
- Create through runtime: `outputs/cone_cpu_full_20260724/`
- Modify after runtime: `README.md`
- Create after runtime: `docs/cone_training_results_20260724.md`

- [ ] **Step 1: Run focused and complete automated verification**

Run: `python -m pytest -q`

Run: `python -m unittest discover -s tests -v`

Run: `python -m ruff check src tests`

Expected: zero failures and zero lint errors.

- [ ] **Step 2: Run real-data smoke training**

Run: `python -m huav.cli.train --config configs/huav_cpu_smoke.yaml --device cpu --output-dir outputs/cone_cpu_smoke_20260724`

Expected: one epoch completes, writes `best.pt`, configuration, history, manifest, metrics, and summary.

- [ ] **Step 3: Run the configured full training and validation**

Run: `python -m huav.cli.train --config configs/huav_base.yaml --device cpu --output-dir outputs/cone_cpu_full_20260724`

Expected: all 99 training and 24 validation files are processed for every configured epoch.

- [ ] **Step 4: Independently reevaluate the best checkpoint**

Run: `python -m huav.cli.evaluate --config outputs/cone_cpu_full_20260724/config.yaml --checkpoint outputs/cone_cpu_full_20260724/best.pt --device cpu --output outputs/cone_cpu_full_20260724/reproduced_metrics.json`

Expected: threshold and both metric groups exactly reproduce `metrics.json`.

- [ ] **Step 5: Record complete results**

Write the real hardware/runtime, data counts, parameter count, per-epoch losses/metrics, best threshold/epoch, independent reevaluation equality, artifact paths, and an honest comparison with the previous dual-view baseline to `docs/cone_training_results_20260724.md` and README.

