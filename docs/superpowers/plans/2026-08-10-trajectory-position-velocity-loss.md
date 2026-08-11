# Trajectory Position and Velocity Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development to implement this plan task-by-task. Execute inline in the current workspace so the user's uncommitted 64-block geometry remains in scope.

**Goal:** Replace residual STC and object-frame recall supervision with differentiable object-frame trajectory position and velocity losses while retaining focal, Dice, and false-alarm behavior.

**Architecture:** Focal and Dice continue to operate on the sampled event set. Trajectory and false-alarm losses operate on the full event logits already produced before sampling; trajectory centers are grouped by `(batch, object_id, floor(timestamp/frame_ms))`, use normalized coordinates and a GT-centered square ROI, and share one center extraction path for position and adjacent-frame velocity supervision.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff, PowerShell.

---

### Task 1: Specify differentiable trajectory behavior

**Files:**
- Modify: `tests/test_training_components.py`

- [ ] Add a failing position-loss test with two events in one object-frame, checking that probability mass near the GT center produces lower SmoothL1 loss than mass on the far ROI event.
- [ ] Add a failing gradient test that calls `trajectory_position_loss(...).backward()` and requires finite, non-zero logit gradients without thresholding.
- [ ] Add a failing velocity-loss test across consecutive frames, checking matching predicted deltas beat a stationary predicted centroid.
- [ ] Add failing edge-case tests for missing adjacent frames, batch/object isolation, empty inputs, invalid shapes, and non-positive frame/ROI/resolution values.
- [ ] Run `python -m pytest -q tests/test_training_components.py` and confirm failure is caused by the missing trajectory APIs and old six-term return contract.

### Task 2: Implement trajectory helpers and simplify EventFocalDiceLoss

**Files:**
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`

- [ ] Remove `_sparse_cone_neighbors`, `spatiotemporal_continuity_weight`, `object_frame_recall_loss`, their imports, exports, validation, constructor fields, and forward use.
- [ ] Implement a private center extractor that validates aligned full-event tensors, computes normalized GT centers by segmented mean, restricts probability-weighted centroids to a GT-centered ROI, and returns differentiable predicted centers plus `(batch, object, frame)` keys.
- [ ] Implement `trajectory_position_loss` as mean coordinate-wise SmoothL1 over valid object-frames.
- [ ] Implement `trajectory_velocity_loss` as mean SmoothL1 over rows sharing batch/object with frame difference exactly one.
- [ ] Make `EventFocalDiceLoss.forward()` return exactly `total`, `focal`, `dice`, `false_alarm`, `trajectory_position`, and `trajectory_velocity`; compute focal without custom weights and use full events for both trajectory terms.
- [ ] Run `python -m pytest -q tests/test_training_components.py` and confirm the new tests pass.

### Task 3: Replace the configuration and engine contract

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `scripts/check_largest_sequence.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_project_integration.py`
- Modify: `tests/test_training_visualization.py`

- [ ] First change tests to require defaults `lambda_dice=0.5`, `lambda_false_alarm=0.05`, `lambda_trajectory_position=1.0`, `lambda_trajectory_velocity=0.5`, and `trajectory_roi_radius=16.0`, with all STC/object-frame fields absent.
- [ ] Run the focused config/engine/integration tests and confirm they fail against the old contract.
- [ ] Update all configs, criterion construction sites, loss accumulators, full-object-id forwarding, resource checker, and visualization fixtures to the six-key contract.
- [ ] Increment the checkpoint schema because saved-config loss behavior changed incompatibly.
- [ ] Run focused tests, then `python -m pytest -q`, `python -m ruff check src tests scripts`, and `git diff --check`.

### Task 4: Complete the repository acceptance loop

**Files:**
- Create at runtime: `outputs/cpu_<timestamp>/`

- [ ] Run the `train_094.npz` forward/loss/backward resource gate with the retained `8x8/64` geometry.
- [ ] Train and independently evaluate `configs/huav_score_optimized_smoke.yaml`; require identical saved and reproduced metric hashes.
- [ ] Train all 99 files and validate all 24 files with `configs/huav_score_optimized.yaml` in one fresh CPU run directory.
- [ ] Independently evaluate that run's `config.yaml` and `best.pt`; require byte-identical metrics and complete event confusion counts.
- [ ] Predict from the same checkpoint, then audit 24 flat TXT files, source `x y t p` order/value equality, binary labels, total rows/positives, and ZIP structure.
- [ ] Re-run pytest, Ruff, and `git diff --check`; report any failed stage as incomplete rather than claiming acceptance.
