# Restore Historical-Best Model and Loss Implementation Plan

> **For Codex:** Execute this plan end to end in the current checkout. Preserve unrelated user edits and all historical output directories.

**Goal:** Remove the complete Token Importance K/V-cropping surface and restore the loss/training behavior used by the historical best 8x8/64, token_dim=64 run.

**Architecture:** Restore full-query/full-key/full-value cone attention from the pre-cropping implementation. Keep the current output lifecycle and evaluation/export infrastructure, but migrate the training criterion back to Focal + per-sample Dice + object-frame recall + full-event false-alarm loss. Increment the checkpoint schema because the model state and loss contract change.

**Tech Stack:** Python, PyTorch, pytest, Ruff, YAML, PowerShell CLI.

---

### Task 1: Lock the rollback contract with failing tests

**Files:**
- Modify: `tests/test_training_components.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cone_attention.py`
- Delete: `tests/test_token_importance.py`

1. Restore object-frame and false-alarm loss tests.
2. Assert production configs expose no Token Importance fields and use the historical-best loss weights.
3. Run focused tests and confirm they fail against the current trajectory/K/V implementation.

### Task 2: Remove K/V cropping completely

**Files:**
- Modify: `src/huav/models/cone_attention.py`
- Modify: `src/huav/models/cone_transformer.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/models/__init__.py`
- Delete: `src/huav/models/token_importance.py`
- Modify: `src/huav/config.py`
- Modify: `configs/*.yaml`

1. Restore the pre-cropping attention path so every token remains Q, K, and V.
2. Remove Token Importance constructor parameters, exports, config fields, validation, YAML keys, and tests.
3. Preserve the existing token geometry and 64-dimensional historical-best production width.

### Task 3: Restore the historical-best loss contract

**Files:**
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`
- Modify: `src/huav/training/engine.py`
- Modify: `scripts/check_largest_sequence.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_project_integration.py`
- Modify: `tests/test_training_visualization.py`

1. Restore object-frame recall and full-event false-alarm loss.
2. Remove trajectory position/velocity code, configuration, logging, and tests.
3. Set score configs to lambda_dice=0.75, lambda_object_frame=0.25, object_frame_logit_margin=2.0, lambda_false_alarm=0.05.
4. Restore checkpoint schema 4 compatibility so the historical-best `best.pt` can be called directly, without disturbing current run-directory, visualization, evaluation, or export behavior.

### Task 4: Verify implementation and resource safety

1. Run focused tests, full pytest, Ruff, and git diff check.
2. Confirm repository-wide active code/config/tests contain no K/V-cropping or trajectory-loss surface.
3. Run the largest-sequence forward/loss/backward gate on `train_094.npz`.
4. Run real-data smoke training and independent smoke evaluation.

### Task 5: Complete the mandatory 99/24 experiment loop

1. Do not retrain: use the historical-best `outputs/outputs_old/cpu_pool_token_ablation_20260730_172827/full_q4_token64/best.pt`.
2. Independently reevaluate that checkpoint using its saved `config.yaml`; require byte-identical metrics.
3. Reuse or regenerate submission artifacts only when requested.
4. Rerun pytest, Ruff, diff check, and status; confirm the reproduced score is 0.9145447953504824.
