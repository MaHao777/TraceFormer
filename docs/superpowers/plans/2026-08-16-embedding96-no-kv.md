# 96-D No-K/V-Cropping Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the pre-K/V-cropping detector behavior and measure whether widening both event embeddings and token vectors from 64 to 96 improves complete 99/24 validation performance.

**Architecture:** Keep the pre-cropping trajectory-loss, q4, 8x8/64, seed-37, 18-epoch contract fixed. Disable Token Importance in all score-oriented entry configs and defaults, widen `event_embed_dim` and `token_dim` together, and retain one output logit for every original event.

**Tech Stack:** Python 3, PyTorch, pytest, Ruff, YAML, PowerShell.

---

### Task 1: Lock the restored model contract

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_cone_attention.py`

- [ ] Change score-config assertions to require `token_dim == 96`, `event_embed_dim == 96`, and `token_importance_enabled is False`.
- [ ] Add a no-cropping attention assertion that retained neighbor counts equal original neighbor counts when Token Importance is disabled.
- [ ] Run `python -m pytest tests/test_config.py tests/test_cone_attention.py -q` and confirm the new assertions fail against the current 64-D/cropping defaults.

### Task 2: Implement the controlled 96-D configuration

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/models/cone_attention.py`
- Modify: `src/huav/models/cone_transformer.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`

- [ ] Make no K/V cropping the default at every constructor/config entry point.
- [ ] Set score-oriented full and smoke configs to `token_dim: 96` and `event_embed_dim: 96` without changing seed, geometry, loss, sampling, epochs, or validation coverage.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Preflight and real-data gates

**Files:**
- Produce: `outputs/cpu_YYYYMMDD_HHMMSS/` smoke artifacts.

- [ ] Run repository pytest, Ruff, and `git diff --check`.
- [ ] Run the `train_094.npz` forward/loss/backward resource gate with the 96-D full config.
- [ ] Train the score-oriented smoke config and independently reevaluate its `best.pt` with its saved `config.yaml`.

### Task 4: Full training and independent evaluation

**Files:**
- Produce: `outputs/cpu_YYYYMMDD_HHMMSS/` full-run artifacts.

- [ ] Train all 99 training files for 18 epochs on CPU with a note identifying the 96/96 no-cropping experiment.
- [ ] Independently evaluate the saved full-run `best.pt` over all 24 validation files.
- [ ] Require byte-identical `metrics.json` and `reproduced_metrics.json`, complete confusion counts, and matching dataset-manifest coverage.

### Task 5: Submission audit and comparison

**Files:**
- Produce: `<run>/submission/*.txt`
- Produce: `<run>/submission/submission.zip`
- Produce: `<run>/submission_audit.json`

- [ ] Generate predictions from the same saved config/checkpoint/threshold.
- [ ] Audit all source `x y t p` values and order, binary labels, row counts, and flat ZIP structure.
- [ ] Compare full-validation and `val_023` results against the direct no-cropping 64-D baseline (`0.909653`) and global historical best (`0.914545`).
- [ ] Run final pytest, Ruff, `git diff --check`, and `git status --short`.
