# Restore 64-Block Geometry with Residual STC Loss Implementation Plan

> **For agentic workers:** Execute inline in the current workspace because the improved Loss is uncommitted state that must remain in scope. Do not create subagents or a separate worktree.

**Goal:** Restore the historical best `8x8/64`, q4/token64 model geometry while retaining the schema-6 restored four-term Loss plus residual `lambda_stc=0.1`, then launch a fresh full CPU training and automatic validation/submission pipeline.

**Architecture:** Only the production and matching smoke geometry changes. Model interfaces, event order, optimizer settings, sampling, epoch count, and the current Loss implementation remain unchanged, yielding a controlled comparison against historical q4/token64.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff, PowerShell.

---

### Task 1: Lock the controlled configuration in tests

**Files:**
- Modify: `tests/test_config.py`

- [ ] Change score-config assertions to require 64 time blocks, slopes `[0.5, 1.0, 1.5, 2.0]`, and max time distances `[4, 4, 6, 8]` while retaining token64, q4, 18 epochs, and `lambda_stc=0.1`.
- [ ] Run `python -m pytest -q tests/test_config.py` and verify it fails against the current 160-block YAML.

### Task 2: Restore historical best geometry

**Files:**
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`

- [ ] Set `num_time_blocks: 64` and restore the four historical cone heads without changing the Loss or other training fields.
- [ ] Run `python -m pytest -q tests/test_config.py` and verify it passes.

### Task 3: Static and resource gates

**Files:**
- Preserve: `outputs/outputs_old/`

- [ ] Run `python -m pytest -q`, `python -m ruff check src tests scripts`, and `git diff --check`.
- [ ] Run `scripts/check_largest_sequence.py` on `train_094.npz`; require finite forward/Loss/backward and exactly 90,234 occupied tokens.
- [ ] Run the real-data smoke config and independently evaluate its own `best.pt`; require identical metric hashes.

### Task 4: Fresh full pipeline

**Files:**
- Create: `outputs/pipeline_cpu_<timestamp>.ps1`
- Create at runtime: `outputs/cpu_<timestamp>/`

- [ ] Launch a hidden PowerShell process for `18 epoch train -> independent evaluate -> metric hash check -> predict -> source-row/ZIP audit -> final pytest/Ruff/diff check`.
- [ ] Pass the audit program over Python stdin so PowerShell cannot strip its string quotes.
- [ ] Confirm the PID is alive, `experiment.md` exists, and at least one real training file has completed before handing control back.

No commit is created because the workspace contains the user's active uncommitted Loss experiment.
