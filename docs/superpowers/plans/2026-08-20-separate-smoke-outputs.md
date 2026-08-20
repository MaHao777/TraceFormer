# Separate Smoke Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep disposable smoke-training artifacts outside the formal `outputs/` archive without deleting diagnostic evidence automatically.

**Architecture:** All maintained smoke configurations and the AutoDL smoke launcher write to a dedicated repository-level `smoke_outputs/` root. Full configurations continue to write to `outputs/`, so each root gets its own `INDEX.md`; documentation explains retention and cleanup policy.

**Tech Stack:** Python configuration loading, YAML, Bash, pytest, Ruff, PowerShell documentation.

---

### Task 1: Lock the output-root contract with tests

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_project_integration.py`

- [x] **Step 1: Add a configuration test**

Load all three maintained smoke YAML files and assert that their resolved `training.output_root` is `<repo>/smoke_outputs`. Load the maintained full YAML files and assert that their output root remains `<repo>/outputs`.

- [x] **Step 2: Add launcher and ignore-policy assertions**

Assert that `scripts/autodl_train.sh` selects `smoke_outputs` in smoke mode and that `.gitignore` ignores `/smoke_outputs/`.

- [x] **Step 3: Run the focused tests and confirm they fail before implementation**

Run: `python -m pytest -q tests/test_config.py tests/test_project_integration.py`

Expected: failures showing that smoke runs still target `outputs` and the launcher/ignore rule have not yet been changed.

### Task 2: Route all maintained smoke entry points to the dedicated root

**Files:**
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Modify: `scripts/autodl_train.sh`
- Modify: `.gitignore`

- [x] **Step 1: Change smoke configuration roots**

Set each smoke YAML's `training.output_root` to `../smoke_outputs`; do not change any model, loss, data-limit, optimizer, or full-training configuration.

- [x] **Step 2: Separate AutoDL smoke artifacts and logs**

In the launcher, use `<repo>/smoke_outputs` for smoke mode and `<repo>/outputs` for full mode, then derive both the run directory and log path from the selected root.

- [x] **Step 3: Ignore generated smoke artifacts**

Add `/smoke_outputs/` to `.gitignore` beside `outputs/`.

- [x] **Step 4: Run the focused tests**

Run: `python -m pytest -q tests/test_config.py tests/test_project_integration.py`

Expected: all focused tests pass.

### Task 3: Update the operator workflow and retention explanation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [x] **Step 1: Update smoke discovery and evaluation commands**

Document smoke checkpoints under `smoke_outputs/<device_timestamp>/`, while full checkpoints remain under `outputs/<device_timestamp>/`.

- [x] **Step 2: Document the retention policy**

State that successful smoke outputs may be deleted after the full run has started, but should not be deleted automatically because failed smoke artifacts are useful for diagnosing data loading, forward/loss/backward, validation, checkpoint, and resource failures.

- [x] **Step 3: Run repository verification**

Run: `python -m pytest -q`

Run: `python -m ruff check src tests`

Run: `git diff --check`

Expected: tests pass, Ruff reports no errors, and the diff has no whitespace errors. No smoke or full training is required because this change only affects artifact placement and documentation.
