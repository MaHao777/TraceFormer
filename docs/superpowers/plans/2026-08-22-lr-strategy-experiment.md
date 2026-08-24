# HUAV Learning-Rate Strategy Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine a stable peak learning rate, the fast-to-fine transition epoch, the fine-tuning schedule, and a validated stopping rule for the current schema-7 six-block HUAV model.

**Architecture:** Add backward-compatible learning-rate controllers and exact optimizer/RNG training-state snapshots to the existing training engine. Drive an adaptive, full-data CUDA BF16 experiment matrix from one durable launcher, then compare the current cosine baseline with the selected strategy on paired seeds 37, 73, and 101.

**Tech Stack:** Python 3.12, PyTorch, PyYAML, pytest, Ruff, Bash, AutoDL RTX 5090 BF16.

---

### Task 1: Learning-rate configuration and controllers

**Files:**
- Modify: `src/huav/config.py`
- Create: `src/huav/training/lr_schedules.py`
- Test: `tests/test_lr_schedules.py`

- [ ] Add validated schedule, fast-phase, fixed-ratio, and plateau fields while preserving the current legacy cosine default.
- [ ] Test exact epoch-level sequences for constant, delayed cosine, delayed exponential, fixed-low, plateau, and legacy cosine behavior.
- [ ] Implement the minimal controller and rerun focused tests.

### Task 2: Exact branch state and optimization diagnostics

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Test: `tests/test_training_engine.py`

- [ ] Add failing tests for complete state artifacts, mutually exclusive resume modes, additive history diagnostics, and deterministic CPU branch equivalence.
- [ ] Save model, AdamW, best checkpoint, histories, controller, dataset contract, and all RNG/generator states atomically.
- [ ] Restore exact state through `--resume-state`, retaining the existing fresh-optimizer `--resume` contract.
- [ ] Record phase, gradient norm/clipping, parameter displacement, and elapsed time per epoch.

### Task 3: Adaptive full experiment orchestrator

**Files:**
- Create: `src/huav/experiments/lr_strategy.py`
- Create: `scripts/run_lr_strategy_experiment.sh`
- Test: `tests/test_lr_strategy_experiment.py`

- [ ] Test instability gates, peak-LR ranking, transition refinement, fine-strategy ranking, paired-seed verdicts, and offline stopping-rule simulation.
- [ ] Implement four six-epoch peak-LR screens plus adaptive boundary/midpoint refinement.
- [ ] Implement the shared constant prefix, 6/12/18/24 transition comparison, three-epoch refinement, and four 18-epoch fine schedules.
- [ ] Run paired finalists on seeds 37/73/101, independently reevaluate all six, and package/audit only the global best.
- [ ] Maintain atomic `status.json`, `leaderboard.json`, `failed.json`, per-run logs, and a preliminary report beneath one experiment root.

### Task 4: Verification and cloud launch

**Files:**
- Modify only when a gate reveals a defect.
- Generate: `smoke_outputs/cuda_<timestamp>/` and `outputs/lr_strategy_cuda_<timestamp>/`

- [ ] Run the full pytest suite, Ruff, and `git diff --check`.
- [ ] Run the largest-sequence forward/loss/backward gate and real-data schedule smoke/evaluation on RTX 5090.
- [ ] Upload a clean snapshot to a new `/root/autodl-tmp/HUAV_lr_strategy_<timestamp>/` directory.
- [ ] Launch the single script with `nohup`, verify only immediate process/log health, and report the exact PID, run root, logs, status files, and monitoring commands.
- [ ] Do not claim final experiment completion until the user returns and all final reevaluation/submission artifacts have been analyzed.
