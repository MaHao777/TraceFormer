# AutoDL RTX 5090 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current HUAV score-oriented model train, reevaluate, and export reliably on the user's AutoDL RTX 5090 instance.

**Architecture:** Keep the model, event ordering, cone geometry, loss, and scoring contract unchanged. Add one shared CUDA runtime preflight used by train/evaluate/predict, make AMP dtype explicit so Blackwell can use BF16 without an unnecessary gradient scaler, and provide dedicated reproducible AutoDL configurations and Linux commands.

**Tech Stack:** Python 3.12, PyTorch CUDA 12.8+, NVIDIA Blackwell SM 12.0, PyYAML, pytest, Ruff, Bash.

---

### Task 1: CUDA runtime preflight

**Files:**
- Create: `src/huav/training/runtime.py`
- Create: `src/huav/cli/check_gpu.py`
- Modify: `pyproject.toml`
- Modify: `src/huav/training/engine.py`
- Test: `tests/test_cuda_runtime.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing runtime and CLI tests**

Test CPU metadata, unavailable CUDA errors, successful CUDA metadata with compute capability and architecture list, and actual CUDA tensor execution failure. Require `python -m huav.cli.check_gpu --help` to expose the diagnostic command.

- [ ] **Step 2: Verify the focused tests fail**

Run: `python -m pytest tests/test_cuda_runtime.py tests/test_cli.py -q`

Expected: FAIL because `huav.training.runtime` and `huav.cli.check_gpu` do not exist.

- [ ] **Step 3: Implement the shared device preparation**

Implement `prepare_device(name, num_threads)` so CUDA requests check `torch.cuda.is_available()`, collect PyTorch/CUDA/GPU/compute-capability/architecture metadata, run and synchronize a small tensor kernel, and return a `torch.device` plus serializable metadata. On failure, direct RTX 50-series users to a CUDA 12.8-or-newer PyTorch build.

- [ ] **Step 4: Use the preflight in every public engine path**

Replace ad-hoc `torch.device(...)` construction in training, resumed training, evaluation, and submission generation with the shared preflight. Record the runtime metadata in `summary.json`.

- [ ] **Step 5: Verify focused tests pass**

Run: `python -m pytest tests/test_cuda_runtime.py tests/test_cli.py -q`

Expected: PASS.

### Task 2: Blackwell BF16 mixed precision

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: Write failing config and AMP policy tests**

Require `training.mixed_precision_dtype` to accept only `float16` and `bfloat16`. Require CUDA BF16 autocast to use `torch.bfloat16`, and require GradScaler only for CUDA FP16.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_config.py tests/test_training_engine.py -q`

Expected: FAIL because the dtype field and AMP policy do not exist.

- [ ] **Step 3: Implement the minimal AMP dtype policy**

Add `mixed_precision_dtype: str = "float16"` to `TrainingConfig`, validate it, select the corresponding PyTorch dtype in both training loops, and disable GradScaler for BF16 while retaining it for FP16.

- [ ] **Step 4: Preserve old CPU configuration behavior**

Add the explicit default dtype to existing YAML files without changing their device or mixed-precision setting.

- [ ] **Step 5: Verify focused tests pass**

Run: `python -m pytest tests/test_config.py tests/test_training_engine.py -q`

Expected: PASS.

### Task 3: Reproducible AutoDL configuration and commands

**Files:**
- Create: `configs/huav_score_optimized_autodl_smoke.yaml`
- Create: `configs/huav_score_optimized_autodl.yaml`
- Create: `scripts/autodl_train.sh`
- Create: `docs/autodl_rtx5090.md`
- Test: `tests/test_config.py`
- Test: `tests/test_project_integration.py`

- [ ] **Step 1: Write failing integration tests**

Require both AutoDL configs to retain the current `8x8/64`, 48-dimension score model; require CUDA, BF16, full-data limits for the full config, bounded limits for the smoke config, and a Bash launcher that performs preflight before training.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_config.py tests/test_project_integration.py -q`

Expected: FAIL because the AutoDL artifacts do not exist.

- [ ] **Step 3: Add configs and launcher**

Create CUDA/BF16 variants of the current score-optimized smoke and full configs. Add a launcher accepting `smoke` or `full`, creating a unique timestamped output directory, logging via `tee`, and refusing to reuse an existing run directory.

- [ ] **Step 4: Document the exact AutoDL closure**

Document upload location, environment check, smoke training and independent reevaluation, largest-sequence forward/backward check, full training, independent metric hash comparison, and submission generation.

- [ ] **Step 5: Verify focused tests pass**

Run: `python -m pytest tests/test_config.py tests/test_project_integration.py -q`

Expected: PASS.

### Task 4: Local and real RTX 5090 verification

**Files:**
- Modify only if a failing check reveals a defect.
- Generate new timestamped artifacts under remote `outputs/`.

- [ ] **Step 1: Run the local mandatory checks**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Expected: all checks pass.

- [ ] **Step 2: Upload a clean snapshot to the AutoDL data disk**

Copy source, configs, tests, scripts, and `data/ev-uav` to a new directory under `/root/autodl-tmp`; do not overwrite any prior remote run.

- [ ] **Step 3: Install the editable project and run preflight**

Run:

```bash
python -m pip install -e ".[dev]"
python -m huav.cli.check_gpu
```

Expected: RTX 5090, capability 12.0, CUDA 12.8+, `sm_120`, and a successful CUDA kernel.

- [ ] **Step 4: Run largest-sequence resource reproduction and smoke closure**

Exercise `train_094.npz` through forward, loss, and backward on CUDA BF16, then run the AutoDL smoke config and independently reevaluate its `best.pt`.

- [ ] **Step 5: Start and monitor the full closure**

Run the full AutoDL config in a persistent remote session. On completion, independently reevaluate the same `best.pt`, require byte-identical metrics, generate submission TXT/ZIP, and validate event rows and ZIP structure before reporting completion.

### Task 5: Eighteen-epoch convergence run and local result retrieval

**Files:**
- Generate a new remote resume run without modifying the baseline run.
- Copy the accepted final run beneath local `outputs/`.

- [ ] **Step 1: Train eighteen continuous epochs from scratch**

Run `configs/huav_score_optimized_autodl.yaml` for 18 epochs so the cosine schedule decays continuously across the original six epochs plus twelve convergence epochs. Inspect every epoch in `history.json`, including total/focal/Dice/object-frame loss, validation competition score, Pd, and Fa.

- [ ] **Step 2: Compare epoch 6 with the later convergence window**

Compare epoch 6 against epochs 7-18 using validation competition score and all score components. Accept the best validation checkpoint across all 18 epochs rather than assuming the last epoch is best.

- [ ] **Step 3: Independently reevaluate and package the best checkpoint**

Require byte-identical `metrics.json` and `reproduced_metrics.json`, complete confusion counts, then generate and validate the submission TXT/ZIP from that same `best.pt`.

- [ ] **Step 4: Copy the complete accepted result back to the local workspace**

Retrieve the final run directory, console log, submission directory, and ZIP into a new local directory under `D:\VS_project\TinyObjDetection\HUAV\outputs`; verify file hashes after transfer.
