# CPU Pool-Query and Token-Dimension Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline and follow the repository `AGENTS.md` train-evaluate-submit contract for every full run.

**Goal:** Compare `q4/token64`, `q8/token48`, and `q8/token64` on the complete HUAV dataset using CPU-only training and independently reproducible competition artifacts.

**Architecture:** Keep the validated `8x8/64` cone geometry, loss, sampling, optimizer, seed, and 18-epoch schedule fixed. Change only `model.num_pool_queries` and `model.token_dim`, first gate the largest sequence, then run the heaviest configuration through smoke training before the three full runs.

**Tech Stack:** Python, PyTorch CPU, SciPy cKDTree, Pytest, Ruff, HUAV train/evaluate/predict CLIs.

---

### Task 1: Establish a clean experimental baseline

**Files:**
- Inspect: `configs/huav_score_optimized.yaml`
- Inspect: `AGENTS.md`

- [ ] **Step 1: Record repository state**

Run:

```powershell
git status --short
```

Expected: preserve every pre-existing path and do not overwrite an existing `outputs` directory.

- [ ] **Step 2: Run repository verification**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Expected: all commands exit with status zero before training begins.

### Task 2: Create isolated experiment configurations

**Files:**
- Create: `outputs/cpu_pool_token_ablation_<stamp>/configs/q4_token64.yaml`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/configs/q8_token48.yaml`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/configs/q8_token64.yaml`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/configs/q8_token64_smoke.yaml`

- [ ] **Step 1: Copy the production configuration values**

Use `configs/huav_score_optimized.yaml` as the only source. Set CPU execution, 18 epochs, no train/validation file limits, and retain the full score-oriented loss and sampling fields.

- [ ] **Step 2: Apply only the intended model overrides**

Use these exact model pairs:

```text
q4_token64: num_pool_queries=4, token_dim=64
q8_token48: num_pool_queries=8, token_dim=48
q8_token64: num_pool_queries=8, token_dim=64
```

The smoke configuration must match `q8_token64` except for `epochs=1`, `max_train_files=4`, and `max_val_files=2`.

- [ ] **Step 3: Validate all configurations**

Run:

```powershell
python -c "from huav.config import load_config; [load_config(path) for path in paths]"
```

Expected: each YAML loads successfully and reports the intended query count, token dimension, epoch count, and file limits.

### Task 3: Gate the largest sequence

**Files:**
- Create: `outputs/cpu_pool_token_ablation_<stamp>/resource_q4_token64.json`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/resource_q8_token48.json`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/resource_q8_token64.json`

- [ ] **Step 1: Run forward, production loss, and backward for `train_094.npz`**

Run once per full configuration:

```powershell
python scripts/check_largest_sequence.py --config <config> --sequence train_094.npz --device cpu
```

Expected: 614,725 events complete forward and backward, every parameter gradient is finite, and the process exits normally.

- [ ] **Step 2: Compare resource reports**

Record parameter count, event count, token count, loss query count, runtime, and any memory observations. Do not start smoke training if the joint configuration fails.

### Task 4: Run the heaviest smoke pipeline

**Files:**
- Create: `outputs/cpu_pool_token_ablation_<stamp>/smoke_q8_token64/`

- [ ] **Step 1: Train the smoke configuration**

Run:

```powershell
python -m huav.cli.train --config <q8_token64_smoke.yaml> --device cpu --output-dir <smoke_dir>
```

Expected: `best.pt`, `config.yaml`, `dataset_manifest.json`, `history.json`, `metrics.json`, `metrics.md`, and `summary.json`.

- [ ] **Step 2: Independently reevaluate the smoke checkpoint**

Run:

```powershell
python -m huav.cli.evaluate --config <smoke_dir>/config.yaml --checkpoint <smoke_dir>/best.pt --device cpu --output <smoke_dir>/reproduced_metrics.json --quiet
```

Expected: `metrics.json` and `reproduced_metrics.json` are byte-identical.

### Task 5: Run the three complete CPU experiments

**Files:**
- Create: `outputs/cpu_pool_token_ablation_<stamp>/full_q4_token64/`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/full_q8_token48/`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/full_q8_token64/`

- [ ] **Step 1: Train `q4/token64` for 18 epochs**

Run:

```powershell
python -m huav.cli.train --config <q4_token64.yaml> --device cpu --output-dir <full_q4_token64>
```

- [ ] **Step 2: Independently reevaluate `q4/token64`**

Run:

```powershell
python -m huav.cli.evaluate --config <run>/config.yaml --checkpoint <run>/best.pt --device cpu --output <run>/reproduced_metrics.json --quiet
```

Expected: byte-identical metrics and complete 24-file validation coverage.

- [ ] **Step 3: Generate and audit the `q4/token64` submission**

Run:

```powershell
python -m huav.cli.predict --config <run>/config.yaml --checkpoint <run>/best.pt --output-dir <submission_dir> --device cpu
```

Expected: 24 root-level TXT entries, 1,424,330 rows, exact source `x y t p` values/order, and binary labels.

- [ ] **Step 4: Repeat Steps 1-3 for `q8/token48`**

Use only `q8_token48.yaml`, its own run directory, checkpoint, reproduced metrics, and submission directory.

- [ ] **Step 5: Repeat Steps 1-3 for `q8/token64`**

Use only `q8_token64.yaml`, its own run directory, checkpoint, reproduced metrics, and submission directory.

### Task 6: Compare results and verify the repository

**Files:**
- Create: `outputs/cpu_pool_token_ablation_<stamp>/comparison.json`
- Create: `outputs/cpu_pool_token_ablation_<stamp>/comparison.md`

- [ ] **Step 1: Build the comparison**

For each experiment record parameter count, best epoch, selected threshold, competition score, IoU, SegAcc, Pd, Fa, event Pd/Fa/IoU/Acc, query count, token dimension, and elapsed training time.

- [ ] **Step 2: Identify the winner by official score**

Use:

```text
0.2*IoU + 0.1*SegAcc + 0.4*Pd + 0.3*exp(-10000*Fa)
```

Do not select by training loss or event IoU alone.

- [ ] **Step 3: Run final repository checks**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
git status --short
```

Expected: tests, Ruff, and diff checks pass; experimental artifacts remain isolated under the unique output root.
