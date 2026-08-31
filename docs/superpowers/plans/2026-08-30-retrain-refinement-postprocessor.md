# Refinement Postprocessor Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the HUAV postprocessing cascade from the schema-7 refinement checkpoint and determine whether its independently evaluated 24-file score exceeds the verified historical postprocessing best of `0.9541170157392043`.

**Architecture:** Keep the detector checkpoint frozen. Generate checkpoint-bound train/validation caches, use the fixed seed-37 79/20 development split and stage gates to train the LightGBM event/component/center cascade, then evaluate only stages permitted by the development report. Compare the final independently scored validation predictions against both the raw checkpoint score and the historical postprocessing report.

**Tech Stack:** Python 3, PyTorch CPU FP32, LightGBM 4.7.0, NumPy, pytest, Ruff, HUAV postprocessing CLI.

---

### Task 1: Pin inputs and preflight the postprocessing runtime

**Files:**
- Read: `outputs/cpu_refinement_20260830_154410/config.yaml`
- Read: `outputs/cpu_refinement_20260830_154410/best.pt`
- Read: `outputs/cpu_refinement_20260830_154410/independent_evaluation.json`
- Read: `outputs/postprocess_event_release_dense_rescue_20260829/validation_release99_dense_rescue/validation_comparison.json`
- Test: `tests/test_postprocessing.py`

- [ ] **Step 1: Record SHA-256 identities for the checkpoint, saved config, current independent evaluation, and historical comparison report.**

Run: `Get-FileHash -Algorithm SHA256 <the four pinned files>`

Expected: four non-empty hashes; the current raw score is `0.9213087871881473` and the historical final score is `0.9541170157392043`.

- [ ] **Step 2: Run the focused postprocessing tests.**

Run: `python -m pytest tests/test_postprocessing.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Verify runtime and data capacity.**

Run: `python -c "import lightgbm, torch; print(lightgbm.__version__, torch.__version__)"`

Expected: LightGBM imports successfully, PyTorch is CPU-capable, the dataset contains 99 train and 24 validation NPZ files, and drive D has enough free space for fresh caches.

### Task 2: Generate fresh checkpoint-bound caches

**Files:**
- Create: `outputs/postprocess_refinement_20260830_215157/cache/config.yaml`
- Create: `outputs/postprocess_refinement_20260830_215157/cache/train/*.npz`
- Create: `outputs/postprocess_refinement_20260830_215157/cache/train/manifest.json`
- Create: `outputs/postprocess_refinement_20260830_215157/cache/val/*.npz`
- Create: `outputs/postprocess_refinement_20260830_215157/cache/val/manifest.json`

- [ ] **Step 1: Cache all 99 labeled training sequences from the frozen checkpoint.**

Run: `python -m huav.cli.postprocess cache --config outputs/cpu_refinement_20260830_154410/config.yaml --checkpoint outputs/cpu_refinement_20260830_154410/best.pt --output-dir outputs/postprocess_refinement_20260830_215157/cache --split train --device cpu`

Expected: `files=99`; the manifest checkpoint hash equals the pinned `best.pt` hash and labels are included only in the training cache.

- [ ] **Step 2: Cache all 24 validation sequences without labels.**

Run: `python -m huav.cli.postprocess cache --config outputs/cpu_refinement_20260830_154410/config.yaml --checkpoint outputs/cpu_refinement_20260830_154410/best.pt --output-dir outputs/postprocess_refinement_20260830_215157/cache --split val --device cpu`

Expected: `files=24`; `labels_included=false`, and the checkpoint/config hashes match the training manifest.

- [ ] **Step 3: Independently score the cached raw predictions.**

Run: `python -m huav.cli.postprocess evaluate --prediction-dir outputs/postprocess_refinement_20260830_215157/cache/val --dataset-root data/ev-uav/val --output outputs/postprocess_refinement_20260830_215157/raw_validation.json`

Expected: aggregate Score exactly matches `0.9213087871881473` and covers 24 files / 1,424,330 events.

### Task 3: Retrain the staged postprocessor

**Files:**
- Create: `outputs/postprocess_refinement_20260830_215157/training/training_report.json`
- Create: `outputs/postprocess_refinement_20260830_215157/training/development/**`
- Create when the event gate passes: `outputs/postprocess_refinement_20260830_215157/training/event_release/**`
- Create when all cascade gates pass: `outputs/postprocess_refinement_20260830_215157/training/release/**`

- [ ] **Step 1: Train with the fixed seed-37 79/20 protocol and original 997/1998 boosting rounds.**

Run: `python -m huav.cli.postprocess train --cache-dir outputs/postprocess_refinement_20260830_215157/cache/train --output-dir outputs/postprocess_refinement_20260830_215157/training --holdout-files 20 --small-rounds 997 --large-rounds 1998`

Expected: the report records the deterministic 79/20 split and a terminal status. Stop immediately if the event-stage gate fails. If a later component/center gate fails, retain the last earlier stage that passed and do not promote the rejected stage.

- [ ] **Step 2: Verify model/report hashes and training provenance.**

Run: read `training_report.json`, `event_release/release_report.json` when present, and every recorded model SHA-256.

Expected: all reported hashes match files on disk, event release uses all 99 training caches, and no validation labels were used for fitting or threshold selection.

- [ ] **Step 3: Train the direct-only verifier as a separately gated candidate after the event stage passes.**

Run: `python -m huav.cli.postprocess train-direct-verifier --cache-dir outputs/postprocess_refinement_20260830_215157/cache/train --training-report outputs/postprocess_refinement_20260830_215157/training/training_report.json --output-dir outputs/postprocess_refinement_20260830_215157/direct_verifier`

Expected: a terminal released/rejected report. A rejected verifier remains diagnostic only and cannot define the formal final score.

### Task 4: Blind inference and independent 24-file comparison

**Files:**
- Create: `outputs/postprocess_refinement_20260830_215157/validation/**`
- Read: `outputs/test_submission_20260828_schema4_best/predictions/*.npz`
- Read: `outputs/postprocess_event_release_dense_rescue_20260829/validation_release99_dense_rescue/validation_comparison.json`

- [ ] **Step 1: Run validation inference with frozen accepted models.**

Run the existing postprocessing evaluation driver using the fresh validation cache, fresh event release, fresh verifier report, the fixed schema-4 reference prediction directory, the raw validation report, and `data/ev-uav/val`.

Expected: prediction directories contain 24 aligned NPZ files; labels remain absent from inference caches. If the verifier is rejected, pass the diagnostic flag but treat only `event_release` as formally eligible.

- [ ] **Step 2: Independently evaluate every produced stage from original validation NPZ labels.**

Run: `python -m huav.cli.postprocess evaluate` for raw, event release, any accepted cascade stage, and the matched dense-rescue candidate.

Expected: each report contains aggregate plus 24 per-file metrics and exactly 1,424,330 evaluated events.

- [ ] **Step 3: Compare against both baselines.**

Compute exact deltas versus raw `0.9213087871881473` and historical postprocessing best `0.9541170157392043`.

Expected: conclude “new historical best” only when the independently evaluated final eligible Score is strictly greater than `0.9541170157392043` under the same 24-file metric implementation.

### Task 5: Final verification and artifact handoff

**Files:**
- Create: `outputs/postprocess_refinement_20260830_215157/comparison_summary.json`
- Verify: all new manifests, reports, predictions, and model files

- [ ] **Step 1: Re-run focused tests and static checks if source code was changed.**

Run: `python -m pytest tests/test_postprocessing.py -q`, `python -m ruff check src tests scripts`, and `git diff --check`.

Expected: all executed checks pass; unrelated user changes remain untouched.

- [ ] **Step 2: Write an auditable comparison summary.**

Record pinned input hashes, stage status, raw/current/historical scores, metric deltas, per-file difficult-domain metrics, and the selected final stage.

Expected: the summary names the exact source checkpoint and every result path without claiming a rejected or diagnostic stage as production-ready.

- [ ] **Step 3: Report the outcome without generating a submission package.**

Expected: answer whether the validation score exceeds history and identify the winning stage. Submission generation is outside this request and remains unperformed.
