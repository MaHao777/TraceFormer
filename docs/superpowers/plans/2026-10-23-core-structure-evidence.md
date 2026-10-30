# Core Structure Evidence Implementation Plan

> **For agentic workers:** Execute with the available collaboration subagents, reviewing each artifact before paper integration. All work stays in the current local checkout; no worktree is used.

**Goal:** Identify the strongest supported source of the current model's gain, complete short controlled experiments, and replace unsupported paper claims with reproducible evidence.

**Architecture:** First audit the existing trained controls, including negative attention results. Then replay the frozen current neural outputs, event-tree scores and trajectory contract over all 24 validation and 24 test sequences. Separate verification, direct acceptance and trajectory-conditioned recovery; compare simple threshold relaxation and targeted temporal interventions. This is a frozen-system intervention study, not a replacement for retraining ablation or a new untouched test set.

**Tech Stack:** Python, NumPy, SciPy, existing TraceFormer metrics/tracking, pytest, Matplotlib, LaTeX and PDF rendering.

---

### Task 1: Audit and freeze the experiment

- [x] Inspect the current production manifest and independently evaluated attention controls.
- [x] Record exact checkpoint, contract, input cache and score hashes before replay.
- [x] Freeze controls before reading their new outcomes: neural; direct; tracks only; direct OR tracks; direct OR tracks without endpoint extension; direct OR tracks without the minimum-duration filter; and direct OR tracks after a deterministic frame permutation (restore event alignment for scoring).
- [x] Include a threshold-only control selected on validation global Score from a fixed grid, then freeze it for test. Keep original production thresholds unchanged.
- [x] State that archived test results have already informed this exploratory hypothesis; do not describe this as preregistered confirmatory testing.

### Task 2: Implement and run the bounded replay

**Files:** Create `scripts/run_vtr_structure_ablation.py` and `tests/test_vtr_structure_ablation.py`.

- [x] Reuse `EventCache`, `blend_scores`, `track_prediction`, `BinaryCounts` and raw-source evaluation. Inference receives no labels; only scoring reads annotations.
- [x] Verify source identity, event order, probability shape/range, release model identity and exact full-system masks against archived production predictions. Do not refit models or tune parameters on test data.
- [x] Test intervention invariants, deterministic time permutation with restored indexing, and validation-only threshold selection.
- [x] Run the controls over both complete splits, saving binary masks, all global/per-file integer counts, hashes, wall time and a machine-readable report in a new output directory.
- [x] Independently rescore saved masks from original NPZ files. Analyze direct/track overlap, recovered positives, false events and target frames. Use sequence-level resampling only as conditional test-set uncertainty, never as training-seed significance.
- [x] Budget: no neural retraining; start with one sequence to measure replay cost. Aim for minutes rather than a new multi-hour training queue; reduce redundant scoring/caching, never sequence coverage or unfavorable controls.

### Task 3: Integrate the supported result into the paper

**Files:** Update relevant `D:/Paper/TraceFormer/sections/*.tex`, `main.tex`, generated tables/figures, `README.md`, and a dedicated `evidence/core_structure_20261023/` bundle.

- [x] Preserve a snapshot of edited paper sources and the original PDF.
- [x] Show the attention controls, including the local-attention result that does not favor multi-cone summaries; correct the unequal-support limitation.
- [x] Report the new frozen-structure comparisons, with both positive and negative outcomes and actual measured experiment cost.
- [x] Align title/abstract/contributions/conclusion with the strongest supported mechanism. Do not rename classical components into a novel neural layer or claim superiority not demonstrated by experiments.
- [x] Compile LaTeX, inspect changed PDF pages visually using the PDF skill, and verify tables against the saved reports.
- [x] Finish with a concise Chinese evidence report, exact commands and remaining limitations. Mark the goal complete only when the actual experiments and paper integration are done.
