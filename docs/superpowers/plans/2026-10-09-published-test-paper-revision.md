# EV-UAV labeled test evaluation and manuscript revision

> **For agentic workers:** Execute the following steps inline in the local repository. User instructions prohibit worktrees; no retraining or external baseline inference is required.

**Goal:** Evaluate the frozen paper system on the published 24-sequence test partition and revise the manuscript around this result and the supplied SpTopoNet paper.

**Architecture:** Reuse the schema-5 main detector, schema-4 reference detector, and released LightGBM/tracking contract. Extend the existing frozen evaluation adapters with an explicit test partition and CPU inference. Test labels enter scoring only; all thresholds remain frozen.

**Tech Stack:** Python, PyTorch CPU FP32, NumPy, LightGBM, Matplotlib, pdfLaTeX/BibTeX.

## Evaluation

- [x] Compare all arrays in the downloaded 99 training and 24 validation files with existing data; both partitions are identical.
- [x] Extend `scripts/evaluate_frozen_neural.py` with `--split test` and `--device cpu`. Verify the original validation identity and checkpoint/config contract, then infer with zeroed annotations. Omit threshold selection on test.
- [x] Extend `scripts/evaluate_frozen_system.py` with `--split test` and strict reference/cache file matching. Preserve the released event and recovery contracts.
- [x] Run both archived detectors from `outputs/paper_cuda_20260928/copyback_20260929`, followed by the complete pipeline, into `outputs/paper_test_20261009`.
- [x] Independently rescore all saved masks from the structured raw events; compare the neural scoring implementation and postprocessing scorer. Record data, model, config, source and prediction hashes.
- [x] Verify annotation isolation on a real test sequence and audit all 24 event-aligned predictions.

## Manuscript

- [x] Save a focused before snapshot in `D:/Paper/TraceFormer/evidence/published_test_20261009/before`.
- [x] Add the test partition to `tables/dataset.tex` and replace the external validation comparison with Table 1 values from the supplied SpTopoNet paper plus the newly evaluated TraceFormer system. Cite each original baseline paper and attribute reproduced publication values once in the caption.
- [x] Revise `main.tex`, introduction, related work, method prose, experiments and conclusion. Follow the reference's problem/motivation/method/results structure without copying sentences or overstating component evidence.
- [x] Replace validation-only headline and qualitative plots with test results; preserve internal mechanism experiments as supporting evidence. Retain the separate 31-file competition result only as provenance.
- [x] Remove the repeated limitations and unused external-comparison appendix, retain essential protocol facts at their point of use, and shorten the conclusion.
- [x] Update the project README and evidence source record, leaving historical experiment evidence intact.

## Verification

- [x] Run Ruff on changed scripts and the relevant postprocessing/metric tests; run `git diff --check`.
- [x] Compile the multi-file manuscript with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`.
- [x] Check bibliography keys, references, numerical tables and LaTeX warnings. Render and inspect every final PDF page, including all modified figures and tables.
- [x] Deliver the evaluation report and final manuscript PDF with the actual measured result.
