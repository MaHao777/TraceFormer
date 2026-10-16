# Frozen system cloud timing plan

> Execute inline in the existing local repository; no worktrees or delegation, per user instructions.

**Goal:** Measure the frozen TraceFormer on the supplied cloud GPU, independently verify predictions, and update Table 3 and its timing scope.

**Architecture:** Extend the existing benchmark with CUDA synchronization and repeated measurements. Upload only the two archived detector sources, frozen models/configurations, released tree models, postprocessing code, 24 published test files, and reference masks. Preserve the CPU report, record cloud hardware and prediction comparisons, and copy back verified evidence before editing the manuscript.

**Tech Stack:** Paramiko, PyTorch CUDA FP32, LightGBM, NumPy, LaTeX, Poppler.

- [x] Inspect the supplied SSH host with scripts/paper_remote.py; pin its host key and record hardware, CUDA kernel check, CPU allocation, memory and dependencies.
- [x] Extend scripts/benchmark_frozen_system.py with --device cuda, --repeats 3, synchronized network boundaries, host/device transfer timing, CPU comparison counts, and saved CUDA predictions for independent scoring. Retain the CPU default and strict CPU mask checks.
- [x] Package a minimal hash-manifested bundle, transfer and verify it under /root/autodl-tmp/traceformer-runtime-20261011, install missing dependencies, and smoke-test the largest sequence.
- [x] Warm up and run all 24 files three times; record per-file/component times and verify repeated outputs, frozen weights/settings and independent test metrics.
- [x] Download and hash-verify the report and predictions; preserve the previous CPU evidence, update paper resource inputs and captions with the actual cloud hardware and scope, and keep all 46 references and explanatory figures.
- [x] Compile and render the changed PDF pages, verify citations, detection metrics and absence of composite Score, then copy the final PDF to D:/Paper/TraceFormer/main.pdf.

Verified result: RTX 3090 + Xeon Gold 6330, FP32, 72 measurements and 96 returned masks. Complete mean 43,182.2674 ms/sequence; neural forwards including CPU view construction/search 6,005.3916 ms/sequence. Every repeated mask matches the original CPU result; independent local rescoring matches all four stages. The final PDF has 22 pages and 46 cited references, no composite Score or layout/citation warnings. Full-page visual review covers the changed pages 12–14; contact sheets cover all pages. Evidence: D:/Paper/TraceFormer/evidence/resource_metrics_gpu_20261011/final_qa.json.
