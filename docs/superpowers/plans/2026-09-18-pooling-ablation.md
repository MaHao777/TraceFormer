# Pooling Ablation Implementation Plan

> Execute inline in the local repository; the user has authorized training and evaluation. No worktree or extra approval is needed.

**Goal:** Train statistical and mean-pooling encoders on CPU and independently evaluate them against verified attention-pooling E1.

**Architecture:** Add an experiment-only tokenizer replacement through the existing ablation contract. Extract one shared block-feature hook in EventTokenizer without changing attention arithmetic. Keep seven statistics, position, event-to-token mapping, decoder inputs, cone backbone, losses, seed 37 and 18 epochs fixed. Use the existing serial queue, frozen source, resume state and independent evaluation.

**Tech Stack:** Python 3.12, PyTorch CPU FP32, pytest, existing experiment runner.

- [x] Extract `_encode_block_features(local_features, inverse, counts, statistics)` with the current attention operations unchanged. Call it after block statistics are available.
- [x] Add experimental statistics `7→64→64` and mean `event MLP→value projection→mean→64→192→64` encoders. Remove unused attention parameters. Preserve upstream RNG and shared initial weights. Mean pooling has 152,010 parameters versus attention's 151,882 (0.084% difference).
- [x] Add a `pooling` field to AblationSpec; omit its default from serialized contracts so existing checkpoints still validate. Reject alternative pooling outside the backbone-only stage.
- [x] Test attention equivalence, arithmetic mean, within-block permutation invariance, empty input, mapping/statistics parity, finite gradients and old checkpoint compatibility.
- [x] Prepare three jobs T0 statistics, T1 mean, T2 attention. Reevaluate the existing fully trained T2 checkpoint using updated runtime before accepting reuse. Verify data/config/checkpoint hashes and frozen model implementation differences.
- [x] Run resource checks on train_096 for both new encoders. Run a small training/resume/evaluation smoke and relevant regression tests.
- [x] Freeze source/config/protocol, launch serial CPU training with automatic full 24-file independent evaluation, per-file diagnostics, fixed-0.5 metrics, tables and plots.

Validation commands: `python -m pytest tests/test_paper_ablation.py tests/test_pooling_ablation.py -q`; `python -m ruff check src/traceformer/experiments src/traceformer/models/event_tokenizer.py scripts/run_paper_ablation.py scripts/check_paper_ablation.py`; `git diff --check`.

Report mean versus attention as approximately parameter matched; the statistics baseline is smaller. Do not claim matched FLOPs. Statistical tokens remove event-level learned pooling, not the event-local decoder inputs. Baseline reuse is only accepted after exact independent metric reproduction. All comparisons remain single-seed validation ablations, with no external postprocessing.

Execution: outputs/paper_pooling_cpu_20260918_234003. Training/evaluation run asynchronously; launch completion does not mean experiment results are complete.
