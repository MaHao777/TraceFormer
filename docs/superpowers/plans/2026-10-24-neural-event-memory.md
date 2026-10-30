# Neural Event Memory Decoder Implementation Plan

> **Execution:** Continue in the current local checkout without worktrees. Existing audits are preserved; the remaining experiments, independent evaluation and manuscript integration are executed locally.

**Goal:** Find and verify a neural-network contribution using fair trained neural controls, then add only supported results to the paper. Postprocessing evidence cannot satisfy this goal.

**Architecture:** Share a verified, pretrained backbone-only A3 encoder. Add an event-memory decoder that reads original event time relations rather than tree scores, tracks or saved event predictions. First retrain small matched readouts on exactly shared cached representations, then perform a matched encoder-adaptive comparison by unfreezing the final encoder block. Inspect validation before deciding which new structure merits the second stage; lock the final evaluation matrix before new test inference. A negative memory result requires a new hypothesis, not a favorable subset or weaker completion criterion.

**Tech Stack:** PyTorch CPU FP32, NumPy, existing training losses and metrics, pytest, Matplotlib, LaTeX.

---

### Task 1: Freeze the neural hypothesis and training contract

- [x] Inspect tokenizer, decoder, refinement, dataset normalization and original training calls; exclude all tree and trajectory components.
- [x] Audit prior neural experiments and relevant primary literature. Ordinary point/global-feature concatenation and time surfaces are established designs, not new inventions here.
- [x] Diagnose train/validation information: token-constant IoU oracle is 0.918735/0.945254, so quantization alone is not yet a demonstrated bottleneck. Same-pixel event gaps show potentially useful foreground/background differences that motivate testing a multi-timescale neural input; no monotonic reliability or learned band-pass behavior is assumed.
- [x] Record the exact A3 checkpoint, source and all 99 train/24 validation data identities. Preserve the current production model and existing edits.
- [x] Before new test inference, save the chosen neural structures, seeds, training steps, validation-selection rule, primary comparisons and claim limits.

### Task 2: Implement and verify a neural event-memory readout

**Files:** Create `src/traceformer/experiments/event_memory.py`, `tests/test_event_memory.py`.

- [x] Compute four label-free event gaps: same/opposite polarity, strictly earlier/later events at the same pixel. Exclude the query itself and all equal-time events so the features do not depend on the ordering of simultaneous events. Missing history/future maps to zero activation.
- [x] Encode each gap at fixed 1/10/50 ms decay constants, giving twelve event-memory inputs. Do not feed original event logits or probabilities.
- [x] Compare the original 6,785-parameter readout with a 6,758-parameter readout (12-to-16 memory encoder, 64-to-16 semantic conditioner, 86-to-45-to-32-to-1 classifier). All allocated parameters must participate.
- [x] U/G share exact initial weights, parameter counts and frozen-stage dropout streams; only additive versus multiplicative fusion differs. C is the additional strong direct-memory control.
- Optional trained S was not included in the final matrix. C/G aligned, block-mean, shuffled and zero inputs were evaluated as explicitly labeled fixed-weight validation sensitivity, not as trained S results. These interventions did not support the necessity of fine event alignment.
- [x] Test gaps against a brute-force reference, timestamp ties, polarity exchange, missing context, shuffled alignment, finite gradients and equal parameter counts. Add simple Chinese comments.

### Task 3: Cache once and train matched neural variants

**Files:** Create `scripts/run_neural_event_memory.py`, `src/traceformer/experiments/cached_loss.py`, `tests/test_cached_loss.py`.

- [x] Extract actual FP32 decoder inputs, original time/polarity, event-token indices, token coordinates and penultimate encoder features from the backbone-only A3. Cache exact FP32 features; do not use quantized postprocessing features or saved event probabilities as a shortcut.
- [x] Precompute annotation grouping for the unchanged focal, Dice, object-frame and false-alarm losses. Numerically compare component losses and gradients against the original criterion before running training.
- [x] Run one largest-file full forward/backward resource gate and a representative small file; measure the actual small-head epoch cost before committing to the queue.
- [x] Start with a fixed eight-epoch seed37 screening comparison using all 99 training and 24 validation files. Preserve identical sequence order, sampling RNG, dropout RNG and schedule for paired new-head controls. Select checkpoints and thresholds on validation only.
- [x] If the new structure has a meaningful validation gain over the standard and same-capacity control, complete paired seeds73/101 under the identical protocol. If short-head training has not stabilized, extend all comparison arms together with an explicit record before any new test evaluation.
- [x] Perform the selected new readout versus standard readout with the same final encoder block unfrozen, identical initial block weights, training budget and losses. This prevents relying solely on a frozen-feature probe for the neural structural claim.
- [x] Independently reload saved checkpoints and reevaluate all validation events. Preserve all complete-data results, including negative controls; no favorable-sequence exclusion.
- [x] Keep the goal active while any required training, independent evaluation or paper update remains incomplete. Aim for a first experiment cycle around one hour using measured costs; this is a planning target, not permission to truncate required evidence.

### Task 4: Evaluate the actual standalone neural networks and update the paper

- [x] Lock the confirmatory matrix before new test inference; this is an already inspected benchmark, so describe the research process honestly rather than claiming a new untouched holdout.
- [x] Infer from original unlabelled event inputs with the complete neural graph, including on-the-fly offline event-memory extraction, no postprocessing. Check that raw-input and cached-adapter predictions agree.
- [x] Evaluate all 24 test sequences at validation-selected operating points, independently rescore saved binary masks, and report IoU/ACC/Pd/Fa/Score, AP and actual runtime/parameters.
- [x] Distinguish head-seed replication, final-block adaptation and full-from-scratch training. Do not call the first two full end-to-end multi-seed training.
- [x] Snapshot the paper, revise neural method/contributions/figures/tables around supported comparisons, retain contrary attention evidence, compile and visually inspect the PDF.
- [x] Completion requires a verified neural result and the paper integration; a plan, frozen intervention or postprocessing result alone is insufficient.

## Execution notes (2026-10-24, before new test inference)

- Add a strong direct-memory concatenation control C (6,734 parameters; 82-to-57-to-32-to-1). The four trained arms are B/C/U/G, with total neural sizes 163,972 / 163,921 / 163,945 / 163,945. The existing 189,900-parameter production detector is not replaced by these experiments.
- The four arms share a symmetric hard-negative curriculum from their mean detached logits. U/G share exact initialization and dropout streams; different layer widths in B/C do not give identical dropout masks.
- Train/val caches are complete (99/24 files; 2.782 GiB); extraction took 99.8/27.7 s. Exact cached losses and logit gradients passed 22 tests. Memory and runner tests passed 20 and 17 cases.
- Measured largest-sequence four-arm resource gates: frozen 3.322 s / 4.670 GiB; last-block adaptation 16.746 s / 9.885 GiB, before persistent training-cache overhead.
- Frozen seed37 completed. Validation Score B/C/U/G = 0.895398727 / 0.900104553 / 0.895264929 / 0.897902510. Original A3 = 0.896777361.
- Frozen seed73 completed. Validation Score B/C/U/G = 0.896297603 / 0.899577260 / 0.897936755 / 0.900533267. C and G change rank. Do not claim stable superiority of multiplicative gating.
- Seed101 and all four adaptive arms completed. All16 models were locked before new candidate test inference. No weights or thresholds changed after that inference.
- Optional validation-only alignment interventions (aligned / token mean / within-token joint shuffle / zero) preserve trained C weights and thresholds. They are sensitivity diagnostics, explicitly not a substitute for trained structural controls.

## Final evidence and interpretation

- Training epoch timers total59.78 min (all four arms per epoch, validation AP included). All16 trained models passed raw-input validation and independent all24-file val/test rescoring. Raw/cached validation logits are bit-exact in all384 model-sequence pairs.
- Initial raw validation exposed construction of inference-mode parameters in the evaluator. A controlled reproduction isolated this to the construction context; normal parameter construction restored bit-exact equivalence without changing tolerance, trained weights, cached tensors or thresholds. Two regression checks passed.
- The pre-test nominated C failed the primary test Score comparison with B: frozen mean0.908859 versus0.909452, and adaptive0.908519 versus0.909842. Preserve the original C candidate and lock records.
- The complete locked matrix identifies G as an exploratory finding: frozen test mean0.912029 versus B0.909452, with positive G-B Score differences in all three head seeds on both partitions. Adaptive G0.911736 versus B0.909842. G is not uniformly better than every fusion/metric/seed. This is not a newly confirmed preselected winner.
- G validation-only post-test sensitivity, like the pre-test C sensitivity, shows zero-memory degradation but no benefit from precise within-token event alignment. Claim useful token-conditioned temporal information, not proof that fine event alignment is essential.
- Six independent full B/C/G CPU graphs were timed over all24 test sequences for three passes each; all timed masks match the independently scored predictions. G timing and G validation sensitivity are explicitly post-test follow-ups with no new training or threshold tuning.
- Source, full tables/plots, primary failure, exploratory G finding and limits are integrated in the manuscript. Final PDF layout verification is recorded in the paper evidence manifest.

- Final delivery verified:34-page PDF, no compiler warnings, all generated input/assets hashes and executed source versions checked. Full raw-input val/test scoring is independently verified for all16 models. Final conclusion is explicitly exploratory G, retaining the failed pre-test C nomination.
