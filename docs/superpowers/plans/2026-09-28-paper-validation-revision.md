# TraceFormer validation and manuscript revision plan

> Execution: inline in the existing local repositories; no worktree. Existing release weights, settings, and archived experiments remain immutable.

**Goal:** Evaluate same-task public baselines, test the attention mechanisms honestly, and explain the selected system in a consistent set of scientific figures.

**Architecture:** External implementations live under ignored `outputs/baselines/upstream/`, with upstream commits and artifact hashes recorded. Mechanism experiments reuse the audited training engine and use separate configurations, source snapshots, and output folders. The manuscript consumes verified reports rather than manually invented results.

**Tech stack:** Python 3.12, PyTorch FP32, NumPy/SciPy, existing official-compatible scorer, Matplotlib, TikZ, pdfLaTeX/latexmk.

## 1. Same-task baselines

- [ ] Download and audit PACT and EV-SpSegNet official code and available checkpoints. Record upstream commit, license, hashes, expected data fields, units, output mapping, and training provenance.
- [ ] Use exactly the local 24 validation files, retain original event order, and compute pooled IoU/foreground recall/Pd/Fa/Score with the same scorer. Different native representations are permitted; resizing, dropped events, or overlapping windows must be mapped explicitly.
- [ ] Report native official operating points and a common validation-threshold selection protocol separately. No test labels may enter selection. Public checkpoints must be labeled as such; retraining results must have their own training records.
- [ ] Evaluate TraceFormer neural and complete system separately. Do not copy TraceFormer-specific LightGBM weights to other embeddings. Any later common-postprocessing comparison needs separately fitted compatible verifiers and an equal development budget.
- [ ] CUDA-dependent upstream code runs only after a compatible GPU environment is verified. A missing CUDA run remains incomplete, never a paper-derived reproduction score.

## 2. Focused mechanism evidence

- [x] Build three paired backbone-only variants for seeds 37/73/101: attention+cone, attention+area-matched fixed window, mean+cone. All use 99/24 data, 18 epochs, the same losses/optimizer, CPU FP32, 14 threads, no Sparse/refinement/external postprocessing.
- [x] Reuse seed-37 runs only after config/data/source/checkpoint/history audits and fresh independent reevaluation. New seeds run serially with resumable state, finite-gradient and largest-sequence resource gates.
- [x] Compute all-sequence geometric diagnostics before and after the actual K=8 neighbor selection, including same-instance target-token capture, background candidates, selected-edge overlap, and temporal support. GT is used only in diagnostics, never in inference.
- [x] Independently reevaluate each saved checkpoint on all 24 files; report paired seed differences, fixed-threshold sensitivity and per-sequence results. Repeated seeds measure training variability, not independent-test generalization.
- [x] Allow a robust-validation wording only if all three paired cone differences are positive, mean gain is at least 0.001, ordinary-sequence pooled gain is nonnegative, and mean val_023 target-frame recall does not fall by more than one percentage point. Failure means results-only wording. Apply the same rule to attention versus mean; never suppress a negative comparison. Both comparisons failed and are reported as results only.

## 3. Manuscript scope

- [x] Explain occupied-block attention pooling, query-centered cone support, and learned verification/trajectory recovery as the main design. Sparse is a conditional implementation detail; E3 refinement belongs to supplementary development experiments.
- [x] Include attention versus capacity-matched mean pooling in the main evidence. Remove unrelated automotive comparison and unsupported superiority/efficiency wording.
- [x] Keep the official 31-file result distinct from reproduced 24-file validation. A local strong-baseline result enters the paper only after the full run and audit.

## 4. Figure consistency and QA

- [x] Use shared blue/orange/green/gray colors, sans-serif labels, consistent font sizes, vector PDF output, and explicit panel labels. Past=blue, future=orange, query=red; qualitative TP=green, FP=orange, FN=red.
- [x] Lead with the complete inference system; show pooling and the cone block separately. Remove the unused refinement head from the selected-release architecture.
- [x] Retain query-centered lattice geometry, real attention, time-space trajectories, stage masks, per-sequence effects and threshold curves. Clearly distinguish geometric allowance from actually selected/weighted edges.
- [x] Run focused pytest/Ruff checks, generate all assets, compile the paper, render affected pages, and inspect legibility, arrows, captions, clipping and cross-references. No new result is inferred from a schematic or favorable example.

## Concrete entry points

- `scripts/prepare_external_baselines.py`: pinned official source/checkpoint download and dependency manifest.
- `scripts/prepare_mechanism_validation.py`: audited paired three-seed experiment setup, reusing existing queue machinery.
- `scripts/analyze_cone_support.py`: all-24-file query-centered geometry and selected-neighbor diagnostics.
- `scripts/run_paper_ablation.py --suite <new-suite>`: serial training plus independent reevaluation.
- `D:/Paper/TraceFormer/scripts/figure_style.py`: common publication palette and typography.
- `D:/Paper/TraceFormer/sections/`: conservative, focused main claims and full negative results.
- `D:/Paper/TraceFormer/main.pdf`: compiled manuscript, visually checked after regeneration.

## Execution snapshot: 2026-09-28

- Official PACT source and checkpoint are pinned and downloaded; EV-SpSegNet source is pinned, but its weight URL returned login HTML. All-24-file data audits passed for both. External CUDA inference is pending, not a reproduced result.
- All three seed-37 checkpoints were independently reevaluated in the new suite. Six additional runs are queued; seed73/cone is training. Queue PID 26452; finite decision reporter PID 9204.
- All-24-file support diagnostics are complete. Actual K=8 capture differs by about 0.16 percentage points, so the manuscript reports geometry and results without effectiveness claims.
- The manuscript mainline and figures are revised. Final QA is recorded separately in the paper evidence directory.

## Verified continuation: 2026-09-29 first heartbeat

PACT and selected TraceFormer main/reference/complete pipelines completed all-24 inference on the same 4090D. Source/data/weights/reports/predictions were hash-verified after download, then independently rescored from local raw NPZ files. Complete masks match the archived release event-wise. PACT public-checkpoint Score is 0.798604 at upstream 0.9, 0.759181 at common 0.5, and 0.803205 at common-protocol selected 0.8250; TraceFormer neural/complete selected-release Scores are 0.915002/0.954117. This is an executed validation comparison with unconfirmed public checkpoint training membership, not matched retraining or a SOTA claim. EV-SpSegNet is still blocked on its official weights.

The manuscript now contains the verified comparison, all-24 sequence plot, four-stage mask table including direct-only filtering, and source/environment provenance. `docs/paper_cuda_supervision.md` records the narrow correction of the original independent-score directory error without rerunning inference, copy-back hash and audit entry points. The frozen CPU queue remains running, four of nine completed, seed73/window epoch 11/18. Do not replace single-seed mechanism language with incomplete multiseed evidence. Latest paper compile/render QA is separate from the previous revision's historical QA.

## Second heartbeat: 2026-09-29 03:30 CST

Five of nine CPU runs are complete, with seed73/window independently verified on all 24 files and all 18 epochs accounted for. Seed73 C-W gains +0.001248 at individually selected thresholds but loses -0.001248 at common 0.5; retain this operating-point sensitivity. Source/configuration and all 123 train/validation data hashes are unchanged. Actual worker 9860 is running seed73/mean at epoch 2/18; healthy queue 26452 and finite reporter 9204 continue. The three-seed gate is incomplete, so the manuscript keeps results-only mechanism wording and the verified partial evidence is stored separately under `evidence/mechanism_progress/20260929_0330`.

Remote 276-entry input/source manifest and completed reports remain verified, with no running prior GPU job. EV-SpSegNet official weights remain the outstanding baseline dependency. No repeated download, GPU rerun or login-authorization change. The existing compiled manuscript has no source/figure edits during this incomplete-results check; no new compilation is required.

## Third heartbeat: 2026-09-29 06:30 CST

Unchanged five complete results; healthy seed73/mean epoch 12/18. Actual queue/worker/reporter are 26452/9860/9204; frozen inputs and independent completed reports remain verified. Remote completed main/reference/system/PACT hashes are unchanged; EV weights remain unavailable. No new result or failure warrants changing the manuscript or restarting a job. Quiet monitoring continues; snapshot under `outputs/paper_mechanism_cpu_20260928/supervision/20260929_0630.json`.

## Fourth heartbeat: 2026-09-29 09:30 CST

Six results complete: new seed73/mean passes full18/full24 independent reevaluation, frozen source/configuration checks and 123 raw-data hashes. Seed73 C-M selected gain is +0.000558, but fixed-0.5 difference is -0.006758; retain this sensitivity. Actual worker 19332 runs seed101/cone epoch 3/18, with healthy queue/reporter 26452/9204. Partial evidence is saved in `evidence/mechanism_progress/20260929_0930`; final gate and manuscript mechanism tables/figures await all3 seeds.

Remote SSH now fails before authentication; one independent no-auth probe confirms port 35631 actively refuses TCP connections. Last remote verification was 06:30 CST, not this heartbeat. User confirmation of instance state/new connection is pending; do not guess a cause or loop retries. Downloaded GPU archive, reports and prediction hashes were checked locally and are intact. Preserve CPU progress; no new resource, server restart, login change or redundant paper compilation. Detailed blocker is in the supervision record.

## Fifth heartbeat: 2026-09-29 12:30 CST

Healthy seed101/cone epoch 13/18, six complete results unchanged and full24 independent reports verified. Actual processes 26452/19332/9204 and frozen source/configuration checks pass. Existing remote refusal persists on one unauthenticated probe; no password retry and no repeated user request. Local GPU reports remain intact, but current remote state cannot be inspected. No manuscript or experiment change; continue monitoring with snapshot `outputs/paper_mechanism_cpu_20260928/supervision/20260929_1230.json`.

## Sixth heartbeat: 2026-09-29 15:30 CST

Seven complete results: seed101/cone passes full18/full24 independent reevaluation and all input/source/weight checks; selected Score 0.909689 at 0.75, fixed-0.5 Score 0.904433. Worker 40688 runs seed101/window epoch 4/18; queue/reporter 26452/9204 are healthy. New verified evidence is in `evidence/mechanism_progress/20260929_1530`; third-seed pairs are still incomplete, so no mechanism claim or final table/figure revision yet. One no-auth probe shows the same remote refusal; existing user requests remain pending, and local GPU report hashes are intact. No duplicate run or frozen-protocol change.

## Seventh heartbeat: 2026-09-29 18:30 CST

Healthy seed101/window epoch 7/18; seven completed full24 results remain verified, with unchanged frozen source/configuration and actual process identities. Remote refusal persists on one no-auth probe; no repeated password or user request. Local GPU reports are intact. No new paired result warrants manuscript or experiment changes; quiet supervision continues with snapshot `outputs/paper_mechanism_cpu_20260928/supervision/20260929_1830.json`.

## Eighth heartbeat: 2026-09-30 09:55 CST

Seven of nine CPU results remain fully verified; seed101/window is healthy at epoch 9/18 with rising log and CPU usage despite slower progress. All 123 data files, frozen source/configs, completed checkpoints and full24 independent reports pass. The 4090D SSH port still refuses the single unauthenticated connection probe, so current remote state remains unknown; locally copied validated GPU evidence is intact. Existing server/EV checkpoint questions await answers. No new paper result or figure change, no experiment rerun, and no redundant compilation. Snapshot: `outputs/paper_mechanism_cpu_20260928/supervision/20260930_0955.json`.

## Ninth heartbeat: 2026-09-30 11:15 CST

Seven CPU results and all frozen source/config/data/checkpoint/full24 independent reports remain verified; seed101/window is healthy at epoch 13/18. Both paired three-seed decisions are still incomplete. The single no-auth 4090D port probe again returned connection refused; current remote state cannot be inspected, but the verified local archive hash and four report pairs remain intact. Existing connection and EV checkpoint requests are pending. No experiment rerun, paper edit, or redundant compilation. See `outputs/paper_mechanism_cpu_20260928/supervision/20260930_1115.json`.

## Tenth heartbeat: 2026-09-30 14:15 CST

Eight of nine CPU runs are verified. Seed101/window completed full18/full24, with exact independent FP32 reevaluation and frozen input/source/weight checks. Three-seed cone-versus-area-matched-window paired selected-threshold gains are +0.001100, +0.001248 and +0.000110 (mean +0.000819), below the preregistered +0.001 mean gate. Therefore report this as a negative mechanism-claim decision, while preserving the small positive numerical result and fixed-0.5 sensitivity. Seed101/mean is healthy at epoch 5/18; final attention-vs-mean and paper tables/figures await it. Evidence: `D:/Paper/TraceFormer/evidence/mechanism_progress/20260930_1415/`. One no-auth SSH probe still refused the existing endpoint; local GPU archive remains verified and prior user requests remain pending. No protocol change, restart or paper compilation.

## Eleventh heartbeat: 2026-09-30 18:05 CST

Eight verified runs remain complete; seed101/mean is healthy at epoch 17/18. Frozen source/config/data/checkpoint histories and full24 independent reports pass. Cone/window preregistered gate failed at mean +0.000819; mean-pooling comparison remains incomplete. The abstract, introduction, mechanism subsection/table, discussion and conclusion need an atomic three-seed revision after the last run. Remote SSH port still refuses the single no-auth probe; local GPU archive/reports remain verified and previously requested connection/checkpoint information is pending. No rerun, protocol change or redundant paper compilation. Runtime snapshot: `outputs/paper_mechanism_cpu_20260928/supervision/20260930_1715.json`.

## Final paired-seed and manuscript milestone: 2026-10-01 00:22 CST

Nine of nine CPU runs finished normally. Frozen 76 source files, nine configs, 123 raw files, nine weights, full18 histories, full24 independent reevaluations, and 216 prediction hashes are audited in `D:/Paper/TraceFormer/evidence/mechanism_three_seed/audit.json`. C−W mean selected Score gain +0.000819 fails the pre-registered +0.001 mean gate despite three positive seeds. C−M mean +0.000678 fails because seed37 is negative and the mean is below +0.001. The manuscript reports both without effectiveness claims and includes paired-seed, per-sequence and operating-point visuals. The 26-page `D:/Paper/TraceFormer/main.pdf` compiles with no relevant warnings, embedded fonts and rendered-page QA; focused tests passed 25/25. See `D:/Paper/TraceFormer/evidence/mechanism_three_seed/qa.json` and `outputs/paper_mechanism_cpu_20260928/supervision/20261001_0022.json`.

The 4090D SSH endpoint still refuses unauthenticated TCP; locally verified PACT/TraceFormer GPU evidence remains intact. A genuine EV-SpSegNet checkpoint and current server connection were requested previously and remain unavailable. These external prerequisites are the only unfinished baseline work; leave the three-hour automation active without duplicate training or repeated failed downloads.


## Quiet supervision after system-focus revision: 2026-10-01 15:32 CST

All nine completed runs remain intact: actual former queue/worker/reporter PIDs 26452/36872/9204 are absent; nine configs/checkpoints, full18 histories, full24 saved/independent metric equality, all 216 prediction hashes, frozen source, all 123 raw data hashes and copied compact evidence pass. Both preregistered mechanism gates remain failed. The downloaded GPU archive retains its verified hash. One no-auth TCP check again returns ConnectionRefusedError/10061; no password session or retry loop. Previously requested server connection and genuine EV-SpSegNet checkpoint remain unavailable; no new user action or repeated notification is warranted.

User-directed narrative revision is complete and verified: `D:/Paper/TraceFormer/main.pdf` is now 25 pages, SHA-256 `27485eb9d61c29a123cb51d5cbb70bce1a3eb4aa8c92e632d57e4f6bc6f0d061`, with current QA at `evidence/system_focus_20261001/qa.json`. Title, abstract, contributions, method ordering, stages/figures, discussion and conclusion emphasize the complete evaluated method, without attributing the overall gain to a neural structure. All four core result tables are unchanged. This was already reported to the user interactively; do not repeat that notification or overwrite it with the older 26-page manuscript.

Updated the existing three-hour heartbeat prompt to preserve this accepted system-level focus and concentrate on remaining external prerequisites. No new experiment or manuscript edit/compilation. Snapshot: `outputs/paper_mechanism_cpu_20260928/supervision/20261001_1532.json`.
