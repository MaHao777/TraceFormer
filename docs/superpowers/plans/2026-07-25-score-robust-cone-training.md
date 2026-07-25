# Score-Robust Cone Training Implementation Plan

> Completed on 2026-07-26 with test-first implementation, smoke training,
> full CPU training, and independent checkpoint reproduction.

**Goal:** Improve the official JinSight score by making the cone detector robust to dense polarity-balanced sequences, preserving long-lag cone evidence, and aligning threshold/model training with object-frame Pd and false-alarm behavior.

**Architecture:** Retain one logit per original event and the sparse cone transformer. Enrich every event/token with sequence-relative polarity features, select temporally diverse cone neighbors, train with hard negatives plus an object-frame max-margin loss, and weight the dense balanced-polarity training domain that the current checkpoint ignores. Use a full-density-safe 8x8/64-block, 48-channel experiment configuration and select thresholds by the complete official score.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy cKDTree, pytest, Ruff

---

### Task 1: Select validation thresholds by official score

**Files:**
- Modify: `src/huav/training/engine.py`
- Test: `tests/test_training_engine.py`

- [x] **Step 1: Write a failing synthetic validation test**

Construct two object-frame instances where a lower threshold has slightly worse
event IoU but detects both objects, while a higher threshold detects only one.
Call the wished-for `_select_competition_threshold` API and assert that it
selects the lower threshold because the official weighted score is larger.

- [x] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_training_engine.py -q
```

Expected: collection fails because `_select_competition_threshold` does not
exist.

- [x] **Step 3: Implement score-based coarse-to-fine threshold search**

Add a helper that evaluates `EventMetricAccumulator`,
`EVUAVMetricAccumulator`, and `competition_metrics` for one threshold without
rerunning inference. Search the configured coarse candidates, refine around the
best candidate, and tie-break toward the configured fallback threshold.

- [x] **Step 4: Verify GREEN**

Run the focused test again and require zero failures.

### Task 2: Preserve sequence-relative polarity information

**Files:**
- Modify: `src/huav/models/event_tokenizer.py`
- Modify: `src/huav/models/event_decoder.py`
- Test: `tests/test_event_tokenizer.py`
- Test: `tests/test_spatiotemporal_cone_model.py`

- [x] **Step 1: Write failing tokenizer tests**

Require `event_local_features` to contain
`[local_x, local_y, local_t, p, p-sequence_mean, sequence_mean]`.
Require token statistics to append token-vs-sequence polarity contrast and
binary polarity entropy.

- [x] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_event_tokenizer.py tests/test_spatiotemporal_cone_model.py -q
```

Expected: shape/value assertions fail against the current four event features
and six token statistics.

- [x] **Step 3: Implement batch-isolated relative features**

Use scatter reductions keyed by `parsed_batches`, so separate samples never
share their sequence polarity mean. Expand the decoder input to consume the
six event features plus log-count, density, contrast, and entropy.

- [x] **Step 4: Verify GREEN**

Run the focused tests and require zero failures.

### Task 3: Keep long-lag evidence under the neighbor cap

**Files:**
- Modify: `src/huav/models/cone_neighbor_builder.py`
- Test: `tests/test_cone_neighbor_builder.py`

- [x] **Step 1: Write a failing temporal-diversity test**

Create more near-time candidates than `K` plus a valid far-time candidate.
Assert that the capped neighborhood contains at least one candidate from each
available time distance before taking second-ranked candidates.

- [x] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_cone_neighbor_builder.py -q
```

Expected: the far-time candidate is absent because the current score orders
all near-time candidates first.

- [x] **Step 3: Implement rank-interleaved lag selection**

Rank candidates primarily by within-slice neighbor rank, then time distance,
then Chebyshev distance. Keep the existing deterministic stable ordering and
batch isolation.

- [x] **Step 4: Verify GREEN**

Run the focused neighbor tests and require zero failures.

### Task 4: Add hard-negative and object-frame training objectives

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/sampling.py`
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/training/__init__.py`
- Test: `tests/test_config.py`
- Test: `tests/test_training_components.py`
- Test: `tests/test_training_engine.py`

- [x] **Step 1: Write failing component tests**

Require the sampler to retain all positives, take a configured fraction of
highest-logit negatives, and fill the remainder randomly. Require
`object_frame_recall_loss` to be lower when every labeled object-frame has one
logit above the configured margin.

- [x] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_training_components.py tests/test_config.py -q
```

Expected: new keyword arguments/functions/config fields are missing.

- [x] **Step 3: Implement the score-aligned objectives**

Add `hard_negative_fraction`, dense-domain weighting thresholds/weight, cosine
minimum learning-rate ratio, `lambda_object_frame`, and
`object_frame_logit_margin`. Move loss sampling after forward inference so
detached logits can choose hard negatives. Group positive events by
`(object_id, floor(t_ms / frame_ms))`, apply a max-logit softplus margin, and
add it to focal plus Dice.

- [x] **Step 4: Add dense-domain sequence weighting and cosine decay**

Detect dense balanced-polarity sequences from event count and background
polarity mean, multiply their training objective by the configured weight, and
record the current learning rate in `history.json`.

- [x] **Step 5: Verify GREEN**

Run component, config, and engine tests and require zero failures.

### Task 5: Add dedicated optimized experiment configurations

**Files:**
- Create: `configs/huav_score_optimized.yaml`
- Create: `configs/huav_score_optimized_smoke.yaml`
- Modify: `README.md`
- Test: `tests/test_config.py`

- [x] **Step 1: Write the failing configuration assertions**

Require the optimized configuration to load an 8x8/64-block tokenizer,
48-dimensional tokens, three transformer layers, eight neighbors, six epochs,
hard-negative sampling, dense-domain weighting, cosine decay, and nonzero
object-frame loss.

- [x] **Step 2: Verify RED**

Run `python -m pytest tests/test_config.py -q`; expect a missing-file failure.

- [x] **Step 3: Add full and smoke YAML files**

The smoke file keeps the same model/loss path with one epoch and a small file
limit. Resolve all paths relative to each config file and use new output
directories only.

- [x] **Step 4: Document the experiment**

Describe why the 8x8/64 grid is required for the 300k-625k-event files, why
dense balanced-polarity files are weighted, and why the result must be compared
by the official score.

- [x] **Step 5: Verify GREEN**

Run the configuration tests and require zero failures.

### Task 6: Verify, train, reproduce, and compare

**Files:**
- Generate: `outputs/score_optimized_cpu_smoke_20260725/`
- Generate: `outputs/score_optimized_cpu_full_20260725/`
- Create: `docs/score_optimized_training_results_20260725.md`

- [x] **Step 1: Run repository verification**

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Require zero test and Ruff failures. Fix the two baseline import-format
findings while touching the affected files.

- [x] **Step 2: Run CPU smoke training and independent evaluation**

```powershell
python -m huav.cli.train --config configs/huav_score_optimized_smoke.yaml --device cpu --output-dir outputs/score_optimized_cpu_smoke_20260725
python -m huav.cli.evaluate --config outputs/score_optimized_cpu_smoke_20260725/config.yaml --checkpoint outputs/score_optimized_cpu_smoke_20260725/best.pt --device cpu --output outputs/score_optimized_cpu_smoke_20260725/reproduced_metrics.json --quiet
```

Require the saved and reproduced metrics to match.

- [x] **Step 3: Run full CPU training**

```powershell
python -m huav.cli.train --config configs/huav_score_optimized.yaml --device cpu --output-dir outputs/score_optimized_cpu_full_20260725
```

Keep per-file progress visible and do not overwrite prior outputs.

- [x] **Step 4: Independently reproduce the best checkpoint**

```powershell
python -m huav.cli.evaluate --config outputs/score_optimized_cpu_full_20260725/config.yaml --checkpoint outputs/score_optimized_cpu_full_20260725/best.pt --device cpu --output outputs/score_optimized_cpu_full_20260725/reproduced_metrics.json --quiet
```

Compare the complete JSON values and require confusion counts to cover all
validation events.

- [x] **Step 5: Write the experiment report and final verification**

Report configuration changes, duration, per-epoch official score, best
threshold, dense-domain diagnostics, baseline delta, and artifact paths. Rerun
the full test/Ruff/diff trio after the report and inspect `git status` without
altering `.vscode/`.
