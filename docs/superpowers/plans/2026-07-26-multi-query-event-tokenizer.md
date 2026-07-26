# Multi-Query Event Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-block mean/max event pooling with event-count-independent, sparse multi-query attention pooling while preserving event order, sparse block identity, downstream token interfaces, and reproducible HUAV training/submission behavior.

**Architecture:** A shared event MLP encodes every event independently. Four learned queries attend to each variable-length block through segmented softmax over the existing `inverse` block index, so no fixed event-count parameter or dense padding is introduced; flattened query outputs are fused and concatenated with seven explicit statistics plus the existing normalized block-position encoding.

**Tech Stack:** Python 3.11, PyTorch scatter operations, dataclass/YAML configuration, pytest, Ruff, HUAV CPU train/evaluate/predict CLIs.

---

### Task 1: Specify attention pooling and statistics behavior

**Files:**
- Modify: `tests/test_event_tokenizer.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add failing tokenizer structure and gradient tests**

Add tests that require four shared learned queries, key/value projections, a fusion layer whose input width is `4 * event_embed_dim`, and gradients on every attention-pooling parameter.

- [ ] **Step 2: Add a failing segmented-attention reference test**

Construct two variable-length groups, make key/value projections identity transforms, and compare `_attention_pool` with an explicit per-group softmax calculation.

- [ ] **Step 3: Add a failing signed-polarity statistics test**

Pass polarities in `{-1, 1}` and assert `positive_fraction`, `polarity_contrast`, and `polarity_entropy` are computed from `(polarity > 0)`.

- [ ] **Step 4: Add failing configuration tests**

Require `num_pool_queries == 4`, reject non-positive values, and require the score-optimized YAML to use `event_embed_dim == 64`.

- [ ] **Step 5: Run the focused tests and confirm expected failures**

Run:

```powershell
python -m pytest tests/test_event_tokenizer.py tests/test_config.py -q
```

Expected: failures for missing query parameters/configuration, the absent segmented attention helper, old 8-statistic polarity behavior, and the old score-config embedding dimension.

### Task 2: Implement sparse multi-query pooling

**Files:**
- Modify: `src/huav/models/event_tokenizer.py`
- Modify: `src/huav/models/event_decoder.py`

- [ ] **Step 1: Replace the event encoder**

Use `Linear(6, D) -> LayerNorm(D) -> GELU -> Linear(D, D)` for shared per-event encoding.

- [ ] **Step 2: Add count-independent attention parameters**

Add `pool_queries[M, D]`, bias-free `key_projection`/`value_projection`, and `query_fusion` from `M * D` to `D`. Validate `M > 0`.

- [ ] **Step 3: Implement segmented softmax**

For each learned query, calculate scaled event logits, use `scatter_reduce_(..., reduce="amax")` and `scatter_add_` over `inverse`, and aggregate projected values without padding or an event-count-dependent parameter.

- [ ] **Step 4: Replace mean/max aggregation**

Flatten the `[num_tokens, M, D]` attention output, fuse it, and concatenate `[attention_feature, statistics, encoded_position]` before `token_projection`.

- [ ] **Step 5: Normalize polarity statistics**

Use `(polarity > 0).to(events.dtype)` for positive fractions and sequence-relative polarity contrast. Emit exactly:

```text
log1p(count), positive_fraction, local_time_mean, local_time_variance,
density, polarity_contrast, polarity_entropy
```

Update the decoder's statistic indices from `(0, 5, 6, 7)` to `(0, 4, 5, 6)`.

- [ ] **Step 6: Run the tokenizer tests**

Run:

```powershell
python -m pytest tests/test_event_tokenizer.py -q
```

Expected: all tokenizer tests pass.

### Task 3: Propagate configuration without changing downstream token interfaces

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `tests/test_spatiotemporal_cone_model.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Add `num_pool_queries` to model configuration**

Default to four, validate it as a positive integer, serialize it with saved run configuration, and pass it through `build_model` into the tokenizer.

- [ ] **Step 2: Select production dimensions**

Keep the resource-safe `8x8/64` spacetime grid and set the score-optimized smoke/full configs to `event_embed_dim: 64` and `num_pool_queries: 4`. Add the query count explicitly to the other maintained YAML files without changing their smaller smoke/base embedding sizes.

- [ ] **Step 3: Preserve return contracts**

Keep `EventTokenizer.forward(events, batch_indices=None)` and every `TokenizedEvents` field unchanged; keep detector outputs unchanged.

- [ ] **Step 4: Run focused model/config/training tests**

Run:

```powershell
python -m pytest tests/test_event_tokenizer.py tests/test_spatiotemporal_cone_model.py tests/test_config.py tests/test_training_engine.py -q
```

Expected: all focused tests pass.

### Task 4: Static and regression gate

**Files:**
- Inspect: all changed source, test, config, and plan files

- [ ] **Step 1: Run the complete repository suite**

```powershell
python -m pytest -q
```

- [ ] **Step 2: Run Ruff**

```powershell
python -m ruff check src tests
```

- [ ] **Step 3: Check patch whitespace**

```powershell
git diff --check
```

- [ ] **Step 4: Inspect the final diff and working tree**

```powershell
git diff -- src tests configs docs/superpowers/plans/2026-07-26-multi-query-event-tokenizer.md
git status --short
```

Expected: only intended files are changed and all checks exit zero.

### Task 5: Real-data resource and training closure

**Files:**
- Create: new timestamped directories beneath `outputs/`

- [ ] **Step 1: Reproduce the largest sequence resource path**

Run `train_094.npz` through the new `8x8/64`, `D=64`, `M=4` tokenizer/model for forward, configured loss, and backward on CPU. Record event/token counts, loss, elapsed time, and peak working-set memory in the smoke output directory.

- [ ] **Step 2: Run score-optimized CPU smoke training**

```powershell
python -m huav.cli.train --config configs/huav_score_optimized_smoke.yaml --device cpu --output-dir $smokeDir
python -m huav.cli.evaluate --config "$smokeDir/config.yaml" --checkpoint "$smokeDir/best.pt" --device cpu --output "$smokeDir/reproduced_metrics.json" --quiet
```

Require byte-identical `metrics.json` and `reproduced_metrics.json`.

- [ ] **Step 3: Run score-optimized CPU full training**

```powershell
python -m huav.cli.train --config configs/huav_score_optimized.yaml --device cpu --output-dir $runDir
```

Require all 99 training and 24 validation files and all standard run artifacts.

- [ ] **Step 4: Independently reevaluate the full best checkpoint**

```powershell
python -m huav.cli.evaluate --config "$runDir/config.yaml" --checkpoint "$runDir/best.pt" --device cpu --output "$runDir/reproduced_metrics.json" --quiet
```

Require byte-identical metrics, full confusion-count coverage, and matching manifest validation-file counts.

- [ ] **Step 5: Generate and validate submission files**

```powershell
python -m huav.cli.predict --config "$runDir/config.yaml" --checkpoint "$runDir/best.pt" --output-dir $submissionDir --device cpu
```

Require one root-level TXT per validation NPZ, five columns per row, unchanged `x y t p` values/order, binary labels, matching event counts, and a flat `submission.zip`.

### Task 6: Final verification and delivery

**Files:**
- Inspect: full-run metrics, manifest, history, summary, submission files/ZIP, and working tree

- [ ] **Step 1: Run final quality gates**

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
git status --short
```

- [ ] **Step 2: Report exact artifacts and metrics**

Report the full run directory, saved config, CPU device, train/validation counts, best epoch/threshold, competition score components, event metrics, exact independent reproduction status, absolute ZIP path, TXT/row/positive counts, ZIP validation result, and final code-quality outputs.
