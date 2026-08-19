# Local Cone Iterative Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-head multi-scale cones and per-layer temporal expansion with independently parameterized, sequential fixed-local-cone blocks whose neighborhoods cover only `t-1`, `t`, and `t+1`.

**Architecture:** Keep the tokenizer, event decoder, event ordering, and detector output contract unchanged. Build one batch-isolated quadrangular-pyramid neighborhood per center token using shared geometry, apply standard multi-head center-query attention to that neighborhood, and pass the complete synchronously updated token tensor through a configurable stack of Pre-LN residual blocks.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy `cKDTree`, dataclass/YAML configuration, pytest, Ruff.

---

### Task 1: Specify the fixed local neighborhood and configuration contract

**Files:**
- Modify: `src/huav/config.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write configuration tests for the new fields**

  Assert that `num_cone_blocks` defaults to `5`, every shipped YAML exposes the same `local_cone.base_radius`, `local_cone.slope`, and `local_cone.max_neighbors_per_time_slice` contract, and depths `1`, `3`, `5`, and `7` validate.

- [ ] **Step 2: Remove the multi-cone configuration surface**

  Delete `ConeHeadConfig`, `cone_heads`, `layer_time_scales`, `num_layers`, and `max_neighbors_per_cone`. Add a frozen `LocalConeConfig` and `ModelConfig.num_cone_blocks`.

- [ ] **Step 3: Update YAML loading and validation**

  Convert the nested `model.local_cone` mapping to `LocalConeConfig`, require non-negative finite radii/slopes and a positive per-time-slice cap, and keep `num_heads` solely as standard attention-head partitioning.

- [ ] **Step 4: Run the configuration tests**

  Run: `python -m pytest tests/test_config.py -q`

  Expected: all configuration tests pass and no shipped YAML contains `cone_heads` or `layer_time_scales`.

### Task 2: Build one synchronous `t-1/t/t+1` neighborhood

**Files:**
- Modify: `src/huav/models/cone_neighbor_builder.py`
- Modify: `tests/test_cone_neighbor_builder.py`

- [ ] **Step 1: Write geometry tests**

  Cover batch isolation, Chebyshev spatial distance, `r(dt) = base_radius + slope * abs(dt)`, exact temporal offsets `{-1, 0, 1}`, per-time-slice neighbor caps, causal suppression of `t+1`, stable token indices, and empty input.

- [ ] **Step 2: Replace per-head/per-horizon construction**

  Return one padded `[M, 3 * max_neighbors_per_time_slice]` index tensor. Query each time slice independently so dense `t` tokens cannot consume the capacity of `t-1` or `t+1`.

- [ ] **Step 3: Run the neighborhood tests**

  Run: `python -m pytest tests/test_cone_neighbor_builder.py -q`

  Expected: every selected neighbor belongs to the same sample and obeys the fixed local pyramid.

### Task 3: Implement local center attention and independent Pre-LN blocks

**Files:**
- Modify: `src/huav/models/cone_attention.py`
- Modify: `src/huav/models/cone_transformer.py`
- Modify: `tests/test_cone_attention.py`
- Modify: `tests/test_spatiotemporal_cone_model.py`

- [ ] **Step 1: Write attention/block tests**

  Verify one shared neighborhood is broadcast across standard attention heads, Q/K/V are calculated from the same current block input, inputs are not mutated in place, every block owns different projection/FFN/norm/`attn_scale` parameters, and block `l+1` receives the full output of block `l`.

- [ ] **Step 2: Replace direction/multi-cone fusion with local attention**

  Compute `Q`, `K`, and `V` from the supplied token tensor, gather all fixed-local neighbors without modifying the source tensor, apply masked scaled dot-product attention, concatenate heads, and apply one output projection. Preserve the detector's past/future debug-stat keys by deriving directional counts and attention mass from the single local attention operation.

- [ ] **Step 3: Implement `LocalConeAttentionBlock`**

  Use `tokens + attn_scale * dropout(attention(LN(tokens)))`, then `tokens + dropout(FFN(LN(tokens)))`. Build `num_cone_blocks` distinct instances in a `ModuleList`, feeding each complete output tensor to the next block.

- [ ] **Step 4: Run model tests**

  Run: `python -m pytest tests/test_cone_attention.py tests/test_spatiotemporal_cone_model.py -q`

  Expected: forward/backward are finite, token/event shapes and indices stay unchanged, and debug-stat list length equals the configured block count.

### Task 4: Wire the detector and training entry point without training

**Files:**
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/training/engine.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Replace constructor plumbing**

  Pass `num_cone_blocks` and the single `local_cone` geometry from `build_model` to the detector and encoder. Do not change tokenizer, decoder, loss, event ordering, or returned field names.

- [ ] **Step 2: Advance the checkpoint schema**

  Increment the schema because previous multi-cone state dictionaries and model configs are intentionally incompatible.

- [ ] **Step 3: Update compact engine fixtures**

  Use the new model configuration in the existing one-file pipeline tests without running any user-data smoke or full training workflow during this task.

### Task 5: Verify the code-only refactor

**Files:**
- Review: all files changed above

- [ ] **Step 1: Run targeted regression tests**

  Run: `python -m pytest tests/test_config.py tests/test_cone_neighbor_builder.py tests/test_cone_attention.py tests/test_spatiotemporal_cone_model.py -q`

- [ ] **Step 2: Run the repository quality gates**

  Run: `python -m pytest -q`, `python -m ruff check src tests`, and `git diff --check`.

- [ ] **Step 3: Audit the final diff**

  Search for removed names (`cone_heads`, `layer_time_scales`, `num_layers`, `max_neighbors_per_cone`) in runtime/config paths, confirm all six YAML files use the new fields, and run a compact synthetic forward/backward for depths `1`, `3`, `5`, and `7` to confirm unchanged output shapes.

- [ ] **Step 4: Report the explicit validation boundary**

  State that no real-data smoke/full training, independent checkpoint reevaluation, or submission packaging was run because the user explicitly requested a code-only change.
