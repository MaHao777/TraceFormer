# Decoder Simplification and Fa Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated block statistics from event decoding and add a differentiable, full-event false-alarm surrogate aligned with the competition's frame/pixel component metric.

**Architecture:** The decoder will classify each original event from its contextual parent token plus its six event-local features; block statistics remain only inside token construction. The new Fa objective will group background events by `(batch, 50 ms frame, x, y)`, reduce each occupied background pixel to its maximum logit, and apply a configurable margin-softplus loss to the 32 hardest pixels per frame. Focal, Dice, and object-frame recall continue to use sampled events, while the Fa surrogate uses all forwarded events so hard background responses cannot escape through sampling or be diluted by easy background.

**Tech Stack:** Python, PyTorch, dataclass/YAML configuration, pytest, Ruff

---

### Task 1: Lock the decoder contract

**Files:**
- Create: `tests/test_event_decoder.py`
- Modify: `src/huav/models/event_decoder.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`

- [ ] **Step 1: Write the failing decoder test**

```python
def test_decoder_uses_parent_tokens_and_event_local_features_only() -> None:
    decoder = EventDecoder(token_dim=16, dropout=0.0)
    logits = decoder(
        torch.randn(2, 16),
        torch.tensor([0, 0, 1]),
        torch.randn(3, 6),
    )
    assert decoder.classifier[0].in_features == 22
    assert logits.shape == (3,)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/test_event_decoder.py -q`

Expected: fail because the existing decoder still requires `token_statistics` and its first layer expects `token_dim + 10`.

- [ ] **Step 3: Remove decoder block-statistic concatenation**

Change `EventDecoder.forward` to accept only `token_features`, `event_token_indices`, and `event_local_features`. Build the classifier input with `token_dim + EventTokenizer.event_feature_dim` values and update `SpatioTemporalConeDetector.forward` accordingly.

- [ ] **Step 4: Run focused model tests**

Run: `python -m pytest tests/test_event_decoder.py tests/test_spatiotemporal_cone_model.py -q`

Expected: all tests pass, including end-to-end gradients and empty input.

### Task 2: Define the Fa-aligned surrogate

**Files:**
- Modify: `tests/test_training_components.py`
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`

- [ ] **Step 1: Write failing behavioral tests**

Add tests proving:

```python
assert false_alarm_pixel_loss(low_background, ...) < false_alarm_pixel_loss(
    high_background, ...
)
```

and proving that repeated events at one `(frame, x, y)` are reduced to one pixel maximum rather than counted repeatedly. Add a criterion-composition test requiring:

```python
total = focal + lambda_dice * dice + lambda_object_frame * object_frame
total = total + lambda_false_alarm * false_alarm
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python -m pytest tests/test_training_components.py -q`

Expected: import or constructor failure because the Fa loss and configuration parameters do not exist.

- [ ] **Step 3: Implement the minimal differentiable loss**

Implement:

```python
pixel_max = scatter_amax(
    logits[background],
    unique(batch, floor(t / frame_ms), x, y),
)
loss = softplus(pixel_max - false_alarm_logit_margin).mean()
```

Use the same integer-millisecond and strict frame-boundary behavior as `EVUAVMetricAccumulator`. Return a graph-connected zero for empty inputs or batches with no eligible background events.

- [ ] **Step 4: Add the objective to `EventFocalDiceLoss`**

Add `lambda_false_alarm` and `false_alarm_logit_margin`. Keep focal/Dice/object-frame inputs sampled, and accept separate full-event logits, targets, batches, timestamps, and coordinates for the Fa term.

- [ ] **Step 5: Re-run the focused tests**

Run: `python -m pytest tests/test_training_components.py -q`

Expected: all tests pass with finite gradients.

### Task 3: Wire configuration and both training paths

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`

- [ ] **Step 1: Write failing config and integration assertions**

Require score-optimized configurations to expose:

```yaml
lambda_false_alarm: 0.05
false_alarm_logit_margin: 0.0
```

Require negative loss weights to be rejected and the training artifact to report a `false_alarm` loss component.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_config.py tests/test_training_engine.py -q`

Expected: fail because the new configuration fields and training history entry do not exist.

- [ ] **Step 3: Wire fresh and resumed training**

Pass the full event logits, labels, batch indices, raw millisecond timestamps, and `events[:, :2]` coordinates to the criterion in both training loops. Track `false_alarm` in epoch totals and saved history.

- [ ] **Step 4: Bump the checkpoint schema**

Change `CHECKPOINT_SCHEMA_VERSION` from `3` to `4` because removing four decoder inputs changes the first decoder weight shape. Update current schema assertions while keeping historical result documents unchanged.

- [ ] **Step 5: Run focused integration tests**

Run: `python -m pytest tests/test_config.py tests/test_training_engine.py -q`

Expected: all tests pass and the smoke training artifact contains finite `false_alarm` values.

### Task 4: Local verification before cloud training

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the current architecture/loss documentation**

Document that event decoding uses `[parent_token, six event-local features]`, that the Fa loss operates on full-event background pixel maxima, and that new checkpoints use schema `4`.

- [ ] **Step 2: Run the requested local verification**

Run:

```powershell
python -m pytest tests/test_event_decoder.py tests/test_spatiotemporal_cone_model.py tests/test_training_components.py tests/test_config.py tests/test_training_engine.py -q
python -m ruff check src tests
git diff --check
git status --short
```

Expected: focused tests, Ruff, and diff checks pass. Do not run full training, independent evaluation, or submission export until the user provides the cloud server entry.
