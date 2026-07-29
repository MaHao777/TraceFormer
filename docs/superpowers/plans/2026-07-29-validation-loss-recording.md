# Validation Loss Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record deterministic validation loss components after every validation pass for both fresh and resumed HUAV training.

**Architecture:** Extend the existing single-pass validation loop so it can optionally evaluate `EventFocalDiceLoss` while logits are still on the active device. Validation uses the same positive-preserving hard-negative sampling and dense-sequence weighting as training, but a separately seeded generator makes the sampled validation objective reproducible and prevents validation from advancing the training sampler. Store the averaged loss components under `history.json[*].validation_loss`; keep threshold metrics under `validation` and `metrics.json` unchanged.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff

---

### Task 1: Specify the history contract

**Files:**
- Modify: `tests/test_training_engine.py`

- [x] **Step 1: Write the failing fresh-training assertion**

Add assertions that one training epoch writes:

```python
assert set(history[0]["validation_loss"]) == {
    "total",
    "focal",
    "dice",
    "object_frame",
    "false_alarm",
}
assert all(value >= 0.0 for value in history[0]["validation_loss"].values())
assert "loss" not in history[0]["validation"]
assert "loss" not in metrics
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_training_engine.py::test_cpu_training_writes_reproducible_metrics_and_checkpoint -q
```

Expected: fail because `validation_loss` is absent.

- [x] **Step 3: Add a resumed-training contract test**

Run one additional epoch with `resume_training`, then assert its first history record has the same five validation-loss components and uses the absolute epoch number after the source checkpoint.

### Task 2: Compute validation loss in the existing validation pass

**Files:**
- Modify: `src/huav/training/engine.py`

- [x] **Step 1: Add optional loss calculation to `evaluate_model`**

Accept an optional `EventFocalDiceLoss`. When present, create a validation-only generator seeded from `config.training.seed`, sample the same positive/hard-negative objective used for training, pass full-event tensors to the Fa loss, apply `_dense_sequence_loss_weight` to `total`, and average all five components over validation files.

- [x] **Step 2: Return loss separately from threshold metrics**

Attach the averaged values temporarily as `result["loss"]`. Callers that do not pass a criterion keep the existing metrics-only response.

- [x] **Step 3: Record fresh-training validation loss**

Pass the criterion from `run_training`, remove `loss` from the metric result, and write it as:

```python
epoch_record["validation_loss"] = validation_loss
```

Include `val_loss=<total>` in the epoch console summary.

- [x] **Step 4: Record resumed-training validation loss**

Apply the same behavior in `resume_training` so every validation path has the same history schema.

### Task 3: Verify behavior and documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_training_engine.py`

- [x] **Step 1: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_training_engine.py -q
```

Expected: all focused tests pass.

- [x] **Step 2: Document loss semantics**

State that `history.json` stores `train_loss`, deterministic sampled `validation_loss`, and threshold-based `validation` metrics separately. Clarify that validation uses the training sampling/weighting objective but never changes optimizer or training sampler state.

- [x] **Step 3: Run repository gates**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Expected: tests, Ruff, and whitespace checks pass.

- [ ] **Step 4: Run resource, smoke, full validation, and submission closure**

Use new output directories, verify the largest sequence, smoke-train and independently evaluate, then complete the repository-required full train/evaluate/predict/audit workflow. Confirm new `history.json` contains validation loss for every epoch while `metrics.json` and `reproduced_metrics.json` remain byte-identical.
