# Dual-View Event Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and CPU-train an event-level HUAV detector that projects each 8-second sequence into multi-scale x-t/y-t views, exchanges view features, queries them at every original event, and reports Pd, Fa, IoU, and Acc under explicit metric definitions.

**Architecture:** The data path loads the structured `ev` array from each NPZ and preserves source order. `MultiScaleEventProjector` creates four-channel x-t/y-t maps at several temporal resolutions; independent CNN/window-attention/FPN encoders and time-local bidirectional cross-attention produce high-resolution view features; `grid_sample` queries those features at each event before an MLP returns event logits. Auxiliary soft projection maps and configurable focal, Dice, consistency, and temporal losses train the two branches without building a dense x-y-t tensor.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, PyYAML, pytest

---

### Task 1: Configuration and NPZ dataset

**Files:**
- Modify: `src/huav/config.py`
- Create: `src/huav/data/dataset.py`
- Modify: `src/huav/data/__init__.py`
- Modify: `configs/huav_base.yaml`
- Test: `tests/test_dual_view_data.py`

- [ ] **Step 1: Write failing dataset/config tests**

```python
def test_npz_dataset_preserves_original_event_order(real_npz):
    item = NPZEventDataset(real_npz.parent)[0]
    assert item.events.shape[1] == 4
    assert torch.all(item.events[1:, 2] >= item.events[:-1, 2])
    assert item.labels.shape[0] == item.events.shape[0]

def test_training_config_exposes_loss_weights_and_background_ratio():
    config = load_config("configs/huav_base.yaml")
    assert config.loss.view_weight == 0.5
    assert config.training.background_ratio == 10
```

- [ ] **Step 2: Run tests and confirm missing imports/config sections fail**

Run: `python -m pytest tests/test_dual_view_data.py -q`

Expected: collection/import failure because the dataset and configuration sections do not exist.

- [ ] **Step 3: Implement data loading and configuration**

Load `ev[x,y,t,p,label,name]` directly, normalize timestamps by configured `duration_ms=8000`, keep all validation events, and expose model/training/loss/evaluation settings as frozen dataclasses. Reject unknown YAML fields and invalid dimensions or weights.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_dual_view_data.py -q`

Expected: all dataset/config tests pass.

### Task 2: Multi-scale projections and soft targets

**Files:**
- Create: `src/huav/models/projection.py`
- Test: `tests/test_projection.py`

- [ ] **Step 1: Write failing projection tests**

```python
def test_projection_contains_positive_negative_mean_and_latest_channels():
    views = projector(events, batch_indices, batch_size=1)
    assert views.xt[8].shape == (1, 4, 8, width)
    assert views.yt[8].shape == (1, 4, 8, height)
    assert views.xt[8][0, 0].sum() == 2
    assert views.xt[8][0, 1].sum() == 1

def test_soft_targets_are_nonzero_near_positive_events():
    xt, yt = build_view_targets(events, labels, ...)
    assert xt.max() == 1
    assert yt.max() == 1
```

- [ ] **Step 2: Run and confirm imports fail**

Run: `python -m pytest tests/test_projection.py -q`

Expected: missing `huav.models.projection`.

- [ ] **Step 3: Implement scatter-based maps**

For each requested temporal scale create `[B,4,T,S]` maps containing log-normalized positive count, log-normalized negative count, mean normalized timestamp, and latest normalized timestamp. Build Gaussian-soft auxiliary targets by separable local smoothing around positive projected cells.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_projection.py -q`

Expected: all projection tests pass.

### Task 3: Dual-view network and event queries

**Files:**
- Create: `src/huav/models/blocks.py`
- Create: `src/huav/models/dual_view.py`
- Modify: `src/huav/models/__init__.py`
- Test: `tests/test_dual_view_model.py`

- [ ] **Step 1: Write failing model tests**

```python
def test_forward_returns_required_event_and_view_outputs():
    output = model(events, batch_indices)
    assert set(output) == {
        "event_logits", "xt_logits", "yt_logits",
        "xt_event_logits", "yt_event_logits", "event_features",
    }
    assert output["event_logits"].shape == (len(events),)

def test_xt_and_yt_encoders_do_not_share_parameters():
    assert next(model.xt_encoder.parameters()) is not next(model.yt_encoder.parameters())
```

- [ ] **Step 2: Run and confirm the model is missing**

Run: `python -m pytest tests/test_dual_view_model.py -q`

Expected: missing `DualViewEventClassifier`.

- [ ] **Step 3: Implement the network**

Use independent shallow multi-scale adapters, residual CNN stages, non-overlapping local-window self-attention, and an FPN top-down path. Exchange time-summary tokens with bidirectional neighbor-masked cross-attention. Upsample features to the configured fine time grid and query `(x,t)` / `(y,t)` using `torch.nn.functional.grid_sample`. Concatenate both features, product, absolute difference, polarity, and normalized timestamp before event classification.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_dual_view_model.py -q`

Expected: required shapes, finite gradients, local attention, and unshared branches pass.

### Task 4: Losses, sampling, and metrics

**Files:**
- Create: `src/huav/training/losses.py`
- Create: `src/huav/training/metrics.py`
- Create: `src/huav/training/sampling.py`
- Modify: `src/huav/training/__init__.py`
- Test: `tests/test_training_components.py`

- [ ] **Step 1: Write failing component tests**

```python
def test_sampler_keeps_all_positives_and_caps_background():
    selected = sample_event_indices(labels, background_ratio=10, ...)
    assert torch.all(torch.isin(torch.where(labels == 1)[0], selected))
    assert (labels[selected] == 0).sum() <= 10 * (labels == 1).sum()

def test_metric_confusion_counts_produce_pd_fa_iou_acc():
    metrics = event_metrics(logits, labels, threshold=0.5)
    assert metrics["iou"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run and confirm modules are missing**

Run: `python -m pytest tests/test_training_components.py -q`

Expected: missing losses, sampler, and metrics.

- [ ] **Step 3: Implement components**

Combine binary focal and soft Dice losses for event/view logits; add L1 agreement between per-view probabilities and weak time-bin smoothness. Use `event + view_weight*view + consistency_weight*consistency + temporal_weight*temporal` (all positive additions). Define event IoU and overall Acc from TP/FP/TN/FN, Pd as positive-event recall, and Fa as `FP/(FP+TN)`. Also compute the EV-UAV-compatible 50-ms object Pd and connected-component spatial Fa as separately named fields.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_training_components.py -q`

Expected: all loss, sampling, and metric tests pass.

### Task 5: CPU trainer, evaluator, and artifact contract

**Files:**
- Create: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Modify: `src/huav/cli/evaluate.py`
- Modify: `src/huav/cli/_command.py`
- Modify: `README.md`
- Test: `tests/test_training_engine.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing engine/CLI tests**

```python
def test_cpu_smoke_run_writes_metrics_checkpoint_and_history(tmp_path):
    summary = run_training(config, output_dir=tmp_path, device="cpu")
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert summary["device"] == "cpu"
```

- [ ] **Step 2: Run and confirm Phase 1 guard causes failure**

Run: `python -m pytest tests/test_training_engine.py tests/test_cli.py -q`

Expected: training engine missing and `huav-train` still exits with Phase 1 error.

- [ ] **Step 3: Implement training and validation**

Train sequence-by-sequence on CPU, sample only event-level loss indices while building view supervision from all positive events, ramp consistency and temporal weights after configured warmup, evaluate every original validation event, choose/checkpoint the best validation IoU, and atomically write `config.yaml`, `history.json`, `metrics.json`, `metrics.md`, and `best.pt` inside a timestamped `outputs` run directory.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_training_engine.py tests/test_cli.py -q`

Expected: CPU toy run and CLI invocation pass.

### Task 6: Full verification and real CPU experiment

**Files:**
- Modify: `README.md`
- Create at runtime: `outputs/cpu_<timestamp>/...`

- [ ] **Step 1: Run static and unit verification**

Run: `python -m pytest -q`

Expected: zero failures.

Run: `python -m ruff check src tests`

Expected: zero errors.

- [ ] **Step 2: Run a bounded real-data CPU smoke experiment**

Run: `python -m huav.cli.train --config configs/huav_cpu_smoke.yaml --device cpu`

Expected: one complete train/validation cycle and all output artifacts.

- [ ] **Step 3: Run the configured CPU experiment**

Run: `python -m huav.cli.train --config configs/huav_base.yaml --device cpu`

Expected: configured epochs complete, validation processes all 24 files without background downsampling, and the output summary contains Pd, Fa, IoU, and Acc.

- [ ] **Step 4: Audit artifacts**

Load `metrics.json`, ensure all values are finite and confusion counts sum to all 1,424,330 validation events, load `best.pt` on CPU, rerun evaluation, and verify the reproduced metrics match the saved best metrics.

