# Official Competition Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every HUAV validation output report and prioritize the score used by the JinSight competition website, while retaining event-level diagnostic metrics.

**Architecture:** Quantize raw event timestamps with `floor(t_ms)` at the EV-UAV frame-metric boundary because the official evaluator frames `ev_loc[:, 2]`, not the floating `ev["t"]`. Build a separate `competition` result from event counts and object/frame counts, including the official false-alarm transform and weighted total score. Training will select the best epoch by this total competition score; independent checkpoint evaluation will use the saved threshold and reproduce the same result.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff

---

### Task 1: Lock the official formulas and timestamp semantics with failing tests

**Files:**
- Modify: `tests/test_training_components.py`
- Modify: `tests/test_training_engine.py`

- [x] **Step 1: Add a fractional-timestamp regression**

Add a metric test with timestamps `49.9`, `50.1`, and `99.9`. The official evaluator floors them to `49`, `50`, and `99`, then applies strict open intervals, so only the first and third events contribute false components.

```python
def test_ev_uav_frames_use_official_integer_millisecond_coordinates() -> None:
    accumulator = EVUAVMetricAccumulator(
        width=4,
        height=3,
        frame_ms=50,
        object_correct_fraction=0.0001,
        threshold=0.5,
    )
    accumulator.update(
        torch.full((3,), 5.0),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([49.9, 50.1, 99.9], dtype=torch.float64),
        torch.tensor([[0, 0], [1, 0], [2, 0]]),
    )
    assert accumulator.compute()["false_components"] == 2
```

- [x] **Step 2: Add an official score regression**

Use the exact website values and require the official score calculator to reproduce `score_fa=0.931529848925` and `score=0.87889372208`.

```python
competition = competition_metrics(
    event={"tp": 57205, "fp": 3332, "fn": 8301},
    object_metrics={
        "detected_objects": 4118,
        "object_instances": 4762,
        "false_components": 2394,
        "frame_count": 3752,
    },
    width=346,
    height=260,
)
assert competition["score_fa"] == pytest.approx(0.931529848925)
assert competition["score"] == pytest.approx(0.87889372208)
```

- [x] **Step 3: Require training artifacts to contain the competition group**

Extend the CPU smoke integration assertion:

```python
assert {
    "iou",
    "seg_acc",
    "pd",
    "fa",
    "score_fa",
    "score",
    "counts",
}.issubset(metrics["competition"])
assert summary["best_score"] == metrics["competition"]["score"]
```

- [x] **Step 4: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_training_components.py tests/test_training_engine.py -q
```

Expected: failures because `competition_metrics` and the `competition` artifact do not exist, and fractional timestamps are not yet quantized.

### Task 2: Implement the website-compatible scorer

**Files:**
- Modify: `src/huav/training/metrics.py`
- Modify: `src/huav/training/__init__.py`

- [x] **Step 1: Quantize timestamps at the frame-metric boundary**

Immediately after moving timestamps to CPU double precision, apply:

```python
timestamps_ms = torch.floor(timestamps_ms.detach().cpu().double())
```

This matches `ev_loc[:, 2]` for all 1,424,330 validation events.

- [x] **Step 2: Add the competition aggregate**

Implement:

```python
def competition_metrics(
    event: Mapping[str, float | int],
    object_metrics: Mapping[str, float | int],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    intersection = int(event["tp"])
    union = int(event["tp"]) + int(event["fp"]) + int(event["fn"])
    gt_positive = int(event["tp"]) + int(event["fn"])
    frames = int(object_metrics["frame_count"])
    false_alarms = int(object_metrics["false_components"])
    objects = int(object_metrics["object_instances"])
    detected_objects = int(object_metrics["detected_objects"])
    iou = intersection / union if union else 0.0
    seg_acc = intersection / gt_positive if gt_positive else 0.0
    pd = detected_objects / objects if objects else 0.0
    fa = false_alarms / (frames * width * height) if frames else 0.0
    score_fa = math.exp(-10000.0 * fa)
    score = 0.2 * iou + 0.1 * seg_acc + 0.4 * pd + 0.3 * score_fa
    return {
        "iou": iou,
        "seg_acc": seg_acc,
        "pd": pd,
        "fa": fa,
        "score_fa": score_fa,
        "score": score,
        "counts": {
            "intersection": intersection,
            "union": union,
            "true_positive": intersection,
            "gt_positive": gt_positive,
            "detected_objects": detected_objects,
            "objects": objects,
            "false_alarms": false_alarms,
            "frames": frames,
        },
    }
```

- [x] **Step 3: Export the aggregate and run focused tests**

Run:

```powershell
python -m pytest tests/test_training_components.py -q
```

Expected: all component metric tests pass.

### Task 3: Make competition scoring primary in validation artifacts

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `tests/test_training_engine.py`

- [x] **Step 1: Attach `competition` to every evaluation**

Compute event and object metric dictionaries once, call `competition_metrics`, and return all three groups:

```python
return {
    "threshold": float(threshold),
    "competition": competition_metrics(
        event,
        object_metrics,
        width=config.resolution.width,
        height=config.resolution.height,
    ),
    "event": event,
    "ev_uav_compatible": object_metrics,
    "validation_files": len(dataset),
}
```

- [x] **Step 2: Put the official score first in `metrics.md`**

Render `score`, `iou`, `seg_acc`, `pd`, `fa`, `score_fa`, and the exact count tuple before the diagnostic event metrics. State the formulas:

```text
score_fa = exp(-10000 * Fa)
score = 0.2 * IoU + 0.1 * SegAcc + 0.4 * Pd + 0.3 * ScoreFa
```

- [x] **Step 3: Select checkpoints by competition score**

Replace `best_iou` comparison with `best_score = metrics["competition"]["score"]`, print the official score in the epoch line, and retain `best_iou` in `summary.json` as a diagnostic alongside:

```python
"best_score": best_score,
```

- [x] **Step 4: Run the focused integration tests**

Run:

```powershell
python -m pytest tests/test_training_engine.py -q
```

Expected: training, checkpoint reload, artifact generation, and competition-score assertions pass.

### Task 4: Correct documentation and existing outputs

**Files:**
- Modify: `README.md`
- Modify: `docs/cone_training_results_20260724.md`
- Generate in each checkpoint run: `official_metrics.json`
- Generate in each checkpoint run: `official_metrics.md`
- Generate in each submission directory: `official_metrics.json`

- [x] **Step 1: Document the official metric**

State that frame metrics use `floor(t_ms)`, `seg_acc` is positive-event recall rather than overall binary accuracy, and the competition score is the primary model-selection value.

- [x] **Step 2: Independently evaluate every saved `best.pt`**

For each output directory containing `best.pt`, load its own `config.yaml` and checkpoint, evaluate with the saved decision threshold, and write `official_metrics.json` plus `official_metrics.md`. Do not overwrite historical `history.json`, because non-best epoch predictions are not retained.

- [x] **Step 3: Score both saved submission directories from their TXT labels**

Join each `val_XXX.txt` to `data/ev-uav/val/val_XXX.npz`, apply the same scorer, and write `official_metrics.json`. Require the latest cone submission to reproduce:

```text
intersection=57205, union=68838
detected_objects=4118, objects=4762
false_alarms=2394, frames=3752
score=0.87889372208
```

- [x] **Step 4: Run full verification**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
```

Expected: the full test suite and Ruff complete with zero failures.

Actual: `pytest` completed with 74 passed and 1 skipped. Full-repository Ruff
reported two pre-existing import-format findings in the user-modified
`src/huav/models/cone_neighbor_builder.py` and untouched
`tests/test_public_api.py`; Ruff passed for every file changed by this plan.
