# Per-Time-Slice Eight-Neighbor Cone Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep up to eight nearest cone neighbors from every eligible target time slice and remove the second cross-time truncation to eight neighbors.

**Architecture:** `ConeNeighborBuilder` will return one variable-width index tensor per attention head and direction. Each head tensor has `max_neighbors_per_time_slice * layer_horizon` slots, arranged in increasing time-distance segments, so no head is padded to the widest head. `BidirectionalConeAttention` will process each head's index tensor independently, apply one softmax over that head's complete cross-time candidate set, then stack the four head aggregates back into the unchanged token output contract.

**Tech Stack:** Python 3.12, NumPy, SciPy `cKDTree`, PyTorch, pytest, Ruff, HUAV CLI.

---

### Task 1: Lock the per-time-slice neighbor contract with tests

**Files:**
- Modify: `tests/test_cone_neighbor_builder.py`
- Modify: `tests/test_cone_attention.py`

- [ ] **Step 1: Write a failing builder test**

Add a case with one center token and three past time slices, each containing more than two valid tokens. Configure `max_neighbors_per_cone=2` and assert that the center retains exactly two neighbors from each time slice, for six total rather than a final global two.

```python
def test_neighbor_cap_is_applied_independently_to_each_time_slice() -> None:
    coords = torch.tensor(
        [
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 0, 1], [1, 0, 1], [2, 0, 1],
            [0, 0, 2], [1, 0, 2], [2, 0, 2],
            [0, 0, 3],
        ],
        dtype=torch.long,
    )
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(4.0, 0.0, 3),),
        max_neighbors_per_cone=2,
    )
    neighbors = builder(coords, torch.zeros(len(coords), dtype=torch.long))
    selected = neighbors.past_indices[0][9]
    selected_times = coords[selected[selected >= 0], 2]
    assert torch.bincount(selected_times, minlength=3).tolist() == [2, 2, 2]
```

- [ ] **Step 2: Run the new test and confirm the old global truncation fails**

Run:

```powershell
python -m pytest tests/test_cone_neighbor_builder.py -q
```

Expected: failure because the current builder returns a single `[M,H,K]` tensor and retains only `K` candidates after merging time slices.

- [ ] **Step 3: Update existing builder assertions for per-head tuples**

Access a head with `neighbors.past_indices[head_index]`. Assert empty inputs return one tensor per head with shape `[0, K * horizon]`, causal future tensors contain only `-1`, and different heads retain their own horizons.

- [ ] **Step 4: Add an attention regression test**

Use debug statistics to assert that one head can report more than `max_neighbors_per_cone` total edges for a center when valid neighbors occur in multiple time slices, while output shape, finite values, and backward gradients remain unchanged.

### Task 2: Preserve every time slice's local nearest neighbors

**Files:**
- Modify: `src/huav/models/cone_neighbor_builder.py`

- [ ] **Step 1: Change neighborhood storage to per-head tensors**

```python
@dataclass(frozen=True)
class ConeNeighborhoods:
    past_indices: tuple[torch.Tensor, ...]
    future_indices: tuple[torch.Tensor, ...]
```

Allocate head `h` with width:

```python
horizon = max(1, math.ceil(head.max_time_distance * layer_time_scale))
width = horizon * max_neighbors_per_cone
```

- [ ] **Step 2: Store every target slice in a dedicated segment**

For `time_distance` starting at one, place that slice's local nearest neighbors in:

```python
start = (time_distance - 1) * max_neighbors_per_cone
stop = start + max_neighbors_per_cone
output[source_indices, start:stop] = global_indices
```

Remove the merged `candidate_scores`, cross-time sort, and final `[:max_neighbors_per_cone]` truncation. Retain `cKDTree.query(..., k=1..K, p=inf, distance_upper_bound=radius)` so each target slice still contributes at most its eight spatially nearest valid tokens.

- [ ] **Step 3: Run builder tests**

Run:

```powershell
python -m pytest tests/test_cone_neighbor_builder.py -q
```

Expected: all builder tests pass.

### Task 3: Attend over variable-width head neighborhoods

**Files:**
- Modify: `src/huav/models/cone_attention.py`
- Test: `tests/test_cone_attention.py`

- [ ] **Step 1: Process one head at a time**

Change `_attend_direction` to accept `tuple[torch.Tensor, ...]`. For each head, gather its Key/Value tensor, construct relative features, compute scores, mask, softmax across all of that head's retained time-slice candidates, and aggregate to `[M, head_dim]`.

```python
for head_index, head_neighbors in enumerate(neighbor_indices):
    mask = head_neighbors >= 0
    safe_indices = head_neighbors.clamp_min(0)
    neighbor_keys = key[safe_indices, head_index]
    neighbor_values = value[safe_indices, head_index]
    # QK + relative bias, one softmax across the complete head candidate set.
```

Stack head outputs and counts to recover `[M,H,D_head]` and `[M,H]`. Preserve past/future separation, direction embeddings, learned direction gate, output projection, residual interface, and all event-order/output contracts.

- [ ] **Step 2: Run attention and integration tests**

Run:

```powershell
python -m pytest tests/test_cone_attention.py tests/test_spatiotemporal_cone_model.py tests/test_project_integration.py -q
```

Expected: all tests pass with finite gradients.

### Task 4: Static and real-data resource gates

**Files:**
- Modify only if a test exposes an implementation defect.

- [ ] **Step 1: Run repository checks**

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Expected: all commands pass before training.

- [ ] **Step 2: Run the largest-sequence resource gate**

```powershell
python scripts/check_largest_sequence.py `
  --config configs/huav_score_optimized.yaml `
  --sequence train_094.npz `
  --device cpu
```

Expected: 90,234 tokens; finite forward loss and gradients; no Windows `c10.dll` access violation or system-memory exhaustion.

### Task 5: Smoke closure and asynchronous full CPU launch

**Files:**
- Generated: `outputs/cpu_YYYYMMDD_HHMMSS/`
- Updated by CLI: `outputs/INDEX.md`

- [ ] **Step 1: Run real-data smoke training and independent evaluation**

```powershell
python -m huav.cli.train `
  --config configs/huav_score_optimized_smoke.yaml `
  --device cpu `
  --note "每个目标时间片保留最多8个邻居，取消跨时间片最终裁切 smoke"
```

Resolve the newly created smoke directory and evaluate its own `config.yaml` and `best.pt` with `python -m huav.cli.evaluate --quiet`. Expected: both commands complete and produce finite metrics.

- [ ] **Step 2: Launch the full 99/24 CPU training in a hidden background process**

Use the repository's full `configs/huav_score_optimized.yaml`, a new lifecycle-managed output directory, and note:

```text
每个目标时间片保留最多8个邻居，取消跨时间片最终裁切
```

Redirect stdout and stderr to explicit log files, save the process ID, and do not wait for training completion.

- [ ] **Step 3: Confirm launch without claiming training completion**

Verify the process remains alive after initial startup, identify the new absolute output directory, inspect the first log lines for immediate errors, and report that full training was launched but independent reevaluation/submission remain pending until training finishes.

