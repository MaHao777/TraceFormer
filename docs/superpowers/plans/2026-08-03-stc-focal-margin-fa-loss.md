# STC-Focal and Margin False-Alarm Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development and verification-before-completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the recall-heavy focal/Dice/object-frame objective with a configurable event-level STC-Focal loss plus a full-event margin false-alarm loss, then validate it through a complete CPU train/evaluate/submission cycle.

**Architecture:** Keep `SpatioTemporalConeDetector` and its output contract unchanged. Build capped, batch-isolated event neighborhoods in the loss from raw pixel coordinates and raw millisecond timestamps using sparse voxel hashing; compute detached neighbor support only for sampled classification queries, while preserving the false-alarm surrogate over every original event. The existing token-cone indices are not reused because they are internal token-level edges and do not map one-to-one to original events.

**Tech Stack:** Python 3, PyTorch scatter reductions and stable BCE-with-logits, YAML dataclass configuration, pytest, Ruff.

---

### Task 1: Specify the new public loss contract with failing tests

**Files:**
- Modify: `tests/test_training_components.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_project_integration.py`

- [ ] Add tests importing `EventSpatiotemporalLoss`, `spatiotemporal_focal_loss`, and `margin_false_alarm_loss`, with no imports or result keys for Dice/object-frame losses.
- [ ] Cover empty, all-positive, all-negative, no-neighbor, batch-isolation, continuous-track support, isolated/background clusters, top-k overflow, float32/autocast finiteness, and backward gradients.
- [ ] Assert `total == stc_focal + lambda_fa * false_alarm` and that the full-event logits receive the false-alarm gradient.
- [ ] Assert score/smoke/AutoDL YAML files expose the same STC and margin-Fa defaults.
- [ ] Run `python -m pytest tests/test_training_components.py tests/test_config.py tests/test_training_engine.py tests/test_project_integration.py -q` and confirm it fails because the new API/config does not exist.

### Task 2: Implement sparse STC-Focal and margin false-alarm losses

**Files:**
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`

- [ ] Add strict shape/dtype/device and finite-hyperparameter validation.
- [ ] Implement fixed-offset sparse voxel lookup in `(batch, time_ms, y_px, x_px)` cells, cap each occupied cell deterministically, filter the exact bidirectional cone, and retain at most `max_neighbors` closest candidates without an `N x N` matrix or per-event Python loop.
- [ ] Compute detached Gaussian neighbor support and stable BCE-with-logits STC-Focal weighting.
- [ ] Collapse background events by `(batch, 50 ms frame, x, y)` using `scatter_reduce_(amax)`, take per-frame top-k, and apply `temperature * softplus((logit - negative_margin) / temperature)`.
- [ ] Replace the composite module and public exports; remove Dice and object-frame implementations.
- [ ] Run the focused loss tests until green.

### Task 3: Rewire validated configuration and training/evaluation calls

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `scripts/check_largest_sequence.py`
- Modify: `configs/huav_base.yaml`
- Modify: `configs/huav_cpu_smoke.yaml`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`

- [ ] Replace old loss fields with `alpha`, `gamma_focal`, `gamma_stc`, `eta_pos`, `eta_neg`, `tau_ms`, `r0_pixels`, `vmax_pixels_per_ms`, `sigma_xy_pixels`, `sigma_t_ms`, `max_neighbors`, `lambda_fa`, `fa_top_k`, `fa_negative_margin`, and `fa_temperature`.
- [ ] Validate ranges and finite values, including strictly positive scales/caps and non-negative weights/radii/speeds.
- [ ] Pass sampled raw pixel coordinates and millisecond timestamps to STC-Focal; continue passing all original-event logits/labels/coordinates/timestamps to the Fa term.
- [ ] Restrict loss logs/history to `total`, `stc_focal`, and `false_alarm`, including resumed training.
- [ ] Run focused config/engine/integration tests until green.

### Task 4: Document units, defaults, and design constraints

**Files:**
- Modify: `README.md`
- Create: `docs/stc_focal_loss.md`

- [ ] Record that dataset `events[:,2]` is normalized to `[0,1]`, while the loss uses retained `timestamps_ms`; x/y remain source pixels.
- [ ] Explain why internal token-cone edges cannot be reused for original-event support without changing the model contract.
- [ ] Derive the default `vmax_pixels_per_ms` from the maximum production cone slope and `8 px / 125 ms` block scale, and document every default/formula.
- [ ] Run `rg -n "dice|object_frame|focal_alpha|lambda_false_alarm|false_alarm_logit_margin" src tests configs scripts README.md docs/stc_focal_loss.md` and remove active stale contract references.

### Task 5: Static, resource, smoke, and full CPU validation

**Files:**
- Create: new timestamped directories under `outputs/`

- [ ] Run `python -m pytest -q`, `python -m ruff check src tests`, and `git diff --check`.
- [ ] Refresh `docs/cone_slope_analysis.json` and `.md` from all 99 training files, retaining robust quantiles.
- [ ] Run `python scripts/check_largest_sequence.py --help` followed by its configured CPU forward/loss/backward gate for `train_094.npz`.
- [ ] Train `configs/huav_score_optimized_smoke.yaml` on CPU in a new timestamped directory and independently reproduce its `best.pt` metrics.
- [ ] Train `configs/huav_score_optimized.yaml` on all 99/24 files on CPU in a new timestamped directory and independently reproduce byte-identical metrics.

### Task 6: Submission and final audit

**Files:**
- Create: a new timestamped submission directory under `outputs/`

- [ ] Run `python -m huav.cli.predict` with the full run's saved config and `best.pt`.
- [ ] Audit one TXT per validation NPZ, exact original `x y t p` values/order, five columns, binary labels, total rows/positives, and flat TXT-only ZIP root.
- [ ] Re-run pytest, Ruff, `git diff --check`, and `git status --short`.
- [ ] Report full metrics, independent reproduction identity, artifacts, submission counts, and any failed or unresolved requirement.
