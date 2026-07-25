# Spatiotemporal Cone Refactor: Training and Validation Results

## Run scope

- Date: 2026-07-24
- Device: CPU (`torch.cuda.is_available() == False`)
- Python: 3.12.7
- PyTorch: 2.11.0+cpu
- PyTorch CPU threads: 14
- Model parameters: 30,570
- Mixed precision: disabled on CPU; the training path enables it only for CUDA
- Full training duration: 33 minutes 17.166 seconds
- Training data: 99 files, 9,324,544 events, 274,122 positive events
- Validation data: 24 files, 1,424,330 events
- Epochs: 3
- Best checkpoint: epoch 2

The dataset manifest in the run folder records every NPZ path, byte size, modification time, and SHA-256 digest.

## Data-derived cone configuration

Target centers were measured from positive events grouped by `(source file, name, time block)`. With 4×4-pixel spatial blocks and 128 time blocks, the Chebyshev motion-slope distribution was:

| Percentile | Spatial blocks / time block |
|---:|---:|
| q50 | 0.640625 |
| q90 | 1.250000 |
| q95 | 1.446429 |
| q99 | 1.875000 |
| max | 23.000000 |

The maximum came from one sparse target-time group jump and was retained in the report but not used to enlarge every hard cone. The implemented heads use slopes `0.5 / 1.0 / 1.5 / 2.0` and base radii `1 / 1 / 2 / 2`.

The CPU training grid uses 8×8-pixel spatial blocks and 64 time blocks. Because spatial and temporal block sizes both doubled, one slope unit still represents approximately 64 pixels/second. Full analysis is in [cone_slope_analysis.md](cone_slope_analysis.md) and `cone_slope_analysis.json`.

## Training configuration

| Setting | Value |
|---|---:|
| Spatial block | 8×8 pixels |
| Time blocks | 64 |
| Token dimension | 32 |
| Event embedding | 16 |
| Cone transformer layers | 3 |
| Attention heads | 4 |
| Max neighbors / head / direction | 8 |
| Layer time scales | 0.5 / 0.75 / 1.0 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clip norm | 1.0 |
| Focal alpha / gamma | 0.75 / 2.0 |
| Dice weight | 1.0 |
| Background / positive loss ratio | 10 |

All events participate in tokenization, sparse cone context, and decoding. The background ratio only selects logits for loss calculation.

## Per-epoch results

| Epoch | Total loss | Focal | Dice | Threshold | Pd | Fa | IoU | Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.558274 | 0.039545 | 0.518729 | 0.80 | 0.779501 | 0.001484 | 0.756216 | 0.988443 |
| 2 | 0.284028 | 0.036819 | 0.247209 | 0.85 | 0.873279 | 0.002452 | **0.831009** | **0.991833** |
| 3 | 0.217790 | 0.035128 | 0.182662 | 0.90 | 0.875416 | 0.002874 | 0.826166 | 0.991529 |

Epoch 3 reduced mean training loss and slightly increased Pd, but its extra false positives reduced validation IoU. The epoch-2 checkpoint was therefore retained.

## Best event-level validation result

- Threshold: 0.8500000238
- Pd: 0.8732787836
- Fa: 0.0024521204
- IoU: 0.8310090357
- Accuracy: 0.9918326511
- Precision: 0.9449592811
- TP / FP / FN / TN: 57,205 / 3,332 / 8,301 / 1,355,492
- Validation events: 1,424,330

## Official JinSight competition result

- Score: 0.878893722080
- IoU: 0.831009035707
- SegAcc: 0.873278783623
- Object Pd: 0.864762704746
- Spatial-component Fa: 0.00000709270455194
- ScoreFa: 0.931529848925
- Detected / object instances: 4,118 / 4,762
- False components / equivalent frames: 2,394 / 3,752

The website score is exactly reproduced by
`ScoreFa = exp(-10000 * Fa)` and
`Score = 0.2 * IoU + 0.1 * SegAcc + 0.4 * Pd + 0.3 * ScoreFa`.
The official evaluator uses 50 ms equivalent frames over the integer timestamp
coordinate `floor(t_ms)` (`ev_loc[:, 2]`) with strict open frame intervals.
The earlier local report used floating raw timestamps and therefore over-counted
3 object instances, 4 detections, and 38 false components.

## Comparison with the previous dual-view run

The previous recorded result is `outputs/cpu_full_final_20260723/metrics.json` and uses the same 24 validation files and 1,424,330 events.

| Event metric | Previous dual-view | New cone model | Absolute change |
|---|---:|---:|---:|
| Pd | 0.845495 | 0.873279 | +0.027784 |
| Fa | 0.004137 | 0.002452 | -0.001685 |
| IoU | 0.778678 | 0.831009 | +0.052331 |
| Acc | 0.988948 | 0.991833 | +0.002885 |
| Precision | 0.907862 | 0.944959 | +0.037098 |

IoU improved by 6.72% relative, while event-level Fa fell by 40.72%. Historical
object-level comparisons in the original run artifacts used floating timestamps;
the corrected official-score artifacts use the integer-millisecond website protocol.

## Independent checkpoint reproduction

The best checkpoint was loaded in a fresh evaluator using the saved model configuration, decision threshold, and validation manifest digest. `reproduced_metrics.json` is exactly equal to `metrics.json` after JSON parsing.

Checkpoint metadata:

```text
schema_version: 2
epoch: 2
decision_threshold: 0.8500000238418579
validation_manifest_sha256:
8011b04cb00eaa8e709130c3eaf19e2766ad35276cf6483dc114d41260cba0d0
```

## Artifacts

- Full run: `outputs/cone_cpu_full_20260724/`
- Best weights: `outputs/cone_cpu_full_20260724/best.pt`
- Exact configuration: `outputs/cone_cpu_full_20260724/config.yaml`
- Dataset manifest: `outputs/cone_cpu_full_20260724/dataset_manifest.json`
- Epoch history: `outputs/cone_cpu_full_20260724/history.json`
- Best metrics: `outputs/cone_cpu_full_20260724/metrics.json`
- Independent metrics: `outputs/cone_cpu_full_20260724/reproduced_metrics.json`
- Summary: `outputs/cone_cpu_full_20260724/summary.json`
- Smoke run: `outputs/cone_cpu_smoke_20260724/`

The smoke run intentionally used only four training files, two validation files, and one epoch. Its IoU of 0.040126 only verifies the real-data execution path and is not a performance result.
