# HUAV Training-Set Spatiotemporal Cone Slope Analysis

## Method

- Data: `99` training sequences, `274122` positive events.
- Grid: `4×4` pixels and `128` time blocks over `8000` ms (`62.5` ms/block).
- Track centers: median positive-event x/y for each `(file, name, time block)` group with at least three events.
- Slope: Chebyshev center displacement divided by time-block distance.
- All measured pairs remain in the JSON. Fixed cone heads are selected from robust quantiles so a single sparse-bin jump does not inflate every cone.

## Results

| Quantity | q50 | q90 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|
| Raw slope | 0.640625 | 1.250000 | 1.446429 | 1.875000 | 23.000000 |
| Required slope, base radius 1 | 0.350000 | 0.968750 | 1.160714 | 1.571429 | 22.000000 |
| Required slope, base radius 2 | 0.125000 | 0.750000 | 0.958333 | 1.375000 | 21.000000 |
| Target 90% half-extent | 0.625000 | 1.050000 | 1.250000 | 1.750000 | 23.450000 |

Unit: spatial blocks per time block for slopes; spatial blocks for extent.

## Configuration decision

- Cone slopes: `0.5, 1.0, 1.5, 2.0`.
- Base radii: `1, 1, 2, 2` blocks.
- The first two heads model common motion; the 1.5 and 2.0 heads cover approximately the 95th and 99th percentile tails.
- The observed maximum is retained in the report but treated as a sparse-group outlier; neighbor caps and attention scores handle such cases without a globally oversized hard cone.

The full per-lag and per-sequence values are in `docs/cone_slope_analysis.json`.
