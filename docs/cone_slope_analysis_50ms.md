# HUAV Training-Set Spatiotemporal Cone Slope Analysis

## Method

- Data: `99` training sequences, `274122` positive events.
- Grid: `8×8` pixels and `160` time blocks over `8000` ms (`50.0` ms/block).
- Track centers: median positive-event x/y for each `(file, name, time block)` group with at least three events.
- Slope: Chebyshev center displacement divided by time-block distance.
- All measured pairs remain in the JSON. Fixed cone heads are selected from robust quantiles so a single sparse-bin jump does not inflate every cone.

## Results

| Quantity | q50 | q90 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|
| Raw slope | 0.250000 | 0.500000 | 0.562500 | 0.750000 | 11.687500 |
| Required slope, base radius 1 | 0.112500 | 0.363281 | 0.437500 | 0.600000 | 10.687500 |
| Required slope, base radius 2 | 0.013158 | 0.275000 | 0.352273 | 0.513889 | 9.687500 |
| Target 90% half-extent | 0.300000 | 0.500000 | 0.612500 | 0.800000 | 11.525000 |

Unit: spatial blocks per time block for slopes; spatial blocks for extent.

## Configuration decision

- Cone slopes: `0.2, 0.4, 0.6, 0.8`.
- Base radii: `1.0, 1.0, 2.0, 2.0` blocks.
- Maximum time distances: `10, 10, 15, 20` blocks.
- The highest-slope head covers the robust q99 motion estimate.
- The observed maximum is retained in the report but treated as a sparse-group outlier; neighbor caps and attention scores handle such cases without a globally oversized hard cone.

The full per-lag and per-sequence values are in `docs/cone_slope_analysis_50ms.json`.
