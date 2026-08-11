# HUAV Training-Set Spatiotemporal Cone Slope Analysis

## Method

- Data: `99` training sequences, `274122` positive events.
- Grid: `8×8` pixels and `64` time blocks over `8000` ms (`125.0` ms/block).
- Track centers: median positive-event x/y for each `(file, name, time block)` group with at least three events.
- Slope: Chebyshev center displacement divided by time-block distance.
- All measured pairs remain in the JSON. Fixed cone heads are selected from robust quantiles so a single sparse-bin jump does not inflate every cone.

## Results

| Quantity | q50 | q90 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|
| Raw slope | 0.612500 | 1.187500 | 1.375000 | 1.796875 | 11.750000 |
| Required slope, base radius 1 | 0.296875 | 0.900000 | 1.083333 | 1.468750 | 10.750000 |
| Required slope, base radius 2 | 0.062500 | 0.687500 | 0.875000 | 1.270833 | 9.750000 |
| Target 90% half-extent | 0.375000 | 0.750000 | 0.850000 | 1.179750 | 11.650000 |

Unit: spatial blocks per time block for slopes; spatial blocks for extent.

## Configuration decision

- Cone slopes: `0.5, 1.0, 1.5, 2.0`.
- Base radii: `1.0, 1.0, 2.0, 2.0` blocks.
- Maximum time distances: `4, 4, 6, 8` blocks.
- The highest-slope head covers the robust q99 motion estimate.
- The observed maximum is retained in the report but treated as a sparse-group outlier; neighbor caps and attention scores handle such cases without a globally oversized hard cone.

The full per-lag and per-sequence values are in `docs/cone_slope_analysis_125ms_stc_control.json`.
