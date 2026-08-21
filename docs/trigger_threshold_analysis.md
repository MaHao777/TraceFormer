# HUAV High-Trigger-Threshold Event Analysis

## Method

- Per-pixel signed contrast integration; ON/OFF increments share one residual.
- A crossing retains the exact source event and subtracts the threshold while carrying residual state.
- Eligibility and event generation are annotation-independent.
- Dense gate: at least `300000` events and maximum polarity fraction no greater than `0.65`.

## Aggregate results

Data: `99` files; `11` eligible dense balanced files.

| threshold ratio | event retention | positive retention | object-frame retention |
|---:|---:|---:|---:|
| 1.5× | 0.223487 | 0.358632 | 0.879491 |
| 2× | 0.173808 | 0.271793 | 0.846557 |
| 3× | 0.062147 | 0.126384 | 0.604042 |

## Decision

- Train with `1.5×` and `2.0×` views.
- Keep `3.0×` as a diagnostic stress level because its target-frame coverage is substantially lower.
