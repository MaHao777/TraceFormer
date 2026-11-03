# TraceFormer

TraceFormer segments tiny UAV targets directly in event streams. The paper model (S1) combines spatial tokenization, multiscale past-cone summaries, event decoding, and a multiscale neural refinement head. It predicts one probability and one binary label for every original event, without external postprocessing.

[中文说明](README.zh-CN.md)

## Model

For a 346 × 260 sensor and an 8 s sequence, occupied 8 × 8 pixel × 125 ms blocks form 64-dimensional tokens. The tokenizer encodes six event features, averages them in a 4 × 4 spatial subcell grid, applies one 3 × 3 depthwise convolution and GELU, and uses event-count-weighted readout. Seven block statistics and positional encoding complete each token.

Three encoder layers use source-gated, square-section past-cone summaries with `(radius, time horizon)` of `(8,2)`, `(4,4)`, and `(2,8)` in token coordinates. A center query fuses the three summaries with four-head attention. The event decoder gathers token context back to the original events. Neural refinement adds context at 2 × 2 × 10 ms, 4 × 4 × 25 ms, and 8 × 8 × 50 ms.

S1 has **143,350 trainable parameters**: tokenizer 14,400; encoder 113,987; decoder 6,785; refinement 8,178. The encoder uses past-only support; sequence statistics and the refinement head use additional temporal information, so the full network is an offline sequence model.

## Installation

Use Python 3.10–3.12. Install a CUDA-enabled PyTorch appropriate for your server first, then run from this repository root:

```bash
python -m pip install -e .
traceformer-train --help
traceformer-evaluate --help
traceformer-predict --help
```

The paper experiments used PyTorch 2.10.0+cu128 on an RTX 4090, with CUDA 12.8, FP32, AMP/TF32 disabled, and eight CPU threads. A compatible installed PyTorch is reused. CPU execution is available with `--device cpu`; it is slower and its timing is not comparable to the paper's GPU timing.

## Data and preprocessing

Provide the EV-UAV NPZ files yourself; data are not included:

```text
/path/to/ev-uav/
  train/*.npz    # 99 labeled sequences in the paper
  val/*.npz      # 24 labeled sequences
  test/*.npz     # 24 sequences; labels needed only for evaluation
```

Each NPZ must contain a one-dimensional structured array named `ev` with integer `x`, `y`, and `p` fields, and numeric `t` in **milliseconds**. Coordinates satisfy `0 <= x < 346`, `0 <= y < 260`; polarity is 0/1. Timestamps are finite, nondecreasing, and within [0,8000]. Training/evaluation additionally require binary integer `label` and integer target identity `name` (0 for background). Empty sequences are rejected.

Preprocessing preserves source order and uses `t / 8000` as the network time coordinate. Spatial blocks, subcells, local event descriptors, and token statistics are computed online. No intermediate voxel dataset or raw-format conversion is required.

```bash
traceformer-preprocess --data-root /path/to/ev-uav --split train val test --output outputs/data_manifest.json
# Check an unlabeled test split:
traceformer-preprocess --data-root /path/to/ev-uav --split test --unlabeled --output outputs/test_manifest.json
```

These commands validate the same preprocessing used during training and write file identities and event counts; they do not rewrite the dataset.

## Pretrained S1

`weights/s1_best.pt` is the original schema-8 checkpoint, selected by validation Score at **epoch 22**. Its contract is `{"variant":"S1","version":4}`. The stored threshold is `0.887499988079071` (0.8875 when rounded).

SHA-256:

```text
6931ab7529907d50c191a560dcd0e65fbba011689354566056ce4383f7e54e10
```

The loader checks the structure and normalization contract and loads every tensor strictly. `num_pool_queries` in the saved model metadata preserves the original initialization contract; the actual S1 network has no query-pooling parameters. Temporary initialization draws retain the paper's shared weights and RNG state.

## Training and resume

```bash
traceformer-train --config configs/s1.yaml --variant S1 --data-root /path/to/ev-uav --device cuda --output-dir outputs/s1
```

The default configuration uses seed37, 24 epochs, AdamW (learning rate 0.0007, weight decay 0.0002), the original cosine schedule, gradient clipping at 1.0, and one complete sequence per optimization step. Focal/Dice supervision samples at most 12 background events per target, equally split between hard and random negatives. The complete event stream is still passed through the network. The joint loss is focal + 0.75 Dice + 0.25 object-frame + 0.05 false-alarm + 0.2 auxiliary cell loss. The original dense-sequence weighting is retained.

Validation selects the best weights and threshold after each epoch. Runtime outputs include `best.pt`, `training_state.pt`, per-epoch states, the effective configuration, dataset manifest, history, and metrics. Output directories must be new; resume into a new output directory:

```bash
traceformer-train --config outputs/s1/config.yaml --variant S1 --data-root /path/to/ev-uav --device cuda --resume-state outputs/s1/training_state.pt --output-dir outputs/s1_resumed
```

Resume restores the model, optimizer, schedule, Python/NumPy/PyTorch RNGs, and sampling/order generators. Keep the same variant, data identity, total epoch budget and training settings. Paths may move; the restored data must have the same content. The training state is a trusted local artifact.

For a pipeline check, use `--device cpu --epochs 1 --max-train-files 2 --max-val-files 2 --num-threads 1` with a fresh output directory. This does not reproduce paper accuracy.

## Evaluation and inference

```bash
traceformer-evaluate --data-root /path/to/ev-uav --split val --device cuda --output-dir outputs/s1_val
traceformer-evaluate --data-root /path/to/ev-uav --split test --device cuda --output-dir outputs/s1_test
traceformer-predict --input-dir /path/to/unlabeled_npz --device cuda --output-dir outputs/prediction
```

Defaults are `--config configs/s1.yaml` and `--checkpoint weights/s1_best.pt`. To evaluate trained/ablation weights, supply the saved `--config` and `--checkpoint`. The variant and threshold come from that checkpoint. Validation evaluation verifies the original validation-data identity; test evaluation needs only the test split. Prediction needs neither annotations nor a validation directory. Labels and object identities are never passed into the model.

Each `predictions/<source-name>.npz` contains aligned `event_probabilities` (float32), `prediction` (boolean), `source_sha256`, and `threshold`. `metrics.json` records identities, settings and, during evaluation, per-file and global metrics. Output directories must not exist.

Evaluation uses the checkpoint threshold without reselection. It aggregates integer counts globally before computing:

```text
Score = 0.2 IoU + 0.1 SegAcc + 0.4 Pd + 0.3 exp(-10000 Fa)
```

SegAcc is positive-event recall in this benchmark. Pd counts detected annotated object-frame instances. Fa is the number of false connected components per pixel-frame, not the event-level false-positive rate. The original strict frame boundaries and reference pixel-count semantics are retained. An independent NumPy/SciPy counter cross-checks the official counts. AP uses all event probabilities.

Rescore saved predictions without another forward pass:

```bash
traceformer-evaluate --data-root /path/to/ev-uav --split test --device cpu --predictions-dir outputs/s1_test/predictions --output-dir outputs/s1_test_rescore
```

## Paper ablations

All nine variants share `configs/s1.yaml`; `configs/ablations.yaml` supplies the selected checkpoint contract. Train each variant from scratch by changing `--variant`; only S1 pretrained weights are included. Do not use S1 weights as a replacement for retraining ablations.

| ID | Change relative to S1 | Parameters | Val Score | Test Score |
|---|---|---:|---:|---:|
| S1 | Full model | 143350 | 0.913654915 | 0.921787678 |
| S0 | Event-embedding mean pooling | 142774 | 0.903369256 | 0.915294960 |
| A1 | Remove encoder cross-token interaction | 79603 | 0.900421183 | 0.889125897 |
| A2 | Remove refinement and its auxiliary loss | 135172 | 0.901181351 | 0.902915097 |
| P1 | 1 × 1 depthwise kernel, retaining GELU/readout | 142838 | 0.906050406 | 0.915515149 |
| G1 | Approximately volume-matched fixed windows | 143350 | 0.911572960 | 0.918347501 |
| G2 | Middle cone only | 142966 | 0.907179309 | 0.913712190 |
| F1 | Summary mean + 64–128–64 MLP | 143158 | 0.908325546 | 0.916658736 |
| T1 | Gated direct attention over the deduplicated cone union | 130102 | 0.910520349 | 0.913340651 |

```bash
for variant in S1 S0 A1 A2 P1 G1 G2 F1 T1; do
  traceformer-train --variant "$variant" --data-root /path/to/ev-uav --device cuda --output-dir "outputs/$variant"
done
```

Structure comparisons use S1/S0/A1/A2; mechanism comparisons use S1/S0/P1/G1/G2/F1/T1. G1 is not exactly volume/occupancy matched. G2 changes both scales and summary slots. T1 has no neighbor cap and does not isolate coordinate anchoring. These are single-seed results, not evidence of statistical significance or improvements on every sequence/metric.

Execution chunking preserves complete supports. Spatial pooling uses `TRACEFORMER_SPATIAL_TOKEN_CHUNK=4096`; T1 uses `TRACEFORMER_S1_DIRECT_QUERY_CHUNK=256`. The original prefix-sum implementation uses 1024-query chunks during training and CPU execution, and 16384 in CUDA inference. Keep settings unchanged for exact resume. S1's published 41.499 ms mean is the complete neural forward on GPU-resident input on the paper RTX 4090; it excludes data loading, transfer and scoring. It is not a newly measured release benchmark.

## Release verification

The release was checked on CPU with PyTorch 2.11.0+cpu. All nine variants matched the original implementation bit for bit in initialized weights, RNG state, outputs and gradients. All nine completed miniature training, interrupted-state recovery and independent evaluation; resumed model/optimizer/RNG states matched uninterrupted training. Isolated installation and unlabeled inference also passed.

Full CPU reevaluation covered all 24 validation and 24 test sequences: event probabilities matched the original CPU implementation bit for bit; every binary mask and official integer count matched the paper artifacts, reproducing the Scores above. CPU AP was 0.9419454783655674 (val) and 0.9602258519858334 (test), identical to the original CPU code. The historical GPU AP was 0.9419454870573791 and 0.9602258511768741; CPU/CUDA floating-point ranking differences leave the masks and official Score unchanged.

CUDA execution of this reorganized release has not been rerun on the CPU-only release host; the historical paper GPU measurements retain their original scope. Test tools, evidence and generated predictions are kept outside this publication repository.
