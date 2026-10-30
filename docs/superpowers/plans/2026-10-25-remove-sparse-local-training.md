# Remove Sparse Branch and Retrain Locally

> Execute in the current local checkout, as explicitly requested. Do not use a worktree or modify `D:/Paper/TraceFormer`.

**Goal:** Remove the conditional Full/Sparse fusion branch, train the remaining network from scratch locally, and evaluate the saved best model on the published test split.

**Architecture:** Keep the tokenizer, three-cone encoder, event decoder, neural refinement head and external postprocessing. Remove Sparse parameters, paired forwarding, auxiliary losses, configuration and training stage. Use checkpoint schema 8 to prevent loading old Sparse weights as the new architecture.

**Tech Stack:** Python 3.12, PyTorch CPU FP32, pytest, Ruff, YAML.

- [x] Preserve the current source/config/test files and original main-model identity under `outputs/no_sparse_local_20261025/before`; record paper file hashes without modifying that directory.
- [x] Remove the branch from `models/spatiotemporal_cone.py`, `config.py`, `training/engine.py`, inference/cache forwarding and current YAML configurations. Delete the retired Sparse module and retire its dedicated tests.
- [x] Update tests for the new parameter/output/checkpoint contracts, training/resume and inference; retain real tests for the tokenizer, attention, refinement, metrics and postprocessing.
- [x] Run `python -m pytest -q`, `python -m ruff check src tests scripts`, and `git diff --check`; fix failures caused by this change.
- [x] Run complete-sequence forward/loss/backward gates for `train_096.npz` and `train_094.npz`, then a real-data smoke run and independent evaluation.
- [x] Freeze the changed source and configuration for the local run. Reuse the original main-model hyperparameters: 99 train/24 validation, seed 37, 24 epochs, AdamW 0.0007/0.0002, CPU FP32 and 8 threads. Remove only Sparse-specific objectives; retain dense-sequence weighting and neural refinement supervision.
- [ ] Train from random initialization with per-epoch recovery states, independently reproduce the saved best validation result, then evaluate all 24 test files at the frozen validation-selected threshold. Save predictions, global/per-file counts, AP and comparison with the original neural model.
- [ ] Verify paper file hashes again. Do not promote the new model or refit external trees as part of this neural-only training comparison.
