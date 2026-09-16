# From-Scratch Competition Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public one-click workflow train every learned neural and LightGBM parameter from the 99 training files, starting from seeded random initialization, before predicting with the newly produced artifacts.

**Architecture:** Add an explicit 18-epoch Full-path base configuration, then run the existing 24-epoch frozen-Full Sparse Mixture stage using only the base checkpoint created earlier in the same output directory. Preserve the existing postprocessing development/release and `TrainingRunDir` inference flow, while adding provenance assertions that distinguish gradient training, validation-only selection/evaluation, and test inference.

**Tech Stack:** Windows PowerShell 5.1+, Python 3.12, PyTorch, PyYAML, LightGBM, pytest, Ruff.

---

### Task 1: Freeze the two-stage configuration contract

**Files:**
- Create: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\configs\train_base.yaml`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\tests\test_project_integration.py`

- [ ] **Step 1: Write the failing configuration test**

```python
def test_from_scratch_base_and_final_configs_form_two_stage_contract() -> None:
    base = load_config(ROOT / "configs" / "train_base.yaml")
    final = load_config(ROOT / "configs" / "train.yaml")
    assert base.model == final.model
    assert base.resolution == final.resolution
    assert base.training.seed == final.training.seed == 37
    assert base.training.epochs == 18
    assert base.sparse_mixture.enabled is False
    assert base.sparse_mixture.freeze_full_model is False
    assert final.training.epochs == 24
    assert final.sparse_mixture.enabled is True
    assert final.sparse_mixture.freeze_full_model is True
```

- [ ] **Step 2: Run the test and confirm the missing base config fails**

Run: `python -m pytest tests/test_project_integration.py::test_from_scratch_base_and_final_configs_form_two_stage_contract -q`

Expected: failure because `configs/train_base.yaml` does not exist.

- [ ] **Step 3: Add the exact historical Full-path recipe as a portable config**

Create `configs/train_base.yaml` with the same 8x8/64 model, seed 37, AdamW hyperparameters, four-term loss and 18 epochs as the recorded Full-path run, but set `dataset.root: ../data/ev-uav`, `sparse_mixture.enabled: false`, and `sparse_mixture.freeze_full_model: false`.

- [ ] **Step 4: Run the focused contract test**

Run: `python -m pytest tests/test_project_integration.py::test_from_scratch_base_and_final_configs_form_two_stage_contract -q`

Expected: pass.

### Task 2: Change one-click training to generate its own base checkpoint

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\run_train.ps1`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\tests\test_project_integration.py`

- [ ] **Step 1: Add failing script assertions**

```python
def test_one_click_training_starts_from_random_initialization() -> None:
    script = (ROOT / "run_train.ps1").read_text(encoding="utf-8-sig")
    assert "configs\\train_base.yaml" in script
    assert "stage1_base" in script
    assert "stage2_final" in script
    assert "weights\\pretrained" not in script
    assert "full_q4_token64_best.pth" not in script
```

- [ ] **Step 2: Run the focused test and confirm the old dependency fails**

Run: `python -m pytest tests/test_project_integration.py::test_one_click_training_starts_from_random_initialization -q`

Expected: failure because the script still loads `weights/pretrained/full_q4_token64_best.pth`.

- [ ] **Step 3: Implement the two-stage PowerShell workflow**

Generate portable runtime copies of `train_base.yaml` and `train.yaml`. Run stage 1 without `--initialize-full` into `<OutputDir>/stage1_base`, independently evaluate its `best.pt`, then run stage 2 into `<OutputDir>/stage2_final` with `--initialize-full <OutputDir>/stage1_base/best.pt`. Point all cache, LightGBM, validation, and bundle paths at `stage2_final`.

- [ ] **Step 4: Extend `reproduction_bundle.json`**

Record `initialization_mode: seeded_random_two_stage`, stage-1 config/checkpoint/metrics, stage-2 config/checkpoint/metrics, the stage-1 checkpoint SHA-256 bound into stage 2, and the explicit declaration that validation is excluded from optimizer batches and backward propagation.

- [ ] **Step 5: Preserve Windows PowerShell 5.1 encoding**

Rewrite `run_train.ps1` as UTF-8 with BOM and verify its first bytes are `EF BB BF`.

### Task 3: Add machine-checkable training provenance

**Files:**
- Create: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\scripts\audit_training_provenance.py`
- Create: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\tests\test_training_provenance.py`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\run_train.ps1`

- [ ] **Step 1: Write failing provenance tests**

Test that the auditor rejects a base checkpoint with `full_initialization != None`, rejects a stage-2 checkpoint whose `full_initialization.sha256` differs from the stage-1 checkpoint, and accepts matching model/resolution/config/manifest contracts.

- [ ] **Step 2: Implement the auditor**

The auditor must verify seeded-random stage 1, disabled Sparse Mixture in stage 1, frozen enabled Sparse Mixture in stage 2, identical model/resolution definitions, identical 99/24 dataset manifest identities, exact stage-1 SHA binding in stage 2, and write `training_provenance.json` with `validation_used_for_gradient_training: false`.

- [ ] **Step 3: Call the auditor before postprocessing**

Make `run_train.ps1` fail closed if provenance auditing fails; add the report path to `reproduction_bundle.json`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_training_provenance.py tests/test_project_integration.py -q`

Expected: all pass.

### Task 4: Rewrite reviewer-facing documentation

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\README.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\提交材料检查表.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\docs\运行说明.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\docs\算法报告.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\other_files\package_audit.json`

- [ ] **Step 1: State the non-cheating boundary precisely**

Document that every learned parameter starts from seeded random initialization or an earlier checkpoint generated in the same run from the 99 training files; validation labels are used only for standard checkpoint/threshold selection and final scoring, never optimizer input or backpropagation; test placeholders are never read as labels.

- [ ] **Step 2: Label supplied historical weights as non-training evidence**

State that `weights/pretrained/full_q4_token64_best.pth` is retained only for historical provenance and is not referenced by `run_train.ps1`.

- [ ] **Step 3: Update all output trees and commands**

Show `stage1_base`, `stage2_final`, `training_provenance.json`, `reproduction_bundle.json`, and the existing `run_test.ps1 -TrainingRunDir` command.

- [ ] **Step 4: Regenerate and visually inspect both PDFs**

Render every page to PNG and reject clipped, overlapping, or unreadable output.

### Task 5: Final package audit

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\MANIFEST.sha256`

- [ ] **Step 1: Run static and unit checks**

Run: `python -m ruff check src tests scripts --no-cache`

Run: `python -m pytest -q -p no:cacheprovider`

Expected: all tests pass, with only the existing environment-dependent skip.

- [ ] **Step 2: Run Windows PowerShell 5.1 parse checks**

Execute both scripts with deliberately missing data paths. Expected: readable Chinese missing-data errors, not parser errors.

- [ ] **Step 3: Remove local caches and rebuild the manifest**

Ensure `.claude`, `__pycache__`, `.pytest_cache`, and `.ruff_cache` are absent. Rebuild `MANIFEST.sha256` over every other delivered file.

- [ ] **Step 4: Run manifest integration tests**

Run: `python -m pytest tests/test_project_integration.py -q -p no:cacheprovider`

Expected: every delivered file and content hash matches the manifest.

