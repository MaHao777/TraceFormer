# Fixed Schema-7 Model Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固定当前 schema-7 模型为“锥主干 + 稀疏混合 + 多尺度事件精修”的唯一结构，删除模块选择开关、历史增强路径、重复恢复训练实现和失效公共接口。

**Architecture:** 模型构造始终创建三个组成部分，配置只记录固定结构参数和一个明确的训练阶段。训练恢复统一使用带优化器/RNG 的 training-state 路径；历史 best.pt 只用于两个明确的分阶段初始化入口。推理兼容性只校验影响参数形状与推理行为的签名，训练策略单独留在运行配置中。

**Tech Stack:** Python 3.10+, PyTorch, dataclasses, PyYAML, pytest, Ruff

---

### Task 1: 固定模型结构与训练阶段

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/training/engine.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_spatiotemporal_cone_model.py`

- [ ] **Step 1: 写固定结构测试**

要求 `SpatioTemporalConeDetector` 无条件创建 `SparseTokenMixture` 和 `MultiScaleEventRefinementHead`，并删除 `sparse_mixture_enabled`、`frame_refinement_enabled` 构造参数。

- [ ] **Step 2: 写显式训练阶段测试**

`TrainingConfig.stage` 只接受 `full`、`sparse_mixture`、`refinement`。当前正式配置必须为 `refinement`；不同阶段分别只开放预期参数。

- [ ] **Step 3: 实现固定构造与阶段冻结**

模型始终实例化稀疏混合和事件精修模块。以 `configure_training_stage(stage)` 统一设置 `requires_grad` 和冻结模块的评估态，删除两个互相叠加的 freeze 布尔开关。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/test_config.py tests/test_spatiotemporal_cone_model.py -q`

Expected: PASS

### Task 2: 删除历史 threshold augmentation 路径

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/experiments/dense_fit.py`
- Modify: `scripts/check_largest_sequence.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_dense_fit_experiment.py`

- [ ] **Step 1: 删除配置和 YAML 解析入口**

移除 `ThresholdAugmentationConfig`、`HUAVConfig.threshold_augmentation` 以及对应校验和 YAML section。

- [ ] **Step 2: 删除训练分支和全零产物**

移除 threshold view、权重调度、辅助 loss、history 字段和 `threshold_augmentation.json`。保留稀疏混合所需的高阈值视图实现。

- [ ] **Step 3: 删除实验脚本中的历史调用**

资源门禁脚本只验证当前固定模型的真实前向与 loss，不再模拟已淘汰增强路径。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/test_training_engine.py tests/test_dense_fit_experiment.py -q`

Expected: PASS

### Task 3: 合并恢复训练入口

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Modify: `tests/test_training_engine.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 删除 best.pt 伪恢复入口**

删除 `resume_training()` 和 `_resume_training_impl()`。`--resume-state` 继续通过 `_run_training_impl()` 恢复模型、优化器、学习率控制器与 RNG。

- [ ] **Step 2: 保留两个明确初始化入口**

`--initialize-full` 仅用于从固定主干初始化 sparse-mixture 阶段；`--initialize-refinement` 仅用于从 schema-5 基线初始化 refinement 阶段。

- [ ] **Step 3: 测试错误阶段和精确恢复**

验证初始化参数与 `training.stage` 必须匹配，并验证 training-state 恢复保持分支契约。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/test_training_engine.py tests/test_cli.py -q`

Expected: PASS

### Task 4: 收缩 checkpoint 契约

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: 定义固定推理签名**

签名包含 `model_config`、分辨率、sparse-mixture 推理门限与 refinement 结构；排除训练阶段、loss 权重、warmup/ramp 和冻结策略。

- [ ] **Step 2: 兼容当前 schema-7 checkpoint 元数据**

从现有 `sparse_mixture_config` 和 `frame_refinement_config` 中提取固定推理字段比较，不引入模块开关或旧模型结构分支。

- [ ] **Step 3: 验证当前正式 checkpoint 可严格加载**

Run: `python -m pytest tests/test_training_engine.py -q`

Expected: PASS

### Task 5: 删除旧体素公共接口与诊断结构开关

**Files:**
- Modify: `src/huav/data/__init__.py`
- Modify: `src/huav/models/__init__.py`
- Modify: `src/huav/__init__.py`
- Modify: `src/huav/inference/__init__.py`
- Modify: `src/huav/models/cone_neighbor_builder.py`
- Modify: `src/huav/models/cone_attention.py`
- Modify: `src/huav/models/cone_transformer.py`
- Modify: `tests/test_contracts.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_cone_attention.py`
- Modify: `tests/test_cone_neighbor_builder.py`

- [ ] **Step 1: 删除未使用公共类型**

移除 `EventWindow`、`EventBatch`、`ModelOutput`、`EventSegmenter`、`Predictor` 和 `SubmissionWriter`；保留真实使用的 `EventSequence` 与提交校验类型。

- [ ] **Step 2: 固定双向注意力**

删除 `causal_mode` 和 `debug_attention_stats` 配置与分支。锥注意力始终建立过去和未来邻域，输出仅保留 token features。

- [ ] **Step 3: 运行模型与契约测试**

Run: `python -m pytest tests/test_contracts.py tests/test_public_api.py tests/test_cone_attention.py tests/test_cone_neighbor_builder.py -q`

Expected: PASS

### Task 6: 收缩配置和 CLI 表面

**Files:**
- Delete: `configs/huav_base.yaml`
- Delete: `configs/huav_cpu_smoke.yaml`
- Delete: `configs/huav_embedding_a_autodl.yaml`
- Delete: `configs/huav_embedding_a_autodl_smoke.yaml`
- Delete: `configs/huav_embedding_b_autodl.yaml`
- Delete: `configs/huav_embedding_b_autodl_smoke.yaml`
- Delete: `configs/huav_embedding_c_autodl.yaml`
- Delete: `configs/huav_embedding_c_autodl_smoke.yaml`
- Delete: `src/huav/cli/inspect.py`
- Delete: `src/huav/cli/package.py`
- Delete: `src/huav/cli/_command.py`
- Modify: `pyproject.toml`
- Modify: `src/huav/cli/train.py`
- Modify: `src/huav/cli/evaluate.py`
- Modify: `src/huav/cli/predict.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 只保留当前模型的 CPU/CUDA、full/smoke 配置**

四份配置之间只允许设备、混合精度、线程、文件上限和输出根目录不同，模型结构完全一致。

- [ ] **Step 2: 删除未实现命令**

从入口点移除 `huav-inspect` 和 `huav-package`，删除 Phase-1 scaffold。

- [ ] **Step 3: 修正 CLI 默认配置**

训练默认使用 `configs/huav_score_optimized.yaml`；评估与预测继续显式支持配置，但默认同样指向固定模型配置。

- [ ] **Step 4: 运行 CLI 和配置测试**

Run: `python -m pytest tests/test_cli.py tests/test_config.py -q`

Expected: PASS

### Task 7: 文档同步和全量验证

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `tests/test_project_integration.py`

- [ ] **Step 1: 更新唯一正式模型说明**

记录 schema-7、64 维 token/event embedding、固定 sparse mixture、固定多尺度 refinement、显式 refinement 训练阶段和 checkpoint-adjacent 配置用法。

- [ ] **Step 2: 删除过期 schema-5/STC/48 维说明**

历史实验细节保留在 `docs/superpowers/plans/` 和输出产物中，不再作为 README 当前行为。

- [ ] **Step 3: 运行完整验证**

Run: `python -m pytest -q`

Expected: all tests pass

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

Run: `git diff --check`

Expected: no output
