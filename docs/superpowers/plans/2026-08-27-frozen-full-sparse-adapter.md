# Frozen Full Sparse Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用历史最佳检查点冻结完整事件分支，只训练稀疏专家修正，并用全量正事件保护损失避免 `val_023` 召回下降。

**Architecture:** 训练启动时只从历史检查点加载 tokenizer、encoder、decoder，稀疏混合模块保持新初始化。Full 分支参数永久冻结且训练态保持 `eval`，优化器只接收稀疏混合模块参数；主损失和辅助损失都作用于融合后的全事件 logits，教师 Full logits 全部 `detach`。

**Tech Stack:** Python 3.12、PyTorch、YAML 配置、pytest、Ruff。

---

### Task 1: 配置和正样本保护损失

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`
- Test: `tests/test_config.py`
- Test: `tests/test_training_components.py`

- [ ] **Step 1: 写失败测试**

验证 `freeze_full_model` 必须是布尔值、保护损失权重非负，并验证保护损失只在融合正事件低于冻结教师指定 margin 时产生梯度。

- [ ] **Step 2: 运行定向测试确认失败**

Run: `python -m pytest tests/test_config.py tests/test_training_components.py -q`
Expected: 新字段和新函数尚不存在，测试失败。

- [ ] **Step 3: 实现最小配置与损失**

在 `SparseMixtureConfig` 增加 `freeze_full_model`、`positive_preservation_weight`、`positive_preservation_margin`；实现 `positive_logit_preservation_loss(fused, full, labels)`，Full logits 必须在函数内断开梯度。

- [ ] **Step 4: 运行定向测试确认通过**

Run: `python -m pytest tests/test_config.py tests/test_training_components.py -q`
Expected: PASS。

### Task 2: 冻结模型与历史 Full 初始化

**Files:**
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Test: `tests/test_spatiotemporal_cone_model.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写失败测试**

验证冻结后 tokenizer、encoder、decoder 无梯度且始终保持 `eval`，稀疏混合模块仍可训练；验证 `--initialize-full` 与恢复参数互斥。

- [ ] **Step 2: 运行定向测试确认失败**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py tests/test_training_engine.py -q`
Expected: 冻结接口和初始化参数尚不存在，测试失败。

- [ ] **Step 3: 实现冻结和严格初始化**

新增 `freeze_full_path()`，覆盖 `train()` 保持 Full 分支为评估态；初始化只允许旧检查点中与 Full 分支完全匹配的权重，拒绝架构或分辨率不匹配。优化器只接收 `requires_grad=True` 的参数，并把初始化路径和 SHA-256 写入运行摘要。

- [ ] **Step 4: 把辅助目标改到可训练路径**

稀疏采样损失使用融合 logits 的对应事件；背景一致性、块级假峰和正样本保护均使用全事件融合 logits，对照 Full logits 断开梯度。冻结模式下不再把无梯度的 Full 辅助分支加入总 loss。

- [ ] **Step 5: 运行定向测试确认通过**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py tests/test_training_engine.py -q`
Expected: PASS。

### Task 3: 正式配置、静态验证和 CPU 后台启动

**Files:**
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Modify: `scripts/check_largest_sequence.py`
- Test: `tests/test_project_integration.py`

- [ ] **Step 1: 更新配置**

启用 Full 冻结，将 `full_aux_loss_weight` 和旧对象帧一致性权重设为 0，把稀疏采样、背景、假峰权重降到保守值，并开启全正事件保护。

- [ ] **Step 2: 运行完整静态验证**

Run: `python -m pytest -q`
Expected: 全部测试通过。

Run: `python -m ruff check src tests scripts`
Expected: 无 lint 错误。

Run: `git diff --check`
Expected: 无空白错误。

- [ ] **Step 3: 运行最大序列资源门禁**

Run: `python scripts/check_largest_sequence.py --config configs/huav_score_optimized.yaml --device cpu --initialize-full outputs/outputs_old/cpu_pool_token_ablation_20260730_172827/full_q4_token64/best.pt`
Expected: 前向、loss、反向均成功，Full 分支无梯度。

- [ ] **Step 4: 后台启动本地 CPU 正式训练**

Run: `python -m huav.cli.train --config configs/huav_score_optimized.yaml --device cpu --initialize-full <历史最佳 best.pt> --output-dir <新目录> --note <实验说明>`
Expected: 进程后台存活，日志出现训练启动行；随后停止轮询并向用户报告 PID、日志和输出目录。
