# Frame Cell Objectness Full CPU Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 schema-5 稀疏专家模型上增加 50 ms × 8×8 的帧网格 objectness 残差头，从现有最佳检查点只初始化并训练新头，随后完成 CPU 全量训练和三组 24 文件对比。

**Architecture:** 新头融合最终 token 特征与事件局部特征，按 `(batch, frame, y_block, x_block)` 聚合均值/最大值，生成 cell logit 并残差加到当前事件 logit。旧 tokenizer、encoder、decoder 和 sparse mixture 从 schema-5 最佳检查点严格加载并冻结；新 checkpoint 升为 schema 6，并显式保存 `frame_objectness_config`。后处理必须基于新模型重新缓存和训练，禁止复用旧模型的 LightGBM 与阈值。

**Tech Stack:** Python 3.12、PyTorch CPU、YAML、pytest、Ruff、LightGBM。

---

### Task 1: 配置与检查点契约

**Files:**
- Modify: `src/huav/config.py`
- Modify: `src/huav/training/engine.py`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写配置失败测试**

验证缺省 `FrameObjectnessConfig(enabled=False)`；正式/烟雾配置均为 `enabled=True, frame_ms=50, spatial_block_size_x=8, spatial_block_size_y=8, loss_weight=0.2, freeze_base_model=True`；拒绝非正尺寸、非正 frame 和负 loss 权重。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_config.py -q`

Expected: `FrameObjectnessConfig` 尚不存在而失败。

- [ ] **Step 3: 实现配置**

```python
@dataclass(frozen=True)
class FrameObjectnessConfig:
    enabled: bool = False
    frame_ms: float = 50.0
    spatial_block_size_x: int = 8
    spatial_block_size_y: int = 8
    loss_weight: float = 0.2
    freeze_base_model: bool = False
```

将其加入 `HUAVConfig`、YAML 解析和 `validate()`；正式与 smoke 配置只改变训练文件上限，objectness 配置必须完全相同。

- [ ] **Step 4: 升级检查点契约**

把 `CHECKPOINT_SCHEMA_VERSION` 从 5 升到 6。schema-6 的训练、恢复、评估、推理和提交路径都严格比较 `frame_objectness_config`；只有 `--initialize-objectness` 可以读取 schema-5 源检查点。

### Task 2: 对象头与检测器前向

**Files:**
- Modify: `src/huav/models/frame_objectness.py`
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/models/__init__.py`
- Modify: `src/huav/training/engine.py`
- Test: `tests/test_frame_objectness.py`
- Test: `tests/test_spatiotemporal_cone_model.py`

- [ ] **Step 1: 写前向失败测试**

覆盖同 cell 共享 logit、跨帧/空间块隔离、空输入、零初始化不改变事件 logits、稀疏专家融合后的 token 特征被对象头使用，以及禁用时输出形状固定。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_frame_objectness.py tests/test_spatiotemporal_cone_model.py -q`

Expected: detector 尚无 objectness 输出字段而失败。

- [ ] **Step 3: 接入 detector**

```python
base_event_logits = event_logits
objectness = self.frame_objectness_head(...)
event_logits = base_event_logits + objectness.event_logits
```

固定输出增加 `base_event_logits`、`frame_objectness_cell_logits`、`frame_objectness_event_cell_indices`、`frame_objectness_event_logits`；末层保持零初始化。

- [ ] **Step 4: 冻结旧模型**

新增 `freeze_base_for_objectness()`，冻结 tokenizer、encoder、decoder、sparse mixture，只保留 `frame_objectness_head` 可训练；`train()` 保证冻结模块保持 eval、对象头保持 train。

- [ ] **Step 5: 运行前向测试**

Run: `python -m pytest tests/test_frame_objectness.py tests/test_spatiotemporal_cone_model.py -q`

Expected: PASS。

### Task 3: Cell loss 与训练/验证记录

**Files:**
- Modify: `src/huav/training/losses.py`
- Modify: `src/huav/training/__init__.py`
- Modify: `src/huav/training/engine.py`
- Test: `tests/test_training_components.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写 loss 失败测试**

验证 cell target 是组内事件标签最大值，正确 logits 的 focal loss 更小，空输入返回可导零值，且 loss 能向 objectness 头回传有限梯度。

- [ ] **Step 2: 实现 loss**

```python
def frame_cell_objectness_loss(
    cell_logits: torch.Tensor,
    event_cell_indices: torch.Tensor,
    event_targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    ...
```

用 `scatter_reduce_(reduce="amax")` 构造 cell target，再复用 `binary_focal_loss`。

- [ ] **Step 3: 接入训练和验证**

对 full-view 输出计算 `frame_objectness`，用 `loss_weight` 加到总 loss；训练与验证 history 都记录该字段。dense sequence 权重作用于事件主损失与 cell loss 的合计。

### Task 4: 从当前最佳只初始化新头

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写初始化失败测试**

构造 schema-5 源检查点，验证 schema、model、sparse mixture、resolution 全部严格匹配；加载后只有 `frame_objectness_head.*` 是新参数，旧参数逐 tensor 相等且冻结，初始化报告包含绝对路径、SHA-256、源 schema 和 epoch。

- [ ] **Step 2: 实现 CLI 与引擎参数**

增加 `--initialize-objectness BEST_PT` 和 `initial_objectness_checkpoint_path`。它与 `--resume-state`、`--initialize-full` 互斥，并要求 `frame_objectness.enabled=True`、`freeze_base_model=True`。

- [ ] **Step 3: 验证首步可训练性**

训练最小 fixture，断言 optimizer 只包含对象头、首步后旧参数不变、对象头参数变化、零初始化时首轮前向与源检查点 logits 完全相同。

- [ ] **Step 4: 运行定向测试**

Run: `python -m pytest tests/test_config.py tests/test_frame_objectness.py tests/test_spatiotemporal_cone_model.py tests/test_training_components.py tests/test_training_engine.py -q`

Expected: PASS。

### Task 5: 完整静态与 CPU 资源门槛

**Files:**
- Modify: `docs/architecture.md`
- Test: `tests/test_project_integration.py`

- [ ] **Step 1: 完整验证**

Run: `python -m pytest -q`

Expected: 全部通过。

Run: `python -m ruff check src tests scripts`

Expected: 无 lint 错误。

Run: `git diff --check`

Expected: 无空白错误。

- [ ] **Step 2: smoke 训练**

从 `outputs/cpu_20260827_230649/best.pt` 用 objectness smoke 配置初始化，只训练新头，确认配置、数据、前向、cell loss、反向、checkpoint 和独立 reevaluation 全链路可用。

- [ ] **Step 3: 最大序列资源门槛**

在 `train_094.npz` 上执行一次 FP32 CPU forward/loss/backward，记录峰值工作集、事件数、token 数、有限 loss 和梯度。

### Task 6: 完整 CPU 训练与独立复评

**Files:**
- Output: `outputs/cpu_YYYYMMDD_HHMMSS/`

- [ ] **Step 1: 启动完整训练**

Run: `python -m huav.cli.train --config configs/huav_score_optimized.yaml --device cpu --initialize-objectness outputs/cpu_20260827_230649/best.pt --note "50ms 8x8 frame objectness head; freeze schema5 base"`

Expected: 99 train / 24 validation、24 epochs、只有对象头可训练，输出新目录、PID 和日志。

- [ ] **Step 2: 独立复评**

训练完成后用保存的 `config.yaml + best.pt` 在 CPU FP32 上独立复评，要求与 `metrics.json` 完全一致，并保留 24 文件 per-file 指标。

### Task 7: 重训后处理与三组对比

**Files:**
- Output: `outputs/postprocess_frame_objectness_YYYYMMDD/`

- [ ] **Step 1: 重新缓存**

使用 objectness `best.pt` 分别生成 99 train 与 24 val 缓存；manifest 必须记录新 checkpoint/config SHA-256。禁止复用 `outputs/postprocess_huav_20260828/cache`。

- [ ] **Step 2: 重训后处理**

按固定 seed 37、79/20、5 folds、997/1998 rounds 重训事件与组件模型。用户已接受较低召回，因此同时保留事件级和组件级验证预测，不以旧 dense-Pd gate 删除候选。

- [ ] **Step 3: 24 文件盲推与独立复评**

冻结后处理参数，对无标签 val 缓存盲推，再从原始 NPZ 独立复评。

- [ ] **Step 4: 三组正式对比**

报告当前原模型、objectness、objectness+事件后处理、objectness+组件剪枝的全局 TP/FP/FN、Pd、Fa、false components、IoU、SegAcc、官方 Score，并单列 `val_015`、`val_023`。结论不得用单文件或内部 holdout 代替完整 24 文件。

---

### Self-review

- 当前 schema-5 sparse mixture、冻结路径、checkpoint 元数据与 CLI 均已覆盖。
- objectness 的 forward、loss、初始化、训练、恢复、评估、推理和提交契约均有测试步骤。
- 后处理明确要求基于新 checkpoint 重新缓存和重训。
- 完整 CPU 训练、独立复评和三组 24 文件比较均在计划内。
