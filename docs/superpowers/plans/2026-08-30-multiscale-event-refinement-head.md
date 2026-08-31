# Multi-Scale Event Refinement Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用冻结的 schema-5 稀疏专家主干训练一个多尺度时空事件精修头，在保住 `val_023` 召回的同时降低对象 cell 内的背景误检。

**Architecture:** 主干继续输出一个原始事件 logit 和 64 维父 Token。精修头只投影一次事件特征，再分别按 `2×2×10 ms`、`4×4×25 ms`、`8×8×50 ms` 稀疏分组；每个尺度融合同一空间 cell 的前、当前、后一时间片，最后由事件级 MLP 输出零初始化残差。`8×8×50 ms` 分支额外输出 cell objectness，作为辅助监督而不再向 cell 内所有事件广播同一个 logit。

**Tech Stack:** Python 3.12、PyTorch、YAML、pytest、Ruff。

---

## 文件边界

- `src/huav/config.py`：只定义和校验尺度/精修配置。
- `src/huav/models/event_refinement.py`：只负责多尺度稀疏聚合、相邻时间片融合和事件残差预测。
- `src/huav/models/spatiotemporal_cone.py`：只负责把主干输出送入精修头并保持固定输出契约。
- `src/huav/training/engine.py`：负责 schema-7 检查点、schema-5 初始化、冻结和 loss 接线。
- `src/huav/training/losses.py`：继续提供可微的 cell focal loss，不加入阈值操作。
- `tests/test_event_refinement.py`：覆盖精修头的纯模块契约。
- 其余现有模型、锥邻居、Tokenizer、稀疏专家和后处理代码不改。

当前仓库已有用户未提交的后处理改动，因此本计划在本地 checkout 直接执行，不创建 worktree，也不自动提交 Git commit。

### Task 1: 配置与 schema-7 契约

**Files:**
- Modify: `src/huav/config.py`
- Modify: `configs/huav_score_optimized.yaml`
- Modify: `configs/huav_score_optimized_smoke.yaml`
- Modify: `configs/huav_score_optimized_autodl.yaml`
- Modify: `configs/huav_score_optimized_autodl_smoke.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写配置失败测试**

```python
def test_frame_refinement_defaults_and_formal_scales() -> None:
    config = load_config("configs/huav_score_optimized.yaml")
    assert config.frame_refinement.enabled
    assert config.frame_refinement.freeze_base_model
    assert config.frame_refinement.context_dim == 16
    assert config.frame_refinement.scales == (
        RefinementScaleConfig(2, 2, 10.0),
        RefinementScaleConfig(4, 4, 25.0),
        RefinementScaleConfig(8, 8, 50.0),
    )


def test_frame_refinement_rejects_invalid_scale() -> None:
    base = HUAVConfig.default()
    with pytest.raises(ValueError, match="refinement scale"):
        replace(
            base,
            frame_refinement=FrameRefinementConfig(
                enabled=True,
                scales=(RefinementScaleConfig(0, 2, 10.0),),
            ),
        ).validate()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -q`

Expected: 因 `FrameRefinementConfig` 尚不存在而失败。

- [ ] **Step 3: 实现强类型配置**

```python
@dataclass(frozen=True)
class RefinementScaleConfig:
    spatial_block_size_x: int
    spatial_block_size_y: int
    frame_ms: float


DEFAULT_REFINEMENT_SCALES = (
    RefinementScaleConfig(2, 2, 10.0),
    RefinementScaleConfig(4, 4, 25.0),
    RefinementScaleConfig(8, 8, 50.0),
)


@dataclass(frozen=True)
class FrameRefinementConfig:
    enabled: bool = False
    context_dim: int = 16
    scales: tuple[RefinementScaleConfig, ...] = DEFAULT_REFINEMENT_SCALES
    loss_weight: float = 0.2
    freeze_base_model: bool = False
```

`HUAVConfig` 使用 `frame_refinement` 字段；YAML 解析时把 `scales` 列表逐项转为 `RefinementScaleConfig`。校验 `context_dim`、空间尺寸、`frame_ms` 为正，尺度按 `frame_ms` 严格递增，最后一个尺度必须是 `8×8×50 ms`，`loss_weight >= 0`，布尔字段必须为布尔值。

- [ ] **Step 4: 更新四个正式/烟雾配置**

```yaml
frame_refinement:
  enabled: true
  context_dim: 16
  scales:
  - spatial_block_size_x: 2
    spatial_block_size_y: 2
    frame_ms: 10.0
  - spatial_block_size_x: 4
    spatial_block_size_y: 4
    frame_ms: 25.0
  - spatial_block_size_x: 8
    spatial_block_size_y: 8
    frame_ms: 50.0
  loss_weight: 0.2
  freeze_base_model: true
```

- [ ] **Step 5: 运行配置测试**

Run: `python -m pytest tests/test_config.py -q`

Expected: PASS。

### Task 2: 多尺度事件精修模块

**Files:**
- Create: `src/huav/models/event_refinement.py`
- Delete: `src/huav/models/frame_objectness.py`
- Create: `tests/test_event_refinement.py`
- Delete: `tests/test_frame_objectness.py`

- [ ] **Step 1: 写模块失败测试**

覆盖以下行为：

```python
def test_refinement_groups_each_scale_and_keeps_event_specific_delta() -> None:
    head = MultiScaleEventRefinementHead(
        token_dim=4,
        event_feature_dim=6,
        context_dim=4,
        width=16,
        height=16,
        duration_ms=100.0,
        scales=(
            RefinementScaleConfig(2, 2, 10.0),
            RefinementScaleConfig(8, 8, 50.0),
        ),
        dropout=0.0,
    )
    output = head(
        token_features=torch.randn(2, 4),
        event_token_indices=torch.tensor([0, 0, 1, 0]),
        event_local_features=torch.randn(4, 6),
        event_xy=torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [1.0, 1.0]]),
        normalized_time=torch.tensor([0.10, 0.20, 0.20, 0.60]),
        event_batch_indices=torch.zeros(4, dtype=torch.long),
        base_event_logits=torch.randn(4),
    )
    assert output.cell_logits.shape == (3,)
    assert output.event_cell_indices.shape == (4,)
    assert output.event_logits.shape == (4,)
    assert torch.equal(output.event_logits, torch.zeros(4))
```

另写测试验证：前一/后一时间片存在时输出上下文不同；两个同 cell 事件可在解除零初始化后得到不同残差；不同 batch 不串联；空输入安全；所有输出有限。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_event_refinement.py -q`

Expected: 因模块不存在而失败。

- [ ] **Step 3: 实现一次事件投影与稀疏 cell 聚合**

模块输入先执行：

```python
event_features = self.event_projection(
    torch.cat(
        (
            token_features[event_token_indices],
            event_local_features,
            base_event_logits[:, None],
        ),
        dim=1,
    )
)
```

每个尺度用 `(batch, frame, y_block, x_block)` 组成单调整数 key，通过 `torch.unique(..., sorted=True, return_inverse=True, return_counts=True)` 聚合 `mean`、`amax` 和 `log1p(count)`；每个尺度独立 MLP 把 `[mean, maximum, log_count]` 投影回 `context_dim`。

- [ ] **Step 4: 实现相邻时间片查找**

对排序后的占用 cell key，目标 key 为当前 key 加减一个 `cells_per_frame`。用 `torch.searchsorted` 查找，先检查 batch 与 frame 边界，再把不存在的邻居填零：

```python
temporal_input = torch.cat((previous, current, following), dim=1)
temporal_context = self.temporal_fusers[scale_index](temporal_input)
event_context = temporal_context[inverse]
```

- [ ] **Step 5: 实现 cell 辅助输出与事件残差**

最后一个 `8×8×50 ms` 尺度产生 `cell_logits`。事件分类器读取共享事件投影、三个尺度的事件上下文、该事件对应的 cell logit 与 base logit，输出每事件独立残差。cell 分类器和事件分类器最后一层权重/偏置都置零，保证初始前向与 schema-5 完全一致。

- [ ] **Step 6: 运行模块测试**

Run: `python -m pytest tests/test_event_refinement.py -q`

Expected: PASS。

### Task 3: 检测器接线与冻结

**Files:**
- Modify: `src/huav/models/spatiotemporal_cone.py`
- Modify: `src/huav/models/__init__.py`
- Test: `tests/test_spatiotemporal_cone_model.py`

- [ ] **Step 1: 写检测器失败测试**

验证固定输出增加：

```python
"event_refinement_cell_logits"
"event_refinement_event_cell_indices"
"event_refinement_event_logits"
```

禁用时返回稳定空/零 Tensor；启用时 `event_logits == base_event_logits + event_refinement_event_logits`；零初始化时两者完全相等。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py -q`

Expected: 旧对象头字段与新字段不一致而失败。

- [ ] **Step 3: 替换检测器对象头**

```python
base_event_logits = event_logits
if self.event_refinement_head is not None:
    refinement = self.event_refinement_head(
        token_features=token_features,
        event_token_indices=tokenized.event_token_indices,
        event_local_features=tokenized.event_local_features,
        event_xy=tokenized.event_xy,
        normalized_time=tokenized.normalized_time,
        event_batch_indices=tokenized.event_batch_indices,
        base_event_logits=base_event_logits,
    )
    event_logits = base_event_logits + refinement.event_logits
```

实现 `freeze_base_for_refinement()`：冻结 tokenizer、encoder、decoder、sparse mixture，保持这些模块为 `eval()`，只让 `event_refinement_head` 训练。中文注释说明冻结原因。

- [ ] **Step 4: 运行检测器测试**

Run: `python -m pytest tests/test_spatiotemporal_cone_model.py -q`

Expected: PASS。

### Task 4: 初始化、CLI 与检查点契约

**Files:**
- Modify: `src/huav/training/engine.py`
- Modify: `src/huav/cli/train.py`
- Modify: `scripts/check_largest_sequence.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写 schema-5 初始化失败测试**

构造 schema-5 checkpoint，验证：model、sparse mixture、resolution 完全匹配；加载后源权重逐 Tensor 相等；只有 `event_refinement_head.*` 可训练；初始化摘要 `kind == "event_refinement"`；保存 checkpoint schema 为 7 且包含 `frame_refinement_config`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_training_engine.py::test_refinement_initialization_loads_schema5_base_and_trains_only_new_head -q`

Expected: 新初始化入口不存在而失败。

- [ ] **Step 3: 升级检查点与初始化入口**

```python
CHECKPOINT_SCHEMA_VERSION = 7
REFINEMENT_SOURCE_SCHEMA_VERSION = 5
```

实现 `_initialize_refinement_head()`：只允许 schema-5；严格比较 `model_config`、`sparse_mixture_config`、`resolution`；源状态不得含 `event_refinement_head.*`；目标缺失键只能来自新头；返回路径、SHA-256、源 schema 和 epoch。

- [ ] **Step 4: 更新 CLI**

删除 `--initialize-objectness`，增加：

```python
resume_group.add_argument(
    "--initialize-refinement",
    metavar="BEST_PT",
    help="initialize schema-5 base weights and train only event refinement",
)
```

它与 resume、resume-state、initialize-full 互斥，并要求 `frame_refinement.freeze_base_model=true`。

- [ ] **Step 5: 更新最大序列门禁脚本**

脚本支持 schema-5 初始化精修头，检查主干无梯度、精修头梯度有限、事件输出数量不变。

- [ ] **Step 6: 运行初始化测试**

Run: `python -m pytest tests/test_training_engine.py -q`

Expected: PASS。

### Task 5: 辅助损失与训练记录

**Files:**
- Modify: `src/huav/training/engine.py`
- Test: `tests/test_training_components.py`
- Test: `tests/test_training_engine.py`

- [ ] **Step 1: 写 loss 接线测试**

继续使用 `frame_cell_objectness_loss`：cell target 为 cell 内事件标签 `amax`，不对概率 threshold。验证精修头 cell classifier 和 event classifier 都获得有限梯度，冻结主干无梯度。

- [ ] **Step 2: 接入训练与验证**

```python
refinement_loss = frame_cell_objectness_loss(
    output["event_refinement_cell_logits"],
    output["event_refinement_event_cell_indices"],
    targets,
    alpha=config.loss.focal_alpha,
    gamma=config.loss.focal_gamma,
)
total = main_loss + config.frame_refinement.loss_weight * refinement_loss
```

训练和验证 history 使用 `event_refinement` 字段。主损失仍作用于最终全事件 logits；稀疏专家、困难负样本、对象帧召回和完整事件假警损失保持不变。

- [ ] **Step 3: 运行定向训练测试**

Run: `python -m pytest tests/test_training_components.py tests/test_training_engine.py -q`

Expected: PASS。

### Task 6: 文档、静态验证与资源门禁

**Files:**
- Modify: `docs/architecture.md`
- Test: `tests/test_project_integration.py`

- [ ] **Step 1: 更新架构文档**

说明三个尺度、相邻时间片查找、cell 辅助监督、事件独立残差、零初始化和 schema-5 冻结来源；删除旧“cell logit 广播给所有事件”的描述。

- [ ] **Step 2: 完整静态验证**

Run: `python -m pytest -q`

Expected: 全部通过。

Run: `python -m ruff check src tests scripts`

Expected: 无 lint 错误。

Run: `git diff --check`

Expected: 无空白错误；只允许本计划文件和精修头相关文件新增差异，原有后处理差异保持不变。

- [ ] **Step 3: 最大序列门禁**

Run:

```powershell
python scripts/check_largest_sequence.py `
  --config configs/huav_score_optimized.yaml `
  --device cpu `
  --initialize-refinement outputs/cpu_20260827_230649/best.pt
```

Expected: `train_094.npz` 前向、主损失、cell 辅助损失和反向成功；主干无梯度；精修头梯度有限；报告峰值内存与运行时间。

### Task 7: 99/24 正式实验与晋级门槛

**Files:**
- Output: `outputs/cpu_YYYYMMDD_HHMMSS/`

- [ ] **Step 1: 启动单变量正式实验**

Run:

```powershell
python -m huav.cli.train `
  --config configs/huav_score_optimized.yaml `
  --device cpu `
  --initialize-refinement outputs/cpu_20260827_230649/best.pt `
  --note "三尺度事件精修头；冻结schema5主干；保持锥结构和训练协议不变"
```

Expected: 99 train / 24 validation、seed 37、24 epoch；只训练精修头；新输出目录不覆盖既有结果。

- [ ] **Step 2: 独立 FP32 复评**

用保存的 `config.yaml + best.pt` 独立复评，要求 `metrics.json` 与 `independent_evaluation.json` 完全一致，并保留 24 文件逐文件指标。

- [ ] **Step 3: 执行预注册晋级门槛**

候选只有同时满足以下条件才可替代当前 schema-6：

- 完整 24 文件 Score `>= 0.9164111564`；
- `val_023` detected objects `>= 253 / 413`；
- `val_023` false components `<= 336`；
- 24 TXT、1,424,330 行、源 `x y t p` 顺序和值、二值标签和 ZIP 根目录审计全部通过。

任一门槛失败就保留当前模型，并停止后续可变形锥实验。

## Self-review

- 规格只覆盖一个独立子系统：冻结主干上的多尺度事件精修头。
- 没有修改锥邻居、Tokenizer、稀疏专家、数据、损失基线或后处理，因此比较仍是单变量。
- 类型名、YAML 字段、模型属性、输出字段、CLI 参数和 checkpoint 字段在各任务中保持一致。
- 空输入、batch 隔离、时间边界、零初始化、冻结梯度、严格 schema 和完整 24 文件验收均有明确测试。
