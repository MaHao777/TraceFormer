# HUAV 固定 schema-7 模型结构

## 1. 事件输入与 Token 化

模型输入为 `[N,4] = [x,y,t,p]` 和独立的 `batch_indices[N]`。时间按固定序列时长归一化。当前空间块为 `8×8`，时间分为 64 段，Token 和事件嵌入维度均为 64。

`EventTokenizer` 只枚举非空的 `(batch,t_block,y_block,x_block)`，不会分配稠密三维体素网格。每个事件保留局部坐标、局部时间、极性及序列极性统计，用于块内池化和最终事件级解码。

## 2. 双向时空锥主干

每层按四组固定锥参数查找过去和未来邻域。邻域构建使用每个时间片的稀疏空间索引，输出规模与 `token_count × heads × max_neighbors` 成正比，不构造完整 Token 两两关系。

中心 Token 产生 Query；过去与未来共享 Key/Value 投影，并分别加入方向编码。相对空间、时间和速度特征为注意力分数提供偏置。两侧结果经过数据相关门控融合，再进入残差 MLP。

该主干始终为双向结构。代码中不再存在因果模式或调试统计开关。

## 3. 固定 Sparse Token 混合

`SparseTokenMixture` 始终属于模型并保存在 checkpoint 中。对于事件数足够且极性分布平衡的序列，预处理会确定性地生成 `1.5×` 高触发阈值子流；普通序列不生成子流，但仍经过同一套固定模型。

Full 与 Sparse 视图共享 Tokenizer、锥编码器和事件解码器。Sparse Token 按离散块坐标对齐到 Full Token，路由器只在 Sparse Token 存在的位置计算残差修正：

```text
gate = sparse_present × sigmoid(router(features))
mixed_token = full_token + residual_scale × gate × sparse_adapter(features)
```

最终始终按 Full 视图的事件索引解码，因此输出数量和原始事件顺序保持不变。

## 4. 固定多尺度事件精修

`MultiScaleEventRefinementHead` 始终属于模型。它复用最终 Token、六维事件局部特征和基础 logit，在以下三个尺度聚合稀疏 cell 上下文：

- `2×2×10 ms`
- `4×4×25 ms`
- `8×8×50 ms`

主尺度输出 cell objectness 辅助监督；事件分类器为每个事件产生独立残差：

```text
event_logit = base_event_logit + refinement_delta[event]
```

精修头只处理被事件占用的 cell，不构造稠密时空体。

## 5. 训练阶段不改变结构

`training.stage` 只设置 `requires_grad` 和冻结模块的评估态：

| 阶段 | 可训练部分 |
|---|---|
| `full` | 全部固定结构 |
| `sparse_mixture` | 仅 Sparse Token 混合 |
| `refinement` | 仅多尺度事件精修头 |

阶段切换不会创建、删除或绕过模型参数。当前正式配置使用 `refinement`。

训练主损失由事件 Focal、Dice、目标帧召回、完整事件假警约束和主尺度精修辅助损失组成。Sparse 子流存在时再计算其专家辅助约束；这些 loss 权重属于训练策略，不属于模型结构开关。

## 6. Checkpoint 契约

正式模型 checkpoint schema 固定为 7，保存完整模型状态和以下推理契约：

- 主干模型结构参数
- 输入分辨率
- Sparse 子流的事件数、极性和阈值门限
- 精修头上下文维度及三个尺度

训练阶段、loss 权重、warmup/ramp 和冻结策略不参与推理兼容性判断。历史 schema-7 元数据中固定关闭的 `causal_mode`、`debug_attention_stats`，以及旧的 `enabled/freeze` 字段只在加载时做受限归一化；它们不再是配置项或代码分支。

`best.pt` 用于推理或明确的分阶段初始化；精确续训统一使用 training-state，以恢复优化器、学习率控制器和随机数状态。
