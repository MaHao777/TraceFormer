# HUAV 稀疏时空锥模型

## 1. 输入与稀疏块

公开输入为浮点张量 `[N,5]`：

```text
[batch_index, x, y, normalized_time, polarity]
```

也可传 `[N,4] = [x,y,t,p]` 和独立 `batch_indices[N]`。`normalized_time` 基于固定 8 秒序列范围归一化到 `[0,1]`。

`EventTokenizer` 计算：

```text
x_block = floor(x / block_size_x)
y_block = floor(y / block_size_y)
t_block = min(floor(t * num_time_blocks), num_time_blocks - 1)
```

复合键包含 batch，因此不同样本即使块坐标相同也会生成不同 Token。`torch.unique` 只枚举非空块并返回 `event_token_indices[N]`，不分配稠密三维网格。

块内每个事件的局部特征为：

```text
[local_x, local_y, local_t, polarity,
 polarity - sequence_polarity_mean, sequence_polarity_mean]
```

Event MLP 编码后使用四个共享查询执行不依赖 padding 的块内注意力池化。块统计为：

```text
[log(1+count), positive_polarity_ratio,
 local_t_mean, local_t_variance, bounded_density,
 token_vs_sequence_polarity_contrast, polarity_entropy]
```

池化特征、统计量和归一化 Token 坐标编码共同投影为 `tokens[M,C]`。

## 2. 数据测得的锥

空间距离使用：

```text
d_space = max(abs(dx), abs(dy))
```

某一头在时间距离 `dt` 的硬边界为：

```text
d_space <= base_radius + slope * dt
```

训练集轨迹统计得到原始斜率 q50/q90/q95/q99 为 `0.640625 / 1.25 / 1.446429 / 1.875`，所以四头配置采用：

| Head | Base radius | Slope | Max time distance |
|---:|---:|---:|---:|
| 0 | 1 | 0.5 | 4 |
| 1 | 1 | 1.0 | 4 |
| 2 | 2 | 1.5 | 6 |
| 3 | 2 | 2.0 | 8 |

三层时间尺度为 `0.5 / 0.75 / 1.0`，浅层先建模短轨迹，深层再扩大时间范围。

## 3. 稀疏邻域

`ConeNeighborBuilder` 对每个 `(batch,time_block)` 的非空空间坐标建立一个 `cKDTree`。每个中心 Token 只查询同 batch、配置时间范围内的过去或未来切片，`p=∞` 对应 Chebyshev 距离。候选按时间距离、空间距离排序并截断到配置的 `K`。

输出索引：

```text
past_indices[M,H,K]
future_indices[M,H,K]
```

无邻居位置填 `-1`。该过程的存储量与 `M×H×K` 成正比，不构造完整 `M×M` 关系。

## 4. 双向多锥中心注意力

中心 Token 只产生 Query。全部 Token 经过同一组 Key/Value 投影，过去和未来复用这些参数，但分别加入可学习的方向 embedding 并执行独立 softmax。

每条边的相对特征为：

```text
[dx, dy, signed_dt, Chebyshev_space, abs_dt, Chebyshev_space / abs_dt]
```

小型 MLP 将其变为注意力分数偏置。全空邻域在 softmax 前显式识别，聚合结果为零，不产生 NaN。

中心 Token 经线性层和 softmax 得到过去/未来门控。没有邻域的分支被屏蔽并重新归一化；`causal_mode=True` 时未来分支恒为零。注意力更新后保留残差：

```text
LayerNorm → bidirectional cone attention → residual
LayerNorm → MLP → residual
```

## 5. 事件解码

每个事件通过 `event_token_indices` 取回更新后的父 Token，并拼接：

```text
[parent_token,
 local_x, local_y, local_t, polarity,
 polarity - sequence_polarity_mean, sequence_polarity_mean]
```

块统计量已经参与初始 Token 构造，不在解码器中重复拼接。轻量 MLP 输出 `event_logits[N]`；同一块内事件仍能根据各自的局部坐标、时间和极性得到不同结果。

### 5.1 稀疏混合专家

分数优化配置为满足事件数和极性平衡门槛的序列生成一个确定性的 `1.5×` 高触发阈值子流。Full 与 sparse 视图共享 `EventTokenizer`、锥编码器和事件解码器；稀疏 Token 通过 `(batch,t_block,y_block,x_block)` 精确对齐到 Full Token，稀疏视图没有占据的 Full Token 不接受专家修正。

稀疏专家读取 Full/Sparse 表示、两者差值、事件数和保留率，由 Token 级 Router 输出门控：

```text
gate = sparse_present * sigmoid(router(features))
mixed_token = full_token + residual_scale * gate * sparse_adapter(features)
```

Router 初始 bias 为 `-2`，残差 LayerScale 初始为 `0.01`。训练前几个 epoch 将 gate 权重固定为零，随后逐步升到一；因此新分支开始训练时模型行为接近原 Full 模型。普通序列直接跳过 Sparse forward。最终仍使用 Full 视图的 `event_token_indices` 解码全部原始事件，输出数量和顺序不受稀疏化影响。

## 6. 训练

模型始终对完整序列前向。背景下采样只生成损失索引：

```text
all positive events + at most ratio × positive background events
```

无正事件序列保留最多 `max_background_without_positive` 个背景损失项。Focal、Dice 和目标帧召回项使用采样事件；Fa 项使用全部事件，先按 50 ms 帧与像素聚合背景响应，再惩罚每帧最高的 32 个像素 logit：

```text
total = focal
      + lambda_dice * dice
      + lambda_object_frame * object_frame
      + lambda_false_alarm * false_alarm
```

启用稀疏混合专家时，融合输出仍承担上述完整主损失，并增加：

```text
total_moe = full_event_main
          + 0.25  * full_branch_aux
          + 0.15  * sparse_branch_aux
          + 0.10  * positive_object_frame_consistency
          + 0.05  * background_upper_bound_consistency
          + 0.05  * sparse_block_false_peak
          + 0.001 * mean_gate
```

正目标帧一致性保护稀疏分支的召回；背景一致性只在 Sparse 背景 logit 高于 detached Full logit 时施加惩罚。块级假警项按 `50 ms × 8×8` 聚合背景最大 logit，再选每帧最高的 8 个空间块，专门抑制稀疏视图造成的局部背景峰。Full 分支辅助损失保证即使关闭稀疏专家，主干仍保持独立判别能力。

优化器为 AdamW，支持梯度裁剪。`mixed_precision=true` 且设备为 CUDA 时启用 autocast 和 GradScaler；CPU 自动保持 FP32。

## 7. 评估与可复现性

验证对全部事件预测；满足门槛的序列使用与训练相同的确定性 Sparse 视图和完整门控权重。输出事件级 Pd/Fa/IoU/Acc 和单独命名的 EV-UAV-compatible 指标。每次训练内验证还使用独立固定种子的采样器，按与训练一致的正样本保留、hard-negative 背景采样、密集序列权重和完整事件 Fa 上下文计算 loss；五个主损失分量写入 `history.json[*].validation_loss`，不会混入 `metrics.json`。训练时选出的事件 IoU 阈值随最佳 checkpoint 保存，复评不会再次使用标签调阈值。schema-5 checkpoint 同时保存模型、稀疏混合专家配置和验证集 manifest 摘要，配置或数据不匹配时拒绝复评。
