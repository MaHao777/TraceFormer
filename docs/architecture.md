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

`ConeNeighborBuilder` 对每个 `(batch,time_block)` 的非空空间坐标建立一个 `cKDTree`。每个中心 Token 只查询同 batch、配置时间范围内的过去或未来切片，`p=∞` 对应 Chebyshev 距离。每个目标时间切片独立保留空间距离最近的至多 `K` 个候选，不再把不同时间切片合并后第二次截断到 `K`；因此较长时间间隔的局部近邻不会被较近切片占满。

输出索引：

```text
past_indices[h][M,K*layer_horizon[h]]
future_indices[h][M,K*layer_horizon[h]]
```

无邻居位置填 `-1`。不同 Head 使用各自的实际时间范围，避免短锥填充到最长锥的宽度；该过程的存储量与 `M×K×sum(layer_horizon)` 成正比，不构造完整 `M×M` 关系。

## 4. 双向多锥中心注意力

中心 Token 只产生 Query。全部 Token 经过同一组 Key/Value 投影，过去和未来复用这些参数，但分别加入可学习的方向 embedding；每个 Head 在其所有目标时间切片保留下来的候选集合上执行一次独立 softmax。

每条边的相对特征为：

```text
[dx, dy, signed_dt, Chebyshev_space, abs_dt, Chebyshev_space / abs_dt]
```

小型 MLP 将其变为注意力分数偏置。全空邻域在 softmax 前显式识别，聚合结果为零，不产生 NaN。

中心 Token 经线性层和 softmax 得到过去/未来门控。没有邻域的分支被屏蔽并重新归一化；`causal_mode=True` 时未来分支恒为零。

## 5. 跨层 Full Attention Residuals

默认 encoder 包含 6 个 `LocalConeAttentionBlock`，每个 block 的锥注意力和 FFN 分别视为一个子层。Tokenizer 输出 `x0` 是第一个历史源，12 个子层分别产生实际参与残差路径的增量：

```text
attention_delta = attn_scale * dropout(cone_attention(LayerNorm(attention_input)))
ffn_delta = dropout(FFN(LayerNorm(ffn_input)))
```

对任意当前子层 `l`，历史源记为 `v_i [M,C]`。每个子层有独立的零初始化伪查询 `w_l [C]` 和 `RMSNorm(eps=1e-6)`：

```text
logit_i[m] = dot(w_l, RMSNorm(v_i[m]))
alpha[:,m] = softmax(logit[:,m], source_dimension)
h_l[m] = sum_i alpha_i[m] * v_i[m]
```

softmax 使用 FP32 计算，权重随后转换回 Token dtype；Value 始终使用未归一化的原始增量。零查询使初始权重在所有可见来源上均匀分布，而不是复现标准残差的求和。第一 attention 子层只看到 `x0`，因此严格保持恒等输入。

实现逐源计算 RMSNorm 和 logit，只堆叠 `[source_count,M]` 分数，不构造 `[source_count,M,C]` 历史副本。六个 block 的 attention 子层依次看到 `1/3/5/7/9/11` 个来源，FFN 依次看到 `2/4/6/8/10/12` 个来源；最后使用第 13 个独立查询聚合 `x0` 和全部 12 个增量，再执行 `final_norm`。

Full AttnRes 不改变 Token 数量、batch 隔离、局部锥邻域或事件回映射。checkpoint schema 为 7，旧 schema-6 权重不能加载到该结构。本次改动只完成代码和静态验证，尚未执行真实数据训练、独立复评或提交生成，不能据此声称指标提升。

## 6. 事件解码

每个事件通过 `event_token_indices` 取回更新后的父 Token，并拼接：

```text
[parent_token,
 local_x, local_y, local_t, polarity,
 polarity - sequence_polarity_mean, sequence_polarity_mean]
```

块统计量已经参与初始 Token 构造，不在解码器中重复拼接。轻量 MLP 输出 `event_logits[N]`；同一块内事件仍能根据各自的局部坐标、时间和极性得到不同结果。

## 7. 训练

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

优化器为 AdamW，支持梯度裁剪。`mixed_precision=true` 且设备为 CUDA 时启用 autocast 和 GradScaler；CPU 自动保持 FP32。

## 8. 评估与可复现性

验证对全部事件预测，输出事件级 Pd/Fa/IoU/Acc 和单独命名的 EV-UAV-compatible 指标。每次训练内验证还使用独立固定种子的采样器，按与训练一致的正样本保留、hard-negative 背景采样、密集序列权重和完整事件 Fa 上下文计算 loss；五个分量写入 `history.json[*].validation_loss`，不会混入 `metrics.json`。训练时选出的事件 IoU 阈值随最佳 checkpoint 保存，复评不会再次使用标签调阈值。checkpoint 同时保存模型配置和验证集 manifest 摘要，配置或数据不匹配时拒绝复评。
