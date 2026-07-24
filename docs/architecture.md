# HUAV 多尺度双视图事件分类架构

## 1. 输入与投影

一个 batch 的原始事件表示为 `events[N,4] = (x,y,t_norm,p)`，另有 `batch_indices[N]`。`t_norm` 是 8 秒序列内的归一化时间。

`MultiScaleEventProjector` 对每个时间尺度 `T` 生成：

- `XT[B,4,T,346]`：在 y 方向聚合；
- `YT[B,4,T,260]`：在 x 方向聚合。

四个通道依次是正极性计数、负极性计数、平均归一化时间、最近归一化时间。模型输入计数经过 `log1p` 压缩，但不会生成 `[B,T,260,346]` 的三维稠密体。

## 2. 独立编码器

每个尺度先由独立浅层卷积映射到 `C=8`，双线性对齐到最高时间分辨率后拼接融合。XT 与 YT 使用两个完全不共享权重的 `HierarchicalViewEncoder`：

1. 高分辨率 3×3 CNN；
2. 非重叠局部窗口自注意力；
3. stride-2 CNN 下采样；
4. 低分辨率局部窗口自注意力；
5. FPN lateral + top-down 上采样；
6. 高分辨率平滑卷积。

因此最终保留 `[B,C,T,346]` 与 `[B,C,T,260]` 的高分辨率轨迹特征。

## 3. 跨视图交互

每个视图先只沿空间轴求平均，得到 `[B,T,C]` 时间 token。双向 Cross-Attention 使用布尔 mask，只允许时间距离不超过 `cross_time_radius` 的 token 互相关注：

```text
XT token = Query, YT token = Key/Value
YT token = Query, XT token = Key/Value
```

交互结果通过可学习门控广播回各自空间轴。这里没有对 XT 全部位置和 YT 全部位置做全局笛卡尔注意力。

## 4. 回到原始事件

对于查询事件 `(x_i,y_i,t_i,p_i)`：

- 在 XT 特征图按 `(x_i,t_i)` 使用 `grid_sample` 双线性插值；
- 在 YT 特征图按 `(y_i,t_i)` 使用 `grid_sample` 双线性插值。

查询保持原事件的 x/y 对应关系。最终事件特征为：

```text
[f_xt, f_yt, f_xt * f_yt, abs(f_xt - f_yt), polarity, normalized_time]
```

MLP 输出 `event_logits[N]`。默认查询全部事件；训练器可传 `query_indices` 仅查询“全部正事件 + 最多 10 倍背景”，而投影图和辅助标签仍由完整序列构造。

## 5. 输出与监督

辅助 1×1 卷积头输出：

- `xt_logits[B,1,T,346]`
- `yt_logits[B,1,T,260]`

正事件被投影到两个平面，并以二维高斯核扩展为软标签。事件主头、两个辅助头、两个单视图事件头共同支持 Focal、Dice、跨视图 L1 一致性和弱时间平滑损失。

总损失中的各项采用正权重相加，两个单视图事件头另有直接标签损失。配置默认让 epoch 1 的 consistency/temporal scale 为 0，epoch 2 为 0.5，epoch 3 为 1.0。

## 6. 评测

验证按文件依次推理，但每个文件的每个事件都进入最终混淆矩阵。阈值可固定，也可在验证候选阈值上按正类 IoU 选择。训练保存验证 IoU 最优 checkpoint，并额外实现 EV-UAV 的 50 ms 目标 Pd 与空间连通分量 Fa 口径。
