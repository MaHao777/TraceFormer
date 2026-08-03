# STC-Focal 与 margin Fa 损失

## 当前数据单位

`NPZEventDataset` 保留两套时间表示：模型输入 `events[:, 2]` 是原始时间除以 `duration_ms=8000` 后的 `[0,1]` 归一化值；`timestamps_ms` 则保留 NPZ 的原始毫秒值。损失邻域使用后者。`events[:, 0:2]` 始终是 346×260 传感器上的原始像素坐标，不是归一化坐标或 Token 内局部坐标。

模型注意力中的前锥/后锥索引不能直接用于本损失：这些边连接的是 `8×8×125 ms` 的稀疏 Token，且只在注意力层内部生成；损失需要一条 logit 对应一个原始事件的事件级支持度。为保持模型主体和输出契约不变，损失在采样事件上用物理坐标单独构造稀疏邻域。

## STC-Focal

对事件 `i`，双向局部锥为：

```text
C_i = {j | batch_j=batch_i,
           |t_j-t_i| <= tau,
           ||xy_j-xy_i||_2 <= r0 + vmax*|t_j-t_i|,
           j != i}
```

实现先按最大锥半径 `r0 + vmax*tau` 和 `tau` 对 `(batch,t,y,x)` 分桶。每个体素确定性均匀保留至多 `max_neighbors` 个代表，只批量检查相邻 27 个体素；每轮合并后保留归一化时空距离最近的 `max_neighbors` 个事件。因此内存随 `N×max_neighbors` 增长，不建立 `N×N` 距离矩阵，也没有逐事件 Python 循环。

邻域支持度为：

```text
K_ij = exp(-d_xy²/(2*sigma_xy²) - d_t²/(2*sigma_t²))
s_i  = sum_j K_ij * sigmoid(z_j).detach() / (sum_j K_ij + eps)
```

无邻居时 `s_i=0`。时空权重及稳定的 focal 形式为：

```text
c_i = eta_pos + (1-eta_pos)*s_i^gamma_stc                 (y_i=1)
c_i = eta_neg + (1-eta_neg)*(1-s_i)^gamma_stc             (y_i=0)

L_stc_focal = mean(alpha_t * c_i * (1-p_t)^gamma_focal
                   * BCEWithLogits(z_i, y_i))
```

邻居概率全部分离梯度，模型无法通过整体抬高背景概率来操纵权重。BCE 使用 `binary_cross_entropy_with_logits`，距离核在 float32 中计算，适用于 float32 和 bfloat16/float16 autocast。

## Margin false-alarm loss

该项始终使用全部原始事件的 `full_logits`。只选择背景事件，按 `(batch, floor(t_ms/50), x, y)` 分组并用 `scatter_reduce_(amax)` 得到每个背景像素的最大 logit。通过两次稳定排序进行 `(frame, logit descending)` 的向量化排名，每帧选择至多 `fa_top_k` 个像素：

```text
L_margin_fa = mean_frame mean_topk(
    temperature * softplus((r_j - negative_margin) / temperature)
)
```

最终训练目标只有：

```text
L_total = L_stc_focal + lambda_fa * L_margin_fa
```

Dice 与 object-frame recall 已从实现、配置、训练调用和日志中移除。

## 默认参数与来源

| 参数 | 默认值 | 单位/依据 |
|---|---:|---|
| `alpha` | 0.55 | 降低旧 `0.75` 的正类偏置 |
| `gamma_focal` | 2.0 | 无量纲 |
| `gamma_stc` | 2.0 | 无量纲 |
| `eta_pos`, `eta_neg` | 0.5 | 无量纲权重下界 |
| `tau` | 50.0 | ms；与比赛局部目标/Fa 帧一致 |
| `r0` | 8.0 | px；生产模型最小空间块宽度 |
| `vmax` | 0.128 | px/ms；最大锥斜率 `2×8 px / 125 ms` |
| `sigma_xy` | 8.0 | px；一个最小空间块 |
| `sigma_t` | 25.0 | ms；半个局部帧 |
| `max_neighbors` | 32 | 每事件邻居上限 |
| `lambda_fa` | 0.5 | Fa 项权重 |
| `fa_top_k` | 32 | 每 batch/帧背景像素数 |
| `fa_negative_margin` | -1.0 | logit |
| `fa_temperature` | 0.25 | logit 平滑尺度 |

训练集目标轨迹的稳健斜率统计仍保存在 `docs/cone_slope_analysis.json` 与 `.md`；稀疏最大值不会用于放大全局邻域。
