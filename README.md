# HUAV：事件相机微小目标检测

本仓库实现了一个面向原始事件的双视图分类模型。模型不会构建稠密三维 `x-y-t` 体素，也不会把两个投影视图的概率图相乘；它先学习 `x-t` 与 `y-t` 轨迹特征，再按每个原始事件自己的 `(x,y,t)` 坐标查询两个视图，最终输出一条事件对应一个 logit。

## 架构

```text
原始事件 (x,y,t,p)
  → 8 秒时间归一化
  → 多尺度 x-t / y-t 投影（正计数、负计数、平均时间、最近时间）
  → 两个不共享参数的 CNN + 局部窗口注意力 + FPN 编码器
  → 相同/相邻时间 bin 内的双向 Cross-Attention
  → grid_sample 查询原始事件的 (x,t) / (y,t) 特征
  → [xt, yt, xt*yt, |xt-yt|, p, t] 事件分类头
```

默认时间尺度为 32、16、8 个 bin。`XTEncoder` 和 `YTEncoder` 结构相同但参数独立；注意力仅在局部二维窗口及相邻时间位置内计算。详细张量流程见 [docs/architecture.md](docs/architecture.md)。

模型 `forward` 返回：

```python
{
    "event_logits": ...,
    "xt_logits": ...,
    "yt_logits": ...,
    "xt_event_logits": ...,
    "yt_event_logits": ...,
    "event_features": ...,
}
```

## 数据与采样

数据根目录默认为同级目录 `../训练集、验证集`，其中包含 `train/*.npz` 与 `val/*.npz`。每个文件读取结构化数组 `ev[x,y,t,p,label,name]`：

- `t` 按 8000 ms 归一化到 `[0,1]`；
- 事件顺序、事件数量、原始毫秒时间戳和目标 ID 全部保留；
- 训练时保留所有正事件，负事件最多取正事件数的 10 倍；
- 验证时不下采样，始终预测每一个原始事件。

当前数据统计：99 个训练序列，共 9,324,544 个事件，正事件比例 2.939790%；24 个验证序列，共 1,424,330 个事件。

## 损失与两阶段训练

总损失为正权重相加：

```text
Total = Event + 0.25 * ViewEvent + 0.5 * View
        + 0.1 * Consistency + 0.01 * Temporal
```

- `Event`：事件级 Focal + Dice；
- `ViewEvent`：两个单视图事件头各自的 Focal + Dice 标签监督；
- `View`：x-t/y-t 高斯软投影标签上的 Focal + Dice；
- `Consistency`：两个单视图事件概率之间的 L1；
- `Temporal`：相邻时间 bin 辅助概率的弱平滑项。

第一个 epoch 只启用事件级和视图损失；第二、三个 epoch 将一致性与时间连续性损失按 `0.5 → 1.0` 渐进加入。两个单视图事件头也直接接受事件标签监督，避免只靠相互一致性收敛到共同错误。所有权重、warmup/ramp epoch 和背景比例都可在 YAML 中修改。

## 安装与训练

```powershell
python -m pip install -e .
huav-inspect --help
huav-train --config configs/huav_base.yaml --device cpu
```

不安装 console script 时也可直接执行：

```powershell
python -m huav.cli.train --config configs/huav_base.yaml --device cpu
```

小规模真实数据烟雾训练：

```powershell
python -m huav.cli.train --config configs/huav_cpu_smoke.yaml --device cpu
```

独立复评 checkpoint：

```powershell
python -m huav.cli.evaluate `
  --config configs/huav_base.yaml `
  --checkpoint outputs/cpu_full_final_20260723/best.pt `
  --device cpu
```

每次训练在 `outputs/cpu_<timestamp>/`（或 `--output-dir` 指定目录）中写入：

- `best.pt`：验证 IoU 最优权重；
- `config.yaml`：本次完整配置快照；
- `history.json`：逐 epoch 损失和验证指标；
- `metrics.json` / `metrics.md`：最佳 checkpoint 指标；
- `summary.json`：运行摘要。

## 指标口径

`metrics.json` 的 `event` 字段给出用户可直接比较的事件级指标：

- `Pd = TP / (TP + FN)`；
- `Fa = FP / (FP + TN)`；
- `IoU = TP / (TP + FP + FN)`；
- `Acc = (TP + TN) / All`。

同时输出 `ev_uav_compatible`：按同目录 EV-UAV 参考代码使用 50 ms 等效帧、目标实例检出比例计算 Pd，并把背景误报的八连通分量数除以等效帧数和 `346×260` 传感器面积得到 Fa。两组指标分开命名，避免把官方的目标级/空间虚警口径误当成普通二分类指标。

本机 CPU 最终完整训练结果位于 `outputs/cpu_full_final_20260723/`。最佳权重来自辅助损失完全启用的 epoch 3，完整 24 文件验证结果为：

| Pd | Fa | IoU | Acc |
| ---: | ---: | ---: | ---: |
| 0.845495 | 0.004137 | 0.778678 | 0.988948 |

阈值是在验证集上从配置的候选阈值中按正类 IoU 选择得到的 `0.65`，并已写入 `best.pt`；独立复评不会再次借助标签调阈值。该结果用于初次架构可行性检查，不等同于独立测试集成绩。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m pytest
```

测试覆盖 NPZ 契约、多尺度投影、软标签、双视图前向与梯度、事件采样、四项损失、指标、CLI 和 CPU 训练/checkpoint 复评。
