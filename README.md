# HUAV：稀疏时空锥事件分割

本仓库使用完整 8 秒事件流，直接为每个原始事件预测微小目标概率。当前模型不再使用双视图投影、多尺度输入、FPN、辅助分割头或跨视图损失；核心是稀疏三维块 Token 和双向多锥中心注意力。

## 架构

```text
原始事件 (batch, x, y, t, p)
  → 仅对非空 (x_block, y_block, t_block) 生成 Token
  → Event MLP + 均值/最大池化 + 块统计 + 位置编码
  → 过去锥 / 未来锥分别做稀疏中心注意力
  → Token 条件的双向门控融合
  → 父 Token + 事件局部坐标/极性/块统计
  → 每个原始事件一个 logit
```

模型不会创建完整三维体素，也不会创建 `M×M` 注意力矩阵。邻域按 batch 和时间切片建立 `cKDTree`，使用 Chebyshev 空间距离，并在每个头、每个方向上限制邻居数量。完整张量流程见 [docs/architecture.md](docs/architecture.md)。

`forward` 返回：

```python
{
    "event_logits": event_logits,
    "token_features": updated_tokens,
    "token_coords": token_coords,
    "token_batch_indices": token_batch_indices,
    "event_token_indices": event_token_indices,
    "past_attention_stats": past_attention_stats,
    "future_attention_stats": future_attention_stats,
}
```

输入既支持 `[N,5] = [batch,x,y,t,p]`，也支持 `[N,4]` 加独立 `batch_indices`。模型层面支持变长 batch、空事件、无目标事件、因果模式和 CUDA 混合精度。

## 数据驱动的锥参数

对 99 个训练序列、274,122 个正事件按 `4×4` 像素和 128 个时间块分析目标中心轨迹。Chebyshev 斜率的中位数及 90/95/99 分位为：

| q50 | q90 | q95 | q99 |
|---:|---:|---:|---:|
| 0.640625 | 1.250000 | 1.446429 | 1.875000 |

单位是“每相隔一个时间块移动多少空间块”。据此使用四种锥斜率 `0.5 / 1.0 / 1.5 / 2.0` 和基础半径 `1 / 1 / 2 / 2`。完整方法、逐时间距离和逐序列结果见 [docs/cone_slope_analysis.md](docs/cone_slope_analysis.md) 与 `docs/cone_slope_analysis.json`。

CPU 完整训练配置使用 `8×8` 像素、64 个时间块；时间和空间同时扩大 2 倍，因此一个斜率单位仍对应约 `64 px/s`，实测斜率数值无需换算。该配置把训练集非空 Token 总量从约 613 万降到约 274 万，同时保留事件级解码。

## 数据与损失

默认数据根目录为 `../训练集、验证集`，包含 `train/*.npz` 和 `val/*.npz`。每个文件读取结构化数组 `ev[x,y,t,p,label,name]`；时间按 8000 ms 归一化，事件顺序和数量保持不变。

训练只使用：

```text
Total Loss = Focal Loss + lambda_dice × per-sample Dice Loss
```

训练时保留全部正事件，并为损失最多选择 10 倍背景事件。所有原始事件仍参与 Token 生成、锥上下文建模和事件解码；背景采样只选择哪些 logits 进入损失。验证始终使用全部事件。

## 安装、分析与训练

```powershell
python -m pip install -e .
huav-inspect --help
huav-train --config configs/huav_base.yaml --device cpu
```

复现锥斜率分析：

```powershell
python -m huav.analysis.cone_slopes data/ev-uav/train `
  --json docs/cone_slope_analysis.json `
  --markdown docs/cone_slope_analysis.md
```

真实数据烟雾训练：

```powershell
python -m huav.cli.train `
  --config configs/huav_cpu_smoke.yaml `
  --device cpu
```

独立复评 checkpoint：

```powershell
python -m huav.cli.evaluate `
  --config outputs/<run>/config.yaml `
  --checkpoint outputs/<run>/best.pt `
  --device cpu
```

每次训练写入：

- `best.pt`：验证事件 IoU 最优的权重、阈值和模型配置；
- `config.yaml`：完整配置快照；
- `dataset_manifest.json`：所有训练/验证 NPZ 的大小和 SHA-256；
- `history.json`：逐 epoch 的 focal、Dice、总损失和验证指标；
- `metrics.json` / `metrics.md`：最佳 checkpoint 指标；
- `summary.json`：设备、参数量、epoch 和最佳结果。

## 指标口径

`metrics.json` 的 `event` 字段包含：

- `Pd = TP / (TP + FN)`；
- `Fa = FP / (FP + TN)`；
- `IoU = TP / (TP + FP + FN)`；
- `Acc = (TP + TN) / All`。

`ev_uav_compatible` 另按 50 ms 等效帧、目标实例检出比例和背景误报连通分量计算，避免把目标级/空间虚警口径与事件二分类指标混用。验证集阈值只在训练阶段选择并写入 checkpoint；独立复评直接使用保存阈值。

## 完整训练结果

2026-07-24 在本机 CPU 上完成 99 个训练文件、24 个验证文件、3 个 epoch 的全量训练，耗时 33 分 17 秒。模型共 30,570 个参数，最佳权重来自 epoch 2：

| Pd | Fa | IoU | Acc | Precision |
|---:|---:|---:|---:|---:|
| 0.873279 | 0.002452 | 0.831009 | 0.991833 | 0.944959 |

最佳阈值为 `0.85`。独立重新加载 checkpoint 后，`reproduced_metrics.json` 与训练期 `metrics.json` 逐字段完全一致。

相较仓库此前双视图完整验证记录，IoU 从 `0.778678` 提升到 `0.831009`，事件 Fa 从 `0.004137` 降至 `0.002452`。逐 epoch 损失、混淆矩阵、EV-UAV-compatible 指标、运行环境和所有产物路径见 [docs/cone_training_results_20260724.md](docs/cone_training_results_20260724.md)。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m pytest
python -m ruff check src tests
```

测试覆盖真实 NPZ 契约、斜率分析、稀疏 Token 映射、batch 隔离、过去/未来锥、多锥头、空锥、因果模式、事件级解码、Focal/Dice 反向传播、训练产物和 checkpoint 独立复评。
