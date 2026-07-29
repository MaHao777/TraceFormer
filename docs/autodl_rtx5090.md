# AutoDL RTX 5090 训练

本仓库的 RTX 5090 配置保持当前 score-oriented 模型、`8x8/64` 锥体几何、原始事件顺序、损失和官网评分口径不变，只把运行设备切换为 CUDA，并启用 BF16 自动混合精度。RTX 5090 的 compute capability 是 12.0，因此 PyTorch 必须使用 CUDA 12.8 或更新构建。

## 环境准备

代码和数据建议放在高速数据盘，例如 `/root/autodl-tmp/HUAV`：

```bash
cd /root/autodl-tmp/HUAV
python -m pip install -e ".[dev]"
python -m huav.cli.check_gpu --device cuda
```

预检必须报告 RTX 5090、`compute_capability: 12.0`、包含 `sm_120` 的 architecture 列表和 `cuda_kernel_verified: true`。若预检失败，不要开始训练；安装 PyTorch 官方 CUDA 12.8 或更新 wheel 后重新检查。

## 强制 smoke 闭环

先用最大训练序列 `data/ev-uav/train/train_094.npz` 做一次 forward、loss、backward 资源复现：

```bash
python scripts/check_largest_sequence.py \
  --config configs/huav_adaptive_hierarchical_autodl.yaml \
  --initialize-detector outputs/<baseline-run>/best.pt \
  --device cuda
```

确认输出的 loss 有限并记录 CUDA 峰值显存后，再运行：

```bash
bash scripts/autodl_train.sh smoke
```

训练完成后，使用 smoke 输出目录中保存的配置和权重独立复评：

```bash
python -m huav.cli.evaluate \
  --config outputs/<smoke-run>/config.yaml \
  --checkpoint outputs/<smoke-run>/best.pt \
  --device cuda \
  --output outputs/<smoke-run>/reproduced_metrics.json \
  --quiet
```

只有 `metrics.json` 和 `reproduced_metrics.json` 完全一致时才进入全量训练。

## 全量训练、独立复评和提交

建议在 `screen` 或 `tmux` 中执行，避免 SSH 断线终止进程：

```bash
bash scripts/autodl_train.sh full
```

全量配置从头训练 18 epoch，使余弦学习率在完整训练区间内连续衰减；第 6 epoch 与后续 12 epoch 的结果都保存在同一个 `history.json` 中，最佳 checkpoint 按验证集官网总分选择。全量训练必须覆盖 99 个训练文件和 24 个验证文件。完成后使用该轮 `config.yaml` 与 `best.pt` 独立复评，并比较 SHA-256：

```bash
RUN_DIR=outputs/<full-run>
python -m huav.cli.evaluate \
  --config "$RUN_DIR/config.yaml" \
  --checkpoint "$RUN_DIR/best.pt" \
  --device cuda \
  --output "$RUN_DIR/reproduced_metrics.json" \
  --quiet
sha256sum "$RUN_DIR/metrics.json" "$RUN_DIR/reproduced_metrics.json"
```

两个哈希一致后，才能用同一个 checkpoint 生成提交：

```bash
SUBMISSION_DIR=outputs/submission_<timestamp>
python -m huav.cli.predict \
  --config "$RUN_DIR/config.yaml" \
  --checkpoint "$RUN_DIR/best.pt" \
  --output-dir "$SUBMISSION_DIR" \
  --device cuda
```

最终还需检查 24 个 TXT、共 1,424,330 行、每行五列、原始 `x y t p` 值与顺序不变、标签仅为 0/1，以及 `submission.zip` 根目录只含 TXT。
