# HUAV 固定 schema-7 事件检测模型

本仓库当前只保留一套正式模型结构：

```text
EventTokenizer
  → 双向时空锥 Transformer
  → Full/Sparse Token 混合
  → 事件解码器
  → 多尺度事件精修头
  → 每个原始事件的二分类 logit
```

模型结构固定为 schema-7。稀疏混合模块和多尺度精修头始终构造并参与推理，不再通过配置选择性增减模块。对不满足密集、极性平衡条件的序列，不生成 Sparse 子流，但模型结构和 checkpoint 参数不变。

## 安装与检查

```powershell
python -m pip install -e .
python -m pip install -e ".[dev]"

huav-train --help
huav-evaluate --help
huav-predict --help
huav-check-gpu --help
```

运行测试与静态检查：

```powershell
python -m unittest discover -s tests -v
python -m pytest
python -m ruff check src tests scripts
git diff --check
```

## 固定配置

仅保留四份当前模型配置：

- `configs/huav_score_optimized.yaml`：本地 CPU 正式配置
- `configs/huav_score_optimized_smoke.yaml`：本地 CPU 冒烟配置
- `configs/huav_score_optimized_autodl.yaml`：CUDA 正式配置
- `configs/huav_score_optimized_autodl_smoke.yaml`：CUDA 冒烟配置

四份配置的模型结构完全相同，只允许设备、混合精度、线程、数据文件上限和输出目录等运行环境字段不同。训练配置中的 `stage` 只控制哪些参数参与优化，不改变模型组成：

- `full`：训练完整固定结构
- `sparse_mixture`：冻结主干和精修头，只训练稀疏混合
- `refinement`：冻结主干和稀疏混合，只训练多尺度精修头

当前正式配置固定为 `refinement`。新建这一阶段的训练必须通过 `--initialize-refinement` 明确提供基线 checkpoint；精确续训只使用 `--resume-state`，它会恢复模型、优化器、学习率控制器和随机数状态。

## 训练、复评与预测

精修阶段示例：

```powershell
huav-train `
  --config configs/huav_score_optimized.yaml `
  --initialize-refinement path/to/schema5_best.pt `
  --output-dir outputs/fixed_schema7_refinement
```

从完整训练状态继续：

```powershell
huav-train `
  --config path/to/run/config.yaml `
  --resume-state path/to/run/training_state.pt `
  --output-dir outputs/resumed_run
```

复评和生成预测时，应优先使用 checkpoint 同目录保存的 `config.yaml`：

```powershell
huav-evaluate `
  --config path/to/run/config.yaml `
  --checkpoint path/to/run/best.pt `
  --device cpu

huav-predict `
  --config path/to/run/config.yaml `
  --checkpoint path/to/run/best.pt `
  --device cpu
```

schema-7 加载会严格检查真实模型结构、分辨率、Sparse 推理门限和精修头结构。历史 checkpoint 元数据中的 `causal_mode=false`、`debug_attention_stats=false`、`enabled=true` 和冻结字段只会被识别为旧记录，不会恢复成运行时开关。

## 输出产物

训练目录保存：

- `best.pt`：固定 schema-7 模型及推理契约
- `config.yaml`：本次运行的完整配置
- `history.json`、`metrics.json`、`metrics.md`
- `dataset_manifest.json`
- `training_state.pt` 与可选的分 epoch 状态
- `summary.json`、`experiment.md`、`training_curves.png`

详细结构见 [docs/architecture.md](docs/architecture.md)。历史实验计划和淘汰方案只作为记录保存在 `docs/superpowers/plans/` 与既有输出目录中，不再进入当前配置和代码路径。
