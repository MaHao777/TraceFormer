# TraceFormer 固定 schema-8 事件检测模型

当前后处理已删除 reference recovery，最终使用直接分支与主轨迹并集；[48 序列独立复评与变更说明](docs/reference_recovery_removal_20261016.md)。论文冻结 schema-5 系统与下述当前 schema-8 模型分开计分。

项目地址：https://github.com/MaHao777/TraceFormer

## 当前主模型

当前主模型固定为新三锥网络及其重训事件后处理：189,900个神经网络参数，
small/large两套树，以及主轨迹与直接分支并集。
[主模型清单](configs/main_model.json)固定权重、原配置和后处理文件哈希，
本地产物在`outputs/main_model`；[使用与核验说明](docs/main_model_20261018.md)。
这是用户指定的主模型，原始验证Fa门槛失败记录仍保留。

默认预测运行完整主模型，无需填写权重和后处理路径：

```powershell
traceformer-predict --device cpu --predictions-only
# 云端使用同一完整管线，网络、特征和树均在GPU运行。
traceformer-predict --device cuda --input-dir /path/to/test --output-dir outputs/main_test
```

默认生成NPZ掩码及经过审计的TXT/ZIP；`--predictions-only`只保存掩码。
`--split val`选择验证集。显式`--checkpoint`用于单网络预测，不包含后处理。
CUDA无梯度推理已默认采用16384分块与层间索引复用，轨迹形态学已改为局部画布；
训练仍用1024分块，模型结构和checkpoint张量键不变。

已合入代码的GPU复核完成：单网络**168.973 ms/文件**、完整管线**442.657 ms/文件**，
全部24测试文件各三轮；48序列精确对照、实际默认入口、独立复评及TXT/ZIP审计通过。
[最新GPU实测与完整回传记录](docs/main_model_gpu_verification_20261018.md)。

## 新三锥网络的事件后处理重训

本地重训入口只读取 train/val，冻结指定网络，用79/20开发划分选择事件树融合与
轨迹阈值，通过后用全部99训练序列发布，并对24个验证序列独立复评：

```powershell
python scripts/retrain_event_postprocess.py --config outputs/prefix_cone/cpu_20261017_full/config.yaml --checkpoint outputs/prefix_cone/cpu_20261017_full/best.pt --output outputs/postprocess_prefix_cone_new
```

`status.json` 记录开发/验证门槛结果。只有 `validation_accepted` 才可运行
`python scripts/package_event_release.py --run-dir outputs/postprocess_prefix_cone_new`。
云端包包含冻结产物、验证掩码、GPU预测/计时命令；本地重训不运行测试集。
事件范围也可通过 `traceformer-postprocess train --event-only` 单独执行。

优化前，2026-10-18在现有RTX 3090服务器完成新模型及重训后处理候选的完整诊断评测，
复用原有数据，并按历史配置对齐为Torch/OMP/MKL 14线程、OpenBLAS 1线程、CPU树参数8。
测试集三轮平均耗时为单模型536.679 ms/文件、完整管线1202.171 ms/文件；
[14线程实测范围、数值核验与结果](docs/gpu_prefix_cone_14t_20261018.md)。
[原8线程记录](docs/gpu_prefix_cone_20261018.md)保留，候选仍保留原验证门槛未通过状态。

[耗时回归定位](docs/prefix_cone_latency_diagnosis_20261018.md)确认默认1024 Token分块的前缀池化是主要瓶颈；
临时16384分块试验使完整管线降至919.527 ms/文件，val/test全部网络Tensor与原输出逐位一致。
上述诊断时尚未改动默认实现；优化现已合入，见主模型说明。

[完整管线进一步提速审计](docs/full_pipeline_speed_audit_20261018.md)验证了轨迹形态学局部裁剪与层间几何索引复用：
在16384分块基础上，同次三轮对照由955.015降至444.187 ms/文件，输出及独立评分不变。
上述为独立进程原型的历史记录；当前代码已合入这两项优化。

Python 包为 `traceformer`，模型入口为 `from traceformer import TraceFormer`，配置类为 `TraceFormerConfig`。命令统一使用 `traceformer-` 前缀。更名前的本地目录名与历史实验记录可继续保留；checkpoint 的张量键和数值结构不因名称变更而改变。

本仓库当前只保留一套正式模型结构：

```text
EventTokenizer
  → 双向时空锥 Transformer
  → 事件解码器
  → 多尺度事件精修头
  → 每个原始事件的二分类 logit
```

模型结构固定为 schema-8，包含 Tokenizer、三锥编码器、事件解码器和多尺度精修头，共 172,150 参数。Full/Sparse 双视图融合及其专属监督已经删除，所有序列均直接处理完整事件流。历史主模型产物保留原字节，通过对应归档源码复现。

## 安装与检查

实验统一使用 `traceformer-experiment`，维护 M1+C4、六组池化/局部注意力、四组欧氏锥对照及S1论文筛选消融。
启动、分阶段运行、续跑、独立上传包和历史归档见[实验使用导航](docs/experiments/README.md)。
旧实验已分类到 `archive/experiments/`，历史冻结源码与结果保留原位。

```powershell
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[experiments]"

traceformer-train --help
traceformer-evaluate --help
traceformer-predict --help
traceformer-check-gpu --help
traceformer-experiment --help
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

- `configs/traceformer_score_optimized.yaml`：本地 CPU 正式配置
- `configs/traceformer_score_optimized_smoke.yaml`：本地 CPU 冒烟配置
- `configs/traceformer_score_optimized_autodl.yaml`：CUDA 正式配置
- `configs/traceformer_score_optimized_autodl_smoke.yaml`：CUDA 冒烟配置

四份配置的模型结构完全相同，只允许设备、混合精度、线程、数据文件上限和输出目录等运行环境字段不同。训练配置中的 `stage` 只控制哪些参数参与优化，不改变模型组成：

- `full`：训练完整固定结构
- `refinement`：冻结主干，只训练多尺度精修头

当前正式配置固定为`full`。如需训练`refinement`阶段，应先在配置中明确指定该阶段，并通过`--initialize-refinement`提供基线checkpoint；精确续训使用`--resume-state`恢复模型、优化器、学习率控制器和随机数状态。

## 训练、复评与预测

新建完整模型训练示例（不改动已冻结的主模型产物）：

```powershell
traceformer-train `
  --config configs/traceformer_score_optimized.yaml `
  --output-dir outputs/new_prefix_cone_training
```

从完整训练状态继续：

```powershell
traceformer-train `
  --config path/to/run/config.yaml `
  --resume-state path/to/run/training_state.pt `
  --output-dir outputs/resumed_run
```

复评和生成预测时，应优先使用 checkpoint 同目录保存的 `config.yaml`：

```powershell
traceformer-evaluate `
  --config path/to/run/config.yaml `
  --checkpoint path/to/run/best.pt `
  --device cpu

traceformer-predict `
  --config path/to/run/config.yaml `
  --checkpoint path/to/run/best.pt `
  --device cpu
```

schema-8 加载严格检查模型结构、分辨率和精修头结构。带 Sparse 参数的历史 schema-7 checkpoint 不作为新结构加载；精确续训只接受匹配的 schema-4 training-state。

## 输出产物

训练目录保存：

- `best.pt`：固定 schema-8 模型及推理契约
- `config.yaml`：本次运行的完整配置
- `history.json`、`metrics.json`、`metrics.md`
- `dataset_manifest.json`
- `training_state.pt` 与可选的分 epoch 状态
- `summary.json`、`experiment.md`、`training_curves.png`

详细结构见 [docs/architecture.md](docs/architecture.md)。历史实验计划和淘汰方案只作为记录保存在 `docs/superpowers/plans/` 与既有输出目录中，不再进入当前配置和代码路径。
