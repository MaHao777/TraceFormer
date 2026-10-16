# GPU 推理适配

CPU 仍是默认后端。GPU 后端只改变运行位置，保持权重、网络结构、阈值、检查点 schema、缓存格式和事件顺序。训练与 OOF 默认仍使用原 CPU 树模型。

## 实现范围

- 邻域：每个事件视图按原输入顺序准备 SciPy KD-tree，一次打包到 GPU；各层在 GPU 进行 Chebyshev 查询和 rank/time/distance 排序。使用原叶序和堆的等距比较规则，避免改为任意 top-k。树结构不跨前向缓存。
- 事件树：加载原 LightGBM 文件，保留 float64 阈值、缺失值规则、叶值和树顺序；每个事件在 CUDA 上顺序累计 float64，再输出原 float32 分数。不重新训练。仅支持本项目发布的数值分裂 binary 模型，不支持的目标或 categorical 分裂报错。
- 特征：分组、稳定事件排序、概率统计、轨迹窗口和父 token 展开放在 GPU。log/sqrt 及整数 log1p 查表沿用 NumPy，保留 CPU 舍入和发布缓存的 float16 token 量化。仍以 200,000 行展开事件特征。
- 归档：冻结 schema 5/4 的源码完整保留，运行时 adapter 只替换邻域查询，并在一次 Encoder 调用中共享几何。模型参数和 state_dict 键不变。
- 轨迹跟踪、参考恢复、直接验证器及旧组件级模型保留原 CPU 实现；所有输出仍按原规则生成。

## GPU 环境与执行

在有 NVIDIA GPU 的环境先安装兼容显卡的 CUDA 版 PyTorch，然后在仓库目录执行：

```bash
python -m pip install -e ".[gpu,dev]"
traceformer-check-gpu --device cuda
```

`gpu` 可选依赖使用 `numba-cuda[cu12]`。CUDA 13 环境可按 [Numba 官方安装说明](https://nvidia.github.io/numba-cuda/user/installation.html)选择 `numba-cuda[cu13]`。内核通过 [CUDA Array Interface](https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html)直接使用 Torch Tensor 的显存，并在 Torch 当前 stream 排队；不要求额外的 CuPy 或 nvForest。

PyTorch `2.8.0+cu128` 环境需将工具链对齐到 CUDA 12.8：

```bash
python -m pip install -e ".[gpu,dev]" "cuda-toolkit==12.8.1"
```

仅安装最新 `cu12` extras 会选择 CUDA 12.9 编译器，而 PyTorch 可能已加载 12.8 nvJitLink，导致真实 JIT 失败。链接库版本必须不低于输入版本，详见 [NVIDIA nvJitLink 兼容规则](https://docs.nvidia.com/cuda/nvjitlink/index.html#compatibility)。以 `torch.version.cuda` 和实际加载的库为准，`nvidia-smi` 的 CUDA 数字表示驱动支持范围。

### 先做真实 GPU 验收

```bash
python scripts/check_gpu_inference.py --output outputs/gpu_check/report.json
```

默认使用小型无标签合成事件、当前 schema-7 权重和发布 large 树。也可通过 `--sequence /path/to/test_023.npz` 检查最大的真实序列；输入标签与身份不会进入推理。检查逐层邻居、同一 CUDA 网络使用 CPU/GPU 邻域时的全部 Tensor 输出、112 维特征和两种特征后端的树分数。只有全部逐位一致时状态为 `verified`。

完整验证集和测试集复用只读权重进行检查，预测写出后再独立评分；验证数据须与检查点保存的 manifest 相同：

```bash
python scripts/check_gpu_inference.py --split all --dataset /path/to/EV-UAV-dataset \
  --output outputs/gpu_check/all_sequences.json
```

CPU 环境可检查同一内核的逻辑，但不能据此宣称 GPU 编译或速度通过：

```bash
python scripts/check_gpu_inference.py --simulate --output outputs/gpu_check/simulator.json
```

### 冻结完整系统

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/benchmark_frozen_system.py \
  --dataset /path/to/EV-UAV-dataset \
  --frozen outputs/paper_test_20261009 \
  --output outputs/gpu_benchmark/runtime_report.json \
  --device cuda --neighbor-backend cuda \
  --prediction-backend cuda --feature-backend cuda \
  --prediction-threads 8 --num-threads 14 --repeats 3 \
  --processor 'actual host CPU'
```

这条命令使用原归档权重、模型哈希、阈值和恢复规则；包含一次完整预热。任何阶段掩码与冻结结果不同都会报错，不能将有差异的结果标为通过。计时包含原 CPU 跟踪、恢复以及设备传输；报告记录邻域、树和特征后端。CUDA 树使用 128 threads/block，`prediction-threads` 只控制 CPU LightGBM。

### 当前模型、缓存和 Track 2

普通神经预测和验证增加 `--neighbor-backend cuda`，并同时指定 `--device cuda`：

```bash
traceformer-evaluate --config configs/traceformer_score_optimized.yaml \
  --checkpoint outputs/cpu_refinement_20260830_154410/best.pt \
  --device cuda --neighbor-backend cuda --output outputs/gpu_check/validation.json
```

`traceformer-predict`、`traceformer-postprocess cache` 支持相同邻域参数。

`traceformer-postprocess predict` 和 `scripts/evaluate_postprocess_stages.py` 接收 `--prediction-backend cuda --feature-backend cuda`，从原 CPU 缓存读取并在 GPU 上做事件树预测。`scripts/predict_postprocess_track2.py` 同时支持 `--device cuda --neighbor-backend cuda --prediction-backend cuda --feature-backend cuda`；参考预测文件接口、验证门槛和 TXT/ZIP 审计仍保留。

## 验证结果与边界

完整 pytest：278 passed、2 skipped；16 项 CUDA simulator 对照通过，Ruff 和 `git diff --check` 通过。跳过项包含真实 CUDA 测试；最后补充的后端、默认行为和不支持模型类型检查为 5 passed。检查记录保存于 `outputs/gpu_adaptation_20261013/checks.json`。

本机 PyTorch 2.11.0+cpu，没有可用 CUDA。Simulator 执行的是与 GPU 相同的内核代码，用来检查多 batch、缺失时间片、重复/等距点、半径边界、不同层尺度、缺失值路由、稳定累计、完整特征和真实发布树分数。

2026-10-13 已在 RTX 3090 24GB、PyTorch 2.8.0+cu128 上完成真实验收：完整 pytest 为 280 passed、1 skipped，真实 CUDA 内核用例为 16 passed；当前模型完整 val/test 共 48 个序列逐项通过，冻结完整系统全部 24 个测试序列各跑三轮，阶段掩码及评分不变。同机 CPU 平均 12.950 秒/序列，完整 CUDA 后端平均 0.872 秒/序列，提速 14.85 倍。计时包括设备传输、CPU 跟踪及恢复，不包括加载、JIT、磁盘 I/O 和评分。环境及细项见 [真实 GPU 验证报告](gpu_validation_20261013.md)。

CUDA KD 查询队列最多 128 个待访问节点，溢出显式报错，不能静默截断邻居。GPU 上下文最多处理 2^24 个事件，确保轨迹整数计数与原 float32 计数一致；当前最大完整训练序列 625,178 个事件在范围内。
