# TraceFormer

TraceFormer 直接对事件流中的微小无人机进行逐事件分割。论文主模型 S1 包含空间 Tokenizer、过去三尺度锥摘要编码器、事件解码器和多尺度神经精修头，每个原始事件输出一个概率及二值标签，不使用外部后处理。

[English README](README.md)

## 模型

输入分辨率为 346 × 260，每段 8 秒。仅为有事件的 8 × 8 像素 × 125 ms 父块构建 64 维 Token。Tokenizer 对六维事件特征编码，在父块内 4 × 4 子格求均值，经单个 3 × 3 depthwise 卷积及 GELU 后按事件数回读，再结合七项块统计和位置编码。

三层编码器使用 Token 坐标下 `(半径, 时间跨度)` 为 `(8,2)/(4,4)/(2,8)` 的过去方截面锥。源 Token 门控得到三尺度摘要，中心 Query 通过四头注意力融合。解码器将上下文映射回原事件；神经精修头再加入 2 × 2 × 10 ms、4 × 4 × 25 ms、8 × 8 × 50 ms 上下文。

S1 共 **143,350 个可训练参数**：Tokenizer 14,400、编码器 113,987、解码器 6,785、精修头 8,178。编码器只读取过去支持域；全序列统计及精修头仍使用其他时间信息，因此完整网络属于离线序列模型。

## 安装

使用 Python 3.10–3.12。云端先安装适配服务器的 CUDA PyTorch，再在本仓库根目录运行：

```bash
python -m pip install -e .
traceformer-train --help
traceformer-evaluate --help
traceformer-predict --help
```

论文环境为 RTX 4090、PyTorch 2.10.0+cu128、CUDA 12.8、FP32、关闭 AMP/TF32、8 个辅助线程。安装复用已存在的兼容 PyTorch。`--device cpu` 可在 CPU 上执行，但较慢，耗时不能与论文 GPU 数值直接比较。

## 数据和预处理

数据不随代码提供，请准备 EV-UAV NPZ：

```text
/path/to/ev-uav/
  train/*.npz    # 论文使用 99 个标注序列
  val/*.npz      # 24 个标注序列
  test/*.npz     # 24 个序列，评估时需要标注
```

每个文件包含名为 `ev` 的一维结构化数组，至少含整数 `x/y/p` 和数值 `t`。时间单位为**毫秒**，范围 [0,8000]，有限且不递减；坐标在 346 × 260 分辨率内，极性为 0/1。训练及评估还需要整数二分类 `label` 和整数目标 ID `name`，背景 ID 为 0。空序列报错。

预处理保持原始行序，以 `t/8000` 生成网络时间坐标；父块、子格、局部事件特征与统计在线计算。不需要中间体素数据，也不提供其他原始事件格式的转换。

```bash
traceformer-preprocess --data-root /path/to/ev-uav --split train val test --output outputs/data_manifest.json
# 无标注测试集仅校验 x/y/t/p：
traceformer-preprocess --data-root /path/to/ev-uav --split test --unlabeled --output outputs/test_manifest.json
```

该命令复用训练预处理，校验数据并生成文件身份及事件数清单，不改写原始文件。

## 最优预训练权重

`weights/s1_best.pt` 是验证 Score 选择的原始 schema-8、第 **22 轮**权重。结构契约为 `{"variant":"S1","version":4}`，保存阈值 `0.887499988079071`，四舍五入为 0.8875。

SHA-256：

```text
6931ab7529907d50c191a560dcd0e65fbba011689354566056ce4383f7e54e10
```

加载器严格校验结构和时间归一化契约，并加载所有张量。保存元数据中的 `num_pool_queries` 用于保持论文原初始化契约；实际 S1 不含查询池化参数。构造时保留临时初始化随机数消耗，以复现共享参数和 RNG 状态。

## 训练与恢复

```bash
traceformer-train --config configs/s1.yaml --variant S1 --data-root /path/to/ev-uav --device cuda --output-dir outputs/s1
```

默认 seed37、24 轮、AdamW，学习率 0.0007、weight decay 0.0002，原余弦日程，梯度裁剪 1.0，每步读取一个完整序列。Focal/Dice 监督每个目标事件最多采样 12 个背景，困难和随机负样本各占一半；网络前向仍读取完整事件流。联合损失为 focal + 0.75 Dice + 0.25 object-frame + 0.05 false-alarm + 0.2 辅助 cell 损失，保留原稠密序列加权。

每轮验证选择最优权重及阈值。运行产物包括 `best.pt`、`training_state.pt`、逐轮状态、实际配置、数据清单、history 和 metrics。输出目录必须是新目录；恢复也写入新目录：

```bash
traceformer-train --config outputs/s1/config.yaml --variant S1 --data-root /path/to/ev-uav --device cuda --resume-state outputs/s1/training_state.pt --output-dir outputs/s1_resumed
```

恢复模型、优化器、学习率控制器、Python/NumPy/PyTorch 随机状态及采样/顺序生成器。保持变体、数据身份、总训练轮数和训练设置一致；数据内容相同时可以迁移路径。训练状态仅加载可信本地产物。

流程检查可使用新输出目录，并增加 `--device cpu --epochs 1 --max-train-files 2 --max-val-files 2 --num-threads 1`；这不代表复现论文精度。

## 评估与推理

```bash
traceformer-evaluate --data-root /path/to/ev-uav --split val --device cuda --output-dir outputs/s1_val
traceformer-evaluate --data-root /path/to/ev-uav --split test --device cuda --output-dir outputs/s1_test
traceformer-predict --input-dir /path/to/unlabeled_npz --device cuda --output-dir outputs/prediction
```

默认使用 `configs/s1.yaml` 和 `weights/s1_best.pt`。评估新训练或消融权重时，显式传入其保存的 `--config` 和 `--checkpoint`，结构与阈值自动读取 checkpoint。验证评估校验原验证集身份；测试评估只需测试集，推理不需要标注或验证目录。网络不接收标签或目标 ID。

每个 `predictions/<源文件名>.npz` 保存原序 `event_probabilities`（float32）、`prediction`（布尔值）、`source_sha256` 和 `threshold`。`metrics.json` 保存身份、设置及评估时的逐文件和全局指标。输出目录必须尚不存在。

评估不重新选阈值。先累加全局整数计数，再计算：

```text
Score = 0.2 IoU + 0.1 SegAcc + 0.4 Pd + 0.3 exp(-10000 Fa)
```

本基准 SegAcc 为目标事件召回率，Pd 为目标帧实例检出率，Fa 为单位像素帧的假警连通分量率，不是逐事件假阳性率。保留原严格帧边界和参考像素计数语义；另一套 NumPy/SciPy 实现独立核对官方计数。AP 使用全量事件概率。

可读取保存概率独立重评分，不重复前向：

```bash
traceformer-evaluate --data-root /path/to/ev-uav --split test --device cpu --predictions-dir outputs/s1_test/predictions --output-dir outputs/s1_test_rescore
```

## 论文消融

九组共用 `configs/s1.yaml`，训练入口读取 `configs/ablations.yaml` 中所选契约。通过 `--variant` 选择变体并从头训练；只提供 S1 预训练权重，不能用 S1 前向开关代替消融重训。

| ID | 相对 S1 的变化 | 参数量 | 验证 Score | 测试 Score |
|---|---|---:|---:|---:|
| S1 | 完整模型 | 143350 | 0.913654915 | 0.921787678 |
| S0 | 事件编码均值池化 | 142774 | 0.903369256 | 0.915294960 |
| A1 | 移除编码器跨 Token 交互 | 79603 | 0.900421183 | 0.889125897 |
| A2 | 移除神经精修及辅助损失 | 135172 | 0.901181351 | 0.902915097 |
| P1 | 1 × 1 depthwise 核，保留 GELU 和回读 | 142838 | 0.906050406 | 0.915515149 |
| G1 | 近似体积匹配的固定窗口 | 143350 | 0.911572960 | 0.918347501 |
| G2 | 仅保留中间单锥 | 142966 | 0.907179309 | 0.913712190 |
| F1 | 有效摘要均值及 64–128–64 MLP | 143158 | 0.908325546 | 0.916658736 |
| T1 | 去重三锥并集的门控直接注意力 | 130102 | 0.910520349 | 0.913340651 |

```bash
for variant in S1 S0 A1 A2 P1 G1 G2 F1 T1; do
  traceformer-train --variant "$variant" --data-root /path/to/ev-uav --device cuda --output-dir "outputs/$variant"
done
```

结构表使用 S1/S0/A1/A2，机制表使用 S1/S0/P1/G1/G2/F1/T1。G1 不是严格等体积/等邻居数；G2 同时改变尺度与摘要槽；T1 不限制邻居数，不单独证明中心坐标锚定。以上仅为单种子结果，不代表统计显著或每个序列/指标均改善。

执行分块不截断支持域。空间池化使用 `TRACEFORMER_SPATIAL_TOKEN_CHUNK=4096`，T1 使用 `TRACEFORMER_S1_DIRECT_QUERY_CHUNK=256`。原前缀和实现训练及 CPU 使用 1024 查询分块，CUDA 推理使用 16384。精确恢复时保持设置一致。论文 S1 的 41.499 ms 均值来自 RTX 4090 上 GPU 常驻输入的完整神经前向，不含加载、传输、评分，不是本次发布重新测得的速度。

## 发布核验

本次在 PyTorch 2.11.0+cpu 上验收。九组初始化权重、RNG 状态、输出和梯度均与原实现逐位一致；九组均完成微型训练、中断恢复和独立评估，恢复后的模型、优化器与 RNG 状态与不中断训练逐位一致。独立安装和无标注推理也已通过。

完整 CPU 复评覆盖全部 24 验证和 24 测试序列：逐事件概率与原 CPU 实现逐位一致，所有二值掩码及官方整数计数与论文产物一致，复现上表 Score。CPU AP 为验证 0.9419454783655674、测试 0.9602258519858334，与原 CPU 代码完全相同；历史 GPU AP 分别为 0.9419454870573791、0.9602258511768741。CPU/CUDA 浮点排序差异未改变掩码和官方 Score。

整理后的版本尚未在本次仅有 CPU 的发布主机重新执行 CUDA 验收；历史论文 GPU 测量保留其原始范围。测试工具、验收证据和生成预测均保存在发布仓库之外。
