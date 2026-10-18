# TraceFormer 核心消融（CPU）

## 2026-10-17 三锥注意力最小对照

使用当前 prefix-cone 实现，三组均关闭 Sparse、精修与外部后处理，从头训练24轮，seed37，CPU FP32、8线程，99训练/24验证。

|ID|编码器交互分支|参数量|
|---|---|---:|
|C0|逐 Token 64→164→64 MLP，保留归一化、残差与原FFN|164077|
|C1|三尺度固定窗口，半径6/3/1、时间跨度2/4/8，六摘要中心注意力|163972|
|C2|三锥 (R,H)=(8,2)/(4,4)/(2,8)，六摘要中心注意力|163972|

C1/C2 共享完全相同的初始参数，仅邻域形状不同；三组公共参数和训练 RNG 起点相同。
窗口半径通过最接近锥平均离散截面积确定，内部网格体积差为 -8.6%/+19.5%/+12.5%；报告实际占用邻居的均值、分位数和空区域比例，不宣称严格等体积或等 FLOPs。
过去/未来各三个槽，排除同时间片，无邻居截断。C0 仅表示编码器不跨 Token 传递，公共 Tokenizer 统计输入保留。

```powershell
python scripts/run_paper_ablation.py --prepare-attention --config configs/traceformer_prefix_cone.yaml --suite outputs/cone_attention_cpu_YYYYMMDD_HHMMSS
python scripts/run_paper_ablation.py --suite outputs/cone_attention_cpu_YYYYMMDD_HHMMSS
```

三组训练完成后，使用发布的24文件带标签测试集比较：

```powershell
python scripts/evaluate_attention_test.py --suite outputs/cone_attention_cpu_YYYYMMDD_HHMMSS --dataset-root D:/VS_project/TinyObjDetection/EV-UAV-dataset --output outputs/cone_attention_test_YYYYMMDD_HHMMSS
```

该入口复用各组 `best.pt`、保存配置和验证阈值；推理标签与目标 ID 清零，不使用测试标签调整阈值，也不调用外部后处理。
模型源码使用训练套件冻结副本，适配器单独保存；每组在另一进程读取概率重算，并由原始事件二值计数实现逐文件交叉核验。
保存测试数据哈希、概率、掩码、逐文件计数和完整对照表。

2026-10-18 实测完成：24测试序列、2,074,586事件，验证阈值 C0/C1/C2=0.8375/0.95/0.90。
测试 Score 分别为 0.851466157330 / 0.895621084427 / 0.906258845580，C2-C1=+0.010637761153。
三锥相对固定窗口的 IoU/Pd 提高、Fa也增加；召回收益主要集中于 test_022/test_023，两文件合计检出目标帧142→235（共388）。其余22文件的合计Score为0.940502→0.941001。
完整24文件主结果及独立报告保存在 `outputs/cone_attention_cpu_20261017_235917/test_20261018_094317/`；分组仅作诊断，不替代主结果。
可随 Git 提交保存的[完整验证/测试指标、实验协议及权重哈希归档](experiments/cone_attention_20261018.json)。

每组启动前核对准备时的数据清单哈希，使用冻结源码，各组独立复评后才能完成。
比较 C1-C0、C2-C1、C2-C0；统一报告最优阈值与0.5阈值、完整24文件全局指标、逐文件结果、参数量、耗时、峰值进程RSS及固定GT片段图。
验证集用于选择epoch与阈值；单种子结果不支持多种子稳定性或统计显著性结论。
下述旧主干实验仅作历史记录，不能作为当前三锥消融的已训练基线。

## 2026-09-18 Pool attention 编码消融

固定时空锥主干，关闭 Sparse 和精修，使用 seed 37、99/24 文件、18 轮 CPU FP32、14 线程。

|ID|编码|训练来源|
|---|---|---|
|T0|七维统计特征 → 7→64→64 MLP|从头训练|
|T1|原逐事件 MLP + value 投影 → 块内算术平均 → 64→192→64 MLP|从头训练|
|T2|原逐事件 MLP + 四查询 Attention pooling|复用 E1 已完成的 18 轮训练，先用当前运行时重新独立复评|

三组保留相同的七维统计直连、位置编码、token/事件映射及事件解码器局部输入。
T0 测试的是 token 编码仅使用统计描述，并非删除整个网络的逐事件信息。
删除各替代方案中不用的 attention 参数；T1 使用实际参与计算的宽 MLP，152,010 参数与 T2 的 151,882 近似匹配（差 0.084%），T0 为 122,762 参数。
T1−T0 检验学习逐事件表示的整体方案（包含容量差异）；T2−T1 在近似同参数量下检验注意力聚合相对平均池化的收益。FLOPs 不宣称匹配。
比较采用相同 checkpoint/阈值选择规则，并同时报告固定阈值 0.5、完整 24 文件、困难域和逐文件结果。

```powershell
python scripts/run_paper_ablation.py --prepare-pooling-from outputs/paper_ablation_simple_cpu_20260917_231507 --suite outputs/paper_pooling_cpu_YYYYMMDD_HHMMSS
python scripts/run_paper_ablation.py --suite outputs/paper_pooling_cpu_YYYYMMDD_HHMMSS
```

新队列先独立复评 T2，复现失败立即停止；随后串行训练 T0/T1，各自独立复评后生成表格和三组固定片段图。
源码快照、实验契约、配置哈希、每轮完整恢复状态与旧队列隔离。正式配置及已有实验权重不改写。

## 2026-09-17 精简执行方案

采用单种子 37 的四组递增对照：E0 固定窗口主干、E1 时空锥主干、E2 主干+Sparse、E3 主干+Sparse+精修。
复用 2026-09-15 队列中已经完整训练和独立复评的 B、BS，仅补跑 window_B 18 轮和 BSR 24 轮。
BR 的 19 轮断点保留为补充记录，不继续训练；原 42 阶段队列保持暂停。

```powershell
python scripts/run_paper_ablation.py --prepare-simple-from outputs/paper_ablation_cpu_20260915_235356 --suite outputs/paper_ablation_simple_cpu_YYYYMMDD_HHMMSS
python scripts/run_paper_ablation.py --suite outputs/paper_ablation_simple_cpu_YYYYMMDD_HHMMSS
```

准备时校验相同训练/模型代码、数据清单哈希、权重哈希、同种子上游关系及已保存的完整复评结果。
新目录中只保存对旧完成产物的引用，不修改旧实验目录。
结果按 E1-E0、E2-E1、E3-E2 汇总；只论证逐步添加的贡献，不论证模块交互或多种子稳定性。
新队列全部完成后自动生成四组表格和固定片段的可视化。

## 实验契约

99 个训练文件、24 个验证文件；随机种子 37/73/101，CPU FP32、14 线程。
所有配置沿用 AdamW、0.0007 学习率、0.0002 权重衰减、legacy cosine 与当前各阶段损失。
阶段轮数为 B=18、S=24、R=24；每阶段固定上游，公共上游按同种子复用。
验证集参与最佳 epoch 与阈值选择，论文应标为验证集消融。

|ID|变体|
|---|---|
|A0|B：Tokenizer + 双向时空锥 + 事件解码器|
|A1|B + Sparse Token 混合|
|A2|B + 多尺度精修|
|A3|B + Sparse + 多尺度精修|
|A4|A3 的时空锥替换为匹配平均截面积的固定窗口|
|A5|A3 的注意力仅使用过去方向，关闭方向 dropout；整网不是因果模型|
|A6|A3 的注意力删除整个相对几何/运动偏置|
|A7|A3 的精修保留三条独立参数分支，全部改用 8x8x50ms|

共 24 个最终结果，42 次阶段训练。A4/A5/A6 分别重训 B、S、R。
移除的模块不保留参数或专属辅助梯度；正式配置没有新增开关。
实验规范保存在 ablation.json 和 checkpoint.ablation 中；正式加载器拒绝实验权重。

## 启动

```powershell
python scripts/run_paper_ablation.py --prepare --suite outputs/paper_ablation_cpu_YYYYMMDD_HHMMSS
python scripts/run_paper_ablation.py --suite outputs/paper_ablation_cpu_YYYYMMDD_HHMMSS
```

prepare 保存每个阶段的独立 YAML、源代码副本/哈希，以及只由 GT 确定的可视化窗口。
运行队列使用该源代码副本；各训练阶段及复评使用独立进程，串行执行。
正式启动前，运行 scripts/check_paper_ablation.py，检查 train_096、train_094 与各结构的前向和反向。

队列状态在 status.json，阶段日志在 logs/。每个 epoch 保存 training_state.pt。
失败时队列停止，并保留已有结果。排除失败原因且确认旧工作进程已停止后，重新执行同一个 --suite 命令。
它会跳过已复评完成的阶段，将未完成阶段从最后一个 epoch 状态恢复到新的 attempt 目录。
没有完整 epoch 状态时，同种子从该阶段起点重跑，旧目录保留。

## 产物

- results.csv / results.md：完整 24 文件指标，按种子逐步汇总。
- groups.csv：按固定稀疏触发门限分组，不根据模型结果分组。
- aggregate.json：均值、样本标准差、同种子差值及模块交互；只有一个种子时标准差留空。
- 各阶段 ablation_report.json：保存阈值与 0.5 阈值结果、逐文件计数、门控分位数、稀疏视图保留统计、精修前后 TP/FP/FN。
- predictions/：按原始事件顺序保存最终及精修前 logit，支持后续绘图和复核。
- qualitative_seed*.png：固定 val_000/015/023 的 GT 最大目标事件帧，各配置共用相同片段。
- ablation_metrics.png：已完成种子的 Score/Pd/Fa 均值及标准差。

独立 FP32 复评必须逐项复现保存的完整验证指标，否则停止队列。
CPU 内存记录为独立复评进程的峰值 RSS（包含验证数据及诊断缓存），不能称作 GPU 显存。
推理耗时包含稀疏子流模拟与 CPU 邻域构建，不包含 NPZ 磁盘读取。

## 结论约束

报告全部预注册配置和全部种子。收益若只发生在密集场景，应限定结论。
模块添加结果同时包含其专属损失与阶段训练预算的贡献；A7 额外控制参数量。
主表不使用 LightGBM 或任何轨迹/密集目标补救后处理。
