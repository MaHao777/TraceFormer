# 均值/空间卷积池化 × 过去锥局部注意力：六组云端实验

> 2026-10-30：启动、续跑和打包已统一到 [实验导航](experiments/README.md)。已有冻结包继续使用原入口。
> 下文历史配置/验证记录保留；六组当前实现仅记录耗时，查询分块默认8192、空间分块4096。


本次只新增独立实验入口，当前正式模型配置与预测入口保持原行为。直接在本地仓库开发，不使用 worktree；冻结源码后整体上传 `remote_upload/local_pooling_six/`。

## 六组结构

|ID|父块内聚合|三层编码器|实际参数量|
|---|---|---|---:|
|S0|M1 事件特征均值|原三尺度门控过去锥摘要注意力|142774|
|S1|单核空间卷积池化|同 S0|143350|
|S2|双核空间卷积池化|同 S0|143926|
|L0|M1 事件特征均值|门控过去锥 Token 局部注意力|129526|
|L1|单核空间卷积池化|同 L0|130102|
|L2|双核空间卷积池化|同 L0|130678|

L 组每层固定过去 `(R,H)=(2,2)`：前一个时间块半径 1，前两个时间块半径 2，读取全部 34 个候选格点中的占用 Token；排除同时间块与未来，不截断或采样。三层共用本次前向的整数邻居索引，QKV 每层投影一次，分块 SDPA 并在训练时重算。原源 Token 门控 `sigmoid(pool_weight(x))+1e-4` 取对数后加入注意力 logit；删除摘要 value 投影、尺度/方向摘要嵌入和中心摘要交互。

卷积仅细分原父块的空间：8×8 像素分为 4×4 子格，各子格先对公共事件编码取均值，再做无偏置、padding=1 的 depthwise 3×3 卷积与 GELU。双核分别激活后均值，按子格原事件数加权回读 64 维；七维统计和位置独立拼接。原事件到 Token 映射、极性解释及事件 Head 输入不变。空子格权重为零，边缘父块按固定像素尺度分格，不重缩放。

## 固定训练与计算门槛

99 train / 24 val / 24 test，seed37，24 轮，CUDA FP32，关闭 TF32，8 辅助线程。其余模型、损失、密集序列权重、采样、AdamW 与学习率调度沿用 M1+C4。实验版本 4 只接受六个 ID；检查点与恢复状态严格匹配结构。

S0 默认复用 `remote_result/26-10-28` 的 GPU 检查点，SHA256 为 `6e67f2d02947ce44a4a52e4bc29df654680c6ee621c4132f9129529fc6fadb6e`。原字节与旧 `M1_C4` 契约保留，先核验完整训练记录、数据身份并重新独立复评验证集，再开始其余五组。缺基线时可显式 `--without-baseline` 从头训练 S0；无效基线直接拒绝，不静默回退。

当前流程默认直接训练，不以相对耗时或显存比例停止实验。显式 `--stage preflight` 才执行训练步骤与验证前向诊断；完成训练后统一测量24测试序列真实网络前向，记录均值、P95、峰值显存与相对S0差值。重评分缓存的耗时不作为网络速度。历史1.25倍预算规则不再执行。

## 打包和云端启动

本地重新生成独立包（拒绝覆盖已有目录）：

```powershell
traceformer-experiment bundle local_pooling_six --output remote_upload/local_pooling_six_new
```

整体上传生成目录，云端进入该目录执行：

```powershell
python -m pip install -r requirements.txt
python -u run.py --data-root "E:/mahao/TraceFormer/EV-UAV-dataset" --test-root "E:/mahao/TraceFormer/EV-UAV-dataset"
```

数据路径为旧云端主机的示例，实际使用包含 `train/val` 和 `test` 的目录。安装依赖不替换主机已有 CUDA PyTorch。也支持 Linux 路径。

```powershell
python -u run.py --stage preflight --data-root /path/to/dataset
python -u run.py --stage train --data-root /path/to/dataset
python -u run.py --stage predict --test-root /path/to/dataset
python -u run.py --resume
```

默认查询分块 8192、卷积父 Token 分块 4096，首次运行前可用 `TRACEFORMER_LOCAL_QUERY_CHUNK` 与 `TRACEFORMER_SPATIAL_TOKEN_CHUNK` 指定正整数；实际值冻结到实验契约，恢复不能改变。每轮保存完整训练状态，中断恢复新建 attempt，已完成组不重训；文件锁拒绝重复协调器。

CPU 仅允许显式 `--smoke --device cpu` 做两文件一轮诊断，写入独立 `output/smoke`，不复用正式 S0，也不证明 GPU 预算或正式精度。

## 交付与解释

回传 `output/results.zip`，包含六组最佳权重、配置、完整恢复状态、训练曲线、逐事件概率/掩码、概率独立重评和原始掩码第二套计数，以及训练后测试前向计时、可选预检报告、环境和数据/源码/产物哈希。复用的 S0 权重随包回传。

`results.md/results.json` 提供验证/测试指标、前向均值/P95、显存、相对 S0 的差值，并计算 L0−S0、S1−S0、L1−L0、S2−S1、L2−L1 与单核交互差值。

本次是单种子筛选。编码器比较同时改变邻域范围与交互方式；卷积比较包含非线性变化。全序列极性统计和多尺度精修 Head 仍使用其他时间上下文，因此只称编码器使用过去信息，不称整网严格因果。本机只有 CPU，真实 CUDA 数值与开销检查需在云端显式预检及训练后基准完成；历史检查证据保留在原交付包 `LOCAL_CHECKS.json`，本轮整理见[验证记录](experiments/reorganization_20261030.md)。
