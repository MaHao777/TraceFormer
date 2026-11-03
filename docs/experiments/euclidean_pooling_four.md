# 四组池化 × 欧氏过去锥注意力

本轮为四种结构、seed37、各24轮的单种子对比，全部重新训练。旧CPU中心注意力与
GPU六组实验采用方形截面，不作为本轮正式结果或预训练权重复用。

|组别|池化|三层编码器|参数量|
|---|---|---|---:|
|E0，基线|事件编码后均值|小欧氏过去锥局部注意力|129526|
|E1|单核空间卷积与GELU|同E0|130102|
|E2|事件编码后均值|三尺度欧氏过去锥中心锚定摘要注意力|142774|
|E3|单核空间卷积与GELU|同E2|143350|

## 结构与控制变量

局部固定 `(R,H)=(2,2)`；中心固定 `(8,2)/(4,4)/(2,8)`。以Token格点为单位，
过去距离d满足 `1 <= d <= H` 且 `H²(dx²+dy²) <= R²d²`。整数平方比较包含边界，
不先对圆盘半径取整。小锥两个截面分别有5、13个候选格点，共18个；三尺度的候选数
分别为246、96、44。只读实际占用Token，不截断；同时间与未来排除。

局部保留源门控 `log(sigmoid(pool_weight)+1e-4)`，直接参与邻居注意力；
中心保留同一源门控的加权摘要、value投影、尺度与过去方向编码及中心Query融合。
圆盘分解为互不重叠的行区间，用前缀和精确求和及伴随反向，不近似成方形。

单核沿用已有实现：8×8像素父块细分4×4子格，子格均值经过单个无偏置depthwise
3×3卷积、GELU，再按子格原事件数回读。七项统计与位置独立拼接；父Token数量、
事件映射、Event Decoder和多尺度精修Head保留。Sparse与外部后处理关闭。

99 train / 24 val / 24 test；64维、3层、4头、64时间块、dropout0.1，CUDA FP32、
关闭TF32、8辅助线程。其余损失、密集序列权重、背景采样、AdamW与学习率日程
冻结自最新六组相同配置。新增版本5契约；打包及每次运行逐位核对公共初值、卷积配对
及构造后的RNG，实际结构记录在每组 `structure.json`，不按模板Query字段解释模型。

## 打包与启动

在本地仓库生成新目录，不覆盖已有上传包：

```powershell
traceformer-experiment bundle euclidean_pooling_four --output remote_upload/euclidean_pooling_four
```

整体上传生成目录，或解压相应ZIP，进入包含 `run.py` 的目录执行：

```bash
python -m pip install -r requirements.txt
python -u run.py --data-root /path/to/dataset --test-root /path/to/dataset
```

`data-root`包含train/val，`test-root`包含test；可以指向同一数据根目录。
依赖清单保留云端已有CUDA PyTorch，不替换它。首次校验全部文件哈希、真实CUDA
内核及环境；没有CUDA时正式运行明确失败。云端主机由用户启动，本地CPU检查不代表
正式精度、显存或CUDA数值验证。

默认all先完成四组预检：同权重CPU/CUDA输出和梯度对照，再在train_094/train_096
上执行完整训练步骤（2次预热、5次计时），以及24验证序列前向检查。之后按
E0→E1→E2→E3串行训练。计时和显存仅记录，没有相对基线预算门槛。

```bash
python -u run.py --stage preflight --data-root /path/to/dataset
python -u run.py --stage train --data-root /path/to/dataset
python -u run.py --stage predict --test-root /path/to/dataset
python -u run.py --resume
```

train也自动执行首次预检；predict复用完成的训练。每轮保存训练状态，中断恢复新建
attempt；已完成组核验后复用，不重新训练。文件锁拒绝重复协调器。

查询分块默认8192，空间池化父Token分块4096；可在首次运行前分别设置
`TRACEFORMER_LOCAL_QUERY_CHUNK`、`TRACEFORMER_SPATIAL_TOKEN_CHUNK`。
实际值、源码、配置、数据和计算环境冻结，恢复时不能改变。OOM或数值异常保留现场，
不自动切换精度或改变锥范围。需要改变分块时创建全新运行目录。

CPU微型自检只用两文件、一轮，输出在独立的output/smoke：

```bash
python -u run.py --smoke --device cpu --data-root /path/to/dataset --test-root /path/to/dataset
```

## 评估、计时与回传

验证集选择最佳epoch与阈值；各组测试沿用其验证阈值，不在测试上重选。
Score由全量整数计数计算：`0.2 IoU + 0.1 SegAcc + 0.4 Pd + 0.3 exp(-10000 Fa)`。
Fa是单位像素帧的背景假警连通分量率，不是逐事件FP率。

全部训练完成后独立预测、概率复评，再用原始掩码第二套计数核验；验证全局计数必须
复现保存的最佳结果。报告Score、IoU、SegAcc、Pd、Fa、AP、阈值、参数量，
各轮训练及验证合计耗时，以及最佳权重在24测试序列上的完整网络前向均值/P95与
峰值显存。推理计时同机独立进程，预热后3次测量，CUDA同步，不包含IO与评分。

results.md/results.json给出E1−E0、E3−E2、E2−E0、E3−E1、E3−E0以及
`(E3−E2)−(E1−E0)`交互差值。val_023与test_022/test_023单列诊断，最终排序
使用全量结果。

回传 `output/results.zip` 和 `output/results.zip.sha256`。归档包括四组最佳权重、
配置、实际结构、完整恢复状态、训练曲线、逐事件概率/掩码、两套独立评分、预检与
训练后计时、环境及完整性审计；ZIP生成后检查CRC并写SHA256。

本轮只支持单种子观察。注意力比较同时改变邻域范围和交互方式；单核比较包含卷积与
GELU。精修Head与全序列极性统计仍读取其他时间上下文，整网不宣称严格因果。

## 本地交付检查（2026-10-30）

共564个用例：560通过、4跳过。按用例身份合并以下批次，不重复计数：

- 欧氏几何、注意力、历史前缀池化与训练恢复：42通过。
- 新旧独立包、统一入口与恢复：37通过。
- 其余回归：481通过、4跳过。
- 最终源码冻结后四组独立包复验：6通过，覆盖前述同名用例，不额外累计。

圆盘与暴力枚举的成员、摘要、注意力输出及梯度对照通过；四组完整微型训练、
独立复评、逐轮恢复、错组拒绝、数据/源码变更拒绝与环境变更拒绝通过。
历史方锥12组输出、成员计数和梯度与提交前源码逐位一致。Ruff和git diff检查通过。

证据在本地忽略目录 `tmp/euclidean_*_checks.xml`、`tmp/euclidean_all_checks.json`
和 `tmp/euclidean_legacy_equivalence.json`。上传包内 `LOCAL_CHECKS.json` 保存批次
计数与证据哈希。本机为PyTorch 2.11.0+cpu，没有执行正式CUDA训练；正式精度、
CUDA数值与显存须由云端默认all阶段完成后报告。
