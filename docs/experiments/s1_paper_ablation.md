# S1论文筛选消融：独立云端包

本包默认seed37，只训练七种新增结构；有效S1/S0按原权重与原checkpoint契约复用。
按用户选择，本轮不自动运行seed38/39，不包含VTR与额外特征/源门控消融。

|ID|定义|正式参数量|来源|
|---|---|---:|---|
|S1|单核空间池化＋三层门控过去三尺度中心摘要＋事件解码＋多尺度精修|143350|核验后复用|
|S0|仅将S1空间聚合替换为事件编码均值|142774|核验后复用，可能保留原M1_C4契约|
|A1|仅保留编码器每层FFN、LN与残差，无跨Token交互|79603|从头训练|
|A2|去神经精修Head及其辅助损失|135172|从头训练|
|P1|4×4子格3×3 depthwise核替换为1×1，保留GELU与按事件数回读|142838|从头训练|
|G1|三尺度固定窗口摘要替代展开锥|143350|从头训练|
|G2|只保留预先指定的中间(4,4)单锥摘要|142966|从头训练|
|F1|有效摘要均值后接64→128→64 MLP，替代中心Query多头融合|143158|从头训练|
|T1|三锥并集内的门控直接Token注意力，集合去重，不限制邻居数|130102|从头训练|

结构消融表为S1/S0/A1/A2；主要创新点表为S1/S0/P1/G1/G2/F1/T1。
两表共享基线，不重复训练。results.json与results.md同时保存完整主表及这两张子表。

## 科学定义

所有组保留相同8×8父块、64时间块、64维、3层、4头、七项块统计、位置编码、
事件特征直连及同一组损失（A2移除Head关联的辅助损失）。无Sparse与外部后处理。
99 train / 24 val / 24 test，24轮、seed37、CUDA FP32、关闭TF32、8辅助线程，
AdamW、采样、密集序列权重与学习率日程沿用原S1配置。

S1使用过去(8,2)/(4,4)/(2,8)方形截面锥，排除同时间与未来。G1采用近似离散体积
匹配的固定窗口(6,2)/(3,4)/(1,8)，不是严格等体积；support_geometry.json逐验证
序列记录每尺度实际占用邻居的均值、P95和总数，并记录直接注意力去重并集的占用数。
该诊断独立于网络耗时测量。

T1读取同一三锥支持域的并集，按过去距离、行、列排序；边界裁剪前有552个候选格点，
重复成员只出现一次。保留源Token门控log(sigmoid(pool_weight)+1e-4)，共用原QKV
及输出投影，去掉摘要value投影和槽位嵌入。不同参数与计算开销如实记录，不称等参数
实验；它检验摘要式与直接式交互，不独立证明中心坐标锚定的贡献。

P1从同一S1初始化中复制3×3核的中心系数，只去掉邻格连接；新卷积与F1融合MLP
不推进公共训练RNG。G2保留中间尺度初始化。所有保留参数及初始化后的RNG逐位核对，
每组实际结构与参数量写入structure.json。A1的Head仍有上下文；原网络全序列统计与
精修Head仍保留其他时间信息，因此不宣称整网严格因果。

## 本地打包

默认核验并打包remote_result/26-10-29/full中的S1/S0；缺失或无效时直接报错，
不静默改成重新训练。S0原字节在该full同级baseline中；不使用失效云端绝对路径。

```powershell
traceformer-experiment bundle s1_paper_ablation --output remote_upload/s1_paper_ablation
```

可以显式--baseline-root指定同格式回传目录。仅当明确使用--without-baseline时，
S1/S0也从头训练，本轮将变为9次训练。既有上传目录不覆盖，原始回传不会被改写。
依赖清单不替换主机已有CUDA PyTorch。

## 云端启动与恢复

解压或上传整个包，进入包含run.py的目录：

```bash
python -m pip install -r requirements.txt
python -u run.py --data-root /path/to/dataset --test-root /path/to/dataset
```

data-root包含train/val，test-root包含test，可为同一个根目录。唯一入口run.py
自带冻结源码路径，不需要安装本地项目。默认输出output/full，最终回传
output/results.zip及output/results.zip.sha256。

all/train先完成每组CPU/CUDA小样本输出和梯度对照，train_094/train_096完整
训练步骤、24验证序列前向预检及实际支持域诊断。随后独立复现S1与S0原验证指标，
两者通过后才开始A1→A2→P1→G1→G2→F1→T1的24轮训练。所有训练完成再统一
预测验证/测试、概率重评分、原始掩码独立计数和训练后测试前向基准。

```bash
python -u run.py --stage preflight --data-root /path/to/dataset
python -u run.py --stage train --data-root /path/to/dataset
python -u run.py --stage predict --test-root /path/to/dataset
python -u run.py --resume
```

每轮保存完整训练状态；中断恢复新建attempt，已完成训练核验后复用。S1/S0原权重、
配置、history、metrics、summary、训练曲线与最终恢复状态按字节保留在输出目录，
不重写原checkpoint契约。配置、源码、数据、主机/GPU/驱动与分块冻结，恢复时不可变。

查询分块默认8192、空间池化父Token分块4096；T1独立使用256查询分块以控制完整
552候选的显存。首次运行前分别可用TRACEFORMER_LOCAL_QUERY_CHUNK、
TRACEFORMER_SPATIAL_TOKEN_CHUNK、TRACEFORMER_S1_DIRECT_QUERY_CHUNK设置。
分块是执行参数，不缩小支持域。OOM/非有限数值保留现场，不自动切精度或截邻居。

## 输出与解释

验证选择checkpoint和阈值；测试沿用验证阈值，Score使用全量整数计数，不平均逐文件
Score。所有组记录IoU、SegAcc、Pd、Fa、AP、参数、完整网络前向均值/P95和显存。
Fa为单位像素帧的假警连通分量率，不是逐事件FP率。单种子差异不作为显著性结论。

CPU微型烟测显式使用--smoke --device cpu，只用2文件、1轮，独立写output/smoke，
不复用正式基线，不验证CUDA数值与正式精度。本地检查通过后才交付；云端真实检查
由默认all阶段执行。只做固定权重前向开关不能替代本包的从头重训消融。

## 本地交付核验（2026-10-31）

581个用例按身份合并：577通过、4跳过。模型专项74通过；其余回归461通过、4跳过；新旧流程兼容批次中的复用夹具序列化问题已修正，两个复用专项及最终独立包5项复验均通过。重复复验不额外计数。Ruff与git diff检查通过。

七种新结构覆盖从头微型训练、独立复评、恢复后模型/优化器/RNG一致和错组拒绝。九组冻结包在离开仓库的目录完成CPU训练、预测、两套评分、两张消融表、ZIP校验及恢复。旧小锥局部注意力的6组输出/梯度/RNG对照逐位一致。

正式S1/S0原始权重和24轮协议已核验并打包；云端仍先独立复现其验证计数。本机只有CPU，正式CUDA数值、资源和24轮精度由云端all阶段检查。详细批次及证据哈希见包内LOCAL_CHECKS.json。

本轮回传建议解压至remote_result/s1_paper_ablation_20261031。
