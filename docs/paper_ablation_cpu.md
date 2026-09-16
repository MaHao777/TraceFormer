# TraceFormer 核心消融（CPU）

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
