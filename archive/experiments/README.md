# 历史实验归档

日常入口见 [实验导航](../../docs/experiments/README.md)。本目录集中保存旧实验的调度与分析代码，
不再保留原 `scripts/` 命令转发。模型库来自登记的历史快照，而不是当前源码。

|系列|归档ID|相关工具|
|---|---|---|
|论文消融与机制|paper_ablation、mechanism|prepare/report/check_paper|
|注意力与中心交互|attention、center_attention|analyze/finish_center_attention|
|旧队列恢复|attention_recovery|run_attention_accelerated、run_attention_fast_track|
|组件与结构|component_core、structure_improvements、no_sparse|check_component_full_equivalence|
|事件记忆|neural_memory|analyze/finish/benchmark_neural_memory、diagnose_memory_alignment|
|原网络读出|original_readout|diagnose_original_neural_paths、diagnose_summary_weighting、summarize_original_readout_recovery|
|冻结轨迹干预|vtr|run_vtr_structure_ablation|
|冻结 CUDA 比较|paper_cuda|audit_paper_cuda|
|一次性上传包修补|mean_past_hotfix|fix_mean_past_attention|

[catalog.json](catalog.json) 登记每个系列的原命令、默认入口、源码、配置/权重、报告与依赖。
可运行状态必须同时满足：快照完整、依赖可用、参数解析通过、登记权重严格加载通过。
缺失证据的条目标记 incomplete，不用当前模型代替历史模型。目录可显式搬迁，源码字节必须匹配。

```powershell
traceformer-experiment list
traceformer-experiment archive-run component_core -- --help
```

`--` 后的参数原样交给归档脚本。需要另一个同字节快照时，在归档ID前使用
`--source-root <目录>`。已冻结包、输出目录和历史报告保持原位；已运行实验应使用原入口续跑。
