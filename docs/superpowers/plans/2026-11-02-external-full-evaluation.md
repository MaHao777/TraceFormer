# External Full Evaluation Implementation Plan

> **For agentic workers:** Execute inline in the current repository; the user requires local development without worktrees. No commit or publication is requested.

**Goal:** 在昨天的原生Windows包上增加三个外部模型的完整val/test预测、可核查指标和单包回传，并预设三个普通测试序列加一个密集测试序列。

**Architecture:** 保留官方源码、权重、固定0.9阈值和现有predict.py。新增独立批跑入口，按split分别调用现有预测及收集器，分别保留完整输出和失败状态；新增CPU指标统计，按基准口径输出全局与逐序列计数及概率诊断。用增量ZIP交付新增文件，避免重传模型或改动已验证环境。

**Tech Stack:** Python标准库、NumPy、SciPy、scikit-learn、现有Windows PyTorch环境。

---

### Task 1: 核对协议和输入

**Files:** `remote/external_qualitative/FULL_RUN.md`, `paper_reference.json`, `qualitative_selection.json`。

- [x] 验证本地val/test各24文件；确认省略`--sequences`已有完整split支持。
- [x] 阅读EV-SpSegNet、PACT、PointEvent官方指标实现及本地论文，记录IoU、前景ACC、Pd、Fa单位与PACT代码差异。
- [x] 保存文献出处、原始指标、PDF哈希和预先固定的`test_000,test_001,test_002,test_022`选择。

### Task 2: 实现增量工具

**Files:** `remote/external_qualitative/tools/benchmark_metrics.py`, `analyze_results.py`, `run_full.py`。

- [x] 基准统计保留整数毫秒、50ms开区间、目标ID、8连通域和uint8计数；测试同时比较官方代码和独立计数。
- [x] 汇总每个split/模型的全局指标、逐文件指标、概率分布、AUC和阈值0.5诊断；0.9推理掩码保持不变。
- [x] 批跑六个独立预测进程，输出分split目录；完整输入审计后运行，可重用身份一致的完整模型任务，失败任务保存到_failed_attempts后重跑，不覆盖已有预测。
- [x] 沿用原收集器生成两个split包，再包含分析、输入清单、选择配置、环境和日志生成一个SHA256可核查回传包；失败时保留已完成结果并记录状态。

Entry point:

```powershell
python .\tools\run_full.py --data-root $EventData --output-root .\results_full_20261102
```

### Task 3: 验证和交付

**Files:** `tests/test_external_full_evaluation.py`, `remote_upload/external_full_patch_20261102.zip`。

- [x] 用小型事件数据验证微平均和宏平均差异、50ms边界、多个目标与虚警连通域、官方PACT Fa缩进差异、失败/恢复行为和完整包身份校验；144份预测的归档测试只验证数据契约，不声称执行了神经推理。
- [x] 将昨天3序列×3模型的真实回传结果交给新增分析器，确认与现有计数及PointEvent原生日志一致；不运行新GPU推理。
- [x] 26项测试、Ruff、diff检查通过；本地完整48输入审计通过，旧包168文件校验通过。增量ZIP为14721字节，8个成员及其哈希检查通过。
- [x] 准备增量补丁、原生PowerShell命令、48文件/144预测的期望清单和回传文件名。完整GPU运行和四行论文图在云端结果返回后完成。

## 验证证据

`tmp/external_full_regression/regression.json` 保存昨天真实回传的9组重算结果。PointEvent全局IoU=0.1884268214、前景ACC=0.1906237260、逐序列平均AUC=0.8838863449，与原生日志舍入结果一致。没有用3个样例生成全文献指标差值。`tmp/external_full_validation/dataset_inventory.json`记录全部48文件身份。`remote_upload/external_full_patch_20261102.zip`仅新增工具和说明，未修改原网络、权重或依赖。
