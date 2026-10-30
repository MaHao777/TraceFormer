# 2026-10-30 实验流程整理与验证

使用方法见 [实验导航](README.md)。本轮直接在本地仓库整理，没有使用 worktree。

## 实现结果

- 日常维护 `mean_past_gated` 和 `local_pooling_six`，统一为 `traceformer-experiment` 的 list、bundle、run、status、archive-run。
- 两套 workflow 复用运行层、独立评分与打包实现。新上传包只有一个 run.py，不再复制旧 workflow、train/predict 或评分脚本。
- 29 个旧实验与辅助脚本移入归档，另有一个归档路径辅助模块；主 scripts 目录由49个 Python 文件减至18个。
- 13个归档条目中，11个通过参数解析及匹配权重严格加载；paper_cuda、mean_past_hotfix 为一次性旧工具，缺少通用运行条件，明确标记 incomplete。
- Windows 子进程完成后清除活动 PID，恢复时核对进程创建时间，修复旧 PID 被复用后误阻止续跑的问题。
- 已安装新增命令，并实际核验帮助、清单、状态及真实归档入口。已有冻结包和输出目录保留原位。

## 数值与产物对照

本轮开始时，将源码、脚本、远端模板、配置和测试共202个文件保存在本地忽略目录
`tmp/experiment_reorganization_baseline/`，并记录逐文件 SHA256。

- 原有 src Python 模块字节未变；新增实验模块承接迁移后的流程。
- 模型、训练、后处理、配置共34个文件与开始时字节一致。
- M1_C4、S0/S1/S2/L0/L1/L2 共2374项张量检查逐位一致，覆盖初始化随机状态、参数、前向、梯度与一次 AdamW 更新。
- 对比整理前后实际独立包完成一轮微型训练的结果：七组共768个模型状态张量逐位一致，验证阈值和独立计数一致。
- 两套独立包均在离开仓库的目录中完成 CPU 训练、预测、概率复评、原始掩码计数、汇总、ZIP核验及恢复。

这些检查不等于重新运行完整数据集或 CUDA 评测，也不用于宣称正式分数不变。

## 测试结果

最终覆盖当前 pytest 收集到的全部543个用例：**539通过、4跳过**。
按以下批次执行，覆盖集合按用例标识核对，既没有遗漏，也不重复计数：

1. 统一 CLI、项目集成、六组独立包：36通过。
2. 排除上述三个模块的其余全套测试：502通过、4跳过；一个旧 CLI 帮助测试受父子进程 UTF-8/GBK 不一致影响。
3. 在 `PYTHONUTF8=1` 下复验该 CLI 用例：1通过。

另外，迁移后的两套独立包专项曾单独执行，30通过、1项CUDA测试跳过；
归档与中心注意力等相关专项28通过。首次基线中的两条旧预算断言已按当前 timing_only 规则更新。

最终用例合并记录与原始日志保存在本地忽略目录：

- `tmp/experiment_reorganization_final.xml`、`experiment_reorganization_final.json`
- `tmp/pid_fix_pytest.xml`、`remaining_reorganization_pytest.xml`、`cli_encoding_pytest.xml`
- `tmp/experiment_equivalence.json`、`trained_pipeline_equivalence.json`

`python -m ruff check src tests scripts archive/experiments/scripts` 与 `git diff --check` 均通过。
真实 CUDA 不可用（本机 PyTorch 2.11.0+cpu）；CPU smoke 不作为 CUDA 验收。

## 保留事项

- 已有运行通过原冻结入口继续恢复；新流程创建新目录，不接管旧产物或改写其哈希。
- 默认完整主模型仍有 schema-7 产物与 schema-8 加载器错配，按本轮范围单独留待修复。
- 新增实验复用 runtime、evaluation 和 packaging，只增加自身的阶段与科学配置，不复制整份旧 workflow。
