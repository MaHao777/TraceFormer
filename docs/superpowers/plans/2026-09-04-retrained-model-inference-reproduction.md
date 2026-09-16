# Retrained Model Inference Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让评委既能用随包冻结权重生成最终提交，也能完整重训后用新 checkpoint、新 LightGBM 和新验证报告重新生成测试结果。

**Architecture:** 训练入口生成神经网络、train/val 缓存、一次事件模型发布、完整 val 后处理复评和机器可读复现清单。测试入口通过可选 `-TrainingRunDir` 在冻结模式和重训模式之间选择同一条预测链路；所有 checkpoint、缓存、LightGBM 和验证报告继续执行哈希一致性检查。

**Tech Stack:** PowerShell 7、Python 3.12、PyTorch、LightGBM、NumPy、pytest、Ruff。

---

### Task 1: 固化重训产物契约

**Files:**
- Create: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\src\huav\postprocessing\reproduction.py`
- Test: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\tests\test_reproduction.py`

- [ ] **Step 1: 写失败测试**

覆盖事件门槛通过后的合法终态、事件门槛失败、使用外部基线的冻结报告和使用 `objectness` 自基线的重训报告。

- [ ] **Step 2: 验证测试先失败**

Run: `python -m pytest tests/test_reproduction.py -q`

Expected: 因 `huav.postprocessing.reproduction` 不存在而失败。

- [ ] **Step 3: 实现最小契约模块**

实现 `require_event_release_eligible()` 和 `validation_gate()`，只允许事件门槛已通过且训练流程已结束的报告进入测试链路。

- [ ] **Step 4: 验证单元测试通过**

Run: `python -m pytest tests/test_reproduction.py -q`

Expected: 全部通过。

### Task 2: 新增重训后完整验证脚本

**Files:**
- Create: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\scripts\evaluate_event_release.py`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\tests\test_project_integration.py`

- [ ] **Step 1: 增加交付契约测试**

要求新验证脚本存在、训练脚本调用它、训练脚本不再重复调用 `release-event`。

- [ ] **Step 2: 实现 event-only 验证**

从无标签 val 缓存生成 `objectness`、`event_release`、`event_release_dense_rescue` 三阶段预测；推理固定后再从原始 val NPZ 计分，并写出 `validation_comparison.json`。

- [ ] **Step 3: 验证帮助入口**

Run: `python scripts/evaluate_event_release.py --help`

Expected: 退出码 0，展示 cache、training、event release、reference、dataset 和 output 参数。

### Task 3: 修复一键训练链路

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\run_train.ps1`
- Add assets: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\weights\reference_predictions_val\val_000.txt` 至 `val_023.txt`

- [ ] **Step 1: 复用内部事件发布产物**

把正式事件模型固定为 `postprocess_training/event_release`，删除第二次 `release-event` 调用；事件门槛未通过时给出明确错误。

- [ ] **Step 2: 消费 val 缓存**

调用 `evaluate_event_release.py` 生成 `postprocess_validation/validation_comparison.json`。

- [ ] **Step 3: 写出重训清单**

生成 `reproduction_bundle.json`，记录新 config、checkpoint、train cache manifest、训练协议、事件发布目录、完整验证报告、网络分数、完整分数及相对冻结结果差值。

- [ ] **Step 4: 给出下一条命令**

训练结束时直接打印 `run_test.ps1 -TrainingRunDir <本次输出目录>` 以及所有关键结果位置。

### Task 4: 让一键测试接受新训练产物

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\run_test.ps1`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\scripts\predict_postprocess_track2.py`

- [ ] **Step 1: 加入 `-TrainingRunDir`**

默认模式继续使用随包冻结权重；重训模式读取 `reproduction_bundle.json` 并解析全部相对路径。

- [ ] **Step 2: 保留严格一致性检查**

继续检查 checkpoint schema/config、cache checkpoint SHA-256、LightGBM SHA-256、99文件发布和验证报告绑定关系。

- [ ] **Step 3: 允许报告验证漂移但不阻断产出**

重训模式把验证门槛结果记录到 `prediction_manifest.json`；即使分数漂移也生成 ZIP，方便评委比较，而冻结模式仍强制门槛通过。

- [ ] **Step 4: 输出模式和产物路径**

控制台明确打印 frozen/retrained 模式、实际 checkpoint、`submission.zip`、审计报告和验证对照。

### Task 5: 更新评委文档与审查材料

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\README.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\提交材料检查表.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\docs\运行说明.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\docs\算法报告.md`
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\other_files\package_audit.json`

- [ ] **Step 1: 分别写出冻结预测和重训预测命令**

冻结预测：`.\run_test.ps1 -TestDir .\data\test`。

重训后预测：`.\run_train.ps1 -Device cpu -OutputDir <新目录>`，随后 `.\run_test.ps1 -TrainingRunDir <新目录> -TestDir .\data\test`。

- [ ] **Step 2: 解释无验证集梯度训练证据**

明确99个 train 用于梯度训练，24个 val 只用于阈值/checkpoint选择和固定预测后的评分，LightGBM 参数仅由 train 内部79/20开发及全99发布得到。

- [ ] **Step 3: 重建并检查 PDF**

Run: `pdftoppm -png docs/算法报告.pdf tmp/pdfs/algorithm` 与运行说明对应命令。

Expected: 无裁切、重叠、乱码和表格溢出。

### Task 6: 最终回归与清单

**Files:**
- Modify: `D:\金晴杯\16-追光逐界队-jinsight-赛道2\MANIFEST.sha256`

- [ ] **Step 1: 执行目标测试**

Run: `python -m pytest tests/test_reproduction.py tests/test_project_integration.py -q -p no:cacheprovider`

Expected: 全部通过。

- [ ] **Step 2: 执行全量回归**

Run: `python -m pytest -q -p no:cacheprovider`

Expected: 全部通过，仅保留既有平台跳过项。

- [ ] **Step 3: 执行静态检查**

Run: `python -m ruff check --no-cache src tests scripts`

Expected: `All checks passed!`

- [ ] **Step 4: 重建 SHA-256 清单并复核**

重建 `MANIFEST.sha256`，运行集成测试逐文件核对名称和内容哈希，并确认提交目录无缓存、代理配置或内部计划。
