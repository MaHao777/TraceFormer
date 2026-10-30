# A/B Structure Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现统计均值主导的单 Query embedding（A）和六层固定几何过去锥、无池化门控编码器（B），训练10/01/11并独立评估验证集和测试集，00不重训。

**Architecture:** 用版本3的显式实验契约在现有单视图Full上做结构替换，复用原训练、断点恢复、逐事件推理与独立计数。基线配置描述替换前的共同模板，实际结构由ablation.json唯一指定；公共参数初值及训练RNG与00一致，新增参数构造不推进训练RNG。

**Tech Stack:** Python 3.12、PyTorch CPU FP32、pytest、Ruff；本地仓库，无worktree。

---

### Task 1: 结构实现

- [x] 修改`src/traceformer/models/event_tokenizer.py`：提取Token投影入口，默认计算不变。
- [x] 新建`src/traceformer/experiments/structure.py`：A统计投影加事件编码均值，75%主支路与25%单Query加权融合，最终投影不重复拼接统计；B删除pool_weight，过去锥均值池化，保留中心摘要注意力，六层均使用(8,2)/(4,4)/(2,8)。
- [x] 修改`src/traceformer/models/prefix_cone.py`：支持显式方向和无门控池化，默认路径保持原计算。
- [x] 修改`src/traceformer/experiments/ablation.py`：增加严格的00/10/01/11契约；Checkpoint显式记录契约。

### Task 2: 行为与训练契约验证

- [x] 新建`tests/test_structure_improvements.py`：检验融合公式、统计仅一次、单Query读取全部块内事件、未来Token干预不影响过去输出、六层固定几何、全部参数有限梯度、公共初值/RNG、00精确等价、Checkpoint恢复及错组拒绝。13项通过。
- [x] 执行`D:/VS_environment/Python.12/python.exe -m pytest`和`-m ruff check src scripts tests`；检查`git diff --check`。405 passed、2 skipped、1个Windows子进程文本解码warning，Ruff/diff通过。

### Task 3: 三组受控训练与预测

- [x] 新建`scripts/run_structure_improvements.py`：复用`audit_baseline`、`source_snapshot`、`queue_lock`、`completed_job`、`evaluate_and_audit`和`resource_gate`。
- [x] 核验00权重、99/24开发数据与24测试文件身份，旧/新源码在24验证序列logits逐事件精确等价；不重训00。
- [x] 冻结源码、配置、数据与结构后，对10/01/11各用train_094和train_096做完整前反向资源预检。6项全部通过。
- [x] 串行随机初始化训练10/01/11，各24轮、seed37、CPU FP32/8线程、AdamW lr0.0007/wd0.0002，保留损失、采样和调度。
- [x] 每组训练结束，独立复评验证集，保存24文件验证/测试概率及掩码，测试固定验证阈值；两套计数实现核验逐文件和全局结果。
- [x] 输出`results.json`、`results.md`和`completion_audit.json`；异常保留日志与恢复状态，重复入口不重复训练。额外完成`final_independent_audit.json`，重新核验610份产物、125份冻结源码、192份掩码及192份概率，8个分组/划分的原始NPZ计数再次完全一致。

### Interpretation

A的Query范围为同一个原有时空块的全部事件，不改变分块、位置编码或事件映射。B各层几何固定不等于堆叠后有效感受野固定，且原精修Head和全序列极性特征仍非因果。B同时改变方向、池化门控和深度，只对这一整体设想作结论。单种子；已有测试查看历史；不根据测试调结构、epoch或阈值。只比较神经网络，不复用不匹配的新结构后处理器。
