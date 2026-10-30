# Mean Pooling + Gated Past Cone Remote Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个可整体上传、无需本地仓库、GPU训练与预测、统一output输出的M1+C4联合实验目录，用户自行传输和运行。

**Architecture:** 复用旧M1纯事件编码均值和旧C4三层门控过去锥，统计特征独立拼接；通过显式M1_C4契约加载和恢复权重。上传包冻结完整traceformer源码、配置、数据哈希与旧结果参考；单一工作流负责GPU预检、续训、验证/测试预测、独立计数与结果归档。

**Tech Stack:** Python 3.10–3.12、PyTorch CUDA、NumPy、SciPy、PyYAML、Matplotlib、scikit-learn、psutil。本地开发，不使用worktree。

---

### Task 1: 联合结构与GPU预检

- [x] 修改`src/traceformer/experiments/ablation.py`与`component_core.py`支持`ComponentCoreSpec('M1_C4')`，复用现有两种替换，不改变旧契约含义。
- [x] `component_suite.production_loss`的batch和采样生成器跟随输入设备；默认CPU结果保持原义。
- [x] 新建`tests/test_mean_past_remote.py`，对联合结构与M1→C4顺序应用逐参数、输出和梯度比较，覆盖无Query、独立统计、三层、门控和过去方向。

### Task 2: 可移植工作流

- [x] 新建`remote/mean_past_gated/workflow.py`、`run.py`、`train.py`、`predict.py`和`README.md`。默认CUDA FP32，显式BF16选项，不自动降到CPU。
- [x] 实现远端启动阶段的真实CUDA内核、合成样本数值对照及train_094/train_096完整损失反向预检。使用99/24/24完整数据哈希，全部结果写`output/full`；smoke单独写`output/smoke`。
- [x] 训练固定seed37、24轮、原AdamW/损失/采样/调度；断点恢复建立新attempt，不覆盖失败现场。预测使用验证阈值，标签隔离，原事件顺序输出概率与掩码。
- [x] 复用`scripts/evaluate_neural_checkpoint.py`并增加可选诊断文件数限制；独立复评分和另一套原始计数核验后生成汇总与`output/results.zip`。

### Task 3: 打包与本地验证

- [x] 新建`scripts/package_mean_past_remote.py`生成`remote_upload/m1_mean_past_gated`，打包完整源码、必要预测脚本、依赖、配置、数据清单、模型摘要及旧结果参考。模型按seed37随机初始化，不携带旧权重。用户已确认远端有数据，包内不复制NPZ。
- [x] 在含空格的独立解压/复制目录中测试离线导入、微型CPU全流程、训练恢复、预测重评分、错数据/错源码/重复运行拒绝；CPU只用于本地自检。
- [x] 执行相关测试、全仓pytest、Ruff和diff检查。414 passed、3 skipped；保存包文件哈希及LOCAL_CHECKS记录。本机无GPU，真实GPU检查由远端启动阶段执行。

### Task 4: 交付

- [x] 提供单文件夹路径、两条数据路径参数、完整运行/仅训练/仅预测/恢复命令，以及需拷回的output目录。
- [x] 更新实际结构、参数量、GPU/CPU分工及FP32/BF16比较边界，不将本地自检当作远端实验完成。
