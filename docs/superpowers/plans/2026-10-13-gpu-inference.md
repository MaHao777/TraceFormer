# GPU 推理适配 Implementation Plan

> **For agentic workers:** 使用本次会话逐项实施；用户已授权直接修改本地仓库，不创建 worktree，不启动其他 agent。

**Goal:** 为当前网络、冻结发布系统和 Track 2 提供可选 GPU 邻域、事件树及事件特征后端，保留 CPU 默认行为和原有检查点契约。

**Architecture:** KD-tree 仍按原顺序在每个视图准备一次，GPU 只负责查询，使用原树的叶序和等距离堆规则。事件树导出为只读数值数组，GPU 按原树顺序以 float64 累计；CUDA 内核通过 Numba 的 CUDA Array Interface 使用 Torch 的内存和当前 stream，避免中途复制。特征构造单独可选，固定原缓存量化与 float32 特征模式。

**Tech Stack:** PyTorch、SciPy、LightGBM、可选 numba-cuda；本机用 CUDA simulator 执行同一批内核，真实 CUDA 验证作为独立验收项。

---

### 1. 邻域查询

- [x] 创建 `src/traceformer/gpu/runtime.py`：懒加载 CUDA、共享 Torch stream、包装 CUDA Array Interface。
- [x] 创建 `src/traceformer/gpu/kernels.py`：二维 Chebyshev KD 查询、原 heap tie 顺序、按 rank/time/distance 保留上限；队列溢出显式报错，不丢邻居。
- [x] 创建 `src/traceformer/models/gpu_neighbors.py`：打包现有 SciPy 树，GPU 输出索引；几何结果不跨前向缓存。
- [x] 修改 `cone_neighbor_builder.py` / `cone_transformer.py`：可选后端和共享 GPU 几何；独立 Attention 仍可自行准备。
- [x] 增加冻结源码的运行时 adapter：只替换 builder 并在 Encoder 一次调用内共享上下文，保留归档文件、模型权重键与参数数目。
- [x] 用 simulator 和 SciPy 原构建器比较多 batch、缺失时间片、重复/等距离、半径边界、不同时间尺度和空输入。

### 2. 树预测与事件特征

- [x] 创建 `src/traceformer/postprocessing/gpu.py`：打包 numerical binary LightGBM 树，保留缺失值路由、double 阈值、sigmoid 和树累计顺序；不支持的目标/分裂显式拒绝。
- [x] GPU 特征在稳定分组内按原事件顺序累计概率，轨迹窗口沿用原 3x3 空间和相邻帧规则；保持缓存半精度量化契约和每块 200,000 行。
- [x] 为 `load_event_models` / `predict_event_scores` 增加可选后端，训练和 OOF 仍默认 CPU；无有效树不导入 CUDA 依赖。
- [x] 用 simulator 对真实发布树、CPU LightGBM、缺失值和小型完整特征进行对照；CUDA 数学库差异通过真实 GPU 核验报告，不承诺未经实测的逐位等价。

### 3. 入口和验收

- [x] 将后端参数接入普通预测/缓存、Track 2、后处理预测/分阶段评估和冻结系统 benchmark；报告记录实际后端。
- [x] GPU 模式保留有限数值、模型哈希、输入顺序和原提交格式检查；依赖缺失或设备不匹配应在创建输出之前报错。
- [x] 增加 GPU 检查入口：分项核对邻居、树分数、特征与模型输出，并可用原冻结 benchmark 全集核对掩码；本轮不重新训练。
- [x] 执行相关测试、完整 pytest、Ruff、`git diff --check`；真实 CUDA 编译、完整 val/test 等价性与速度明确列为本机未执行项目。

CPU 入口默认不改变。GPU 运算可能改变 log/sigmoid 最后一位，不能依据 simulator 自动宣称真实 GPU 精确等价；冻结系统核验必须如实报告差异并阻止通过验收。

2026-10-13 已在用户提供的 RTX 3090 上补齐真实 CUDA 编译、stream 互操作、当前模型完整 val/test 核验及冻结系统三轮全链路计时。结果见 [真实 GPU 验证报告](../../gpu_validation_20261013.md)，原始记录在 `outputs/gpu_remote_20261013`。Track 2 GPU 全集提交生成和训练吞吐仍未测量，不能外推该速度比。
