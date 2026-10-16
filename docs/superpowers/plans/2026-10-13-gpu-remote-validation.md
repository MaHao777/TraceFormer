# GPU 远端验证与计时方案

> 在本次会话直接执行；本地代码修改，不创建 worktree，不启动其他 agent。凭据只用于内存中的 SSH 登录。

**Goal:** 在用户提供的 RTX 3090 服务器验证 GPU 后端的真实编译、输出一致性与完整系统耗时，复制回可复核证据。

**Architecture:** 新建 `/root/autodl-tmp/traceformer-gpu-20261013`，复用已核验的冻结权重及测试数据。安装独立 venv，保持同一权重、阈值、输入和线程设置，对比 CPU、仅网络 CUDA、全部已适配 CUDA 三种路径；先验收小输入及最大真实序列，再进行全集重复计时。任何差异先定位并修复本地代码，不能跳过一致性检查。

**Tech Stack:** Paramiko、PyTorch 2.8.0+cu128、numba-cuda、LightGBM、pytest。

- [x] SSH 登录，核对 RTX 3090、CUDA 12.8、14 核 CPU 配额和已有任务。
- [x] 上传本地当前源码及补充数据，用 SHA256 清单核对代码、冻结模型、输入和掩码。
- [x] `python -m pytest tests/test_gpu_inference.py -q` 与 `python scripts/check_gpu_inference.py --output synthetic.json`；检查真实 JIT 和 stream 互操作。
- [x] 用 `--sequence dataset/test/test_023.npz` 检查最大真实测试序列的邻居、网络输出、特征和树分数；实际最大文件按审计选取。
- [x] 在同一服务器运行 `scripts/benchmark_frozen_system.py` 的 CPU、网络 CUDA 和完整 CUDA 路径。相同线程、完整预热；CPU 与完整 CUDA 每条序列三次，中间路径一次用于诊断。独立核对全部 24 个测试序列与冻结掩码及指标。
- [x] 记录网络、事件特征、事件树、跟踪、参考恢复和传输时间；必要时细分现有计时字段，不改计时边界和决策。
- [x] 核对当前模型的完整验证集与测试集，保留原阈值，不重新训练。
- [x] 复制回环境、日志、计时报告和预测，核对传输哈希并在本地独立重新评分；运行相关测试、Ruff 和 diff 检查。
- [x] 交付真实 GPU 效果、每部分耗时、速度比及剩余限制；不以历史机器计时代替同机比较。

结果见 [真实 GPU 验证报告](../../gpu_validation_20261013.md)：CPU 12.950 秒/序列，完整 GPU 0.872 秒/序列，14.85 倍；当前 48 个 val/test 序列及冻结 24 个测试序列掩码一致，420 个实验文件回传并本地独立复评通过。
