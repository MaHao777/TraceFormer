# 同任务外部基线复现（更新至 2026-09-29）

此记录替代早期 250 ms 密集体素方案的规划。当前 TraceFormer 使用完整 8 秒输入和占用块 token；外部方法保留官方原生输入与网络，不统一成 TraceFormer 表示。

## 已获取的官方基线

|方法|官方来源|冻结提交|当前状态|
|---|---|---|---|
|PACT|https://github.com/fulongcai/PACT|02e9e4d2025097b9b54708f052a6f09072cefabf|4090D 全 24 序列完成，服务器及下载后本地复评通过|
|EV-SpSegNet|https://github.com/ChenYichen9527/EV-UAV|33e2affb51a204676536e119803ab8494e2eb0a9|源码已下载；Google Drive 权重链接返回登录 HTML|

PACT 的公开 checkpoint 已用 `torch.load(weights_only=True)` 验证为原始 state dict，SHA256 为 `337779c666f326f989bbf93223480686957060e4ca6ac62b3c874efb8f148d2a`。EV-SpSegNet 的网页另存为 `.download.html`，不作为模型。PACT 源码提供 Apache-2.0 LICENSE；该许可不自动覆盖数据与权重。EV-UAV 此提交未提供顶层 LICENSE。

下载清单为 `outputs/baselines/upstream_manifest.json`，包括源码提交、归档/权重哈希与失败原因。获取入口是 `scripts/prepare_external_baselines.py`。M2E-UAV 官方仓库目前仅 README，无可执行模型，因此本轮未列为可复现基线。

## 统一比较契约

- 使用相同的 `val_000` 到 `val_023`，所有原始事件保留，统一通过现有官方兼容评分器累积全局计数。
- 官方稀疏卷积网络读取 `evs_norm[:, :4]`；`ev_loc` 为原始 x/y 与整数毫秒。两套模型的 24 文件字段、事件顺序、注释和哈希已审计通过。注释仅进入评分器，网络只接收事件体素。
- 保留官方 HAIS 体素化与 352×288×8192 稀疏形状。输出必须按 `p2v_map` 映射回完整事件；若 decoder 改变体素索引顺序，适配器停止，不能隐式错配。
- PACT 上游默认 0.9、共同固定 0.5 与统一 validation 阈值搜索分别报告。公开权重训练集成员身份仍未独立证实，不把它们称为同训练协议重训。
- 冻结的 TraceFormer 神经及完整系统已在同一 4090D 主机复评。CPU 消融保持原协议，不能用于同环境速度比较。
- 不复制 TraceFormer LightGBM 到外部 embedding；外部后处理公平比较必须另行训练匹配的分类器并记录相同开发预算。

适配器 `scripts/evaluate_external_baseline.py` 已通过 PACT 严格权重加载、原生体素化、完整 24 序列事件映射和计分。PACT 输出为 `complete`；EV-SpSegNet 仍为 `data_audited_inference_pending`，不能把后者当作复现成绩。

## 准备阶段的历史限制（现已获得 CUDA 主机）

本机 PyTorch 为 2.11.0+cpu，CUDA 不可用。官方网络需要 CUDA、spconv 和编译 HAIS；已有 SSH 地址在本次连接时被拒绝。未安装不兼容的 CPU 替代算子，未把论文数字充当复现结果。

待可用 Linux CUDA 主机和 EV-SpSegNet 权重后，先验证 CUDA/扩展及严格加载，运行全部 24 文件，再独立评分和核对映射。Linux 需把配置的数据路径及 manifest 的本地路径更新为实际挂载位置。

```powershell
python scripts/evaluate_external_baseline.py --model PACT --config outputs/paper_mechanism_cpu_20260928/configs/seed37_cone.yaml --output outputs/baselines/PACT_validation_20260928
python scripts/evaluate_external_baseline.py --model EV-SpSegNet --config outputs/paper_mechanism_cpu_20260928/configs/seed37_cone.yaml --output outputs/baselines/EV-SpSegNet_validation_20260928
```

## 2026-09-28 CUDA 执行更新

已连接用户授权的 4090D 24 GB 服务器，在独立 venv 使用 PyTorch 2.8.0+cu128、CUDA 12.8、spconv-cu126 2.3.8、LightGBM 4.6.0。24 验证文件、公开 PACT 权重、匹配 schema-4/5 的历史神经源码和冻结树模型已上传并逐文件验哈希。PACT 的 HAIS 编译只修正 ABI 布尔值输出为整数，未改网络算法，且通过完整 val_000 的严格权重加载与前向 smoke；smoke 不算基线成绩。

TraceFormer 主/参考网络均完成全24文件 GPU 推理和独立评分，发布阈值 Score 分别为 0.9150020823918557 / 0.9145447953504824。完整系统和 PACT 按队列继续运行；EV-SpSegNet 官方权重在服务器重试也超时，仍缺少真实权重。结果不得提前写作完整外部复现。

监督自动化每三小时在当前聊天检查真实进程、报告和完成后核验，再按实测结果回填论文。远端启动和故障恢复记录见 `docs/paper_cuda_supervision.md`；本地证据目录为 `outputs/paper_cuda_20260928`。未保存登录密码或添加持久 SSH 登录密钥。

## 2026-09-29 完整结果与论文回填

服务器批次已完成，GPU 空闲，不重复启动。原始完整系统独立命令误用了 NPZ 的上层目录，有限审计进程仅修正该计分入口并得到完全相同的计分报告；原始失败日志保留，不重跑前推。下载包哈希与服务器一致，276 个输入/源码文件、权重、缓存和预测哈希核验通过。`scripts/verify_paper_cuda_copyback.py` 从本地原始 24 文件重新计算所有阶段和概率阈值结果，`outputs/paper_cuda_20260928/copyback_20260929/local_verification.json` 为 verified。

|方法 / 操作点|阈值|IoU (%)|SegAcc (%)|Pd (%)|Fa (10^-6)|Score|
|---|---:|---:|---:|---:|---:|---:|
|PACT 公开权重 / 上游默认|0.9000|46.189|79.585|85.363|5.063|0.798604|
|PACT / 共同固定|0.5000|43.773|84.119|90.403|28.368|0.759181|
|TraceFormer neural / 共同固定|0.5000|86.647|93.707|94.771|11.744|0.912843|
|PACT / 相同验证搜索|0.8250|46.531|81.554|87.841|7.896|0.803205|
|TraceFormer neural / 相同验证搜索|0.3875|86.756|94.349|95.590|12.488|0.915002|
|TraceFormer complete / 冻结规则|不适用|92.236|95.359|96.808|4.411|0.954117|

PACT 有 705401 个神经参数。选定阈值下 val_015 贡献 47794/49306 个误报事件，val_023 仅恢复 56/413 个目标帧实例；必须结合逐序列图解释聚合差距。训练成员及开发预算没有独立匹配，因此这些结果只描述所执行的公开权重与发布权重在验证集上的表现，不证明 SOTA 或架构普遍优越，也不能与 31 文件官方测试成绩混排。

论文已加入同环境表、全 24 序列图、direct 单独阶段及公开权重来源/ABI 构建兼容说明，并编译渲染检查。EV-SpSegNet 仍缺真实官方权重，已向用户请求可访问文件；该请求未回复前不重复无效下载。CPU 三种子消融继续正常运行，未改冻结协议。最新 QA 在 `D:/Paper/TraceFormer/evidence/cuda_comparison_qa.json`。
