# 三个外部模型的原生 Windows 预测包（RTX 4090）

本版按用户的 **Windows + RTX 4090、驱动 526.47、不能使用 WSL2** 环境修正。默认使用 CUDA 11.8 配套依赖；三个模型的源码和权重均已包含。

## 这版实际做了什么

上一版错误地默认了 Linux。除了命令不同，官方 HAIS 和 Mamba 扩展也不能直接沿用。现在提供 `torch-reference` 推理后端：

| 模型 | Windows 适配 | 本机已验证 |
| --- | --- | --- |
| EV-SpSegNet | PyTorch mode=4 体素均值，保持首次出现体素顺序及事件逆映射 | 官方模型构造、458 个状态条目严格加载 |
| PACT | 同上；运动估计与传播网络仍使用官方定义 | 官方模型构造、479 个状态条目严格加载 |
| PointEvent | PyTorch 分段 mean/max、状态递推 scan；保留官方序列化及模型权重 | 官方模型构造、278 个状态条目严格加载 |

这条路线使用 spconv 的官方 Windows 轮子，不编译 HAIS，不安装 Linux Mamba/Causal-Conv 扩展。PointEvent 固定配置实际使用仓库自带的 `SimpleAwareStateSpaceModel`；其卷积本来就是 PyTorch 实现，替换的是 scan 扩展边界，不是换一个网络。

**验证边界：** 本机为 Windows、PyTorch 2.11.0+cpu。三个模型均通过 CPU 严格权重加载；参考算子与逐步状态递推、体素聚合及导出对齐检查通过。使用真实 PointEvent 权重的 517-token SSM 前向与串行公式的最大绝对误差约为 `2.98e-8`。本机没有 GPU，尚未执行 4090 整序列推理。按云端旧驱动改用 PyTorch 2.3.1/cu118 配套环境，须先运行下面的 GPU 前向预检，尚未验证该 GPU 组合的完整推理。

参考实现与融合 CUDA 算子的浮点累加顺序不同，不能声称与原官方扩展逐位一致或已经复现文献分数。`report.json` 会明确记录 `backend=torch-reference`。此版用于取得定性图的候选预测；论文归档时应保留后端说明，若要声称官方推理结果一致，仍需做相同输入下的 GPU 数值核对。

## 1. 解压、设置路径与安装

当前已确认原报错的直接原因是 `torch==2.8.0+cpu`。驱动 526.47 也不适合之前的 cu128 默认路线，因此同时切换到 `torch==2.3.1+cu118`、`torchvision==0.18.1+cu118`、`spconv-cu118` 和 `cumm-cu118`。`nvidia-smi` 显示的 CUDA 12.0 不是当前 Python 安装的 CUDA 版本。

普通依赖使用清华 PyPI 源。实测清华没有对应 CUDA 11.8 PyTorch 轮子，GPU 的 torch/torchvision 使用上海交大维护的国内专用镜像；这两个包必须显式带 `+cu118`，避免再次装成 CPU 版。已核对 Python 3.12 / Windows 两个 wheel 的存在和 ZIP 文件头。若要求所有文件只能来自清华，当前无法提供这一组合，不能把 CPU 包当作替代。

打开 **Anaconda PowerShell Prompt**。下面两处 `D:` 路径按你的机器修改：

```powershell
Set-Location 'D:\external_qualitative_windows_20261101'
$QualBundle = (Get-Location).Path
$EventData = 'D:\EV-UAV'
$env:PYTHONUTF8 = '1'

conda create -n trace_qual_windows python=3.12 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate trace_qual_windows
python -m pip install "torch==2.3.1+cu118" "torchvision==0.18.1+cu118" -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/ --timeout 120
python -m pip install -r .\requirements-windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --only-binary=spconv-cu118,cumm-cu118 --timeout 120
python .\tools\verify_bundle.py
python .\tools\check_environment.py
```

已有 `trace_qual_windows` 环境时，省去 create，激活后先删除旧的 CPU torch 和 CUDA 12 依赖，再执行上面两个 pip install 命令：

```powershell
python -m pip uninstall -y torch torchvision spconv-cu126 cumm-cu126
```

安装后先确认 `torch=2.3.1+cu118`、`cuda=11.8`、`available=True`：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`check_environment.py` 会实际执行一次 GPU 稀疏卷积，检查 CUDA、spconv DLL、GPU 内核及三个推理权重，并把报告写到 `results/environment_windows.json`。成功后再运行预测。若失败，先查看该报告里的实际错误；失败不代表已经产生任何模型预测。

这三个模型可以共用这个 Windows 环境。旧的分模型 Linux 环境说明保留在 `README_LINUX.md`，Windows 按本页执行即可。

## 2. 数据目录

`$EventData` 指向包含 `test/` 或 `val/` 的上层目录：

```text
D:\EV-UAV\test\test_000.npz
D:\EV-UAV\test\test_022.npz
D:\EV-UAV\test\test_023.npz
```

NPZ 必须保留官方的 `ev`、`evs_norm`、`ev_loc`；`evs_norm` 为 N×6。脚本核对原始坐标、整数毫秒、极性、标签、事件行序和输入 SHA256。保留原始整段数据，不重算归一化，不预先裁剪时间。权重与数据集是分开交付的，本包不包含数据或 TraceFormer。

先做输入审计，无需 GPU 前向：

```powershell
python .\tools\predict.py --model pact --data-root $EventData --split test --sequences test_000 test_022 test_023 --audit-only --output-root .\audit_results
```

## 3. 运行三个模型

三个命令选取相同的完整序列，统一使用官方阈值 **0.9**，保存完整概率、掩码和事件行号：

```powershell
Set-Location $QualBundle
foreach ($Model in @('ev_spsegnet', 'pact', 'pointevent')) {
    python .\tools\predict.py --model $Model --backend torch-reference --data-root $EventData --split test --sequences test_000 test_022 test_023
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed: $Model; inspect results/$Model/report.json" }
}
```

如需先试一个完整序列，给相应命令添加 `--output-root .\smoke_results`，并只指定 `--sequences test_000`。正式运行使用默认 `results/`，避免与 smoke 混在一起。脚本拒绝覆盖同一模型已有的 `report.json`；失败后选择新的 `--output-root` 再重试。

只有验证集时，三个模型统一改为 `--split val --sequences val_000 val_022 val_023`。本地输入审计用的是这三份验证文件；没有把它们冒充测试集。

Windows 数据暂存优先使用无需管理员权限的硬链接，跨盘时自动复制。PointEvent 使用零个 DataLoader 子进程，并显式保持种子 1024，避免 Windows spawn 及链接权限问题。官方网格采样、分片概率累积和 `inverse` 回原始事件逻辑保持原路径。

## 4. 打包结果带回来

```powershell
python .\tools\collect_results.py --results .\results --archive .\external_qualitative_results.zip
```

将这两个文件下载回本地：

- `external_qualitative_results.zip`
- `external_qualitative_results.zip.sha256`

收集器核查三个方法的输入文件哈希、事件数、完整行号、概率范围和 0.9 阈值掩码。原始事件与 GT 只保存一份，PointEvent 的大份原生 NPZ 留在云端；回传包包含统一预测、环境、日志和每个模型的报告。部分模型尚未完成时也能先回传，`summary.json` 会列出缺失模型。

```text
results/inputs/test_000.npz                 # ev, gt, source_sha256, time_unit
results/ev_spsegnet/predictions/test_000.npz # probability, prediction, event_index, threshold
results/pact/predictions/test_000.npz
results/pointevent/predictions/test_000.npz
results/<model>/report.json                # 后端、输入/权重哈希、环境和真实完成状态
results/environment_windows.json           # GPU 稀疏卷积预检
```

三维图之后用相同原始 x/y/t、时间范围、事件显示采样和视角绘制。时间单位为毫秒，统一换秒时除以 1000。TraceFormer 也必须使用相同源文件及其预先确定的模型/阈值。普通/高密度场景依据输入统计确认，不按某个模型的预测好坏挑选。Table 3 的文献 IoU 不能当成本次运行成绩。

## 权重与来源

- EV-SpSegNet：用户提供 `best_iou_seed37.pt`，3,944,557 字节，SHA256 `af394a2bdf34959bd2a46013bf7600c4c9e3d71b3f4fc51f68a0b85f159efd94`。
- PACT：官方 Release 的 `checkpoint_improved.pt`，3,005,964 字节，SHA256 `337779c666f326f989bbf93223480686957060e4ca6ac62b3c874efb8f148d2a`。
- PointEvent：保留官方 `model_best.pth`，默认使用已逐张量核对一致的 `model_inference.pth`，避免未发布的训练 scheduler 对象。提取来源和方法见 `manifest.json`、`tools/extract_point_weights.py`。

官方模型文件保留原样；Windows 导入适配发生在内存中，严格权重加载仍启用。原始源码提交、权重哈希、Windows 检查证据与全部文件校验分别记录在 `manifest.json`、`verification.json`、`provenance/` 和 `SHA256SUMS`。

[EV-SpSegNet 官方源码](https://github.com/ChenYichen9527/EV-UAV) · [PACT 官方源码](https://github.com/fulongcai/PACT) · [PointEvent 官方源码](https://github.com/wzz-z/PointEvent)

[spconv-cu118 官方 Windows 轮子](https://pypi.org/project/spconv-cu118/2.3.8/) · [Mamba v1.2.2 官方 Linux 要求](https://github.com/state-spaces/mamba/tree/v1.2.2#installation) · [官方 scan 参考公式](https://github.com/state-spaces/mamba/blob/v1.2.2/mamba_ssm/ops/selective_scan_interface.py) · [上海交大 PyTorch 镜像说明](https://mirror.sjtu.edu.cn/docs/pytorch-wheels) · [PyTorch 官方版本配套](https://pytorch.org/get-started/previous-versions/)
