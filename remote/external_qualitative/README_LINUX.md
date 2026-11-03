# EV-UAV 定性对比：三个官方外部模型（RTX 4090）

本包用于把官方预训练模型的完整逐事件预测带回本地绘图。打包日期：2026-11-01。

## 已放入的内容

| 模型 | 源码提交 | EV-UAV 权重 |
| --- | --- | --- |
| EV-SpSegNet | `33e2affb51a204676536e119803ab8494e2eb0a9` | 用户提供的 `best_iou_seed37.pt`，已通过 CPU 权重读取 |
| PACT | `02e9e4d2025097b9b54708f052a6f09072cefabf` | 官方 Release 的 `checkpoint_improved.pt` |
| PointEvent | `f0781fb2154a8724411049192336d9782cf99ec1` | 官方 `model_best.pth`，另附张量完全一致的推理副本 |

三个 `models/` 子目录均为上述提交的官方源代码，排除了上游误提交的 `build/`、`__pycache__/`、`.egg-info/` 和编译二进制。保留原始 README、配置及已有许可证。完整来源、哈希及本地检查结果见 `manifest.json`、`verification.json` 和 `SHA256SUMS`。

```text
external_qualitative_20261101/
  models/ev_spsegnet/checkpoints/best_iou_seed37.pt
  models/pact/checkpoints/checkpoint_improved.pt
  models/pointevent/exp/EV-UAV/model/model_best.pth
  models/pointevent/exp/EV-UAV/model/model_inference.pth
  tools/predict.py
  tools/collect_results.py
  tools/setup_hais.py
  tools/extract_point_weights.py
  manifest.json
  SHA256SUMS
```

EV-SpSegNet 权重已放入：

```text
models/ev_spsegnet/checkpoints/best_iou_seed37.pt
```

[官方百度权重链接](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4)，提取码 `ztp4`。用户提供文件与公开分享页的文件名和 **3,944,557 字节**大小一致；本地 SHA256 为 `af394a2bdf34959bd2a46013bf7600c4c9e3d71b3f4fc51f68a0b85f159efd94`。已使用 `torch.load(weights_only=True)` 读取，包含 458 个张量条目；尚未执行 GPU 严格加载与推理。[官方 Google Drive 链接](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view) 在打包时返回登录页面，最终文件由用户手动提供。

PointEvent 原文件包含优化器和未发布的 `utils.scheduler.OneCycleLR` 对象，因此默认使用 `model_inference.pth`。提取工具仅对已核查 SHA256 的官方文件截取 `epoch` 与 `state_dict`，不加载训练对象，原样保留张量存储，再使用 `weights_only=True` 读取。已核对 **278 个张量逐一一致**，原始 checkpoint 同时保留。

本包不含数据集、TraceFormer 代码或其权重。数据使用云端现有的官方 EV-UAV NPZ；后续 TraceFormer 预测需使用相同源文件与已确定的模型和阈值。

## 上传后先设置数据路径

上传整个文件夹或解压同名 ZIP，在 Linux 终端运行：

```bash
cd /你的上传目录/external_qualitative_20261101
export QUAL_BUNDLE="$PWD"
export EVUAV_DATA=/root/autodl-tmp/EV-UAV
sha256sum -c SHA256SUMS
```

`EVUAV_DATA` 应指向包含 `test/`、`val/` 的目录，**不是**直接指向 `test/`。

```text
EV-UAV/test/test_000.npz
EV-UAV/test/test_022.npz
EV-UAV/test/test_023.npz
```

每个文件必须保留官方 `ev`、`evs_norm`、`ev_loc` 三个字段，`evs_norm` 为 N×6。模型只读取官方事件特征；标签用于原始事件对齐检查与结果保存。不要先裁剪时间或重算归一化。脚本会检查原始事件行序、整数毫秒坐标、极性、标签及文件 SHA256，不符合时停止。

## 1. PACT 环境与预测

PACT 的本地历史证据已记录 **4090D + PyTorch 2.8.0/cu128 + spconv-cu126 2.3.8** 完成全验证集推理；见 `provenance/pact_4090d_environment.json`。本包新导出入口尚未在 GPU 上执行。建议使用带 **CUDA 12.8 开发工具链** 的 4090 镜像，`nvcc --version` 应显示 12.8；仅有 `nvidia-smi` 显示 CUDA 版本并不代表安装了 nvcc。

```bash
conda create -n trace_qual_pact python=3.12 -y
conda activate trace_qual_pact
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$QUAL_BUNDLE/requirements-pact.txt"
apt-get update
apt-get install -y build-essential libsparsehash-dev
nvcc --version
export TORCH_CUDA_ARCH_LIST=8.9
cd "$QUAL_BUNDLE/models/pact/lib/hais_ops"
python "$QUAL_BUNDLE/tools/setup_hais.py" build_ext --inplace
cd "$QUAL_BUNDLE"
python tools/predict.py --model pact --data-root "$EVUAV_DATA" --split test --sequences test_000 test_022 test_023
```

如果镜像未安装 nvcc，可在当前环境安装匹配开发工具链后再编译：

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
```

`setup_hais.py` 使用历史已验证的编译方式，将 ABI 宏输出为整数 0/1。网络和体素算子实现保持官方代码。推理逐文件、batch=1，保留全序列，避免 PACT 官方 batch=2 的额外峰值显存。

## 2. PointEvent 独立环境与预测

官方没有提供锁定版本的环境文件。下列组合根据其 `spconv-cu118`、旧版 Mamba/Causal-Conv 接口整理，**仍需在云端做首次 GPU 验证**。保持独立环境，避免安装时替换 PACT 的 PyTorch。

```bash
conda create -n trace_qual_point python=3.10 -y
conda activate trace_qual_point
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$QUAL_BUNDLE/requirements-pointevent.txt"
python -m pip install torch-scatter==2.1.2 --only-binary=torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=8.9
python -m pip install causal-conv1d==1.2.2.post1 --no-build-isolation
python -m pip install mamba-ssm==1.2.2 --no-build-isolation
python -m pip check
cd "$QUAL_BUNDLE"
python tools/predict.py --model pointevent --data-root "$EVUAV_DATA" --split test --sequences test_000 test_022 test_023
```

如机器已有 CUDA 11.8 开发工具链，省去 conda toolkit 安装，并把 `CUDA_HOME` 指到该工具链目录。CUDA 11.8 编译需兼容的编译器，例如 GCC/G++ 11；系统默认版本过新时安装 `gcc-11 g++-11` 并设置 `CC=gcc-11 CXX=g++-11`。

入口直接运行官方 `tools/test.py`，保留配置里的网格采样、分片累积及 `inverse` 回原始事件映射。只设置路径、所选序列、固定种子 1024、单个 worker，并关闭官方 TXT/PNG 导出。为兼容官方只识别 `test+` 文件名前缀的逻辑，临时链接统一放在 `test/` 下；原始文件名、数据及报告中的真实 split 不变。统一 NPZ 保存完整概率，三维图在本地画；官方 `native/` 输出也保留在云端供排查。

## 3. EV-SpSegNet 独立环境与预测

可从 PACT 的基础环境复制一个独立环境，再编译 **EV-SpSegNet 自己的** HAIS 扩展：

```bash
conda create -n trace_qual_ev --clone trace_qual_pact -y
conda activate trace_qual_ev
export TORCH_CUDA_ARCH_LIST=8.9
cd "$QUAL_BUNDLE/models/ev_spsegnet/lib/hais_ops"
python "$QUAL_BUNDLE/tools/setup_hais.py" build_ext --inplace
cd "$QUAL_BUNDLE"
python tools/predict.py --model ev_spsegnet --data-root "$EVUAV_DATA" --split test --sequences test_000 test_022 test_023
```

若 PACT 使用 conda 安装的开发工具链，需要重新设置 `CUDA_HOME="$CONDA_PREFIX"`。EV-SpSegNet 两个 C++ 头文件含已移除的 `THC/THC.h`，编译助手仅删除这两个未使用的 include，使其与相同提交的 PACT 头文件一致，并写出 `compatibility_patch.json`。此适配尚未做 EV-SpSegNet 的权重严格加载与 GPU 验证。模型权重始终严格加载；发生版本或形状不匹配会停止，不会跳过参数。

## 输入检查与执行范围

安装 NumPy 后可先做输入审计，无需 GPU 或权重；审计输出另放目录，避免占用正式结果：

```bash
python tools/predict.py --model pact --data-root "$EVUAV_DATA" --split test --sequences test_000 test_022 test_023 --audit-only --output-root audit_results
```

首次可以对一个完整序列做 smoke，再在新的输出目录跑三组；同一模型已有 `report.json` 时拒绝覆盖。三个模型应使用相同的 `--sequences`。需要更换序列时三个模型一起重跑，不能按预测好坏给每个模型单独选样本。

本地只有 train/val 数据，因此本地实际输入审计使用 `val_000`、`val_022`、`val_023`；未声称已检查云端 `test_000/022/023`。若缺 test 或公开 GT，也可将三个命令都改成 `--split val --sequences val_000 val_022 val_023`。`test_022/023` 是否高密度，须根据真实输入统计确认，当前不预先定性。

## 固定的比较协议

- 三个外部模型均采用官方 **0.9** 阈值，保留完整概率；不提供测试样本调阈值入口。
- 使用相同 NPZ、完整序列上下文及原始事件顺序，保留每个方法自己的官方预处理。
- 三维图统一原始 x/y/t、时间范围、坐标范围、视角及用于显示的事件采样索引。数据时间单位为毫秒，绘图换秒时统一除以 1000。
- 所有预测输出与 `inputs/<序列>.npz` 中的 `ev` 按行一一对应。GT 单独保留为 `gt`。
- 论文 Table 3 的 IoU 是文献报告，不能当作本次公开权重的预测成绩。公开权重训练成员及开发协议未独立核实；本包不输出复现分数。
- 云端 GPU 推理尚未执行。依赖安装、严格权重加载、整序列显存及 PointEvent CUDA 接口兼容性仍需在 4090 上确认。发生 OOM 时先更换显存条件，不要静默裁剪序列或删事件。

## 结果回传

预测写入 `results/`，三个模型共用一份原始事件：

```text
results/inputs/test_000.npz                 # ev, gt, source_sha256, time_unit
results/pact/predictions/test_000.npz       # probability, prediction, event_index, threshold, source_sha256
results/pact/report.json                   # 真实环境、权重哈希、输入哈希、运行状态
results/pointevent/predictions/test_000.npz
results/pointevent/native/test.log          # 官方执行日志
results/ev_spsegnet/predictions/test_000.npz # EV-SpSegNet 结果
```

任意装好 NumPy 的环境均可打包：

```bash
cd "$QUAL_BUNDLE"
python tools/collect_results.py --results results --archive external_qualitative_results.zip
```

把 **`external_qualitative_results.zip` 和 `.zip.sha256`** 下载回来即可。收集器检查输入文件哈希、事件数、完整行索引、概率范围、阈值和二值掩码的一致性，再打包原始事件、GT、统一预测与日志；不重复打包 PointEvent 的大份原生 NPZ。也可以先收集已完成的部分模型，`summary.json` 会明确列出缺失模型。

## 官方资源

- [EV-SpSegNet / EV-UAV](https://github.com/ChenYichen9527/EV-UAV)
- [PACT](https://github.com/fulongcai/PACT) · [EV-UAV Release](https://github.com/fulongcai/PACT/releases/tag/v1.0)
- [PointEvent](https://github.com/wzz-z/PointEvent) · [原始 checkpoint](https://github.com/wzz-z/PointEvent/blob/main/exp/EV-UAV/model/model_best.pth)
- [Mamba 1.2.2 安装要求](https://pypi.org/project/mamba-ssm/1.2.2/) · [spconv-cu118 2.3.8](https://pypi.org/project/spconv-cu118/2.3.8/)
