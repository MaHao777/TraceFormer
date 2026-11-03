# 实验使用导航

日常维护四套实验。安装 `pip install -e ".[experiments]"` 后，统一使用
`traceformer-experiment`；`python -m traceformer.cli.experiment` 与它等价。

|ID|用途|变体|
|---|---|---|
|mean_past_gated|M1 均值池化 + 三层门控过去锥|M1_C4|
|local_pooling_six|池化与过去锥局部注意力对照|S0/S1/S2/L0/L1/L2|
|euclidean_pooling_four|均值/单核池化 × 小局部/三尺度中心欧氏过去锥|E0/E1/E2/E3|
|s1_paper_ablation|S1论文结构与创新点筛选消融|S1/S0＋7种新增结构|

新增四组的结构、启动和解释边界见 [欧氏过去锥四组实验](euclidean_pooling_four.md)。
四组全部重新训练，all/train自动执行首次预检，不接受旧基线或BF16。
云端已完成的四组结果与本地独立复核见 [2026-10-31回传分析](euclidean_pooling_four_results_20261031.md)。
S1论文筛选包固定seed37、复用有效S1/S0，见 [结构及创新点消融说明](s1_paper_ablation.md)。

历史模型结构与 checkpoint 契约保留原定义，新欧氏实验使用版本5消融契约。workflow只规定阶段顺序；日志、锁、
子进程、恢复、概率复评和原始掩码独立核验由公共运行层提供。
此前流程整理的数值对照、539项通过测试及限制见 [整理验证记录](reorganization_20261030.md)。

## 本地运行

`--output` 必填，并使用尚不存在的目录。首次运行先在其中的 `source/` 冻结完整执行包，
后续 worker 只导入这份源码。配置默认来自现有无 Sparse 完整训练；也可明确传入配置。

```powershell
traceformer-experiment list
traceformer-experiment run mean_past_gated --output outputs/experiments/mean_past_run --data-root data/ev-uav --test-root ../EV-UAV-dataset
traceformer-experiment run local_pooling_six --output outputs/experiments/pooling_run --data-root data/ev-uav --test-root ../EV-UAV-dataset --baseline-root remote_result/26-10-28
traceformer-experiment status --output outputs/experiments/pooling_run
```

阶段为 `all`（默认）、`preflight`、`train`、`predict`。首次只训练可指定 `--stage train`；
`--stage predict` 复用完成的训练，完成验证/测试预测、独立复评、计时、汇总与归档。

恢复使用同一输出目录及 `--resume`，不再传基线/参考指标等打包选项。中断训练新建
`attempt_*`，已完成训练核验后复用；配置、数据身份或源码变化会拒绝恢复。
若工作副本已经变化，直接运行原冻结 `source/run.py --output <原输出目录> --resume`。
活动 worker 或锁冲突会拒绝重复启动，不覆盖原状态。

```powershell
traceformer-experiment run local_pooling_six --output outputs/experiments/pooling_run --resume
```

CPU 诊断显式使用 `--smoke --device cpu`，仅运行两个文件、一轮训练，写入独立 `smoke/`。
它不证明正式精度或 CUDA 数值一致性。正式结果在 `full/`：

```text
<运行目录>/
  source/                    冻结源码、原始配置及文件身份
  full/ 或 smoke/
    experiment.json          实际配置、精度、分块及源码身份
    status.json, workflow.log
    [各变体]/training/attempt_001/
    [各变体]/val/、test/      概率、掩码和两套独立计数
    results.json, results.md, completion_audit.json
  results.zip 或 smoke_results.zip
```

M1+C4 没有变体子目录。训练与预测分别记录状态；只有 `complete` 代表整个闭环完成。

## 生成远端包

```powershell
traceformer-experiment bundle mean_past_gated --output remote_upload/mean_past_new
traceformer-experiment bundle local_pooling_six --output remote_upload/pooling_new --baseline-root remote_result/26-10-28
```

打包器拒绝覆盖目录。默认使用仓库的数据路径；可用 `--config`、`--data-root`、`--test-root`
指定输入。六组默认核验并复用原 M1+C4 基线；需要从头训练 S0 时明确使用 `--without-baseline`。
无效基线直接报错。M1+C4 的历史参考指标可通过 `--references <JSON>` 携带，不参与训练选择。

整体上传生成目录，安装 `requirements.txt` 后执行唯一的 `run.py`：

```bash
python -u run.py --data-root /path/to/data --test-root /path/to/data
python -u run.py --stage train --data-root /path/to/data
python -u run.py --stage predict --test-root /path/to/data
python -u run.py --resume
```

依赖清单不替换已有 CUDA PyTorch。远端默认写入包内 `output/full/`，回传 `output/results.zip`。
包内没有旧 `workflow.py`、`legacy_workflow.py`、`train.py`、`predict.py` 或评分脚本副本。

## 两套实验的执行规则

- M1+C4 保留真实数值与最大序列预检，默认 CUDA FP32，可显式 `--precision bf16`；独立预测仍使用 FP32。
- 六组固定 CUDA FP32，先完成所有组训练，再统一评分。默认不执行耗时预检；显式 `--stage preflight` 才诊断。
- 六组耗时和显存只记录，不使用旧的 1.25 倍/百分位预算中止规则。完成后在测试集测真实网络前向，读取缓存重评分的时间不算网络速度。
- 查询分块默认 8192，空间 Token 分块默认 4096；对应环境变量为 `TRACEFORMER_LOCAL_QUERY_CHUNK`、`TRACEFORMER_SPATIAL_TOKEN_CHUNK`。首次实际值冻结到契约，恢复时不能改变。
- S0 复用保留原 `M1_C4` 契约、权重和训练记录字节，并在新组训练前重新独立复评验证集。
- 所有测试预测使用冻结验证阈值，不在测试集重新选择参数。

## 历史实验

运行 `traceformer-experiment list` 查看归档用途、原命令、匹配源码、权重、报告和缺失依赖。
详细登记在 [catalog.json](../../archive/experiments/catalog.json)，分类说明在
[历史实验归档](../../archive/experiments/README.md)。

```powershell
traceformer-experiment archive-run component_core -- --help
traceformer-experiment archive-run --source-root <相同字节的源码快照> component_core -- --help
```

归档入口在独立进程中使用已核验的历史模型源码；缺失依赖或产物时明确报错。
已有训练继续使用原冻结入口恢复，新入口不接管旧目录。历史报告和计划原文保留，
原始 `remote_result/` 与已有 `remote_upload/` 保留在本地，不进入源码版本库。

默认完整主模型仍存在 schema-7 产物与 schema-8 加载器错配，此项单独待修复，
不属于本次实验流程整理的验收范围。
