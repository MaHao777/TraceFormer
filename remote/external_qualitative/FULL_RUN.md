# 三个外部模型完整验证集、测试集预测

在昨天已经跑通的 **Windows + RTX4090 + trace_qual_windows** 环境中使用。本补丁只新增批跑、指标分析和回传工具，不修改已有模型源码、权重、参考算子、predict.py或依赖；无需重新下载模型、安装环境。

## 直接运行

将增量ZIP解压到昨天的包目录，确保该目录下新增 `tools/run_full.py` 和本文件。在已激活的环境中执行，数据路径改成云端实际目录：

```powershell
Set-Location 'E:\mahao\TraceFormer\external_qualitative_windows_20261101'
$EventData = 'E:\mahao\TraceFormer\EV-UAV-dataset'
$env:PYTHONUTF8 = '1'
python .\tools\run_full.py --data-root $EventData --output-root .\results_full_20261102
```

`$EventData` 必须同时包含完整的 `val/*.npz` 和 `test/*.npz`，各24文件。只上传昨天三个测试文件时，需要先补齐另外21个测试文件和全部24个验证文件。数据必须保留原来的 `ev,evs_norm,ev_loc`，不能只保留原始事件。

先审计48个输入文件、暂不执行GPU预测：

```powershell
python .\tools\run_full.py --data-root $EventData --output-root .\results_full_20261102 --audit-only
```

程序顺序运行 `val × 三模型` 和 `test × 三模型` 六个独立进程，共144份逐事件预测。每个split都使用全部24文件；不传`--sequences`。阈值保持0.9、种子1024、batch=1，沿用已经跑通的torch-reference后端。验证集与测试集分别保存，昨天的results目录不会被覆盖。

完成一个模型/split会有独立report.json，重跑同一命令时复用身份一致的完整任务。失败或中断的模型目录先移动到当前split的 `_failed_attempts/` 保留，再重新预测该模型的整个split；不保证从某个序列继续。已经完成的模型会在最终收集时再次验证概率、掩码和行号。损坏的完整任务或输入/权重身份变化会报错，此时使用新的输出目录。

某个任务失败后仍继续另外五个任务，并尽可能产生标记partial的回传包；终端会以非零状态结束。请同时查看 `run_status.json` 和 `logs/`。再次运行同一命令可重试失败任务。数据文件必须在运行期间保持不变。

## 拿回哪些文件

成功后将输出目录同级的两个文件拷回本地：

- `results_full_20261102_return.zip`
- `results_full_20261102_return.zip.sha256`

如果重跑后旧回传包已存在，会生成带数字后缀的新包，保留旧包。ZIP内容：

```text
external_full_results/
  summary.json                  # complete/partial、每个split的完整模型清单
  dataset_inventory.json        # 48份源NPZ的SHA256、事件数、GT数与时间范围
  qualitative_selection.json    # 三个普通+一个密集测试序列预设
  paper_reference.json          # 三篇论文原始数值、页码和口径说明
  analysis/metrics.json         # 全局计数、逐文件指标、概率诊断、论文数值差
  analysis/metrics_summary.csv  # 6行全局汇总，比例单位为0到1
  analysis/metrics_per_sequence.csv  # 144行逐文件指标和概率诊断
  val.zip                       # 沿用昨天的结果包结构，含24份公共输入和72份预测
  test.zip                      # 同上，独立保存测试集
  environment_windows.json
  run_status.json
  logs/
  SHA256SUMS                    # 外层成员逐文件校验；两个split包还有各自清单
```

每个split包仍以 `external_qualitative_results/` 为前缀。公共原始事件/GT只保存一份，各模型预测保留完整float32概率、0.9掩码、事件行号和源文件哈希。PointEvent原生的双类别分数及重复原始数组保留在云端 `results_full_20261102/<split>/pointevent/native/npz/`；回传包已有逐事件目标类概率，可继续分析阈值、AUC和绘图。

只重新分析/打包已完成结果，无需重新跑GPU：

```powershell
python .\tools\analyze_results.py --results .\results_full_20261102 --archive .\external_full_reanalysis.zip
```

## 与论文如何比较

| 模型 | IoU(%) | 前景ACC(%) | Pd(%) | Fa×10⁴ |
|---|---:|---:|---:|---:|
| EV-SpSegNet | 55.18 | 65.02 | 77.53 | 1.63 |
| PACT | 75.90 | 80.05 | 91.84 | 0.76 |
| PointEvent | 75.56 | 81.61 | 91.75 | 0.12 |

数值直接核对本地下载的论文：EV-SpSegNet PDF第7页Table 2、PACT第9页Table 1、PointEvent第6页Table 1。各文献来源信息保存在paper_reference.json。只给完整24文件测试集生成论文差值，验证集单独报告。

- IoU=TP/(TP+FP+FN)，ACC/SegAcc=TP/(TP+FN)。先合并完整split的计数，再计算比值；另存逐序列平均值用于排查口径差异。ACC不是含大量背景的整体准确率。
- Pd/Fa遵循EV-UAV基准：使用整数毫秒；50ms开区间；目标恢复比例阈值0.0001；每帧8连通虚警区域；Fa分母为基准帧数×346×260。保留原始uint8像素计数语义。
- **PACT当前固定源码中的Fa统计存在缩进差异**：连通域计数位于逐虚警事件循环内，会重复累加。这里按EV-SpSegNet基准每帧统计一次，明确保留差异说明；PACT源码打印的Fa不能直接视为同口径数值。网络推理未被修改。
- PointEvent官方测试器打印全局前景IoU/ACC和逐文件AUC平均值。分析器同时提供全局AUC和逐文件AUC平均值，Pd/Fa独立按共同基准重算。
- 所有模型保留官方检查点和0.9阈值；不搜索验证集/测试集阈值。0.5的阳性数量只用于判断置信度是否整体偏低，没有生成新的正式预测。
- 已验证的Windows推理结果不代表与原始融合CUDA扩展逐值一致。完整集数值可以用于核查差距，不能仅凭接近就宣称训练协议、算子和论文复现全部一致。

## 后续四行点云图

预设行顺序：`test_000,test_001,test_002,test_022`，前三个为普通事件密度，最后一个为密集样例。依据原始输入事件数预先固定，记录在qualitative_selection.json；不根据预测成绩挑选。

回传全部结果后再更新论文为四行六列：Input、EV-SpSegNet、PACT、PointEvent、Ours、GT。继续使用完整原始事件、同一时间范围和视角；此次只调整展示样例，不修改预测数组。PointEvent密集样例若仍为空，应先核查复现原因，保留异常说明。
