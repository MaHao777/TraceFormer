# 主模型生产代码GPU复核与完整回传（2026-10-18）

当前主模型的实际默认代码已完成GPU复核，并完整回传结果。
单网络平均 **168.973 ms/文件**，完整管线平均 **442.657 ms/文件**；
val/test共48序列的网络Tensor、缓存、树分数及最终掩码均与固定参考精确一致。
两套树、主轨迹和直接分支全部参与完整管线。

## 实测耗时

复用原RTX 3090 24 GiB服务器与原始数据，未重新上传数据集。
CPU cgroup配额14核；Torch/OMP/MKL均为14线程，OpenBLAS为1线程，
CPU树参数8，实际特征与树运行于GPU。PyTorch 2.8.0+cu128，FP32，TF32关闭。

每个模式先完整预热最大文件`test_023`一次，再对全部24个测试文件各跑三轮。
主计时从内存中的输入归一化到最终事件掩码，包含Sparse与数据传输，
完整模式另含缓存构建、特征、两套事件树、CPU轨迹和直接分支并集。
磁盘IO、模型加载、正确性检查、评分、TXT/ZIP导出与审计均不计入。
正式总计时仅在起止同步CUDA，未用阶段插桩值估算总耗时。

| 模式 | 优化前14线程均值 ms/文件 | 当前代码均值 ms/文件 | 耗时减少 | 当前24文件平均秒数 |
|---|---:|---:|---:|---:|
| 单网络 | 536.679 | **168.973** | **68.52%** | 4.055 |
| 完整管线 | 1202.171 | **442.657** | **63.18%** | 10.624 |

优化前后使用同一冻结权重、配置、后处理合同和数据，改变的是已合入的推理实现。
优化前数字来自此前[14线程实测](gpu_prefix_cone_14t_20261018.md)。

| 模式 | 中位数 ms | P95 ms | 最大 ms | 事件/秒 | 峰值分配显存 GiB |
|---|---:|---:|---:|---:|---:|
| 单网络 | 69.191 | 1035.103 | 1225.304 | 511568.234 | 3.504 |
| 完整管线 | 240.642 | 2239.664 | 2667.108 | 195277.984 | 3.522 |

中位数、P95与最大值均先对每文件三轮取平均，再对24文件统计。
完整管线的442.657 ms是文件均值，长序列仍可能耗时超过2秒。
三轮全24文件的实际秒数为：

- 单网络：3.997、4.227、3.942。
- 完整管线：10.999、10.521、10.351。

此前独立原型444.187 ms/文件保留为[历史试验](full_pipeline_speed_audit_20261018.md)。
本报告442.657 ms来自已合入的实际代码，数值接近但不复用原型计时。

## 数值、默认入口与独立复评

- 与冻结旧编码器逐一比较val/test全部48序列，全部网络Tensor逐位一致；
  量化缓存、small/large分数和最终掩码亦精确一致。
- 实际默认CLI另完整运行val/test共48序列，全部最终掩码与固定参考逐位一致，
  实际Torch线程均为14，活动事件模型均为small/large。
- 单网络与完整计时各24文件、三轮，重复预测和固定参考掩码均逐位一致。
- 测试集默认CLI输出TXT/ZIP，逐值审计通过，共24个TXT、2,074,586行，ZIP根目录平铺。
- 回传后从本地原始NPZ独立复评，逐文件及全局指标、计数与此前14线程参考一致：
  验证完整Score **0.9305560045979603**，测试完整Score **0.9477523361441047**，
  测试单网络Score **0.9137197184621895**。
- 源码快照132文件、数据48文件和冻结产物9文件均核验通过。
  本地所有执行代码、配置与主模型选择清单均和实测快照相同；README仅有后续说明更新。
- 主模型仍是用户指定的新三锥网络与重训后处理；原Fa门槛和选择记录保持原值，
  GPU执行核验通过不改写自动验证门槛。

## 回传与证据

原关机时最终评测尚未完成。本次重连后复用已有源码、环境、数据及权重补齐，
保留早期失败尝试和中断日志，新增312个普通文件已全部回传并逐文件校验SHA256。
原有977文件备份保留在`outputs/server_results_20261018`。
远端计算进程已结束，打包前确认GPU空闲。

本地：`outputs/main_model/integration/gpu_return_20261018`。
远端：`/root/autodl-tmp/traceformer-main-integrated-20261018`。

- [主模型核验收据](../outputs/main_model/verification.json)
- [本地完整核验汇总](../outputs/main_model/integration/gpu_return_20261018/summary.json)
- [回传大小与SHA256](../outputs/main_model/integration/gpu_return_20261018/return_report.json)
- [执行源码清单](../outputs/main_model/integration/gpu_return_20261018/evidence/source_manifest.json)
- [48序列Tensor/缓存/分数/掩码记录](../outputs/main_model/integration/gpu_return_20261018/evidence/outputs/main_model/integration/final/tensor_cache_scores_masks.json)
- [单网络三轮原始计时](../outputs/main_model/integration/gpu_return_20261018/evidence/outputs/main_model/integration/final/test_network.json)
- [完整管线三轮原始计时](../outputs/main_model/integration/gpu_return_20261018/evidence/outputs/main_model/integration/final/test_pipeline.json)
- [测试集独立复评](../outputs/main_model/integration/gpu_return_20261018/test_pipeline_metrics.json)
- [测试集提交包审计](../outputs/main_model/integration/gpu_return_20261018/evidence/outputs/main_model/integration/final/cli_test/submission_audit.json)
- [已审计的提交ZIP](../outputs/main_model/integration/gpu_return_20261018/evidence/outputs/main_model/integration/final/cli_test/submission.zip)
- [本地复核脚本](../outputs/main_model/integration/verify_gpu_return.py)

归档大小49,879,726字节，SHA256：
`96b4b3958d1cb2787f7082a8cf4fc0d250b21e88d6a99c1f8f9e7c9749985b16`。
