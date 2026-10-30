# 当前主模型（2026-10-18）

按用户指定，主模型为`prefix_cone_20261018_optimized`：冻结的新三锥网络、
基于该网络重训的small/large事件树、主轨迹与直接分支并集。
入口不加载额外组件验证器或历史reference recovery，也不调整冻结阈值。

## 冻结来源

- 清单：[`configs/main_model.json`](../configs/main_model.json)。
- 权重与原字节配置：`outputs/main_model/best.pt`、`config.yaml`。
- 后处理：`outputs/main_model/event_release`，训练使用全部99序列。
- 原独立验证：`outputs/main_model/validation.json`。
- 原来源：`outputs/postprocess_prefix_cone_20261018_202349`。
- 完整回传：`outputs/server_results_20261018/complete`，977个文件逐一核验，含96份派生缓存/分数。

神经网络189,900参数；small为997棵树、62,811叶节点，large为1998棵树、253,746叶节点。
树的数量是后处理规模统计，不与神经网络可训练参数合计。
文件哈希、发布训练来源及验证报告关联均在默认入口加载时核验。
运行线程单独记录在清单`runtime.num_threads=14`，不修改冻结训练配置的原始字节；
即使原训练配置记录8线程，默认推理仍使用14。可用`--num-threads`显式覆盖，CPU树参数默认8。

原独立验证Fa为6.589045498544248e-6，超过自动门槛6e-6；
主模型选择记录明确为`user_requested`，保留`validation_gate.passed=false`。
这次用户指定不改写原自动门槛，也不改变其他候选的自动发布规则。

## 默认预测

```powershell
# 本地完整管线，仅保存测试掩码。
traceformer-predict --device cpu --predictions-only

# 验证集；可用--input-dir覆盖输入目录。
traceformer-predict --split val --predictions-only --output-dir outputs/main_val

# 云端完整管线，网络、特征、两套树使用GPU，轨迹使用CPU。
traceformer-predict --device cuda --input-dir /path/to/test --output-dir outputs/main_test
```

未指定输入目录时，清单给出相对清单文件的本地val/test路径，不依赖终端当前目录。
未指定输出目录时，写入`outputs/main_model/predictions/<split>_<timestamp>`，不会覆盖已有结果。
默认导出NPZ、TXT与平铺ZIP并逐值审计；`--predictions-only`跳过导出。
网络阈值取自冻结checkpoint；轨迹、融合与直接分支阈值取自冻结发布合同。
输入不需要label/name字段，推理构造的标注占位为零。
复现历史云端计时应同时设置环境`OMP_NUM_THREADS=14`、`MKL_NUM_THREADS=14`、`OPENBLAS_NUM_THREADS=1`。

显式指定`--checkpoint`继续执行单网络入口，用于独立模型分析；它不代表完整主模型结果。
加载主模型时不会训练、按测试分数选择参数或修改后处理合同。

## 已合入的推理优化

1. CUDA无梯度推理默认16384 Token分块；CPU和训练保留1024，显式分块参数仍可用于诊断。
2. 当前前向的`PrefixSample`保存整数矩形索引，各编码器层共享；下一次前向重新创建，训练不缓存。
3. 每帧轨迹画布裁到候选点包围盒加完整膨胀半径，裁剪传感器边界，节点继续使用全局坐标。

没有合入低收益的成员数缓存和无收益的树分数合并回传。
网络数学定义、特征量化、树文件、阈值、分支并集与权重键均不变。
原型完整实测955.015→444.187 ms/文件，详见[审计报告](full_pipeline_speed_audit_20261018.md)。
合入后的独立核验记录保存在`outputs/main_model/integration`。
当前生产代码已完成GPU复核，单网络168.973、完整管线442.657 ms/文件；
详见[完整GPU报告](main_model_gpu_verification_20261018.md)。

## 本次合入核验

- 四个既有远端运行目录977份文件全部回传，归档及逐文件SHA256均通过；96份派生缓存/分数在本地。
- 最终代码：pytest **268通过、2跳过**，Ruff和`git diff --check`通过。
- 使用本次生产后处理与已回传冻结GPU缓存/分数重放val/test全部48序列，掩码与固定参考逐位一致。
  从本地原始NPZ独立复评，验证Score **0.9305560045979603**、测试Score **0.9477523361441047**，
  逐文件及全局计数、指标均不变。这项验证未重跑48序列网络。
- 普通`val_000`与启用Sparse的高密度`test_023`，使用真实冻结权重在CPU比较合入前后全部网络Tensor，逐位一致。
- 两个序列另通过实际默认CLI运行完整网络与CPU两套树、轨迹和直接分支；最终掩码与原GPU参考逐位一致，
  实际Torch线程为14；TXT/ZIP逐值与平铺根目录审计通过。
- 后续重连已补齐并回传生产代码GPU复核312份文件：全部48序列网络Tensor、缓存、树分数与最终掩码精确一致，
  实际默认CLI另跑48序列通过，测试集24份TXT/ZIP审计通过。
- 当前GPU正式三轮实测单网络**168.973 ms/文件**、完整管线**442.657 ms/文件**；
  本地从原始数据独立复评，逐文件和全局计数、指标均与参考一致。

证据：[本地核验汇总](../outputs/main_model/integration/local/summary.json)、
[最终pytest](../outputs/main_model/integration_pytest_final.xml)、
[完整回传核验](../outputs/server_results_20261018/verified.json)、
[最新GPU核验汇总](../outputs/main_model/integration/gpu_return_20261018/summary.json)。
