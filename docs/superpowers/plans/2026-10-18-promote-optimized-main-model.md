# 更新主模型实施计划

> 在当前会话逐项执行；遵循用户要求，直接在本地仓库开发，不使用worktree。

**目标：** 完整回传服务器结果，将冻结的新网络与small/large后处理及已验证加速合入默认推理入口。

**方案：** 网络推理复用每次几何对象内的整数索引，CUDA无梯度推理采用16384分块；训练仍用1024。
轨迹只改变形态学画布范围。用`configs/main_model.json`固定本地主模型产物与哈希，
默认`traceformer-predict`执行完整事件管线，显式`--checkpoint`仍支持单网络预测。
此次主模型由用户指定，不把原Fa失败改写为自动验收成功。

## 1. 完整回传与冻结产物

- [x] 回传四个远端运行目录的全部普通文件，包括96份派生缓存/分数；不复制原始数据和Python缓存。
- [x] 验证归档SHA256与逐文件清单，将固定权重、原字节配置、事件发布、独立验证报告复制到`outputs/main_model`。
- [x] 写入`configs/main_model.json`：相对路径、文件SHA256、用户指定来源与原验证门槛值。

## 2. 合入等价优化

文件：`src/traceformer/models/prefix_cone.py`、`src/traceformer/postprocessing/tracking.py`、`tests/test_prefix_cone.py`、`tests/test_tracking_optimization.py`。

- [x] 先增加无梯度层间几何复用、不同几何隔离、局部画布边界/连通域顺序的回归检查，确认现有实现不满足提速约束。
- [x] 为`PrefixSample`增加仅随当前前向存活的索引字典；无梯度几何创建时启用，梯度路径不缓存。
  `prefix_cone_pool`默认按设备/梯度状态选择`16384 if values.is_cuda and not torch.is_grad_enabled() else 1024`。
- [x] 将每帧种子包围盒向外扩展`cluster_radius`并裁到260×346传感器边界，再执行原膨胀和连通域算法；节点仍使用全局坐标。
- [x] 运行`python -m pytest tests/test_prefix_cone.py tests/test_tracking_optimization.py -q`，梯度与边界检查全部通过。

## 3. 默认完整主模型入口

文件：`src/traceformer/primary.py`、`src/traceformer/cli/predict.py`、`tests/test_primary_model.py`、`README.md`。

- [x] 增加主模型产物加载：核对固定文件哈希、发布来源、独立验证报告与选择记录。
- [x] 复用`infer_event_cache`、`predict_event_scores`、`event_branches`，最终掩码为`tracked | direct`，不引入新阈值或验证器。
- [x] 默认预测命令使用主模型；显式checkpoint继续原网络入口。主模型预测只读输入字段，不使用标签。
- [x] 增加默认入口分派、文件篡改拒绝、原Fa记录保留与完整分支输出检查。

## 4. 最终验证与回传

- [x] 本地运行pytest、Ruff与`git diff --check`。
- [x] 上传独立的新源码快照并启动GPU复核；首次因关机中断，后续重连已补齐并完整回传312份证据。
- [x] 改用本地生产后处理重放全部48序列并独立复评；普通/Sparse两个真实网络对照及默认CLI导出审计通过。
  更新主模型说明与完成记录，明确本地验证范围。
- [x] 生产代码GPU全48序列Tensor、缓存、树分数和掩码精确对照通过；实际默认CLI全48序列及测试TXT/ZIP审计通过。
- [x] 回传后哈希及本地独立复评通过，更新三轮实测：单网络168.973、完整管线442.657 ms/文件。

最终结果：268 passed、2 skipped；48缓存重放掩码精确、两例真实完整默认CLI掩码精确且为14线程。
报告：[当前主模型](../../main_model_20261018.md)。完整回传977文件，未上传原始数据。
后续新增GPU完整回传312文件；[生产代码GPU实测](../../main_model_gpu_verification_20261018.md)。
