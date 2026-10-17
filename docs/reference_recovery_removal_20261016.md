# Reference recovery 删除与独立复评（2026-10-16）

当前最终输出为 `direct | tracked`。删除恢复函数、参考模型加载/复用、参考预测 CLI 参数、报告字段、专用测试和当前论文生成器中的分支；检测器、活动 large 树、阈值及主轨迹规则保持不变。历史 outputs 与论文 evidence 归档保留，不参与当前执行。

使用冻结 schema-5 主模型的已保存事件缓存重新运行事件特征、LightGBM 和主轨迹；随后独立进程从原始带标签 NPZ 重新读取并评分，逐事件比较旧 complete 与 event_track。没有重新训练、搜索阈值或重新运行神经前向。

| 数据集 | 改变事件 / 全部事件 | IoU 前→后 (%) | ACC 前→后 (%) | Pd 前→后 (%) | Fa 前→后 (1e-6) | Score 前→后 |
|---|---:|---:|---:|---:|---:|---:|
| test（24 序列） | 216 / 2,074,586（0.0104%） | 91.924405 → 92.167531 | 97.271976 → 97.260773 | 97.723795 → 97.723795 | 7.881585 → 7.666551 | 0.949278997 → 0.950350896 |
| val（24 序列） | 580 / 1,424,330（0.0407%） | 92.236135 → 91.461703 | 95.359204 → 94.518059 | 96.808064 → 96.157077 | 4.411461 → 4.367020 | 0.954117016 → 0.949250656 |

验证集变化全部来自 `val_023`：少 551 个正确事件和 29 个误报事件，少检出 31 个目标帧。测试集变化仅在 `test_022`（168 个）与 `test_023`（48 个）：少 8 个正确事件和 208 个误报，目标帧检出不变。变化的事件占比很小，但验证集存在真实性能损失，不能称为无损删除。此简化发生在既有测试结果已知之后，测试改善属于回顾性比较。

48 个序列的新 complete 均与旧 event_track 逐事件完全一致，无额外新增正例。逐文件变化、掩码哈希、计数与独立指标见 [comparison.json](../outputs/no_reference_20261016/comparison.json)。新版掩码位于 `outputs/no_reference_20261016/{val,test}/complete/`。

论文同步删除参考分支公式、流程框、阶段行和文字，三阶段图表由新掩码重新生成。组合神经参数为 169,632，活动树仍为 1,998 棵、253,746 叶。旧 871.9 ms 含已删除分支，因此当前组合耗时留空；185.5 ms 仅保留为未改变主网络的历史阶段实测。当前机器为 CPU PyTorch，本轮没有新版 GPU 全链路计时。

## 复现

```powershell
python scripts/evaluate_frozen_system.py --cache outputs/paper_test_20261009/main/cache --release outputs/paper_cuda_20260928/copyback_20260929/artifacts/event_release --dataset-root D:/VS_project/TinyObjDetection/EV-UAV-dataset --split test --output outputs/no_reference_repeat/test
python scripts/evaluate_frozen_system.py --cache outputs/paper_cuda_20260928/copyback_20260929/results/TraceFormer-main/cache --release outputs/paper_cuda_20260928/copyback_20260929/artifacts/event_release --dataset-root D:/VS_project/TinyObjDetection/EV-UAV-dataset --split val --output outputs/no_reference_repeat/val
```

Track 2 入口不再接受 `--reference-prediction-dir`。验证门槛读取 `event_release` / `event_release_direct_gate`，拒绝只提供旧恢复阶段的报告。本轮验证覆盖 val/test 各 24 序列，未重新生成无标签 Track 2 提交 ZIP。

## 检查结果

完整 pytest：267 passed、2 skipped；删除分支后的专项（含新密集/稀疏输出契约）38 passed。Ruff 与两个仓库的 diff 空白检查通过。论文编译为 24 页，检查全部页面缩略图及流程/阶段图，无未解析引用或溢出版面警告。

## 2026-10-17 提交前复核

上表记录的是冻结 schema-5 系统的后处理变化，不代表新三锥模型的评测结果。本次重新核对 val/test 各 24 份完整预测掩码的 SHA-256，均与原复评记录一致；原独立评分与对照汇总的全局指标一致，未重新推理或调参。

当前仓库完整 pytest：231 passed、2 skipped；Ruff 与 diff 空白检查通过。检查日志位于 `outputs/reference_recovery_commit_20261017/`。本次提交仅整理参考恢复分支删除及其配套说明，三锥结构升级已单独提交。
