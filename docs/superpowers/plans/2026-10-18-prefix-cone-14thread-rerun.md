# 新模型按原 GPU 线程设置重测

目标：按用户要求，把新模型计时的线程环境与 2026-10-13 运行脚本逐项对齐。
直接在当前本地仓库执行，复用服务器已有数据、环境和冻结模型，保存独立结果。

- [x] 计时入口区分 Torch 与 CPU 树线程，记录实际线程及环境变量，并将其绑定验证证据。
  命令固定为 `--num-threads 14 --prediction-threads 8 --repeats 3 --profile --diagnostic`；
  环境为 `OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=1`。
- [x] 运行 `python -m pytest tests/test_event_system_benchmark.py -q` 和针对修改文件的 Ruff 检查。
- [x] 连接原服务器，核验 48 个数据文件、固定权重及代码哈希、GPU 空闲和14核容器配额。
- [x] 顺序执行 val/network、val/pipeline、test/network、test/pipeline，各24文件三轮及额外阶段分析。
  使用最大序列预热；测试集读取本次14线程验证证据，所有掩码对照此前8线程结果。
- [x] 证据包回传并验证哈希；本地从原始数据独立复评四组结果，比较8线程和历史旧系统。
- [x] 更新运行报告和README，保留旧8线程记录与候选未通过验证门槛的状态。

本地输出：`outputs/gpu_prefix_cone_14t_20261018_214457`。
远端输出：`/root/autodl-tmp/traceformer-prefix-14t-20261018_214457`。

结果：测试集单模型536.679 ms/文件、完整管线1202.171 ms/文件；比8线程记录分别减少2.01%和4.21%。
96份掩码逐位一致，四组本地独立评分与远端一致。完整报告：`docs/gpu_prefix_cone_14t_20261018.md`。
