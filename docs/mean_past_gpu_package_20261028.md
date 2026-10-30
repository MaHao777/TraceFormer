# M1纯均值 + 三层门控过去锥：远端交付

> 2026-10-30：启动、续跑和打包已统一到 [实验导航](experiments/README.md)。已有冻结包继续使用原入口。
> 下文历史配置/验证记录保留；六组当前实现仅记录耗时，查询分块默认8192、空间分块4096。


交付目录：`remote_upload/m1_mean_past_gated/`，整体上传这一个文件夹。主机已有数据，所以不携带NPZ或旧权重，也没有本地机器的固定数据路径。训练按seed37随机初始化。

模型为旧M1与旧C4的精确组合：0个Query、7项统计特征独立拼接、3层三尺度过去锥、保留池化门控和中心摘要注意力、保留事件Decoder与多尺度精修Head。实际参数142774；默认CUDA FP32、24轮、原损失/采样/AdamW/调度。BF16为显式可选项。

远端首次安装非torch依赖后，执行：

```bash
python -m pip install -r requirements.txt
python -u run.py --data-root /path/containing/train_and_val --test-root /path/containing/test --smoke
python -u run.py --data-root /path/containing/train_and_val --test-root /path/containing/test
```

`run.py --stage train`仅训练，`run.py --stage predict`单独预测；`run.py --resume`恢复失败或中断阶段。启动时核验147个源NPZ的身份、真实CUDA内核、小样本CPU/GPU输出及梯度、两条最大训练序列完整反向。训练/网络预测在GPU上，文件读取/官方计数在CPU上。不会自动改CPU、裁事件或更换结构。

正式结果在`output/full/`，归档在`output/results.zip`。拷回整个output或results.zip即可，包含权重、完整训练状态、曲线、逐事件概率/掩码、独立评分、配置、环境、数据身份和产物哈希。

历史交付包本地验证：414 passed、3 skipped，Ruff和diff检查通过；独立目录微型CPU训练/预测/双重计数/归档、断点恢复与拒绝重复运行均通过；最终交付目录另经离开仓库的独立导入与142774参数模型数值接口核验。真实CUDA测试因本机无GPU未执行，由远端启动阶段完成。历史详情记录在原交付目录LOCAL_CHECKS.json。本轮整理验证见[验证记录](experiments/reorganization_20261030.md)。

可再生入口为`traceformer-experiment bundle mean_past_gated --output <新目录>`；流程位于`src/traceformer/experiments/workflows/`，完整模型源码在`src/traceformer/`。已训练快照不能边训边改，打包器拒绝覆盖已有交付目录。
