# A/B结构改进与对照实验

00复用`outputs/no_sparse_local_20261025/training/best.pt`，不重新训练。只随机初始化训练10（A）、01（B）、11（A+B）。保持当前单视图网络与多尺度事件精修Head，不加载旧LightGBM等不匹配后处理。

A在原有8×8空间块、64时间块分组内实现：

`block = 0.75 * (mean(event_encoder(events)) + Linear(7项统计)) + 0.25 * single_query(events)`

单Query读取该块的全部事件，保持原有事件局部特征、位置编码、分块和事件映射。主支路统计为`log1p(count), positive_fraction, local_time_mean, local_time_variance, density, polarity_contrast, polarity_entropy`。最终Token投影只拼接block和位置编码，统计量不重复输入。删除原4Query融合MLP，固定3:1融合，不增加可学习融合门控。

B保留中心Query对锥摘要的多头注意力；删除摘要池化的`pool_weight/sigmoid`门控，过去锥内改用算术均值；删除未来方向及其参数。编码器由3层变6层，每层几何相同：(R,H)=(8,2)/(4,4)/(2,8)，保持圆锥随时间距离展开的形状，不改为圆柱窗口。当前00本来就是各层相同几何；层叠仍会扩大有效感受野。原精修Head和全序列极性特征保留，因此不将整网称为严格因果网络。

|组|实际Query数|实际编码层数|方向|摘要池化门控|参数量|
|---|---:|---:|---|---|---:|
|00|4|3|过去+未来|有|172150|
|10|1|3|过去+未来|有|151222|
|01|4|6|过去|无|285427|
|11|1|6|过去|无|264499|

共同配置是原3层/4Query构造模板，`ablation.json`的版本3契约指定替换，`suite.json`同时保存实际结构。先初始化相同Full再替换，公共权重初值相同；A在10/11中相同，B新增三层在01/11中相同。替换构造不推进外部训练RNG；不同深度训练中的Dropout调用数仍不同。

固定协议：seed37，99训练/24验证/24测试，24轮，CPU FP32/8线程，AdamW lr0.0007、wd0.0002。损失、密集序列权重、采样和学习率调度继承00。只有验证集选择epoch和阈值，测试固定使用验证阈值；测试标签和object ID在模型输入副本中清零。

入口：

```powershell
D:/VS_environment/Python.12/python.exe scripts/run_structure_improvements.py --root outputs/structure_improvements_20261027
```

启动前需要与当前源码指纹一致的`implementation_checks.json`。入口避免重复启动，后台协调器持有文件锁。运行先冻结源码/配置/数据哈希，对00旧新源码24验证序列逐事件logits和映射做精确等价检查，随后对10/01/11分别在train_094和train_096完整前反向预检，检查有限梯度与RAM门槛。通过后依次执行训练、独立验证、验证/测试预测及两套独立计数。

主要输出：

- `status.json`、`logs/`：当前阶段及实时日志。
- `suite.json`、`source/`、`source_hashes.json`、`baseline_equivalence.json`：实际结构和00复用证据。
- `resource_gate.json`、`preflight/`：6次完整序列资源预检。
- `seed37/10`、`seed37/01`、`seed37/11`：最佳权重、配置、训练曲线和逐轮恢复状态。
- `val/<组>/predictions/*.npz`、`test/<组>/predictions/*.npz`：保持原事件顺序的二值prediction。
- `val/<组>/probabilities/*.pt`、`test/<组>/probabilities/*.pt`：原顺序逐事件概率及源NPZ哈希。
- 各预测目录下`independent_report.json`、`raw_independent_report.json`：概率重新评分与另一套原始掩码计数。
- `results.md`、`results.json`：验证/测试两张对照表；`completion_audit.json`表示全部阶段完成。

单种子实验，不声称统计显著性；测试集有历史查看记录。B同时改变方向、门控和深度，只能评判这一整体改进设想。测试结果不用于调整结构或训练协议。

## 本次运行

运行目录为`outputs/structure_improvements_20261027/`。405项全仓测试通过、2项跳过，Ruff和diff检查通过。00旧新源码在24验证文件上逐事件logits和映射完全相同。三组共6次最大序列完整前反向预检全部通过；实际资源结果保存在`resource_gate.json`。

协调器自动依次训练10/01/11及完成验证/测试评估，无须手动执行后处理。阶段以`status.json`为准，不能把启动或预检通过当作实验已完成。本聊天运行期间使用每小时完成检查（`huav-a-b`），正常运行时保持安静；该检查已在全部完成并通过独立审计后删除。

## 2026-10-28 完成结果

10/01/11均完成24轮，00未重训。最终独立审计通过：610份产物和125份冻结源码哈希一致；四组两个划分各24个文件，共192份掩码、192份概率，全部源数据身份、概率到冻结阈值掩码的对应关系、逐文件及全局官方计数均一致。验证集各组阈值直接用于测试，没有测试集阈值选择。审计脚本和结果保存在运行目录`verify_completion.py`、`final_independent_audit.json`和`final_independent_audit.log`。

|组|验证Score|相对00|测试Score|相对00|
|---|---:|---:|---:|---:|
|00|0.903844558|0|0.909915319|0|
|10|0.903354269|-0.000490288|0.911912015|+0.001996696|
|01|0.892189430|-0.011655128|0.902232865|-0.007682454|
|11|0.901408896|-0.002435662|0.911277196|+0.001361877|

完整IoU/SegAcc/Pd/Fa/AP、参数量和阈值见运行目录`results.md`/`results.json`。A的参数量减少约12.16%，验证Score接近但略低于00，测试Score提高；单独B在两个划分均下降。组合的Score在验证和测试均没有超过单独A。本次单种子实验不据测试优势直接替换主模型，不将差值解释为统计显著或稳定收益。
