# HUAV 模型修改后的强制训练、验证与提交闭环

本文件适用于整个仓库。凡是完成模型结构或训练方法修改，必须在交付前执行下面的完整闭环。不得仅凭单元测试、烟雾训练、训练过程中的验证结果或历史 checkpoint 宣称修改完成。

## 触发条件

以下任一变化都会触发完整闭环：

- `src/huav/models/` 下的网络结构、Token、邻域、注意力、解码器或前向传播变化；
- `src/huav/training/` 下的损失、采样、优化、学习率、训练循环、指标、阈值搜索或 checkpoint 选择变化；
- 影响模型或训练行为的配置字段、数据预处理、标签语义、验证协议变化；
- 提交预测或 TXT/ZIP 导出逻辑变化。

纯文档、注释、格式整理或不影响上述行为的测试修改不触发长时间全量训练，但仍须运行相关测试、Ruff 和 `git diff --check`。如果无法确定是否影响模型结果，按触发处理。

## 基本约束

- 开始前先运行 `git status --short`，保留用户已有改动，不覆盖历史 `outputs/` 结果。
- 每轮使用新的时间戳目录；全量训练、独立复评、提交 TXT 和 ZIP 必须能够对应到同一个 `best.pt`。
- 当前生产配置使用 checkpoint schema `3`。旧 schema `2` 只作为历史产物保留，不能用当前模型代码直接加载。
- 始终保持每个原始事件及其 `x y t p` 顺序不变，模型输出必须对应每个原始事件。背景采样只能影响损失样本选择，不能删减前向上下文或提交事件。
- 新的锥体几何或邻域尺度必须先基于真实训练数据统计确定，不能根据单个极端样本拍脑袋设置。
- 默认使用当前分数优化配置完成 CPU 全量流程。若任务明确要求 CUDA，可将下列命令中的设备统一改为 `cuda`，但必须记录设备，且不能减少训练/验证数据。

## 强制执行流程

### 1. 静态检查与回归测试

在仓库根目录运行：

```powershell
git status --short
python -m pytest -q
python -m ruff check src tests
git diff --check
```

任一命令失败时先修复，不得启动全量训练。

若修改了锥体几何、时间/空间分块或邻域尺度，还要先运行真实训练集统计，并保存报告：

```powershell
python -m huav.analysis.cone_slopes data/ev-uav/train `
  --json docs/cone_slope_analysis.json `
  --markdown docs/cone_slope_analysis.md `
  --spatial-block-x 4 `
  --spatial-block-y 4 `
  --time-blocks 128 `
  --duration-ms 8000 `
  --max-time-distance 8
```

参数应依据稳健分位数选择，同时在报告中保留但不追随稀疏极端值。

### 2. 真实数据烟雾训练与复评

若修改了分块、邻域、Token 数量或注意力图规模，先对事件数最大的训练序列（当前重点为 `train_094.npz`）完成一次 forward、loss 和 backward 资源复现。除非新的实测分析证明更细网格安全，否则沿用已验证可完成全量训练的 `8x8` 空间分块和 64 时间分块；不得把曾触发 Windows `c10.dll` `0xc0000005` 的 `4x4/128` 大图直接投入全量训练。

然后建立本轮唯一目录名，并跑通训练和验证链路：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$smokeDir = "outputs/score_optimized_cpu_smoke_$stamp"
$runDir = "outputs/score_optimized_cpu_full_$stamp"
$submissionDir = "outputs/submission_$stamp"

python -m huav.cli.train `
  --config configs/huav_score_optimized_smoke.yaml `
  --device cpu `
  --output-dir $smokeDir

python -m huav.cli.evaluate `
  --config "$smokeDir/config.yaml" `
  --checkpoint "$smokeDir/best.pt" `
  --device cpu `
  --output "$smokeDir/reproduced_metrics.json" `
  --quiet
```

烟雾流程失败时先修复，不能直接跳到全量训练。烟雾训练只检查链路，不替代全量训练。

### 3. 完整训练与完整验证

必须使用无 `max_train_files`、无 `max_val_files` 限制的全量配置：

```powershell
python -m huav.cli.train `
  --config configs/huav_score_optimized.yaml `
  --device cpu `
  --output-dir $runDir
```

确认全量训练覆盖当前全部训练文件和验证文件，并在 `$runDir` 下生成：

- `best.pt`
- `config.yaml`
- `dataset_manifest.json`
- `history.json`
- `metrics.json`
- `metrics.md`
- `summary.json`

不得复用修改前的 `best.pt`，也不得用烟雾训练权重代替本轮全量训练权重。

### 4. 独立复评最佳 checkpoint

训练结束后，必须使用本轮输出目录中保存的配置和最佳权重重新启动一次独立验证：

```powershell
python -m huav.cli.evaluate `
  --config "$runDir/config.yaml" `
  --checkpoint "$runDir/best.pt" `
  --device cpu `
  --output "$runDir/reproduced_metrics.json" `
  --quiet
```

验收要求：

- `metrics.json` 与 `reproduced_metrics.json` 的完整 JSON 结果一致；
- `event.tp + event.fp + event.fn + event.tn == event.event_count`；
- `event.event_count` 覆盖验证集全部事件；
- `validation_files` 与 `dataset_manifest.json` 中的验证文件数一致；
- 分开报告 `competition`、`event` 和 `ev_uav_compatible` 指标，不混用不同口径。

可用下面的 PowerShell 检查字节级复现；不一致时必须调查，不能继续生成提交包：

```powershell
$savedHash = (Get-FileHash "$runDir/metrics.json" -Algorithm SHA256).Hash
$reproducedHash = (Get-FileHash "$runDir/reproduced_metrics.json" -Algorithm SHA256).Hash
if ($savedHash -ne $reproducedHash) {
    throw "Independent evaluation does not reproduce metrics.json"
}
```

### 5. 生成比赛 TXT 和 ZIP

独立复评通过后，使用同一轮保存的配置、`best.pt` 和 checkpoint 内保存的最佳阈值生成提交文件：

```powershell
python -m huav.cli.predict `
  --config "$runDir/config.yaml" `
  --checkpoint "$runDir/best.pt" `
  --output-dir $submissionDir `
  --device cpu
```

不要使用 `python -m huav.cli.package`；该入口当前只是未实现的脚手架。

必须检查：

- 每个验证 NPZ 恰好对应一个同名 TXT；
- 每行严格为 `x y t p predicted_label` 五列；
- 行数与源 NPZ 的事件数一致，`x y t p` 的值和顺序完全一致；
- `predicted_label` 只能是 `0` 或 `1`；
- `$submissionDir/submission.zip` 存在；
- ZIP 根目录直接包含全部 TXT，不能含子目录、非 TXT 文件或额外包装目录；
- ZIP 内 TXT 数量与验证文件数一致。

至少运行以下 ZIP 结构检查：

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path $submissionDir "submission.zip"
$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entries = @($archive.Entries)
    if ($entries.Count -eq 0) { throw "Submission ZIP is empty" }
    $invalidEntries = @($entries | Where-Object {
        $_.FullName -ne $_.Name -or
        [System.IO.Path]::GetExtension($_.Name).ToLowerInvariant() -ne ".txt"
    })
    if ($invalidEntries.Count -ne 0) {
        throw "Submission ZIP must contain only root-level TXT files"
    }
} finally {
    $archive.Dispose()
}
```

### 6. 完成前复查与交付

生成所有运行产物后再次运行：

```powershell
python -m pytest -q
python -m ruff check src tests
git diff --check
git status --short
```

最终回复必须给出：

- 本轮全量运行目录；
- 使用的配置、设备、训练/验证文件数、最佳 epoch 和阈值；
- `competition.score`、IoU、SegAcc、Pd、Fa，以及事件级 Pd/Fa/IoU/Acc；
- 独立复评是否与训练保存指标完全一致；
- 可直接提交的 `submission.zip` 绝对路径；
- TXT 文件数、总行数、预测为 `1` 的总数和 ZIP 结构检查结果；
- 测试、Ruff、`git diff --check` 的最终结果；
- 若某一步失败，明确报告失败和现存产物，不能把流程描述为已完成。

## 禁止的完成方式

- 不得只跑单元测试后宣称模型修改完成；
- 不得只跑烟雾训练或只验证少量文件；
- 不得仅引用训练阶段的验证指标而跳过 `best.pt` 独立复评；
- 不得复用历史 checkpoint 或历史提交 ZIP 充当本轮结果；
- 不得在指标未复现时生成或交付比赛提交包；
- 不得手工重排、过滤或聚合提交事件；
- 不得覆盖历史输出目录，除非用户明确要求。
