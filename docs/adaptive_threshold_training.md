# 自适应阈值 AutoDL 训练与选择方案

本实验只比较阈值校准方式，不改变已验证的 `8x8` 空间块、64 个锥体时间块、检测器结构、数据采样或官网评分口径。局部阈值使用 160 个时间格，对应 50 ms；所有配置均在 RTX 5090 上使用 CUDA BF16。当前产物为 checkpoint schema `5`；schema `4` 非自适应 checkpoint 可作为检测器初始化源，schema `3` 被拒绝。

## 变量和 GPU 预检

在仓库根目录设置本轮唯一目录和非自适应基线。`BASE_PT` 必须来自当前数据清单、相同分辨率和相同检测器模型的 schema 4 或 schema 5 checkpoint。

```bash
STAMP=$(date +"%Y%m%d_%H%M%S")
REPO_ROOT=$(pwd)
BASE_CONFIG=outputs/<baseline-run>/config.yaml
BASE_PT=outputs/<baseline-run>/best.pt
ROOT=outputs/adaptive_threshold_${STAMP}
mkdir -p "$ROOT"
python -m huav.cli.check_gpu --device cuda | tee "$ROOT/gpu_preflight.log"
```

GPU 预检必须看到 RTX 5090、CUDA 12.8 或更新版本、`sm_120` 和真实 CUDA kernel 成功。

## 0. 非自适应基线重新搜索阈值

schema 4 基线只读评估时必须显式重新搜索 61 点配置下的阈值，作为同数据、同评分代码的 control：

```bash
BASE_EVAL_CONFIG="$ROOT/baseline_reopt61.yaml"
cp "$BASE_CONFIG" "$BASE_EVAL_CONFIG"
sed -i "s/^  threshold_steps: .*$/  threshold_steps: 61/" "$BASE_EVAL_CONFIG"
python -m huav.cli.evaluate \
  --config "$BASE_EVAL_CONFIG" \
  --checkpoint "$BASE_PT" \
  --device cuda \
  --reoptimize-threshold \
  --output "$ROOT/baseline_reoptimized_metrics.json" \
  --quiet
```

## 1. 最大序列资源门禁

在任何 adaptive smoke/full 训练之前，用完整 `train_094.npz` 跑真实 forward、生产总损失和 backward。门禁会复用训练引擎的初始化校验、criterion、阈值正则、密集序列权重和采样路径：

```bash
python scripts/check_largest_sequence.py \
  --config configs/huav_adaptive_hierarchical_autodl.yaml \
  --initialize-detector "$BASE_PT" \
  --sequence train_094.npz \
  --device cuda | tee "$ROOT/largest_sequence_hierarchical.json"
```

只有 `event_count == all_event_logits_count`、`finite_loss=true`、`finite_gradients=true`、`detector_frozen=true`、`calibrator_only_gradients=true` 时才继续，并保留 token/group/neighbor 数量、两种 CUDA 峰值、source checkpoint 绝对路径和 SHA-256。

## 2. 初始化链路 smoke

```bash
SMOKE="$ROOT/smoke_hierarchical"
python -m huav.cli.train \
  --config configs/huav_adaptive_hierarchical_autodl_smoke.yaml \
  --initialize-detector "$BASE_PT" \
  --device cuda \
  --output-dir "$SMOKE"
python -m huav.cli.evaluate \
  --config "$SMOKE/config.yaml" \
  --checkpoint "$SMOKE/best.pt" \
  --device cuda \
  --output "$SMOKE/reproduced_metrics.json" \
  --quiet
cmp "$SMOKE/metrics.json" "$SMOKE/reproduced_metrics.json"
```

smoke 只验证完整初始化、训练、保存和复评链路，不参与结构选择。

## 3. 四种冻结校准器消融

四轮都从同一个 `BASE_PT` 初始化，固定 seed 37、6 epoch、全量 99/24 文件；检测器冻结，只训练校准器：

- global：`configs/huav_adaptive_global_autodl.yaml`
- local：`configs/huav_adaptive_local_autodl.yaml`
- local_neighbor：`configs/huav_adaptive_local_neighbor_autodl.yaml`
- hierarchical：`configs/huav_adaptive_hierarchical_autodl.yaml`

```bash
for MODE in global local local_neighbor hierarchical; do
  RUN="$ROOT/frozen_${MODE}_seed37"
  python -m huav.cli.train \
    --config "configs/huav_adaptive_${MODE}_autodl.yaml" \
    --initialize-detector "$BASE_PT" \
    --device cuda \
    --output-dir "$RUN"
  python -m huav.cli.evaluate \
    --config "$RUN/config.yaml" \
    --checkpoint "$RUN/best.pt" \
    --device cuda \
    --output "$RUN/reproduced_metrics.json" \
    --quiet
  cmp "$RUN/metrics.json" "$RUN/reproduced_metrics.json"
done
```

选择规则按以下严格顺序比较独立复评 JSON：

1. `competition.score` 最高者胜出；
2. 分数相同时优先 IoU 较高者；
3. 再相同时优先 Pd 较高者；
4. 再相同时优先 Fa 较低者；
5. 仍相同则选择更简单的 `global → local → local_neighbor → hierarchical`。

事件级 IoU/Pd/Fa/Acc 和校准偏移只用于诊断，不能替代官网 `competition.score`。

## 4. winner 联合微调三种种子

把 `WINNER_MODE` 设置为上一步胜出的 mode。对应 joint 配置把校准器学习率降到 `1e-4`，检测器学习率为 `3e-5`。三轮都从同一个冻结 winner 的 `best.pt` 恢复，并额外训练 8 epoch：

```bash
WINNER_MODE=hierarchical  # 替换为 frozen 消融胜出的 mode
FROZEN="$ROOT/frozen_${WINNER_MODE}_seed37/best.pt"
for SEED in 37 73 137; do
  JOINT_CONFIG="$ROOT/joint_${WINNER_MODE}_seed${SEED}.yaml"
  cp "configs/huav_adaptive_${WINNER_MODE}_joint_autodl.yaml" "$JOINT_CONFIG"
  sed -i "s|root: ../data/ev-uav|root: ${REPO_ROOT}/data/ev-uav|" "$JOINT_CONFIG"
  sed -i "s/^  seed: 37$/  seed: ${SEED}/" "$JOINT_CONFIG"
  RUN="$ROOT/joint_${WINNER_MODE}_seed${SEED}"
  python -m huav.cli.train \
    --config "$JOINT_CONFIG" \
    --resume "$FROZEN" \
    --additional-epochs 8 \
    --device cuda \
    --output-dir "$RUN"
  python -m huav.cli.evaluate \
    --config "$RUN/config.yaml" \
    --checkpoint "$RUN/best.pt" \
    --device cuda \
    --output "$RUN/reproduced_metrics.json" \
    --quiet
  cmp "$RUN/metrics.json" "$RUN/reproduced_metrics.json"
done
```

三种种子仍按上一节的官方分数优先规则选最佳联合模型。最终候选还必须与冻结 winner 和 baseline control 比较；如果联合模型的 `competition.score` 更低，就交付更好的冻结模型，不因训练阶段更复杂而强行选择 joint。

## 5. 最终独立复评、预测和逐事件审计

```bash
FINAL_RUN="$ROOT/<winning-run>"
FINAL_SUBMISSION="$ROOT/submission"
python -m huav.cli.evaluate \
  --config "$FINAL_RUN/config.yaml" \
  --checkpoint "$FINAL_RUN/best.pt" \
  --device cuda \
  --output "$FINAL_RUN/reproduced_metrics.json" \
  --quiet
cmp "$FINAL_RUN/metrics.json" "$FINAL_RUN/reproduced_metrics.json"
python -m huav.cli.predict \
  --config "$FINAL_RUN/config.yaml" \
  --checkpoint "$FINAL_RUN/best.pt" \
  --output-dir "$FINAL_SUBMISSION" \
  --device cuda
```

然后审计 `submission.zip`：24 个验证 NPZ 必须恰好对应 24 个根目录 TXT，总行数必须等于 1,424,330；每行严格五列 `x y t p predicted_label`，前四列逐值逐序与源 NPZ 一致，标签只能为 0/1，ZIP 不得含子目录或非 TXT 文件。记录 TXT 数、总行数、正类总数、ZIP member 数及 SHA-256；任一项失败都不得交付。

```bash
python - "$FINAL_SUBMISSION" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
import zipfile
import numpy as np

source_dir = Path("data/ev-uav/val")
submission_dir = Path(sys.argv[1])
sources = sorted(source_dir.glob("*.npz"))
txt_files = sorted(submission_dir.glob("*.txt"))
assert len(sources) == len(txt_files) == 24
assert [path.stem for path in sources] == [path.stem for path in txt_files]

total_rows = 0
positive_rows = 0
for source_path, txt_path in zip(sources, txt_files, strict=True):
    with np.load(source_path, allow_pickle=False) as archive:
        source = archive["ev"]
    rows = np.loadtxt(txt_path, dtype=np.float64, ndmin=2)
    assert rows.shape == (len(source), 5)
    assert np.array_equal(rows[:, 0], source["x"])
    assert np.array_equal(rows[:, 1], source["y"])
    assert np.array_equal(rows[:, 2], source["t"].astype(np.float64))
    assert np.array_equal(rows[:, 3], source["p"])
    assert np.isin(rows[:, 4], (0.0, 1.0)).all()
    total_rows += len(rows)
    positive_rows += int(rows[:, 4].sum())

assert total_rows == 1_424_330
zip_path = submission_dir / "submission.zip"
with zipfile.ZipFile(zip_path) as archive:
    members = archive.namelist()
assert members == [path.name for path in txt_files]
assert all("/" not in name and name.endswith(".txt") for name in members)
digest = sha256(zip_path.read_bytes()).hexdigest()
print({
    "txt_count": len(txt_files),
    "total_rows": total_rows,
    "positive_rows": positive_rows,
    "zip_members": len(members),
    "zip_sha256": digest,
})
PY
```

最终报告必须同时列出 baseline、四个 frozen、三个 joint 的最佳 epoch、阈值、`competition.score`、IoU、SegAcc、Pd、Fa，以及最终模型的事件级 Pd/Fa/IoU/Acc、独立复评一致性和提交 ZIP 绝对路径。
