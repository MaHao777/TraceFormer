# Event Release And Dense Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已通过门槛的事件重评分器用全部 99 个训练序列发布，并用仅限 dense 序列的历史长轨迹稀疏救回提高 Pd，同时保持 Fa 门槛。

**Architecture:** 开发阶段仍用固定 79/20 划分选择事件融合和轨迹参数；事件阶段通过后，立即使用冻结参数在全部训练缓存上重训两个 LightGBM 并写入独立 `event_release/`。预测时只读取发布模型；dense 救回以历史 schema-4 二值预测为候选，只补充当前预测缺失且形成至少 8 帧严格轨迹的原始事件，不扩张到候选外事件。

**Tech Stack:** Python 3.12、NumPy、SciPy、LightGBM、PyTorch 缓存、pytest、Ruff。

---

### Task 1: 固定事件发布契约

**Files:**
- Modify: `src/huav/postprocessing/pipeline.py`
- Test: `tests/test_postprocessing.py`

- [ ] **Step 1: 写失败测试**

通过 monkeypatch 让 `train_event_models` 记录训练序列，断言事件门槛通过后会以全部缓存名调用一次发布训练，并将模型报告写入 `event_stage.release`；组件门槛失败不得删除这个发布结果。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_postprocessing.py -k event_release -q`

Expected: FAIL，当前报告没有 `event_stage.release`，且发布训练只位于所有级联通过之后。

- [ ] **Step 3: 最小实现**

在事件门槛通过后执行：

```python
release_events = output_dir / "event_release" / "events"
_, release_report = train_event_models(cache_dir, names, release_events, protocol)
report["event_stage"]["release"] = {
    "model_dir": str(release_events.resolve()),
    "training_files": len(names),
    "model": release_report,
}
```

保持组件阶段原门槛和状态不变，避免把被拒绝的组件剪枝误发布。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_postprocessing.py -k event_release -q`

Expected: PASS。

### Task 2: 实现 dense-only 历史长轨迹救回

**Files:**
- Modify: `src/huav/postprocessing/tracking.py`
- Test: `tests/test_postprocessing.py`

- [ ] **Step 1: 写失败测试**

构造 8 帧连续移动的 reference-only 事件、7 帧短轨迹和当前已预测事件，验证：

```python
rescued, record = rescue_reference_tracks(
    locations,
    current_prediction,
    reference_prediction,
    density_group="dense",
)
assert rescued[long_track].all()
assert not rescued[short_track].any()
assert np.all(rescued >= current_prediction)
assert record["added_events"] == 8
```

另加 sparse 输入测试，必须原样返回当前预测且新增数为 0。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_postprocessing.py -k reference_track_rescue -q`

Expected: FAIL，函数尚不存在。

- [ ] **Step 3: 最小实现**

新增 `rescue_reference_tracks`，候选严格限定为 `reference & ~current`；使用 `cluster_radius=1`、`max_component_span=12`、`max_gap=2`、`max_distance=4`、`minimum_track_frames=8`，并以 `half_width=2`、`half_height=2`、`extension_frames=0` 只写回 reference 中已有的事件。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_postprocessing.py -k reference_track_rescue -q`

Expected: PASS。

### Task 3: 让验证和 Track 2 预测使用发布模型与可选救回

**Files:**
- Modify: `scripts/evaluate_postprocess_stages.py`
- Modify: `scripts/predict_postprocess_track2.py`
- Test: `tests/test_postprocessing.py`

- [ ] **Step 1: 写失败测试**

测试模型目录解析必须优先要求 `event_release/events`，且训练报告中的 `training_files` 必须等于全部训练缓存数；测试 reference 预测按源文件 stem 从 NPZ/TXT 加载。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_postprocessing.py -k "event_model_dir or reference_prediction" -q`

Expected: FAIL，当前脚本硬编码 `development/events`。

- [ ] **Step 3: 接入验证脚本**

增加 `--reference-prediction-dir`。输出 `event_release` 与 `event_release_dense_rescue` 两个阶段，记录模型哈希、发布训练文件数、逐文件新增事件和轨迹数；救回前后均从原始验证 NPZ 独立计分。

- [ ] **Step 4: 接入 Track 2 脚本**

要求发布模型存在；增加 `--reference-prediction-dir`，仅 dense 文件调用救回。manifest 将方案名、救回参数、模型哈希、逐文件新增量和总新增量写全，继续使用 `%.9f` 官方格式。

- [ ] **Step 5: 运行定向测试**

Run: `python -m pytest tests/test_postprocessing.py tests/test_submission.py -q`

Expected: PASS。

### Task 4: 静态验证

**Files:**
- Verify: all changed source, script, test and plan files

- [ ] **Step 1: 完整测试**

Run: `python -m pytest -q`

Expected: all tests pass。

- [ ] **Step 2: 风格与补丁检查**

Run: `python -m ruff check src scripts tests`

Run: `git diff --check`

Expected: both commands exit 0。

### Task 5: 训练、独立复评与提交门槛

**Files:**
- Create: `outputs/postprocess_event_release_dense_rescue_20260829/`

- [ ] **Step 1: 复用现有 schema-5 训练/验证缓存**

核对缓存 checkpoint/config SHA-256 与 `outputs/cpu_20260827_230649/best.pt`、`config.yaml` 一致；不重训主神经网络。

- [ ] **Step 2: 用冻结协议训练 99 序列事件发布模型**

复用 `small_rounds=997`、`large_rounds=1998`、seed 37 以及既有事件权重、阈值和分密度轨迹参数，不重新读取验证标签调参。

- [ ] **Step 3: 24 文件独立复评**

分别评估：79 序列旧事件模型、99 序列发布模型、99 序列发布模型 + dense 救回。验收门槛：最终全局 Score 高于 `0.946594016911914`，`val_023` Pd 不低于旧事件阶段，且全局 Fa 不高于 `6.0e-6`。

- [ ] **Step 4: 生成候选提交**

仅当门槛通过时，使用同一冻结模型和救回参数生成 31 个 TXT，按 `%.9f` 保存时间戳并打包为 ZIP 根目录平铺文件。

- [ ] **Step 5: 独立审计**

运行 `scripts/audit_track2_submission.py`，要求 31 个 TXT、2,265,422 行、源 `x y t p` 顺序逐行一致、标签二值、ZIP 根目录无子目录；记录 ZIP SHA-256。

### Self-review

- 需求覆盖：99 序列发布、dense 救回、完整验证门槛、Track 2 格式审计均有独立任务。
- 无占位实现：参数、目录、命令、验收阈值均已固定。
- 类型一致：救回函数输入输出均为对齐事件的一维布尔数组和 JSON 可序列化记录。
- 执行方式：用户已要求开始实施；受仓库 `AGENTS.md` 约束，在当前本地仓库内 inline 执行，不创建 worktree 或子代理。
