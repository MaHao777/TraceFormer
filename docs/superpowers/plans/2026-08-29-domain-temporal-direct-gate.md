# Domain-Matched Temporal Direct Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变轨迹预测和 dense 历史救回的前提下，只审查没有轨迹支持的 direct 高置信组件，用校准不敏感的时序证据降低 Track 2 域偏移误警。

**Architecture:** 新增独立 `direct_verifier` 模块，从事件模型 OOF 分数构造 `direct & ~tracked` 候选组件，使用相对概率秩、前后帧支持、极性差异和热像素持续性训练小型 LightGBM。开发模型使用79个序列OOF训练、20个独立留出序列选阈值；发布模型使用全部99个严格样本外分数重训。推理仅在 sparse/ordinary 序列门控 direct-only 候选，dense 结果与既有长轨迹救回保持不变。

**Tech Stack:** Python 3.12、NumPy、SciPy、LightGBM、pytest、现有 HUAV 后处理缓存。

---

### Task 1: 固定 direct-only 候选契约

**Files:**
- Create: `src/huav/postprocessing/direct_verifier.py`
- Modify: `tests/test_postprocessing.py`

- [ ] **Step 1: 写失败测试**

构造同一帧内一个 tracked 组件和一个 direct-only 组件，断言候选批次只包含后者；构造 dense 序列，断言应用门控后结果完全不变。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_postprocessing.py -k direct_verifier -q`

Expected: FAIL，模块和接口尚不存在。

- [ ] **Step 3: 实现最小候选接口**

实现：

```python
@dataclass(frozen=True)
class DirectCandidateBatch:
    features: np.ndarray
    members: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray | None
    critical: np.ndarray | None

def direct_candidate_features(
    cache: EventCache,
    direct_probability: np.ndarray,
    tracked_prediction: np.ndarray,
    direct_prediction: np.ndarray,
    *,
    include_targets: bool,
) -> DirectCandidateBatch:
    ...

def apply_direct_verifier(
    batch: DirectCandidateBatch,
    tracked_prediction: np.ndarray,
    direct_prediction: np.ndarray,
    model: Any,
    threshold: float,
    *,
    density_group: str,
) -> np.ndarray:
    ...
```

候选严格等于 `direct_prediction & ~tracked_prediction`；输出始终包含全部 tracked 事件；dense 序列直接返回 `tracked | direct`。

- [ ] **Step 4: 运行定向测试**

Run: `python -m pytest tests/test_postprocessing.py -k direct_verifier -q`

Expected: PASS。

### Task 2: 实现域不敏感特征与反事实饱和视图

**Files:**
- Modify: `src/huav/postprocessing/direct_verifier.py`
- Modify: `tests/test_postprocessing.py`

- [ ] **Step 1: 写特征测试**

验证特征不包含绝对 `x/y/time`，同一几何在单调 logit 饱和变换后概率秩保持一致；前后帧有支持的组件与单帧孤立组件得到不同的时序支持特征。

- [ ] **Step 2: 实现39维特征**

特征只包含组件形状、帧内概率秩、相对分位间隔、组件极性相对序列极性、前后 `1/2/4` 帧邻域支持、像素活跃帧数以及归一化序列密度/极性。不得加入绝对坐标或绝对时间。

- [ ] **Step 3: 实现反事实饱和**

```python
def saturate_probability(
    probability: np.ndarray, *, scale: float, bias: float
) -> np.ndarray:
    logit = np.log(p / (1 - p))
    return sigmoid(scale * logit + bias)
```

训练使用原始视图以及 `(scale, bias)=(1.5, 0.5)`、`(2.0, 0.75)`；仅对 `density < 800` 且序列正极性比例大于 `0.8` 的真实训练序列启用增强。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_postprocessing.py -k "direct_verifier or saturation" -q`

Expected: PASS。

### Task 3: 训练、阈值选择与发布契约

**Files:**
- Modify: `src/huav/postprocessing/direct_verifier.py`
- Modify: `src/huav/postprocessing/__init__.py`
- Modify: `src/huav/cli/postprocess.py`
- Modify: `tests/test_postprocessing.py`

- [ ] **Step 1: 写训练契约测试**

使用 monkeypatch 验证79个训练序列只消费OOF分数，20个留出序列只用于阈值选择；发布模型训练集合必须等于99个序列，报告保存模型哈希、特征维数、开发阈值和门槛结果。

- [ ] **Step 2: 训练开发模型**

每个候选组件一行，负组件天然按Fa等权；唯一命中目标的正组件使用 `1 + 12 * critical` 权重。低密度且极性失衡序列额外乘3倍域权重。LightGBM固定seed 37、600轮、31叶节点。

- [ ] **Step 3: 选择阈值**

在20个留出序列上遍历 `0.05..0.95`。必须满足完整官方Score增加至少 `0.001`、检出目标数不下降、两个留出半集增益非负；dense分支因不门控而必须逐事件一致。

- [ ] **Step 4: 训练发布模型**

将79个OOF分数和20个开发模型留出分数组成99个严格样本外分数，冻结阈值后训练发布模型并写 `direct_verifier.txt`、`training_report.json`。

- [ ] **Step 5: 接入CLI**

新增命令：

```powershell
python -m huav.cli.postprocess train-direct-verifier `
  --cache-dir <train-cache> `
  --training-report <event-training-report> `
  --output-dir <output>
```

### Task 4: 接入24文件验证和Track 2预测

**Files:**
- Modify: `scripts/evaluate_postprocess_stages.py`
- Modify: `scripts/predict_postprocess_track2.py`
- Modify: `tests/test_postprocessing.py`

- [ ] **Step 1: 接入验证脚本**

增加 `--direct-verifier-dir`，同时输出 `event_release`、`event_release_direct_gate`、`event_release_direct_gate_dense_rescue` 三阶段。盲推阶段不读取验证标签，独立计分阶段再读取原始NPZ。

- [ ] **Step 2: 接入Track 2脚本**

加载并校验门控模型与报告哈希；先生成 tracked/direct，再门控direct-only，最后执行既有dense救回。manifest记录门控候选数、保留数、删除数和模型哈希。

- [ ] **Step 3: 运行静态验证**

Run: `python -m pytest tests/test_postprocessing.py tests/test_submission.py -q`

Run: `python -m pytest -q`

Run: `python -m ruff check src scripts tests`

Run: `git diff --check`

Expected: 全部退出码0。

### Task 5: 正式训练、45分钟唤醒与独立复评

**Files:**
- Create: `outputs/postprocess_direct_temporal_gate_20260829/`

- [ ] **Step 1: 复用已验证缓存和事件模型**

复用 `outputs/postprocess_huav_20260828/cache/train`、`cache/val`、既有79序列OOF和99序列事件发布模型，不重训主神经网络或事件重评分器。

- [ ] **Step 2: 启动门控训练**

将stdout/stderr写入独立日志并异步运行；记录PID、命令、输入哈希和输出目录。

- [ ] **Step 3: 设置45分钟后线程唤醒**

建立一次性heartbeat，只检查训练是否完成、日志是否报错和产物是否齐全，不重复启动训练。

- [ ] **Step 4: 24文件独立复评**

正式基线为 `0.9541170157392043`。候选必须满足：Score至少提高 `0.001`，`val_023` Pd不低于 `0.7917675545`，22个sparse验证文件Pd不低于 `0.9857934024`，Fa不高于 `4.4114608e-6`。

- [ ] **Step 5: Track 2候选与审计**

仅在门槛通过时生成31个TXT和ZIP，继续使用 `%.9f`，要求31文件、2,265,422行、源 `x y t p` 顺序一致、标签二值、ZIP根目录仅TXT。

### Self-review

- Spec coverage: direct-only门控、时序特征、域匹配训练、45分钟唤醒、完整验证与Track 2审计均有独立任务。
- Placeholder scan: 所有接口、参数、命令与验收阈值已经明确。
- Type consistency: 候选批次统一使用对齐事件索引和一维布尔预测；训练与推理共享同一特征函数。
- Repository constraint: 直接在当前本地checkout实施，不创建worktree。当前工作区已有未提交实验改动，因此本计划不自动创建Git提交，避免混入用户现有工作。
