"""LightGBM事件重评分及按序列OOF训练。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import EventCache, PostprocessProtocol, fold_of, sha256_file
from .features import EVENT_FEATURE_DIM, build_event_features, make_context


def require_lightgbm():
    """延迟导入，避免普通TraceFormer训练强制依赖LightGBM。"""

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError(
            'postprocessing requires: python -m pip install -e ".[postprocess]"'
        ) from error
    return lgb


def _sequence_rng(seed: int, name: str) -> np.random.Generator:
    token = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(token[:16], 16))


def sample_event_rows(cache: EventCache, protocol: PostprocessProtocol) -> np.ndarray:
    if cache.labels is None:
        raise ValueError(f"{cache.name}: training samples require labels")
    positive = np.flatnonzero(cache.labels)
    negative = np.flatnonzero(~cache.labels)
    maximum_negative = min(
        len(negative),
        max(
            protocol.minimum_negative_samples,
            protocol.negative_to_positive_ratio * len(positive),
        ),
    )
    hard = negative[cache.base_probability[negative] >= protocol.hard_negative_threshold]
    hard_limit = min(len(hard), maximum_negative // 2)
    rng = _sequence_rng(protocol.seed, cache.name)
    hard_selected = (
        rng.choice(hard, hard_limit, replace=False) if hard_limit else np.empty(0, dtype=np.int64)
    )
    remaining = np.setdiff1d(negative, hard_selected, assume_unique=False)
    random_limit = maximum_negative - len(hard_selected)
    random_selected = (
        rng.choice(remaining, random_limit, replace=False)
        if random_limit
        else np.empty(0, dtype=np.int64)
    )
    selected = np.concatenate((positive, hard_selected, random_selected))
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _training_matrix(
    cache_dir: Path,
    names: Iterable[str],
    protocol: PostprocessProtocol,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    report: list[dict[str, int]] = []
    for name in names:
        cache = EventCache.load(cache_dir / name, allow_labels=True)
        rows = sample_event_rows(cache, protocol)
        context = make_context(cache.locations, cache.base_probability)
        features.append(build_event_features(cache, indices=rows, context=context))
        assert cache.labels is not None
        labels.append(cache.labels[rows].astype(np.uint8))
        report.append(
            {
                "events": len(cache.locations),
                "selected": len(rows),
                "positives": int(cache.labels[rows].sum()),
            }
        )
    return np.concatenate(features), np.concatenate(labels), report


def _booster_parameters(*, size: str, seed: int) -> dict[str, Any]:
    common: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "bagging_freq": 1,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "verbosity": -1,
        "num_threads": 8,
        "deterministic": True,
        "force_col_wise": True,
    }
    if size == "small":
        return common | {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
        }
    if size == "large":
        return common | {
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "lambda_l1": 0.2,
            "lambda_l2": 2.0,
        }
    raise ValueError("size must be small or large")


def train_event_models(
    cache_dir: str | Path,
    names: Iterable[str],
    output_dir: str | Path,
    protocol: PostprocessProtocol,
    *,
    name_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """训练两套事件模型并返回模型与可审计报告。"""

    protocol.validate()
    lgb = require_lightgbm()
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_names = sorted(names)
    x, y, sampling = _training_matrix(cache_dir, ordered_names, protocol)
    if x.shape[1] != EVENT_FEATURE_DIM:
        raise RuntimeError("training feature schema mismatch")
    boosters: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for offset, size in enumerate(("small", "large")):
        parameters = _booster_parameters(size=size, seed=protocol.seed + offset * 10)
        rounds = protocol.small_rounds if size == "small" else protocol.large_rounds
        booster = lgb.train(
            parameters,
            lgb.Dataset(x, label=y),
            num_boost_round=rounds,
        )
        model_path = output_dir / f"{name_prefix}{size}.txt"
        booster.save_model(str(model_path))
        boosters[size] = booster
        records.append(
            {
                "name": size,
                "path": model_path.name,
                "rounds": rounds,
                "parameters": parameters,
                "sha256": sha256_file(model_path),
            }
        )
    report = {
        "protocol": asdict(protocol),
        "training_sequences": ordered_names,
        "training_samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "feature_dim": int(x.shape[1]),
        "sampling": sampling,
        "models": records,
    }
    report_path = output_dir / f"{name_prefix}training_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return boosters, report


def required_event_models(contract: dict[str, Any]) -> tuple[str, ...]:
    """只选择 direct 或 track 融合中实际参与决策的树模型。"""

    return tuple(
        name
        for index, name in enumerate(("small", "large"))
        if any(contract[key][index] != 0 for key in ("direct_weights", "track_weights"))
    )


def positive_prediction_threads(value: str) -> int:
    """CLI 线程数必须为正整数，禁止 LightGBM 隐式使用全部线程。"""

    threads = int(value)
    if threads <= 0:
        raise ValueError("prediction threads must be a positive integer")
    return threads


def load_event_models(
    model_dir: str | Path,
    *,
    prefix: str = "",
    names: Iterable[str] = ("small", "large"),
    backend: str = "cpu",
    device: str = "cuda",
) -> dict[str, Any]:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("prediction backend must be cpu or cuda")
    names = tuple(names)
    if any(name not in ("small", "large") for name in names):
        raise ValueError("event model names must be small or large")
    # 纯 base 配置无需加载树模型，也无需导入 LightGBM。
    if not names:
        return {}
    lgb = require_lightgbm()
    model_dir = Path(model_dir)
    models = {
        size: lgb.Booster(model_file=str(model_dir / f"{prefix}{size}.txt"))
        for size in names
    }
    if backend == "cuda":
        from .gpu import CudaEventModel

        models = {name: CudaEventModel(model, device=device) for name, model in models.items()}
    return models


def predict_event_scores(
    cache: EventCache,
    boosters: dict[str, Any],
    *,
    chunk_size: int = 200_000,
    num_threads: int = 1,
    feature_backend: str = "cpu",
) -> dict[str, np.ndarray]:
    """分块展开事件特征，控制密集序列峰值内存。"""

    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if feature_backend not in {"cpu", "cuda"}:
        raise ValueError("feature backend must be cpu or cuda")
    # 无树模型时跳过上下文和事件特征构造。
    if not boosters:
        return {}
    if feature_backend == "cuda":
        from .gpu import CudaEventFeatures, CudaEventModel

        if not all(isinstance(model, CudaEventModel) for model in boosters.values()):
            raise ValueError("CUDA features require CUDA event trees")
        context = CudaEventFeatures(cache, device=next(iter(boosters.values())).device)
    else:
        context = make_context(cache.locations, cache.base_probability)
    scores = {name: np.empty(len(cache.locations), dtype=np.float32) for name in boosters}
    for start in range(0, len(cache.locations), chunk_size):
        stop = min(start + chunk_size, len(cache.locations))
        if feature_backend == "cuda":
            features = context.rows(start, stop)
        else:
            rows = np.arange(start, stop, dtype=np.int64)
            features = build_event_features(cache, indices=rows, context=context)
        for name, booster in boosters.items():
            scores[name][start:stop] = booster.predict(
                features, num_threads=num_threads
            ).astype(np.float32)
    return scores


def generate_oof_scores(
    cache_dir: str | Path,
    names: Iterable[str],
    output_dir: str | Path,
    protocol: PostprocessProtocol,
) -> dict[str, Any]:
    """每个序列只使用未见过该序列的LightGBM产生下游分数。"""

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_names = sorted(names)
    folds: list[dict[str, Any]] = []
    for fold in range(protocol.folds):
        validation = [
            name
            for name in ordered_names
            if fold_of(name, seed=protocol.seed, folds=protocol.folds) == fold
        ]
        training = [name for name in ordered_names if name not in set(validation)]
        if not training or not validation:
            raise ValueError(f"fold {fold} is empty")
        model_dir = output_dir / "models" / f"fold_{fold}"
        boosters, report = train_event_models(
            cache_dir,
            training,
            model_dir,
            protocol,
            name_prefix="",
        )
        for name in validation:
            cache = EventCache.load(cache_dir / name, allow_labels=True)
            scores = predict_event_scores(cache, boosters)
            np.savez_compressed(output_dir / name, **scores)
        folds.append(
            {
                "fold": fold,
                "training": training,
                "validation": validation,
                "model_report": report,
            }
        )
    report = {"protocol": asdict(protocol), "folds": folds}
    (output_dir / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def convex_weights(step: float) -> list[tuple[float, float, float]]:
    """枚举small、large、base三项非负且和为一的权重。"""

    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("blend step must divide one exactly")
    values = []
    for small in range(units + 1):
        for large in range(units - small + 1):
            base = units - small - large
            values.append((small / units, large / units, base / units))
    return values


def blend_scores(
    base: np.ndarray,
    small: np.ndarray | None,
    large: np.ndarray | None,
    weights: tuple[float, float, float],
) -> np.ndarray:
    terms = []
    for name, score, weight in zip(("small", "large"), (small, large), weights[:2]):
        if score is None and weight != 0:
            raise ValueError(f"missing nonzero-weight event scores: {name}")
        # 以零标量补齐缺失项，保留原来的 small + large + base 运算顺序。
        terms.append(np.float32(0) if score is None else weight * score)
    return (terms[0] + terms[1] + weights[2] * base).astype(np.float32)
