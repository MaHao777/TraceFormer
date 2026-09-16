"""当前TraceFormer模型的完整后处理训练与盲推编排。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .contracts import (
    EventCache,
    PostprocessProtocol,
    density_group,
    make_internal_split,
    sha256_file,
)
from .learning import (
    blend_scores,
    convex_weights,
    generate_oof_scores,
    load_event_models,
    predict_event_scores,
    require_lightgbm,
    train_event_models,
)
from .metrics import BinaryCounts, score_binary
from .refinement import (
    apply_component_model,
    center_candidates,
    component_features,
    render_center_candidates,
    train_center_model,
    train_component_model,
)
from .submission import load_prediction, write_prediction_npz
from .tracking import track_prediction

SEED_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85)
LOW_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)
DIRECT_THRESHOLDS = (0.85, 0.88, 0.90, 0.92, 0.94, 0.96)
FALLBACK_THRESHOLDS = (0.01, 0.02, 0.03, 0.05)
GROUPS = ("sparse", "ordinary", "dense")


def _cache_names(cache_dir: Path) -> list[str]:
    names = sorted(path.name for path in cache_dir.glob("*.npz"))
    if not names:
        raise FileNotFoundError(f"no cache files found in {cache_dir}")
    return names


def _aggregate(
    caches: dict[str, EventCache], predictions: dict[str, np.ndarray], names: Iterable[str]
) -> dict[str, float | int]:
    counts = BinaryCounts()
    for name in names:
        cache = caches[name]
        if cache.labels is None or cache.target_ids is None:
            raise ValueError("training evaluation requires labels")
        counts += score_binary(cache.locations, cache.labels, cache.target_ids, predictions[name])
    return counts.metrics()


def _event_screen(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: list[str],
    weights: tuple[float, float, float],
    threshold: float,
) -> float:
    tp = fp = fn = 0
    for name in names:
        cache = caches[name]
        assert cache.labels is not None
        probability = blend_scores(
            cache.base_probability, scores[name]["small"], scores[name]["large"], weights
        )
        prediction = probability >= threshold
        tp += int(np.sum(prediction & cache.labels))
        fp += int(np.sum(prediction & ~cache.labels))
        fn += int(np.sum(~prediction & cache.labels))
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.2 * iou + 0.1 * recall


def _direct_predictions(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: Iterable[str],
    weights: tuple[float, float, float],
    threshold: float,
) -> dict[str, np.ndarray]:
    return {
        name: blend_scores(
            caches[name].base_probability,
            scores[name]["small"],
            scores[name]["large"],
            weights,
        )
        >= threshold
        for name in names
    }


def _select_direct(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: list[str],
    step: float,
) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    """先用事件计数筛选权重，再对少量候选计算完整官方分数。"""

    screened: list[tuple[float, tuple[float, float, float]]] = []
    for weights in convex_weights(step):
        best = max(
            _event_screen(caches, scores, names, weights, threshold)
            for threshold in DIRECT_THRESHOLDS
        )
        screened.append((best, weights))
    candidates = [weights for _, weights in sorted(screened, reverse=True)[:12]]
    records: list[dict[str, Any]] = []
    for weights in candidates:
        for threshold in DIRECT_THRESHOLDS:
            prediction = _direct_predictions(caches, scores, names, weights, threshold)
            metrics = _aggregate(caches, prediction, names)
            records.append({"weights": list(weights), "threshold": threshold, "metrics": metrics})
    winner = max(records, key=lambda row: float(row["metrics"]["score"]))
    return (
        tuple(winner["weights"]),
        float(winner["threshold"]),
        {
            "winner": winner,
            "evaluated": records,
        },
    )


def _tracking_settings(seed: float, low: float, fallback: float) -> dict[str, Any]:
    return {
        "seed_threshold": seed,
        "low_threshold": low,
        "hot_minimum_frames": 12,
        "cluster_radius": 2,
        "max_component_span": 24,
        "max_gap": 7,
        "max_distance": 8.0,
        "minimum_track_frames": 8,
        "half_width": 5,
        "half_height": 4,
        "extension_frames": 9,
        "box_power": 2.75,
        "minimum_events_per_center": 2,
        "fallback_threshold": fallback,
    }


def _stage1_predictions(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: Iterable[str],
    *,
    track_weights: tuple[float, float, float],
    direct_weights: tuple[float, float, float],
    direct_threshold: float,
    tracking: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name in names:
        cache = caches[name]
        group, _ = density_group(cache.locations)
        track_probability = blend_scores(
            cache.base_probability,
            scores[name]["small"],
            scores[name]["large"],
            track_weights,
        )
        direct_probability = blend_scores(
            cache.base_probability,
            scores[name]["small"],
            scores[name]["large"],
            direct_weights,
        )
        tracked = track_prediction(cache.locations, track_probability, **tracking[group])
        predictions[name] = tracked | (direct_probability >= direct_threshold)
        probabilities[name] = track_probability
    return predictions, probabilities


def _select_track_weights(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: list[str],
    direct_weights: tuple[float, float, float],
    direct_threshold: float,
    step: float,
) -> tuple[float, float, float]:
    # 轨迹代价高，只对事件筛选最优的前6组权重做完整比较。
    ranked = sorted(
        (
            (_event_screen(caches, scores, names, weights, 0.75), weights)
            for weights in convex_weights(step)
        ),
        reverse=True,
    )[:6]
    default = {group: _tracking_settings(0.75, 0.03, 0.02) for group in GROUPS}
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for _, weights in ranked:
        prediction, _ = _stage1_predictions(
            caches,
            scores,
            names,
            track_weights=weights,
            direct_weights=direct_weights,
            direct_threshold=direct_threshold,
            tracking=default,
        )
        candidates.append((float(_aggregate(caches, prediction, names)["score"]), weights))
    return max(candidates)[1]


def _tune_tracking(
    caches: dict[str, EventCache],
    scores: dict[str, dict[str, np.ndarray]],
    names: list[str],
    *,
    track_weights: tuple[float, float, float],
    direct_weights: tuple[float, float, float],
    direct_threshold: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """按密度组坐标搜索新模型相关阈值，固定轨迹几何。"""

    settings = {group: _tracking_settings(0.75, 0.03, 0.02) for group in GROUPS}
    records: list[dict[str, Any]] = []
    for group in GROUPS:
        group_names = [name for name in names if density_group(caches[name].locations)[0] == group]
        if not group_names:
            continue
        for key, values in (
            ("seed_threshold", SEED_THRESHOLDS),
            ("low_threshold", LOW_THRESHOLDS),
            ("fallback_threshold", FALLBACK_THRESHOLDS),
        ):
            candidates: list[tuple[float, float, dict[str, Any]]] = []
            for value in values:
                candidate = {name: dict(item) for name, item in settings.items()}
                candidate[group][key] = value
                prediction, _ = _stage1_predictions(
                    caches,
                    scores,
                    group_names,
                    track_weights=track_weights,
                    direct_weights=direct_weights,
                    direct_threshold=direct_threshold,
                    tracking=candidate,
                )
                metrics = _aggregate(caches, prediction, group_names)
                candidates.append((float(metrics["score"]), value, metrics))
            score, value, metrics = max(candidates, key=lambda item: item[0])
            settings[group][key] = value
            records.append(
                {
                    "group": group,
                    "parameter": key,
                    "value": value,
                    "score": score,
                    "metrics": metrics,
                }
            )
    return settings, records


def _gate(
    caches: dict[str, EventCache],
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    names: list[str],
    *,
    minimum_gain: float,
    strict_cross_checks: bool,
) -> dict[str, Any]:
    baseline = _aggregate(caches, before, names)
    candidate = _aggregate(caches, after, names)
    gain = float(candidate["score"]) - float(baseline["score"])
    result: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "gain": gain,
        "passed": gain >= minimum_gain,
    }
    dense_names = [name for name in names if density_group(caches[name].locations)[0] == "dense"]
    if dense_names:
        dense_before = _aggregate(caches, before, dense_names)
        dense_after = _aggregate(caches, after, dense_names)
        result["dense_pd_gain"] = float(dense_after["pd"]) - float(dense_before["pd"])
        result["passed"] = bool(result["passed"] and result["dense_pd_gain"] >= 0)
    if strict_cross_checks:
        half_gains = []
        for half in (names[::2], names[1::2]):
            if half:
                half_gains.append(
                    float(_aggregate(caches, after, half)["score"])
                    - float(_aggregate(caches, before, half)["score"])
                )
        leave_one_out = []
        for omitted in names:
            subset = [name for name in names if name != omitted]
            if subset:
                leave_one_out.append(
                    float(_aggregate(caches, after, subset)["score"])
                    - float(_aggregate(caches, before, subset)["score"])
                )
        result["half_gains"] = half_gains
        result["leave_one_sequence_min_gain"] = min(leave_one_out, default=0.0)
        result["passed"] = bool(
            result["passed"]
            and all(value >= 0 for value in half_gains)
            and result["leave_one_sequence_min_gain"] > 0
        )
    return result


def _save_predictions(directory: Path, predictions: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name, prediction in predictions.items():
        write_prediction_npz(directory / name, prediction)


def _load_oof_scores(directory: Path, names: Iterable[str]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        with np.load(directory / name, allow_pickle=False) as saved:
            result[name] = {"small": saved["small"], "large": saved["large"]}
    return result


def _load_control(directory: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in names:
        stem = Path(name).stem
        candidates = (directory / name, directory / f"{stem}.txt")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing control prediction for {name}")
        result[name] = load_prediction(path)
    return result


def release_event_models(
    cache_dir: str | Path,
    training_report_path: str | Path,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """冻结开发参数，并用全部训练缓存重训事件模型。"""

    cache_dir = Path(cache_dir)
    training_report_path = Path(training_report_path)
    output_dir = Path(output_dir)
    source = json.loads(training_report_path.read_text(encoding="utf-8"))
    event_stage = source.get("event_stage", {})
    if not event_stage.get("gate", {}).get("passed", False):
        raise ValueError("event stage did not pass its development gate")
    names = _cache_names(cache_dir)
    declared = sorted(source.get("training_names", []) + source.get("holdout_names", []))
    if names != declared:
        raise ValueError("training cache names do not match the development split")

    protocol = PostprocessProtocol(**source["protocol"])
    protocol.validate()
    output_dir.mkdir(parents=True, exist_ok=False)
    contract_snapshot = output_dir / "development_contract.json"
    shutil.copy2(training_report_path, contract_snapshot)
    progress(f"使用全部{len(names)}个训练序列发布事件重评分模型")
    model_dir = output_dir / "events"
    _, model_report = train_event_models(cache_dir, names, model_dir, protocol)
    contract = {
        key: event_stage[key]
        for key in ("direct_weights", "track_weights", "direct_threshold", "tracking")
    }
    release = {
        "source_training_report": str(contract_snapshot.resolve()),
        "source_training_report_sha256": sha256_file(contract_snapshot),
        "training_files": len(names),
        "model_dir": str(model_dir.resolve()),
        "model": model_report,
        "event_contract": contract,
        "event_model_sha256": {
            name: sha256_file(model_dir / name) for name in ("small.txt", "large.txt")
        },
    }
    (output_dir / "release_report.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return release


def train_pipeline(
    cache_dir: str | Path,
    output_dir: str | Path,
    protocol: PostprocessProtocol,
    *,
    control_dir: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """按预注册门槛训练；任一级失败即停止，不发布替换模型。"""

    protocol.validate()
    require_lightgbm()
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    names = _cache_names(cache_dir)
    training_names, holdout_names = make_internal_split(
        names, seed=protocol.seed, holdout_files=protocol.holdout_files
    )
    # 首级只常驻20个留出缓存，避免在门槛通过前解压全部99个token缓存。
    caches = {
        name: EventCache.load(cache_dir / name, allow_labels=True)
        for name in holdout_names
    }
    report: dict[str, Any] = {
        "protocol": asdict(protocol),
        "training_names": training_names,
        "holdout_names": holdout_names,
        "status": "running",
    }

    progress("训练79序列事件重评分模型")
    development_events = output_dir / "development" / "events"
    boosters, event_report = train_event_models(
        cache_dir, training_names, development_events, protocol
    )
    holdout_scores = {name: predict_event_scores(caches[name], boosters) for name in holdout_names}
    direct_weights, direct_threshold, direct_report = _select_direct(
        caches, holdout_scores, holdout_names, protocol.blend_step
    )
    track_weights = _select_track_weights(
        caches,
        holdout_scores,
        holdout_names,
        direct_weights,
        direct_threshold,
        protocol.blend_step,
    )
    tracking, tracking_search = _tune_tracking(
        caches,
        holdout_scores,
        holdout_names,
        track_weights=track_weights,
        direct_weights=direct_weights,
        direct_threshold=direct_threshold,
    )
    stage1, holdout_probability = _stage1_predictions(
        caches,
        holdout_scores,
        holdout_names,
        track_weights=track_weights,
        direct_weights=direct_weights,
        direct_threshold=direct_threshold,
        tracking=tracking,
    )
    baseline = {name: caches[name].prediction for name in holdout_names}
    stage1_gate = _gate(
        caches, baseline, stage1, holdout_names, minimum_gain=0.005, strict_cross_checks=False
    )
    control_metrics = None
    if control_dir is not None:
        control = _load_control(Path(control_dir), holdout_names)
        control_metrics = _aggregate(caches, control, holdout_names)
        stage1_gate["control"] = control_metrics
        stage1_gate["passed"] = bool(
            stage1_gate["passed"]
            and float(stage1_gate["candidate"]["score"]) >= float(control_metrics["score"])
        )
    report["event_stage"] = {
        "model": event_report,
        "direct_search": direct_report,
        "direct_weights": list(direct_weights),
        "track_weights": list(track_weights),
        "direct_threshold": direct_threshold,
        "tracking": tracking,
        "tracking_search": tracking_search,
        "gate": stage1_gate,
        "control_supplied": control_dir is not None,
    }
    _save_predictions(output_dir / "development" / "stage1_holdout", stage1)
    if not stage1_gate["passed"]:
        report["status"] = "rejected_event_stage"
        (output_dir / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    # 事件级门槛独立于后续组件级门槛；通过后立即发布全99序列模型。
    training_report_path = output_dir / "training_report.json"
    training_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    event_release_dir = output_dir / "event_release"
    report["event_stage"]["release"] = release_event_models(
        cache_dir,
        training_report_path,
        event_release_dir,
        progress=progress,
    )
    training_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    progress("生成79序列OOF分数并训练组件剪枝")
    caches.update(
        {
            name: EventCache.load(cache_dir / name, allow_labels=True)
            for name in training_names
        }
    )
    oof_dir = output_dir / "development" / "oof_79"
    generate_oof_scores(cache_dir, training_names, oof_dir, protocol)
    oof_scores = _load_oof_scores(oof_dir, training_names)
    oof_stage1, oof_probability = _stage1_predictions(
        caches,
        oof_scores,
        training_names,
        track_weights=track_weights,
        direct_weights=direct_weights,
        direct_threshold=direct_threshold,
        tracking=tracking,
    )
    component_path = output_dir / "development" / "component.txt"
    component_report = train_component_model(
        ((name, caches[name], oof_probability[name], oof_stage1[name]) for name in training_names),
        component_path,
        seed=protocol.seed,
    )
    lgb = require_lightgbm()
    component_model = lgb.Booster(model_file=str(component_path))
    component_candidates: list[tuple[float, dict[str, np.ndarray], dict[str, Any]]] = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        prediction = {}
        for name in holdout_names:
            batch = component_features(
                caches[name], holdout_probability[name], stage1[name], include_targets=False
            )
            prediction[name] = apply_component_model(
                batch, stage1[name], component_model, float(threshold)
            )
        gate = _gate(
            caches, stage1, prediction, holdout_names, minimum_gain=0.001, strict_cross_checks=True
        )
        component_candidates.append(
            (
                float(gate["candidate"]["score"]),
                prediction,
                {"threshold": float(threshold), "gate": gate},
            )
        )
    _, component_prediction, component_winner = max(component_candidates, key=lambda item: item[0])
    report["component_stage"] = {"model": component_report, **component_winner}
    _save_predictions(output_dir / "development" / "component_holdout", component_prediction)
    if not component_winner["gate"]["passed"]:
        report["status"] = "rejected_component_stage"
        (output_dir / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    progress("训练稳定轨迹中心补检模型")
    component_threshold = float(component_winner["threshold"])
    oof_component: dict[str, np.ndarray] = {}
    for name in training_names:
        batch = component_features(
            caches[name], oof_probability[name], oof_stage1[name], include_targets=False
        )
        oof_component[name] = apply_component_model(
            batch, oof_stage1[name], component_model, component_threshold
        )
    center_path = output_dir / "development" / "center.txt"
    center_report = train_center_model(
        (
            (name, caches[name], oof_probability[name], oof_component[name])
            for name in training_names
        ),
        center_path,
        seed=protocol.seed,
    )
    center_model = lgb.Booster(model_file=str(center_path))
    center_thresholds = {group: 0.95 for group in GROUPS}
    center_prediction = dict(component_prediction)
    center_search: list[dict[str, Any]] = []
    for group in GROUPS:
        group_names = [
            name for name in holdout_names if density_group(caches[name].locations)[0] == group
        ]
        if not group_names:
            continue
        candidates = []
        for threshold in np.arange(0.50, 1.0, 0.05):
            trial = dict(center_prediction)
            for name in group_names:
                tracks, features, _ = center_candidates(
                    caches[name], holdout_probability[name], component_prediction[name]
                )
                scores = (
                    center_model.predict(features, num_threads=1) if len(features) else np.empty(0)
                )
                trial[name] = render_center_candidates(
                    caches[name],
                    holdout_probability[name],
                    component_prediction[name],
                    tracks,
                    scores,
                    float(threshold),
                )
            metrics = _aggregate(caches, trial, holdout_names)
            candidates.append((float(metrics["score"]), float(threshold), trial, metrics))
        _, threshold, center_prediction, metrics = max(candidates, key=lambda item: item[0])
        center_thresholds[group] = threshold
        center_search.append({"group": group, "threshold": threshold, "metrics": metrics})
    center_gate = _gate(
        caches,
        component_prediction,
        center_prediction,
        holdout_names,
        minimum_gain=0.001,
        strict_cross_checks=True,
    )
    report["center_stage"] = {
        "model": center_report,
        "thresholds": center_thresholds,
        "search": center_search,
        "gate": center_gate,
    }
    _save_predictions(output_dir / "development" / "center_holdout", center_prediction)
    if not center_gate["passed"]:
        report["status"] = "rejected_center_stage"
        (output_dir / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    progress("所有留出门槛通过，使用99序列重训发布级联")
    release = output_dir / "release"
    release_events = release / "events"
    shutil.copytree(event_release_dir / "events", release_events)
    release_oof = release / "oof_99"
    generate_oof_scores(cache_dir, names, release_oof, protocol)
    release_scores = _load_oof_scores(release_oof, names)
    release_stage1, release_probability = _stage1_predictions(
        caches,
        release_scores,
        names,
        track_weights=track_weights,
        direct_weights=direct_weights,
        direct_threshold=direct_threshold,
        tracking=tracking,
    )
    release_component_path = release / "component.txt"
    train_component_model(
        ((name, caches[name], release_probability[name], release_stage1[name]) for name in names),
        release_component_path,
        seed=protocol.seed,
    )
    release_component_model = lgb.Booster(model_file=str(release_component_path))
    release_component: dict[str, np.ndarray] = {}
    for name in names:
        batch = component_features(
            caches[name], release_probability[name], release_stage1[name], include_targets=False
        )
        release_component[name] = apply_component_model(
            batch, release_stage1[name], release_component_model, component_threshold
        )
    train_center_model(
        (
            (name, caches[name], release_probability[name], release_component[name])
            for name in names
        ),
        release / "center.txt",
        seed=protocol.seed,
    )
    frozen = {
        "protocol": asdict(protocol),
        "direct_weights": list(direct_weights),
        "track_weights": list(track_weights),
        "direct_threshold": direct_threshold,
        "tracking": tracking,
        "component_threshold": component_threshold,
        "center_thresholds": center_thresholds,
        "validation_labels_accessed_at_inference": False,
    }
    (release / "frozen.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["status"] = "released"
    report["release"] = frozen
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def predict_pipeline(
    cache_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """冻结参数盲推；验证缓存契约禁止携带标签。"""

    cache_dir = Path(cache_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((model_dir / "frozen.json").read_text(encoding="utf-8"))
    event_models = load_event_models(model_dir / "events")
    lgb = require_lightgbm()
    component_model = lgb.Booster(model_file=str(model_dir / "component.txt"))
    center_model = lgb.Booster(model_file=str(model_dir / "center.txt"))
    records: dict[str, Any] = {}
    for name in _cache_names(cache_dir):
        cache = EventCache.load(cache_dir / name, allow_labels=False)
        scores = predict_event_scores(cache, event_models)
        group, density = density_group(cache.locations)
        track_probability = blend_scores(
            cache.base_probability, scores["small"], scores["large"], tuple(frozen["track_weights"])
        )
        direct_probability = blend_scores(
            cache.base_probability,
            scores["small"],
            scores["large"],
            tuple(frozen["direct_weights"]),
        )
        prediction = track_prediction(
            cache.locations, track_probability, **frozen["tracking"][group]
        ) | (direct_probability >= float(frozen["direct_threshold"]))
        batch = component_features(cache, track_probability, prediction, include_targets=False)
        prediction = apply_component_model(
            batch, prediction, component_model, float(frozen["component_threshold"])
        )
        tracks, features, _ = center_candidates(cache, track_probability, prediction)
        center_scores = (
            center_model.predict(features, num_threads=1) if len(features) else np.empty(0)
        )
        prediction = render_center_candidates(
            cache,
            track_probability,
            prediction,
            tracks,
            center_scores,
            float(frozen["center_thresholds"][group]),
        )
        output_path = write_prediction_npz(output_dir / name, prediction)
        records[name] = {
            "events": len(prediction),
            "positive": int(prediction.sum()),
            "density_group": group,
            "events_per_frame": density,
            "path": str(output_path.resolve()),
        }
        progress(f"{name}: {int(prediction.sum()):,} positive events")
    report = {
        "validation_labels_accessed": False,
        "model_dir": str(model_dir.resolve()),
        "files": records,
    }
    (output_dir / "inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
