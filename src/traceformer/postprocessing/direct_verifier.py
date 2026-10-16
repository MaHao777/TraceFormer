"""只审查缺少轨迹支持的 direct 高置信组件。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import EventCache, density_group, sha256_file
from .learning import blend_scores, load_event_models, predict_event_scores, require_lightgbm
from .metrics import BinaryCounts, score_binary
from .refinement import connected_components
from .tracking import track_prediction

DIRECT_VERIFIER_FEATURE_DIM = 39
SATURATION_VIEWS = ((1.0, 0.0), (1.5, 0.5), (2.0, 0.75))


@dataclass(frozen=True)
class DirectCandidateBatch:
    features: np.ndarray
    members: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray | None
    critical: np.ndarray | None


def saturate_probability(
    probability: np.ndarray, *, scale: float, bias: float
) -> np.ndarray:
    """在 logit 空间模拟域外高置信饱和，保持事件排序不变。"""

    values = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    shifted = np.clip(scale * logits + bias, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-shifted))).astype(np.float32)


def _pixel_frame_support(locations: np.ndarray, frames: np.ndarray) -> np.ndarray:
    pixels = locations[:, 1] * 346 + locations[:, 0]
    pairs = np.unique(np.column_stack((pixels, frames)), axis=0)
    active_pixels, counts = np.unique(pairs[:, 0], return_counts=True)
    lookup = np.zeros(346 * 260, dtype=np.float32)
    lookup[active_pixels] = counts.astype(np.float32)
    return lookup[pixels]


def direct_candidate_features(
    cache: EventCache,
    direct_probability: np.ndarray,
    tracked_prediction: np.ndarray,
    direct_prediction: np.ndarray,
    *,
    include_targets: bool,
) -> DirectCandidateBatch:
    """提取不含绝对坐标/时间的时序证据，避免域外校准漂移。"""

    probability = np.asarray(direct_probability, dtype=np.float32)
    tracked = np.asarray(tracked_prediction, dtype=bool)
    direct = np.asarray(direct_prediction, dtype=bool)
    count = len(cache.locations)
    if probability.shape != (count,) or tracked.shape != (count,) or direct.shape != (count,):
        raise ValueError("direct verifier inputs must align with the event cache")

    candidate = direct & ~tracked
    frames, grouped = connected_components(cache.locations, candidate)
    unique_frames = np.unique(frames)
    frame_count = max(len(unique_frames), 1)
    sequence_density = count / frame_count
    polarity = cache.raw_features[:, 3]
    positive_fraction = float(np.mean(polarity > 0.5))
    pixel_support = _pixel_frame_support(cache.locations, frames) / frame_count
    frame_indices = {int(frame): np.flatnonzero(frames == frame) for frame in unique_frames}
    frame_sorted = {
        frame: np.sort(probability[indices]) for frame, indices in frame_indices.items()
    }
    frame_quantiles = {
        frame: np.quantile(probability[indices], (0.5, 0.9, 0.99))
        for frame, indices in frame_indices.items()
    }

    rows: list[list[float]] = []
    member_parts: list[np.ndarray] = []
    offsets = [0]
    labels: list[bool] = []
    critical: list[int] = []
    union = tracked | direct
    for frame, components in grouped.items():
        all_in_frame = frame_indices[frame]
        predicted_in_frame = max(int(np.sum(direct[all_in_frame])), 1)
        sorted_values = frame_sorted[frame]
        q50, q90, q99 = frame_quantiles[frame]
        for indices in components:
            points = cache.locations[indices, :2]
            unique_points = np.unique(points, axis=0)
            width = int(np.ptp(points[:, 0]) + 1)
            height = int(np.ptp(points[:, 1]) + 1)
            values = probability[indices]
            ranks = np.searchsorted(sorted_values, values, side="right") / len(sorted_values)
            component_polarity = float(np.mean(polarity[indices] > 0.5))
            bounded_polarity = float(np.clip(component_polarity, 1e-6, 1.0 - 1e-6))
            polarity_entropy = -(
                bounded_polarity * np.log(bounded_polarity)
                + (1.0 - bounded_polarity) * np.log(1.0 - bounded_polarity)
            )
            center = points.mean(axis=0)
            temporal: list[float] = []
            for gap in (1, 2, 4):
                for direction in (-1, 1):
                    other = frame_indices.get(frame + direction * gap)
                    if other is None:
                        temporal.extend((0.0, 0.0))
                        continue
                    distance = np.linalg.norm(cache.locations[other, :2] - center, axis=1)
                    near = other[distance <= 4.0 * gap]
                    temporal.extend(
                        (
                            float(probability[near].max()) if len(near) else 0.0,
                            float(np.log1p(np.sum(direct[near]))) if len(near) else 0.0,
                        )
                    )
            row = [
                float(np.log1p(len(indices))),
                float(np.log1p(len(unique_points))),
                float(width),
                float(height),
                len(unique_points) / max(width * height, 1),
                float(np.log1p(len(all_in_frame))),
                len(indices) / predicted_in_frame,
                float(values.min()),
                float(values.mean()),
                float(values.max()),
                float(values.std()),
                *np.quantile(values, (0.25, 0.5, 0.75)).tolist(),
                float(ranks.min()),
                float(ranks.mean()),
                float(ranks.max()),
                float(values.max() - q50),
                float(values.max() - q90),
                float(values.max() - q99),
                component_polarity,
                component_polarity - positive_fraction,
                float(polarity_entropy),
                float(np.log1p(sequence_density) / np.log1p(5000.0)),
                positive_fraction,
                *temporal,
                float(pixel_support[indices].mean()),
                float(pixel_support[indices].max()),
            ]
            if len(row) != DIRECT_VERIFIER_FEATURE_DIM:
                raise RuntimeError("unexpected direct verifier feature count")
            rows.append(row)
            member_parts.append(indices.astype(np.int32))
            offsets.append(offsets[-1] + len(indices))
            if include_targets:
                if cache.labels is None or cache.target_ids is None:
                    raise ValueError("direct verifier targets require a training cache")
                labels.append(bool(np.any(cache.labels[indices])))
                lost = 0
                target_ids = np.unique(
                    cache.target_ids[indices][
                        cache.labels[indices] & (cache.target_ids[indices] > 0)
                    ]
                )
                for target_id in target_ids:
                    target = (frames == frame) & cache.labels & (cache.target_ids == target_id)
                    predicted_total = int(np.sum(union[target]))
                    predicted_here = int(
                        np.sum(
                            union[indices]
                            & cache.labels[indices]
                            & (cache.target_ids[indices] == target_id)
                        )
                    )
                    if predicted_total > 0 and predicted_here == predicted_total:
                        lost += 1
                critical.append(lost)

    features = np.asarray(rows, dtype=np.float32)
    if not len(features):
        features = np.empty((0, DIRECT_VERIFIER_FEATURE_DIM), dtype=np.float32)
    return DirectCandidateBatch(
        features=features,
        members=(np.concatenate(member_parts) if member_parts else np.empty(0, dtype=np.int32)),
        offsets=np.asarray(offsets, dtype=np.int32),
        labels=np.asarray(labels, dtype=bool) if include_targets else None,
        critical=np.asarray(critical, dtype=np.int16) if include_targets else None,
    )


def apply_direct_verifier(
    batch: DirectCandidateBatch,
    tracked_prediction: np.ndarray,
    direct_prediction: np.ndarray,
    model: Any,
    threshold: float,
    *,
    density_group: str,
) -> np.ndarray:
    """保留全部轨迹结果，只删除验证器拒绝的 direct-only 组件。"""

    tracked = np.asarray(tracked_prediction, dtype=bool)
    direct = np.asarray(direct_prediction, dtype=bool)
    result = tracked | direct
    if density_group == "dense" or not len(batch.features):
        return result
    keep = model.predict(batch.features, num_threads=1) >= threshold
    for index, selected in enumerate(keep):
        if not selected:
            members = batch.members[batch.offsets[index] : batch.offsets[index + 1]]
            result[members] = False
    result[tracked] = True
    return result


def event_branches(
    cache: EventCache,
    scores: dict[str, np.ndarray],
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按冻结事件契约拆分轨迹与 direct 两条支路。"""

    group, _ = density_group(cache.locations)
    track_probability = blend_scores(
        cache.base_probability,
        scores.get("small"),
        scores.get("large"),
        tuple(contract["track_weights"]),
    )
    direct_probability = blend_scores(
        cache.base_probability,
        scores.get("small"),
        scores.get("large"),
        tuple(contract["direct_weights"]),
    )
    tracked = track_prediction(
        cache.locations,
        track_probability,
        **contract["tracking"][group],
    )
    direct = direct_probability >= float(contract["direct_threshold"])
    return track_probability, direct_probability, tracked, direct


def _aggregate(
    caches: dict[str, EventCache], predictions: dict[str, np.ndarray], names: Iterable[str]
) -> dict[str, float | int]:
    counts = BinaryCounts()
    for name in names:
        cache = caches[name]
        if cache.labels is None or cache.target_ids is None:
            raise ValueError("direct verifier evaluation requires labels")
        counts += score_binary(cache.locations, cache.labels, cache.target_ids, predictions[name])
    return counts.metrics()


def _train_model(
    items: Iterable[
        tuple[str, EventCache, np.ndarray, np.ndarray, np.ndarray]
    ],
    output_path: Path,
    *,
    direct_threshold: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    sequences: list[str] = []
    view_count = 0
    for name, cache, direct_probability, tracked, direct in items:
        group, density = density_group(cache.locations)
        positive_fraction = float(np.mean(cache.raw_features[:, 3] > 0.5))
        views = SATURATION_VIEWS if group != "dense" and density < 800 and positive_fraction > 0.8 else SATURATION_VIEWS[:1]
        for scale, bias in views:
            shifted = saturate_probability(direct_probability, scale=scale, bias=bias)
            shifted_direct = shifted >= direct_threshold
            batch = direct_candidate_features(
                cache,
                shifted,
                tracked,
                shifted_direct,
                include_targets=True,
            )
            if not len(batch.features):
                continue
            assert batch.labels is not None and batch.critical is not None
            domain_weight = 3.0 if density < 800 and positive_fraction > 0.8 else 1.0
            feature_parts.append(batch.features)
            label_parts.append(batch.labels.astype(np.uint8))
            weight_parts.append(
                domain_weight * (1.0 + 12.0 * batch.critical.astype(np.float32))
            )
            view_count += 1
        sequences.append(name)
    if not feature_parts:
        raise ValueError("no direct-only candidates available for training")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    weights = np.concatenate(weight_parts)
    lgb = require_lightgbm()
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 2.0,
            "seed": seed,
            "verbosity": -1,
            "num_threads": 8,
            "deterministic": True,
            "force_col_wise": True,
        },
        lgb.Dataset(features, label=labels, weight=weights),
        num_boost_round=600,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    return model, {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "sequences": sorted(sequences),
        "candidates": int(len(labels)),
        "positive_candidates": int(labels.sum()),
        "feature_dim": int(features.shape[1]),
        "training_views": view_count,
        "saturation_views": [list(item) for item in SATURATION_VIEWS],
    }


def _load_score(directory: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(directory / name, allow_pickle=False) as saved:
        return {
            "small": saved["small"].astype(np.float32),
            "large": saved["large"].astype(np.float32),
        }


def train_direct_verifier(
    cache_dir: str | Path,
    training_report_path: str | Path,
    output_dir: str | Path,
    *,
    progress=print,
) -> dict[str, Any]:
    """使用79个OOF序列开发，并用99个样本外序列发布门控器。"""

    cache_dir = Path(cache_dir)
    training_report_path = Path(training_report_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads(training_report_path.read_text(encoding="utf-8"))
    event_stage = source.get("event_stage", {})
    if not event_stage.get("gate", {}).get("passed", False):
        raise ValueError("event stage must pass before training the direct verifier")
    training_names = list(source["training_names"])
    holdout_names = list(source["holdout_names"])
    names = sorted(training_names + holdout_names)
    cache_names = sorted(path.name for path in cache_dir.glob("*.npz"))
    if names != cache_names:
        raise ValueError("direct verifier cache names do not match the development split")
    contract = {
        key: event_stage[key]
        for key in ("direct_weights", "track_weights", "direct_threshold", "tracking")
    }
    oof_dir = training_report_path.parent / "development" / "oof_79"
    development_models = load_event_models(training_report_path.parent / "development" / "events")
    holdout_caches = {
        name: EventCache.load(cache_dir / name, allow_labels=True) for name in holdout_names
    }
    holdout_scores = {
        name: predict_event_scores(holdout_caches[name], development_models)
        for name in holdout_names
    }
    holdout_branches = {
        name: event_branches(holdout_caches[name], holdout_scores[name], contract)
        for name in holdout_names
    }

    def training_items():
        for name in training_names:
            cache = EventCache.load(cache_dir / name, allow_labels=True)
            branch = event_branches(cache, _load_score(oof_dir, name), contract)
            yield name, cache, branch[1], branch[2], branch[3]

    progress("训练79序列 direct-only 时序门控器")
    development_model, development_model_report = _train_model(
        training_items(),
        output_dir / "development" / "direct_verifier.txt",
        direct_threshold=float(contract["direct_threshold"]),
        seed=int(source["protocol"]["seed"]),
    )

    baseline = {
        name: holdout_branches[name][2] | holdout_branches[name][3]
        for name in holdout_names
    }
    candidates: list[dict[str, Any]] = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        prediction: dict[str, np.ndarray] = {}
        deleted = 0
        for name in holdout_names:
            _, direct_probability, tracked, direct = holdout_branches[name]
            cache = holdout_caches[name]
            group, _ = density_group(cache.locations)
            batch = direct_candidate_features(
                cache,
                direct_probability,
                tracked,
                direct,
                include_targets=False,
            )
            prediction[name] = apply_direct_verifier(
                batch,
                tracked,
                direct,
                development_model,
                float(threshold),
                density_group=group,
            )
            deleted += int(np.sum(baseline[name] & ~prediction[name]))
        baseline_metrics = _aggregate(holdout_caches, baseline, holdout_names)
        metrics = _aggregate(holdout_caches, prediction, holdout_names)
        half_gains = []
        for half in (holdout_names[::2], holdout_names[1::2]):
            before_half = _aggregate(holdout_caches, baseline, half)
            after_half = _aggregate(holdout_caches, prediction, half)
            half_gains.append(float(after_half["score"]) - float(before_half["score"]))
        gain = float(metrics["score"]) - float(baseline_metrics["score"])
        passed = bool(
            gain >= 0.001
            and int(metrics["detected_objects"]) >= int(baseline_metrics["detected_objects"])
            and all(value >= 0 for value in half_gains)
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "baseline": baseline_metrics,
                "candidate": metrics,
                "gain": gain,
                "half_gains": half_gains,
                "deleted_events": deleted,
                "passed": passed,
            }
        )
    winner = max(candidates, key=lambda row: float(row["candidate"]["score"]))

    progress("使用全部99个样本外序列训练发布门控器")

    def release_items():
        yield from training_items()
        for name in holdout_names:
            branch = holdout_branches[name]
            yield name, holdout_caches[name], branch[1], branch[2], branch[3]

    _, release_model_report = _train_model(
        release_items(),
        output_dir / "direct_verifier.txt",
        direct_threshold=float(contract["direct_threshold"]),
        seed=int(source["protocol"]["seed"]),
    )
    report = {
        "status": "released" if winner["passed"] else "rejected_development_gate",
        "source_training_report": str(training_report_path.resolve()),
        "source_training_report_sha256": sha256_file(training_report_path),
        "training_files": len(names),
        "development_training_files": len(training_names),
        "development_holdout_files": len(holdout_names),
        "event_contract": contract,
        "development_model": development_model_report,
        "development_gate": winner,
        "threshold_search": candidates,
        "threshold": float(winner["threshold"]),
        "release_model": release_model_report,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_direct_verifier(directory: str | Path) -> tuple[Any, dict[str, Any]]:
    directory = Path(directory)
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    model_path = directory / "direct_verifier.txt"
    if sha256_file(model_path) != report["release_model"]["sha256"]:
        raise ValueError("direct verifier model hash mismatch")
    model = require_lightgbm().Booster(model_file=str(model_path))
    return model, report
