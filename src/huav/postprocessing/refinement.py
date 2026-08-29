"""组件剪枝和稳定轨迹中心补检。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from .contracts import EventCache, density_group, sha256_file
from .learning import require_lightgbm
from .tracking import candidate_tracks, render_tracks, track_features


@dataclass(frozen=True)
class ComponentBatch:
    features: np.ndarray
    members: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray | None
    critical: np.ndarray | None


def connected_components(
    locations: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
    """按50 ms帧提取8连通组件，并保留原事件索引。"""

    locations = np.asarray(locations, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=bool)
    frames = (locations[:, 2] // 50).astype(np.int32)
    grouped: dict[int, list[np.ndarray]] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for frame in np.unique(frames[prediction]):
        indices = np.flatnonzero(prediction & (frames == frame))
        mask = np.zeros((260, 346), dtype=bool)
        mask[locations[indices, 1], locations[indices, 0]] = True
        component_map, count = ndimage.label(mask, structure=structure)
        point_components = component_map[locations[indices, 1], locations[indices, 0]]
        grouped[int(frame)] = [
            indices[point_components == component] for component in range(1, count + 1)
        ]
    return frames, grouped


def component_features(
    cache: EventCache,
    probability: np.ndarray,
    prediction: np.ndarray,
    *,
    include_targets: bool,
) -> ComponentBatch:
    """提取形状、概率、邻帧拓扑和当前64维token聚合特征。"""

    frames, grouped = connected_components(cache.locations, prediction)
    maximum_frame = max(int(frames.max()), 1)
    centers: dict[int, np.ndarray] = {}
    sizes: dict[int, np.ndarray] = {}
    maximums: dict[int, np.ndarray] = {}
    coordinate_sets: dict[int, list[set[int]]] = {}
    for frame, components in grouped.items():
        centers[frame] = np.asarray(
            [cache.locations[index, :2].mean(axis=0) for index in components],
            dtype=np.float32,
        )
        sizes[frame] = np.asarray([len(index) for index in components], np.float32)
        maximums[frame] = np.asarray([probability[index].max() for index in components], np.float32)
        coordinate_sets[frame] = [
            set(
                (
                    cache.locations[index, 0].astype(np.int64) * 260 + cache.locations[index, 1]
                ).tolist()
            )
            for index in components
        ]

    rows: list[list[float]] = []
    member_parts: list[np.ndarray] = []
    offsets = [0]
    labels: list[bool] = []
    critical: list[int] = []
    event_tokens = cache.token_features[cache.event_token_indices]
    for frame, components in grouped.items():
        frame_count = int(np.sum(frames == frame))
        predicted_count = int(np.sum(prediction & (frames == frame)))
        for local_index, indices in enumerate(components):
            points = cache.locations[indices, :2]
            unique_points = np.unique(points, axis=0)
            width = int(np.ptp(points[:, 0]) + 1)
            height = int(np.ptp(points[:, 1]) + 1)
            values = probability[indices]
            token = event_tokens[indices]
            row = [
                np.log1p(len(indices)),
                np.log1p(len(unique_points)),
                width,
                height,
                len(unique_points) / max(width * height, 1),
                float(points[:, 0].mean()) / 346.0,
                float(points[:, 1].mean()) / 260.0,
                frame / maximum_frame,
                np.log1p(frame_count),
                np.log1p(predicted_count),
                len(indices) / max(predicted_count, 1),
                float(values.min()),
                float(values.mean()),
                float(values.max()),
                float(values.std()),
                *np.quantile(values, (0.25, 0.5, 0.75)).tolist(),
                *cache.raw_features[indices].mean(axis=0).tolist(),
                *cache.raw_features[indices].std(axis=0).tolist(),
                *token.mean(axis=0).tolist(),
                *token.std(axis=0).tolist(),
                *token.max(axis=0).tolist(),
            ]
            center = centers[frame][local_index]
            coordinates = coordinate_sets[frame][local_index]
            for gap in (1, 2, 3):
                for direction in (-1, 1):
                    other_frame = frame + direction * gap
                    if other_frame not in centers or not len(centers[other_frame]):
                        row.extend((50.0, 0.0, 0.0, 0.0))
                        continue
                    distance = np.linalg.norm(centers[other_frame] - center, axis=1)
                    nearest = int(np.argmin(distance))
                    row.extend(
                        (
                            min(float(distance[nearest]) / gap, 50.0),
                            np.log1p(
                                min(
                                    float(sizes[other_frame][nearest] / len(indices)),
                                    20.0,
                                )
                            ),
                            float(maximums[other_frame][nearest] - values.max()),
                            float(bool(coordinates & coordinate_sets[other_frame][nearest])),
                        )
                    )
            rows.append(row)
            member_parts.append(indices.astype(np.int32))
            offsets.append(offsets[-1] + len(indices))
            if include_targets:
                if cache.labels is None or cache.target_ids is None:
                    raise ValueError("component targets require training cache")
                labels.append(bool(np.any(cache.labels[indices])))
                lost = 0
                target_ids = np.unique(
                    cache.target_ids[indices][
                        cache.labels[indices] & (cache.target_ids[indices] > 0)
                    ]
                )
                for target_id in target_ids:
                    target = (frames == frame) & cache.labels & (cache.target_ids == target_id)
                    if int(prediction[target].sum()) == int(
                        np.sum(cache.labels[indices] & (cache.target_ids[indices] == target_id))
                    ):
                        lost += 1
                critical.append(lost)
    feature_dim = 18 + 8 + 64 * 3 + 24
    feature_array = np.asarray(rows, dtype=np.float32)
    if not len(feature_array):
        feature_array = np.empty((0, feature_dim), dtype=np.float32)
    return ComponentBatch(
        features=feature_array,
        members=(np.concatenate(member_parts) if member_parts else np.empty(0, dtype=np.int32)),
        offsets=np.asarray(offsets, dtype=np.int32),
        labels=np.asarray(labels, dtype=bool) if include_targets else None,
        critical=np.asarray(critical, dtype=np.int16) if include_targets else None,
    )


def apply_component_model(
    batch: ComponentBatch, prediction: np.ndarray, model: Any, threshold: float
) -> np.ndarray:
    """组件模型只做整组件保留或删除。"""

    result = np.asarray(prediction, dtype=bool).copy()
    if not len(batch.features):
        return result
    keep = model.predict(batch.features, num_threads=1) >= threshold
    for index, selected in enumerate(keep):
        if not selected:
            result[batch.members[batch.offsets[index] : batch.offsets[index + 1]]] = False
    return result


def train_component_model(
    items: Iterable[tuple[str, EventCache, np.ndarray, np.ndarray]],
    output_path: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """使用OOF事件结果训练组件保护模型。"""

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    sequences: list[str] = []
    for name, cache, probability, prediction in items:
        batch = component_features(cache, probability, prediction, include_targets=True)
        if not len(batch.features):
            continue
        assert batch.labels is not None and batch.critical is not None
        feature_parts.append(batch.features)
        label_parts.append(batch.labels.astype(np.uint8))
        # 唯一命中目标的组件加大权重，避免以降Fa为代价损伤Pd。
        weight_parts.append(1.0 + 8.0 * batch.critical.astype(np.float32))
        sequences.append(name)
    if not feature_parts:
        raise ValueError("no connected components available for training")
    lgb = require_lightgbm()
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    weights = np.concatenate(weight_parts)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
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
        num_boost_round=800,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "sequences": sorted(sequences),
        "components": int(len(labels)),
        "positive_components": int(labels.sum()),
        "feature_dim": int(features.shape[1]),
    }


def center_candidates(
    cache: EventCache, probability: np.ndarray, prediction: np.ndarray
) -> tuple[list[list[Any]], np.ndarray, np.ndarray]:
    """只从稳定轨迹节点生成尚未被当前结果覆盖的中心候选。"""

    tracks, _, _ = candidate_tracks(
        cache.locations,
        probability,
        seed_threshold=0.10,
        hot_threshold=0.75,
        hot_minimum_frames=12,
        cluster_radius=2,
        max_component_span=24,
        max_gap=7,
        max_distance=8.0,
    )
    stable_tracks = [track for track in tracks if len(track) >= 5]
    rows: list[np.ndarray] = []
    labels: list[bool] = []
    frames = (cache.locations[:, 2] // 50).astype(np.int32)
    for track in stable_tracks:
        vector = track_features(track)
        for node_index, node in enumerate(track):
            frame_indices = np.flatnonzero(frames == node.frame)
            distance = np.hypot(
                cache.locations[frame_indices, 0] - node.x,
                cache.locations[frame_indices, 1] - node.y,
            )
            near = frame_indices[distance <= 5.0]
            already_detected = bool(np.any(prediction[near]))
            rows.append(
                np.concatenate(
                    (
                        vector,
                        np.asarray(
                            [
                                node.confidence,
                                node_index / max(len(track) - 1, 1),
                                node.x / 346.0,
                                node.y / 260.0,
                                np.mean(probability[near]) if len(near) else 0.0,
                                np.max(probability[near]) if len(near) else 0.0,
                                np.log1p(len(near)),
                                float(already_detected),
                            ],
                            dtype=np.float32,
                        ),
                    )
                )
            )
            if cache.labels is not None:
                labels.append(bool(np.any(cache.labels[near])) and not already_detected)
    feature_array = np.asarray(rows, dtype=np.float32)
    if not len(feature_array):
        feature_array = np.empty((0, 33), dtype=np.float32)
    return stable_tracks, feature_array, np.asarray(labels, dtype=bool)


def render_center_candidates(
    cache: EventCache,
    probability: np.ndarray,
    prediction: np.ndarray,
    tracks: list[list[Any]],
    center_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """按中心分数选择轨迹，回填仍限制在轨迹邻域。"""

    result = np.asarray(prediction, dtype=bool).copy()
    offset = 0
    selected = np.zeros(len(tracks), dtype=bool)
    for index, track in enumerate(tracks):
        values = center_scores[offset : offset + len(track)]
        selected[index] = bool(np.any(values >= threshold))
        offset += len(track)
    added = render_tracks(
        cache.locations,
        probability,
        tracks,
        selected,
        low_threshold=0.03,
        fallback_threshold=0.01,
        half_width=5,
        half_height=4,
        extension_frames=2,
    )
    # 这里是受候选轨迹约束的补检，不是与原模型结果的盲目并集。
    result[added] = True
    return result


def train_center_model(
    items: Iterable[tuple[str, EventCache, np.ndarray, np.ndarray]],
    output_path: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    groups: dict[str, int] = {"sparse": 0, "ordinary": 0, "dense": 0}
    for _, cache, probability, prediction in items:
        _, features, labels = center_candidates(cache, probability, prediction)
        if len(features):
            feature_parts.append(features)
            label_parts.append(labels.astype(np.uint8))
            group, _ = density_group(cache.locations)
            groups[group] += len(features)
    if not feature_parts:
        raise ValueError("no stable center candidates available for training")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    lgb = require_lightgbm()
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 30,
            "lambda_l2": 2.0,
            "seed": seed,
            "verbosity": -1,
            "num_threads": 8,
            "deterministic": True,
            "force_col_wise": True,
        },
        lgb.Dataset(features, label=labels),
        num_boost_round=600,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    report = {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "candidates": int(len(labels)),
        "positive_candidates": int(labels.sum()),
        "groups": groups,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
