"""模型无关的50 ms事件轨迹生成与回填。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackNode:
    frame: int
    x: float
    y: float
    confidence: float


def track_features(track: list[TrackNode]) -> np.ndarray:
    """把一条轨迹压缩为不依赖神经网络结构的几何特征。"""

    if not track:
        raise ValueError("track must not be empty")
    frames = np.asarray([node.frame for node in track], dtype=np.float32)
    x = np.asarray([node.x for node in track], dtype=np.float32)
    y = np.asarray([node.y for node in track], dtype=np.float32)
    confidence = np.asarray([node.confidence for node in track], dtype=np.float32)
    gaps = np.diff(frames)
    dx = np.diff(x)
    dy = np.diff(y)
    steps = np.hypot(dx, dy)
    duration = max(float(frames[-1] - frames[0] + 1), 1.0)
    if len(track) >= 3:
        fitted_x = np.polyval(np.polyfit(frames, x, 1), frames)
        fitted_y = np.polyval(np.polyfit(frames, y, 1), frames)
        residual = np.hypot(x - fitted_x, y - fitted_y)
        velocity = np.column_stack((dx / np.maximum(gaps, 1), dy / np.maximum(gaps, 1)))
        velocity_change = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
    else:
        residual = np.zeros(len(track), dtype=np.float32)
        velocity_change = np.zeros(1, dtype=np.float32)
    return np.asarray(
        [
            len(track),
            duration,
            len(track) / duration,
            confidence.min(),
            confidence.mean(),
            confidence.max(),
            confidence.std(),
            *np.quantile(confidence, (0.25, 0.5, 0.75)).tolist(),
            confidence.sum(),
            gaps.mean() if len(gaps) else 0.0,
            gaps.max() if len(gaps) else 0.0,
            steps.mean() if len(steps) else 0.0,
            steps.std() if len(steps) else 0.0,
            steps.max() if len(steps) else 0.0,
            np.hypot(x[-1] - x[0], y[-1] - y[0]),
            np.ptp(x),
            np.ptp(y),
            residual.mean(),
            residual.max(),
            velocity_change.mean(),
            velocity_change.max(),
            x.mean() / 346.0,
            y.mean() / 260.0,
        ],
        dtype=np.float32,
    )


def render_tracks(
    locations: np.ndarray,
    probability: np.ndarray,
    tracks: list[list[TrackNode]],
    selected: np.ndarray,
    *,
    low_threshold: float,
    fallback_threshold: float,
    half_width: int = 5,
    half_height: int = 4,
    extension_frames: int = 9,
    box_power: float = 2.75,
) -> np.ndarray:
    """只在被选轨迹的椭圆邻域内回填事件。"""

    locations = np.asarray(locations, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(tracks),):
        raise ValueError("selected must align with tracks")
    frames = (locations[:, 2] // 50).astype(np.int32)
    order = np.argsort(frames, kind="stable")
    split_points = np.flatnonzero(np.diff(frames[order])) + 1
    frame_indices = {
        int(frames[group[0]]): group for group in np.split(order, split_points) if len(group)
    }
    centers: dict[int, list[tuple[float, float]]] = {}
    for track, keep in zip(tracks, selected):
        if not keep:
            continue
        track_frames = np.asarray([node.frame for node in track], dtype=np.int32)
        x = np.asarray([node.x for node in track], dtype=np.float32)
        y = np.asarray([node.y for node in track], dtype=np.float32)
        first = max(int(track_frames.min()) - extension_frames, int(frames.min()))
        last = min(int(track_frames.max()) + extension_frames, int(frames.max()))
        full_frames = np.arange(first, last + 1)
        full_x = np.interp(full_frames, track_frames, x)
        full_y = np.interp(full_frames, track_frames, y)
        for frame, center_x, center_y in zip(full_frames, full_x, full_y):
            centers.setdefault(int(frame), []).append((float(center_x), float(center_y)))

    prediction = np.zeros(len(locations), dtype=bool)
    for frame, frame_centers in centers.items():
        indices = frame_indices.get(frame)
        if indices is None:
            continue
        for center_x, center_y in frame_centers:
            delta_x = np.abs(locations[indices, 0] - center_x)
            delta_y = np.abs(locations[indices, 1] - center_y)
            inside = (delta_x / max(half_width, 1)) ** box_power + (
                delta_y / max(half_height, 1)
            ) ** box_power <= 1.0
            chosen = inside & (probability[indices] >= low_threshold)
            prediction[indices[chosen]] = True
            if not np.any(chosen):
                fallback = np.flatnonzero(inside & (probability[indices] >= fallback_threshold))
                if len(fallback):
                    best = fallback[np.argmax(probability[indices[fallback]])]
                    prediction[indices[best]] = True
    return prediction


def _hot_pixels(
    locations: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    minimum_frames: int,
) -> np.ndarray:
    frames = (locations[:, 2] // 50).astype(np.int32)
    pixels = locations[:, 1] * 346 + locations[:, 0]
    chosen = probability >= threshold
    pairs = np.unique(np.column_stack((pixels[chosen], frames[chosen])), axis=0)
    if not len(pairs):
        return np.empty(0, dtype=np.int64)
    unique_pixels, counts = np.unique(pairs[:, 0], return_counts=True)
    return unique_pixels[counts >= minimum_frames]


def _frame_nodes(
    locations: np.ndarray,
    probability: np.ndarray,
    seed_mask: np.ndarray,
    frame_indices: dict[int, np.ndarray],
    cluster_radius: int,
    max_span: int,
) -> dict[int, list[TrackNode]]:
    nodes: dict[int, list[TrackNode]] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for frame, all_indices in frame_indices.items():
        indices = all_indices[seed_mask[all_indices]]
        if not len(indices):
            continue
        points = locations[indices, :2]
        mask = np.zeros((260, 346), dtype=bool)
        mask[points[:, 1], points[:, 0]] = True
        if cluster_radius:
            mask = ndimage.binary_dilation(
                mask,
                structure=np.ones((2 * cluster_radius + 1, 2 * cluster_radius + 1), dtype=bool),
            )
        component_map, _ = ndimage.label(mask, structure=structure)
        component_ids = component_map[points[:, 1], points[:, 0]]
        frame_nodes: list[TrackNode] = []
        for component_id in np.unique(component_ids):
            selected = indices[component_ids == component_id]
            selected_points = locations[selected, :2]
            width = np.ptp(selected_points[:, 0]) + 1
            height = np.ptp(selected_points[:, 1]) + 1
            if width > max_span or height > max_span:
                continue
            weights = np.maximum(probability[selected], 1e-4)
            center = np.average(selected_points, axis=0, weights=weights)
            frame_nodes.append(
                TrackNode(
                    frame=int(frame),
                    x=float(center[0]),
                    y=float(center[1]),
                    confidence=float(probability[selected].max()),
                )
            )
        nodes[int(frame)] = frame_nodes
    return nodes


def _link_tracks(
    nodes: dict[int, list[TrackNode]], max_gap: int, max_distance: float
) -> list[list[TrackNode]]:
    tracks: list[list[TrackNode]] = []
    active: list[list[TrackNode]] = []
    for frame in sorted(nodes):
        current = nodes[frame]
        active = [track for track in active if frame - track[-1].frame <= max_gap + 1]
        matched_nodes: set[int] = set()
        if active and current:
            costs = np.full((len(active), len(current)), 1e6, dtype=np.float32)
            for row, track in enumerate(active):
                gap = frame - track[-1].frame
                velocity_x = velocity_y = 0.0
                if len(track) >= 2:
                    previous_gap = max(track[-1].frame - track[-2].frame, 1)
                    velocity_x = (track[-1].x - track[-2].x) / previous_gap
                    velocity_y = (track[-1].y - track[-2].y) / previous_gap
                predicted_x = track[-1].x + velocity_x * gap
                predicted_y = track[-1].y + velocity_y * gap
                for column, node in enumerate(current):
                    distance = np.hypot(node.x - predicted_x, node.y - predicted_y)
                    if distance <= max_distance * gap:
                        costs[row, column] = distance
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] >= 1e6:
                    continue
                active[row].append(current[column])
                matched_nodes.add(int(column))
        for column, node in enumerate(current):
            if column not in matched_nodes:
                track = [node]
                tracks.append(track)
                active.append(track)
    return tracks


def candidate_tracks(
    locations: np.ndarray,
    probability: np.ndarray,
    *,
    seed_threshold: float,
    hot_threshold: float | None = None,
    hot_minimum_frames: int = 12,
    cluster_radius: int = 2,
    max_component_span: int = 24,
    max_gap: int = 7,
    max_distance: float = 8.0,
) -> tuple[list[list[TrackNode]], dict[int, np.ndarray], np.ndarray]:
    locations = np.asarray(locations, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    frames = (locations[:, 2] // 50).astype(np.int32)
    order = np.argsort(frames, kind="stable")
    split_points = np.flatnonzero(np.diff(frames[order])) + 1
    frame_indices = {
        int(frames[group[0]]): group for group in np.split(order, split_points) if len(group)
    }
    pixels = locations[:, 1] * 346 + locations[:, 0]
    hot = _hot_pixels(
        locations,
        probability,
        seed_threshold if hot_threshold is None else hot_threshold,
        hot_minimum_frames,
    )
    allowed = ~np.isin(pixels, hot)
    nodes = _frame_nodes(
        locations,
        probability,
        (probability >= seed_threshold) & allowed,
        frame_indices,
        cluster_radius,
        max_component_span,
    )
    return _link_tracks(nodes, max_gap, max_distance), frame_indices, allowed


def track_prediction(
    locations: np.ndarray,
    probability: np.ndarray,
    *,
    seed_threshold: float = 0.75,
    low_threshold: float = 0.03,
    hot_minimum_frames: int = 12,
    cluster_radius: int = 2,
    max_component_span: int = 24,
    max_gap: int = 7,
    max_distance: float = 8.0,
    minimum_track_frames: int = 8,
    minimum_motion: float = 0.0,
    half_width: int = 5,
    half_height: int = 4,
    extension_frames: int = 9,
    linear_extension: bool = True,
    box_power: float = 2.75,
    minimum_events_per_center: int = 2,
    fallback_threshold: float = 0.023,
    hot_threshold: float | None = None,
) -> np.ndarray:
    """从高置信节点连轨，并在轨迹附近回填低置信事件。"""

    tracks, frame_indices, allowed = candidate_tracks(
        locations,
        probability,
        seed_threshold=seed_threshold,
        hot_threshold=hot_threshold,
        hot_minimum_frames=hot_minimum_frames,
        cluster_radius=cluster_radius,
        max_component_span=max_component_span,
        max_gap=max_gap,
        max_distance=max_distance,
    )
    locations = np.asarray(locations, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    frames = (locations[:, 2] // 50).astype(np.int32)
    low_mask = (probability >= low_threshold) & allowed
    centers_by_frame: dict[int, list[tuple[float, float]]] = {}
    for track in tracks:
        if len(track) < minimum_track_frames:
            continue
        xs = np.asarray([node.x for node in track])
        ys = np.asarray([node.y for node in track])
        if float(np.hypot(np.ptp(xs), np.ptp(ys))) < minimum_motion:
            continue
        track_frames = np.asarray([node.frame for node in track], dtype=np.int32)
        first = max(int(track_frames.min()) - extension_frames, int(frames.min()))
        last = min(int(track_frames.max()) + extension_frames, int(frames.max()))
        full_frames = np.arange(first, last + 1)
        full_x = np.interp(full_frames, track_frames, xs)
        full_y = np.interp(full_frames, track_frames, ys)
        if linear_extension and len(track) >= 2:
            first_gap = max(track_frames[1] - track_frames[0], 1)
            last_gap = max(track_frames[-1] - track_frames[-2], 1)
            first_velocity = np.array((xs[1] - xs[0], ys[1] - ys[0])) / first_gap
            last_velocity = np.array((xs[-1] - xs[-2], ys[-1] - ys[-2])) / last_gap
            before = full_frames < track_frames[0]
            after = full_frames > track_frames[-1]
            full_x[before] = xs[0] + first_velocity[0] * (full_frames[before] - track_frames[0])
            full_y[before] = ys[0] + first_velocity[1] * (full_frames[before] - track_frames[0])
            full_x[after] = xs[-1] + last_velocity[0] * (full_frames[after] - track_frames[-1])
            full_y[after] = ys[-1] + last_velocity[1] * (full_frames[after] - track_frames[-1])
        for frame, center_x, center_y in zip(full_frames, full_x, full_y):
            centers_by_frame.setdefault(int(frame), []).append((center_x, center_y))

    prediction = np.zeros(len(locations), dtype=bool)
    for frame, centers in centers_by_frame.items():
        indices = frame_indices.get(frame)
        if indices is None:
            continue
        selected = np.zeros(len(indices), dtype=bool)
        for center_x, center_y in centers:
            delta_x = np.abs(locations[indices, 0] - center_x)
            delta_y = np.abs(locations[indices, 1] - center_y)
            inside = (
                (delta_x / max(half_width, 1)) ** box_power
                + (delta_y / max(half_height, 1)) ** box_power
            ) <= 1.0
            chosen = inside & low_mask[indices]
            selected |= chosen
            missing = minimum_events_per_center - int(np.sum(chosen))
            if missing > 0:
                fallback = (
                    inside
                    & allowed[indices]
                    & (probability[indices] >= fallback_threshold)
                    & ~selected
                )
                fallback_rows = np.flatnonzero(fallback)
                if len(fallback_rows):
                    order = np.argsort(probability[indices[fallback_rows]], kind="stable")[::-1]
                    selected[fallback_rows[order[:missing]]] = True
        prediction[indices[selected]] = True
    return prediction
