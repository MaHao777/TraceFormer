"""后处理二值结果的官方全局计数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp

import numpy as np
from scipy import ndimage


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    detected_objects: int = 0
    objects: int = 0
    false_components: int = 0
    frames: int = 0

    def __iadd__(self, other: BinaryCounts) -> BinaryCounts:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def metrics(self, *, width: int = 346, height: int = 260) -> dict[str, float | int]:
        union = self.tp + self.fp + self.fn
        positives = self.tp + self.fn
        iou = self.tp / union if union else 0.0
        seg_acc = self.tp / positives if positives else 0.0
        pd = self.detected_objects / self.objects if self.objects else 0.0
        denominator = self.frames * width * height
        fa = self.false_components / denominator if denominator else 0.0
        score_fa = exp(-10000.0 * fa)
        score = 0.2 * iou + 0.1 * seg_acc + 0.4 * pd + 0.3 * score_fa
        return {
            **asdict(self),
            "iou": iou,
            "seg_acc": seg_acc,
            "pd": pd,
            "fa": fa,
            "score_fa": score_fa,
            "score": score,
        }


def _false_component_count(points: np.ndarray) -> int:
    if not len(points):
        return 0
    counts: dict[tuple[int, int], int] = {}
    for x, y in points[:, :2].astype(np.int64):
        key = (int(x), int(y))
        counts[key] = (counts.get(key, 0) + 1) & 0xFF
    mask = np.zeros((260, 346), dtype=bool)
    for (x, y), count in counts.items():
        if count:
            mask[y, x] = True
    _, components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(components)


def score_binary(
    locations: np.ndarray,
    labels: np.ndarray,
    target_ids: np.ndarray,
    prediction: np.ndarray,
    *,
    frame_ms: float = 50.0,
    object_correct_fraction: float = 0.0001,
) -> BinaryCounts:
    """保持与仓库EV-UAV兼容指标一致的严格帧边界。"""

    locations = np.asarray(locations, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=bool)
    if not (
        locations.shape == (len(labels), 3) and labels.shape == target_ids.shape == prediction.shape
    ):
        raise ValueError("binary metric arrays must align")
    times = locations[:, 2].astype(np.float64)
    frame_count = int((times[-1] - times[0]) / frame_ms)
    result = BinaryCounts(
        tp=int(np.sum(prediction & labels)),
        fp=int(np.sum(prediction & ~labels)),
        fn=int(np.sum(~prediction & labels)),
        tn=int(np.sum(~prediction & ~labels)),
        frames=frame_count,
    )
    for frame_index in range(frame_count + 1):
        start = frame_index * frame_ms
        stop = (frame_index + 1) * frame_ms
        frame_mask = (times > start) & (times < stop)
        for target_id in np.unique(target_ids[frame_mask]):
            if target_id == 0:
                continue
            target = frame_mask & labels & (target_ids == target_id)
            if not np.any(target):
                continue
            result.objects += 1
            if float(np.mean(prediction[target])) >= object_correct_fraction:
                result.detected_objects += 1
        false_points = locations[frame_mask & ~labels & prediction]
        result.false_components += _false_component_count(false_points)
    return result
