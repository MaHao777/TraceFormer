"""Streaming event-level classification metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


class EventMetricAccumulator:
    """Accumulate standard binary metrics without retaining all predictions."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        self.threshold = float(threshold)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 1 or labels.ndim != 1 or len(logits) != len(labels):
            raise ValueError("logits and labels must be aligned one-dimensional tensors")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("logits must be finite")
        prediction = torch.sigmoid(logits.detach()) >= self.threshold
        target = labels.detach() > 0.5
        self.tp += int(torch.sum(prediction & target).item())
        self.fp += int(torch.sum(prediction & ~target).item())
        self.fn += int(torch.sum(~prediction & target).item())
        self.tn += int(torch.sum(~prediction & ~target).item())

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def compute(self) -> dict[str, float | int]:
        total = self.tp + self.fp + self.fn + self.tn
        return {
            "pd": self._ratio(self.tp, self.tp + self.fn),
            "fa": self._ratio(self.fp, self.fp + self.tn),
            "iou": self._ratio(self.tp, self.tp + self.fp + self.fn),
            "acc": self._ratio(self.tp + self.tn, total),
            "precision": self._ratio(self.tp, self.tp + self.fp),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "event_count": total,
        }


def _connected_component_count(points: set[tuple[int, int]]) -> int:
    """Count 8-connected components in a sparse pixel set."""
    count = 0
    while points:
        count += 1
        pending = [points.pop()]
        while pending:
            x, y = pending.pop()
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor = (x + delta_x, y + delta_y)
                    if neighbor in points:
                        points.remove(neighbor)
                        pending.append(neighbor)
    return count


class EVUAVMetricAccumulator:
    """Compatibility implementation of the reference EV-UAV Pd/Fa protocol."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_ms: float = 50.0,
        object_correct_fraction: float = 0.0001,
        threshold: float = 0.5,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame_ms = float(frame_ms)
        self.object_correct_fraction = float(object_correct_fraction)
        self.threshold = float(threshold)
        self.object_instances = 0
        self.detected_objects = 0
        self.false_components = 0
        self.frame_count = 0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        object_ids: torch.Tensor,
        timestamps_ms: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> None:
        if not (
            logits.ndim
            == labels.ndim
            == object_ids.ndim
            == timestamps_ms.ndim
            == 1
        ):
            raise ValueError("EV-UAV metric vectors must be one-dimensional")
        if coordinates.shape != (len(logits), 2):
            raise ValueError("coordinates must have shape [N,2]")
        prediction = torch.sigmoid(logits.detach().cpu()) >= self.threshold
        labels = labels.detach().cpu() > 0.5
        object_ids = object_ids.detach().cpu().long()
        timestamps_ms = torch.floor(timestamps_ms.detach().cpu().double())
        coordinates = coordinates.detach().cpu().long()
        if not len(logits):
            return
        duration = float(timestamps_ms[-1] - timestamps_ms[0])
        reference_frame_count = int(duration / self.frame_ms)
        self.frame_count += reference_frame_count

        for frame_index in range(reference_frame_count + 1):
            frame_start = frame_index * self.frame_ms
            frame_end = (frame_index + 1) * self.frame_ms
            frame_mask = (timestamps_ms > frame_start) & (timestamps_ms < frame_end)
            frame_objects = torch.unique(object_ids[frame_mask])
            for object_id in frame_objects.tolist():
                if object_id == 0:
                    continue
                object_mask = frame_mask & (object_ids == object_id) & labels
                positive_count = int(object_mask.sum().item())
                if positive_count == 0:
                    continue
                self.object_instances += 1
                detected_fraction = float(prediction[object_mask].float().mean().item())
                if detected_fraction >= self.object_correct_fraction:
                    self.detected_objects += 1

            false_mask = frame_mask & ~labels & prediction
            false_coordinates = coordinates[false_mask].tolist()
            point_counts: dict[tuple[int, int], int] = {}
            for x, y in false_coordinates:
                point = (int(x), int(y))
                point_counts[point] = (point_counts.get(point, 0) + 1) & 0xFF
            points = {point for point, count in point_counts.items() if count}
            self.false_components += _connected_component_count(points)

    def compute(self) -> dict[str, float | int]:
        pd = (
            self.detected_objects / self.object_instances
            if self.object_instances
            else 0.0
        )
        denominator = self.frame_count * self.width * self.height
        fa = self.false_components / denominator if denominator else 0.0
        return {
            "pd": float(pd),
            "fa": float(fa),
            "detected_objects": self.detected_objects,
            "object_instances": self.object_instances,
            "false_components": self.false_components,
            "frame_count": self.frame_count,
        }


def competition_metrics(
    event: Mapping[str, float | int],
    object_metrics: Mapping[str, float | int],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Return the exact JinSight competition metrics and weighted score."""
    intersection = int(event["tp"])
    union = intersection + int(event["fp"]) + int(event["fn"])
    gt_positive = intersection + int(event["fn"])
    detected_objects = int(object_metrics["detected_objects"])
    objects = int(object_metrics["object_instances"])
    false_alarms = int(object_metrics["false_components"])
    frames = int(object_metrics["frame_count"])

    iou = intersection / union if union else 0.0
    seg_acc = intersection / gt_positive if gt_positive else 0.0
    pd = detected_objects / objects if objects else 0.0
    fa_denominator = frames * int(width) * int(height)
    fa = false_alarms / fa_denominator if fa_denominator else 0.0
    score_fa = math.exp(-10000.0 * fa)
    score = 0.2 * iou + 0.1 * seg_acc + 0.4 * pd + 0.3 * score_fa
    return {
        "iou": float(iou),
        "seg_acc": float(seg_acc),
        "pd": float(pd),
        "fa": float(fa),
        "score_fa": float(score_fa),
        "score": float(score),
        "counts": {
            "intersection": intersection,
            "union": union,
            "true_positive": intersection,
            "gt_positive": gt_positive,
            "detected_objects": detected_objects,
            "objects": objects,
            "false_alarms": false_alarms,
            "frames": frames,
        },
    }


__all__ = [
    "EVUAVMetricAccumulator",
    "EventMetricAccumulator",
    "competition_metrics",
]
