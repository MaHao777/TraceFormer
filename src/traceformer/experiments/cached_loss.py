"""缓存固定单序列的标签分组，保持原Focal/Dice/目标帧/虚警损失。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from traceformer.training.losses import (
    EventFocalDiceLoss,
    binary_dice_loss,
    binary_focal_loss,
)


class CachedSequenceLoss(nn.Module):
    """固定单序列损失；query_indices必须唯一并保留所有正事件。

    config可为完整TraceFormerConfig、LossConfig或EventFocalDiceLoss。
    单序列batch恒为0；全部固定张量在构造时复制，可用.to(device)迁移。
    """

    def __init__(self, labels, object_ids, timestamps_ms, coordinates, config, *, frame_ms=None):
        super().__init__()
        if isinstance(config, EventFocalDiceLoss):
            self.criterion = config
        else:
            loss = getattr(config, "loss", config)
            names = ("focal_alpha", "focal_gamma", "lambda_dice", "lambda_object_frame",
                     "object_frame_logit_margin", "lambda_false_alarm", "false_alarm_logit_margin",
                     "false_alarm_topk_per_frame")
            self.criterion = EventFocalDiceLoss(**{name: getattr(loss, name) for name in names})
        if frame_ms is None:
            frame_ms = getattr(getattr(config, "evaluation", None), "frame_ms", 50.0)
        self.frame_ms = float(frame_ms)
        if not math.isfinite(self.frame_ms) or self.frame_ms <= 0:
            raise ValueError("frame_ms must be finite and positive")
        labels = torch.as_tensor(labels).detach().clone()
        device = labels.device
        ids = torch.as_tensor(object_ids, device=device)
        timestamps = torch.as_tensor(timestamps_ms, device=device)
        coordinates = torch.as_tensor(coordinates, device=device)
        count = len(labels)
        if labels.shape != (count,) or ids.shape != (count,) or timestamps.shape != (count,):
            raise ValueError("fixed labels, IDs and timestamps must have shape [N]")
        if coordinates.shape != (count, 2):
            raise ValueError("coordinates must have shape [N,2]")
        self.register_buffer("labels", labels)
        self.register_buffer("positive_rows", torch.where(labels > 0.5)[0])
        integer_time = torch.floor(timestamps.to(dtype=torch.float64))
        inside = torch.remainder(integer_time, self.frame_ms) > 0
        frames = torch.div(integer_time, self.frame_ms, rounding_mode="floor").long()

        object_rows = torch.where((labels > 0.5) & (ids > 0) & inside)[0]
        object_groups = torch.stack((ids[object_rows].long(), frames[object_rows]), dim=1)
        unique_objects, object_inverse = torch.unique(object_groups, dim=0, return_inverse=True)
        object_group_by_event = torch.full((count,), -1, dtype=torch.long, device=device)
        object_group_by_event[object_rows] = object_inverse
        self.register_buffer("object_group_by_event", object_group_by_event)
        self.object_group_count = len(unique_objects)

        background_rows = torch.where((labels <= 0.5) & inside)[0]
        pixel_groups = torch.cat((
            torch.zeros((len(background_rows), 1), dtype=torch.long, device=device),
            frames[background_rows, None], coordinates[background_rows].long()), dim=1)
        unique_pixels, pixel_inverse = torch.unique(pixel_groups, dim=0, return_inverse=True)
        _, frame_inverse = torch.unique(unique_pixels[:, :2], dim=0, return_inverse=True)
        self.register_buffer("background_rows", background_rows)
        self.register_buffer("pixel_inverse", pixel_inverse)
        self.pixel_group_count = len(unique_pixels)
        # unique的字典序使同帧像素连续；split避免每步重新构造布尔分组。
        self.frame_pixel_counts = tuple(torch.bincount(frame_inverse).tolist())

    def forward(self, logits: torch.Tensor, query_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.shape != self.labels.shape or logits.device != self.labels.device:
            raise ValueError("logits must align with the cached sequence and device")
        query = torch.as_tensor(query_indices, dtype=torch.long, device=logits.device)
        if query.ndim != 1 or bool(((query < 0) | (query >= len(logits))).any()):
            raise ValueError("query_indices must contain valid event indices")
        # 正样本丢失会改变object-frame定义，因此显式拒绝不合约的采样。
        covered = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
        covered[query] = True
        if int(covered.sum()) != len(query):
            raise ValueError("query_indices must be unique")
        if not bool(covered[self.positive_rows].all()):
            raise ValueError("query_indices must preserve all positive events")

        criterion = self.criterion
        selected = logits[query]
        targets = self.labels[query]
        focal = binary_focal_loss(selected, targets, alpha=criterion.focal_alpha,
                                  gamma=criterion.focal_gamma)
        dice = binary_dice_loss(selected, targets)
        zero = selected.sum() * 0.0
        object_frame = zero
        if criterion.lambda_object_frame and self.object_group_count:
            query_groups = self.object_group_by_event[query]
            eligible = query_groups >= 0
            maxima = logits.new_full((self.object_group_count,), -torch.inf)
            # 沿用query分支和顺序，连total梯度的浮点相加顺序也保持一致。
            maxima.scatter_reduce_(0, query_groups[eligible], selected[eligible],
                                   reduce="amax", include_self=True)
            object_frame = F.softplus(criterion.object_frame_logit_margin - maxima).mean()
        false_alarm = logits.sum() * 0.0 if criterion.lambda_false_alarm else zero
        if criterion.lambda_false_alarm and self.pixel_group_count:
            maxima = logits.new_full((self.pixel_group_count,), -torch.inf)
            maxima.scatter_reduce_(0, self.pixel_inverse, logits[self.background_rows],
                                   reduce="amax", include_self=True)
            frame_losses = []
            for pixels in maxima.split(self.frame_pixel_counts):
                peaks = torch.topk(pixels, k=min(criterion.false_alarm_topk_per_frame, len(pixels)),
                                  largest=True, sorted=False).values
                frame_losses.append(F.softplus(peaks - criterion.false_alarm_logit_margin).mean())
            false_alarm = torch.stack(frame_losses).mean()
        total = (focal + criterion.lambda_dice * dice
                 + criterion.lambda_object_frame * object_frame
                 + criterion.lambda_false_alarm * false_alarm)
        return {"total": total, "focal": focal, "dice": dice,
                "object_frame": object_frame, "false_alarm": false_alarm}


__all__ = ["CachedSequenceLoss"]
