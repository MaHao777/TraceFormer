"""从逐事件掩码计算EV-UAV基准指标，保留官方时间边界和uint8计数。"""

import numpy as np
from scipy.ndimage import label


def event_columns(raw):
    if raw.dtype.names:
        return [raw[key] for key in ("x", "y", "t", "label", "name")]
    if raw.ndim == 2 and raw.shape[1] >= 6:
        return [raw[:, i] for i in (0, 1, 2, 4, 5)]
    raise ValueError("EV-UAV Pd/Fa require original x,y,t,label,object-name fields")


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def from_counts(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return dict(iou=ratio(tp, tp + fp + fn), seg_acc=ratio(tp, tp + fn),
                precision=ratio(tp, tp + fp),
                pd=ratio(counts["detected_objects"], counts["objects"]),
                fa=ratio(counts["false_components"], counts["frames"] * 346 * 260))


def measure(raw, prediction):
    x, y, time, truth, object_id = event_columns(raw)
    truth, prediction = truth.astype(bool), np.asarray(prediction, dtype=bool)
    if prediction.shape != truth.shape or not len(truth):
        raise ValueError("Require a nonempty full-event prediction vector")
    # 官方评估使用ev_loc里的整数毫秒，开区间排除50ms边界上的事件。
    time = time.astype(np.int64)
    if np.any(np.diff(time) < 0) or np.any(x < 0) or np.any(x >= 346):
        raise ValueError("Unsorted timestamps or invalid EV-UAV X coordinates")
    if np.any(y < 0) or np.any(y >= 260):
        raise ValueError("Invalid EV-UAV Y coordinates")
    if np.any((object_id > 0) != truth):
        raise ValueError("Object identities must agree with binary foreground labels")
    counts = dict(tp=int((prediction & truth).sum()), fp=int((prediction & ~truth).sum()),
                  fn=int((~prediction & truth).sum()), tn=int((~prediction & ~truth).sum()),
                  detected_objects=0, objects=0, false_components=0,
                  frames=int((time.max() - time.min()) / 50))
    for frame in range(counts["frames"] + 1):
        start = np.searchsorted(time, frame * 50, side="right")
        end = np.searchsorted(time, (frame + 1) * 50, side="left")
        ids, selected = object_id[start:end], prediction[start:end]
        for identity in np.unique(ids):
            if identity == 0:
                continue
            target = ids == identity
            counts["objects"] += 1
            counts["detected_objects"] += int(selected[target].mean() >= 0.0001)
        false = selected & ~truth[start:end]
        if np.any(false):
            positions = y[start:end][false].astype(np.int64) * 346 + x[start:end][false]
            # 兼容官方uint8像素计数的256回绕，再统计一次8连通域。
            image = np.bincount(positions, minlength=346 * 260).astype(np.uint8).reshape(260, 346)
            counts["false_components"] += int(label(image != 0, np.ones((3, 3)))[1])
    return dict(counts=counts, **from_counts(counts))
