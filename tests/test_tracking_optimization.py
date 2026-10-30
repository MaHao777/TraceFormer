"""以完整传感器画布核验局部形态学的边界、节点顺序与加权中心。"""

import numpy as np
import pytest
from scipy import ndimage

from traceformer.postprocessing.tracking import TrackNode, _frame_nodes


@pytest.mark.parametrize("radius", [0, 1, 2])
@pytest.mark.parametrize("max_span", [2, 20])
def test_roi_nodes_equal_full_canvas_at_sensor_edges(monkeypatch, radius, max_span):
    locations = np.array([[0, 0, 0], [1, 0, 0], [9, 2, 0], [12, 2, 0],
                          [345, 259, 50], [344, 259, 50], [0, 259, 100],
                          [345, 0, 100], [100, 100, 150]])
    probability = np.linspace(0.1, 0.9, len(locations), dtype=np.float32)
    seeds = np.ones(len(locations), dtype=bool)
    seeds[-1] = False
    frames = {i: np.flatnonzero(locations[:, 2] == i * 50) for i in range(4)}
    expected = {}
    # 参考始终在全传感器画布操作，不依赖局部原点换算。
    for frame, all_indices in frames.items():
        indices = all_indices[seeds[all_indices]]
        if not len(indices):
            continue
        points = locations[indices, :2]
        canvas = np.zeros((260, 346), dtype=bool)
        canvas[points[:, 1], points[:, 0]] = True
        if radius:
            canvas = ndimage.binary_dilation(canvas, np.ones((2 * radius + 1,) * 2))
        labels, _ = ndimage.label(canvas, np.ones((3, 3)))
        ids = labels[points[:, 1], points[:, 0]]
        expected[frame] = []
        for group in np.unique(ids):
            chosen = indices[ids == group]
            if np.any(np.ptp(locations[chosen, :2], axis=0) + 1 > max_span):
                continue
            center = np.average(locations[chosen, :2], axis=0,
                                weights=np.maximum(probability[chosen], 1e-4))
            expected[frame].append(TrackNode(frame, float(center[0]), float(center[1]),
                                              float(probability[chosen].max())))
    seen = []
    original = ndimage.label

    def record_shape(mask, *args, **kwargs):
        seen.append(mask.shape)
        return original(mask, *args, **kwargs)

    monkeypatch.setattr(ndimage, "label", record_shape)
    assert _frame_nodes(locations, probability, seeds, frames, radius, max_span) == expected
    assert seen[0][0] * seen[0][1] < 260 * 346
    assert seen[1][0] * seen[1][1] < 260 * 346
