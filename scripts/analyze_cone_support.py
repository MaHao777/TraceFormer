"""在全部验证序列上检查锥型支持与实际 K=8 选边；GT 仅用于诊断。"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from traceformer.config import load_config
from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder
from traceformer.training.engine import _dataset, _file_sha256, _write_json, build_model


def partition(events, config):
    """完全沿用 tokenizer 的整数键顺序，不依赖模型预测或目标标签。"""
    m = config.model
    nx = math.ceil(config.resolution.width / m.spatial_block_size_x)
    ny = math.ceil(config.resolution.height / m.spatial_block_size_y)
    x = events[:, 0].long() // m.spatial_block_size_x
    y = events[:, 1].long() // m.spatial_block_size_y
    t = (events[:, 2] * m.num_time_blocks).long().clamp(max=m.num_time_blocks - 1)
    keys, inverse = torch.unique((t * ny + y) * nx + x, sorted=True, return_inverse=True)
    coords = torch.stack((keys % nx, keys // nx % ny, keys // (nx * ny)), dim=1)
    return coords, inverse.numpy()


def target_index(inverse, labels, object_ids):
    pairs = np.unique(np.column_stack((inverse[labels], object_ids[labels])), axis=0)
    instances, tokens = defaultdict(list), defaultdict(set)
    for token, instance in pairs:
        instances[int(instance)].append(int(token))
        tokens[int(token)].add(int(instance))
    return {key: np.array(value) for key, value in instances.items()}, tokens


def describe(coords, selected, head, horizon, instance_tokens, token_instances, trees):
    """统计目标查询的许可候选、同实例候选、选中邻居及时间偏移。"""
    queries = np.array(sorted(token_instances), dtype=int)
    counts = np.zeros((horizon + 1, 4), dtype=np.int64)
    # 四列依次为可能同实例对、许可同实例对、实际同实例对、全部实际选边。
    for query in queries:
        pool = np.unique(np.concatenate([instance_tokens[i] for i in token_instances[query]]))
        dt = np.abs(coords[pool, 2] - coords[query, 2])
        within = (dt > 0) & (dt <= horizon)
        pool, dt = pool[within], dt[within]
        spatial = np.max(np.abs(coords[pool, :2] - coords[query, :2]), axis=1)
        allowed = spatial <= head.base_radius + head.slope * dt + 1e-9
        counts[:, 0] += np.bincount(dt, minlength=horizon + 1)
        counts[:, 1] += np.bincount(dt[allowed], minlength=horizon + 1)
        neighbors = selected[query]
        neighbors = neighbors[neighbors >= 0]
        edge_dt = np.abs(coords[neighbors, 2] - coords[query, 2])
        same = np.isin(neighbors, pool)
        counts[:, 2] += np.bincount(edge_dt[same], minlength=horizon + 1)
        counts[:, 3] += np.bincount(edge_dt, minlength=horizon + 1)
    occupied = np.zeros(horizon + 1, dtype=np.int64)
    for time in np.unique(coords[queries, 2]):
        group = queries[coords[queries, 2] == time]
        for distance in range(1, horizon + 1):
            radius = head.base_radius + head.slope * distance + 1e-9
            for direction in (-1, 1):
                tree = trees.get(int(time + direction * distance))
                if tree is not None:
                    occupied[distance] += tree.query_ball_point(
                        coords[group, :2], radius, p=np.inf, workers=1, return_length=True
                    ).sum()
    return [{"dt": distance, "possible_same": int(row[0]), "eligible_same": int(row[1]),
             "selected_same": int(row[2]), "selected_all": int(row[3]),
             "eligible_all": int(occupied[distance])}
            for distance, row in enumerate(counts) if distance]


def run(config_path: Path, output: Path) -> None:
    torch.set_num_threads(2)
    config = load_config(config_path)
    dataset = _dataset(config, "val")
    if len(dataset) != 24:
        raise ValueError("support analysis requires all 24 validation sequences")
    records, overlap, data_hashes = [], [], {}
    for file_index, sequence in enumerate(dataset):
        name = sequence.source_path.stem
        data_hashes[name] = _file_sha256(sequence.source_path)
        coordinate_tensor, inverse = partition(sequence.events, config)
        if file_index == 0:
            # 与真实 tokenizer 对齐检查，防止手写诊断发生坐标或顺序漂移。
            with torch.no_grad():
                actual = build_model(config).tokenizer(sequence.events)
            if not torch.equal(coordinate_tensor, actual.token_coords) or not np.array_equal(
                inverse, actual.event_token_indices.numpy()
            ):
                raise ValueError("diagnostic partition differs from actual tokenizer")
        coords = coordinate_tensor.numpy()
        instances, token_instances = target_index(
            inverse, sequence.labels.numpy() > 0, sequence.object_ids.numpy()
        )
        queries = np.array(sorted(token_instances), dtype=int)
        trees = {int(t): cKDTree(coords[coords[:, 2] == t, :2]) for t in np.unique(coords[:, 2])}
        batches = torch.zeros(len(coords), dtype=torch.long)
        for layer, scale in enumerate(config.model.layer_time_scales, start=1):
            window_heads = []
            for head in config.model.cone_heads:
                horizon = max(1, math.ceil(head.max_time_distance * scale))
                radius = math.sqrt(sum((head.base_radius + head.slope * d) ** 2
                                       for d in range(1, horizon + 1)) / horizon)
                window_heads.append(replace(head, base_radius=radius, slope=0.0))
            selected_sets = {}
            for mode, heads in (("cone", config.model.cone_heads), ("window", tuple(window_heads))):
                builder = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=8,
                                              layer_time_scale=scale)
                neighbors = builder(coordinate_tensor, batches)
                selected_sets[mode] = neighbors
                joined = torch.cat((neighbors.past_indices, neighbors.future_indices), dim=2).numpy()
                for head_index, head in enumerate(heads):
                    horizon = max(1, math.ceil(head.max_time_distance * scale))
                    rows = describe(coords, joined[:, head_index], head, horizon,
                                    instances, token_instances, trees)
                    records.append({"sequence": name, "layer": layer, "head": head_index + 1,
                                    "mode": mode, "queries": len(queries), "by_dt": rows})
            for direction in ("past_indices", "future_indices"):
                left = getattr(selected_sets["cone"], direction)[queries].numpy()
                right = getattr(selected_sets["window"], direction)[queries].numpy()
                common = ((left[..., :, None] == right[..., None, :]).any(-1) & (left >= 0)).sum(-1)
                union = (left >= 0).sum(-1) + (right >= 0).sum(-1) - common
                identical = common == union
                active = union > 0
                overlap.append({"sequence": name, "layer": layer, "direction": direction,
                                "common_edges": int(common.sum()), "union_edges": int(union.sum()),
                                "identical_nonempty_sets": int((identical & active).sum()),
                                "nonempty_sets": int(active.sum())})
        print(f"{name}: {len(coords)} tokens, {len(queries)} GT-containing queries", flush=True)
    _write_json(output, {"schema": 1, "records": records, "overlap": overlap,
                         "data_sha256": data_hashes, "config_sha256": _file_sha256(config_path),
                         "note": "Micro counts over query/layer/head/direction strata; GT-containing tokens may also contain background. Support capture is not detection performance."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
