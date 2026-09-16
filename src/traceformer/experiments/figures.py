"""固定验证片段与论文消融图，不按模型结果挑选样例。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traceformer.config import TraceFormerConfig, load_config
from traceformer.training.engine import _dataset, _write_json


def select_windows(config: TraceFormerConfig, root: Path) -> None:
    windows = []
    for sequence in _dataset(config, "val"):
        if sequence.source_path.stem not in {"val_000", "val_015", "val_023"}:
            continue
        # 在任何预测生成前，只按 GT 选择目标事件最多的 50ms 帧，平局取最早。
        frames = torch.floor(sequence.timestamps_ms / config.evaluation.frame_ms).long()
        counts = torch.bincount(frames[sequence.labels > 0.5])
        selected = int(counts.argmax()) if len(counts) else 0
        windows.append({"sequence": sequence.source_path.stem, "frame": selected,
                        "frame_ms": config.evaluation.frame_ms})
    _write_json(root / "visualization_windows.json", windows)


def render_figures(root: Path, seed: int) -> None:
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    config = load_config(suite["jobs"][0]["config"])
    variants = ["A0", "A1", "A3", "A7"]
    paths = {}
    for job in suite["jobs"]:
        if job["seed"] != seed or job["result_id"] not in variants:
            continue
        done = root / job["id"] / "done.json"
        if not done.exists():
            return
        record = json.loads(done.read_text(encoding="utf-8"))
        paths[job["result_id"]] = Path(record["checkpoint"]).parent / "predictions"
    if len(paths) != len(variants):
        return
    windows = json.loads((root / "visualization_windows.json").read_text(encoding="utf-8"))
    dataset = _dataset(config, "val")
    selected = {p.stem: i for i, p in enumerate(dataset.files)}
    fig, axes = plt.subplots(len(windows), 5, figsize=(16, 3.2 * len(windows)), squeeze=False)
    for row, window in enumerate(windows):
        name = window["sequence"]
        sequence = dataset[selected[name]]
        mask = torch.floor(sequence.timestamps_ms / window["frame_ms"]).long() == window["frame"]
        xy = sequence.events[mask, :2].numpy()
        target = sequence.labels[mask].numpy() > 0.5
        for column, variant in enumerate(["GT", *variants]):
            ax = axes[row, column]
            ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#dddddd", rasterized=True)
            if variant == "GT":
                ax.scatter(xy[target, 0], xy[target, 1], s=5, c="#16814b")
            else:
                record = torch.load(paths[variant] / f"{name}.pt", weights_only=True)
                predicted = record["event_logits"][mask].sigmoid().numpy() >= record["threshold"]
                for chosen, color in ((target & predicted, "#16814b"),
                                      (~target & predicted, "#d74b36"), (target & ~predicted, "#2369bd")):
                    ax.scatter(xy[chosen, 0], xy[chosen, 1], s=5, c=color)
            ax.set_title(variant)
            ax.set(xlim=(0, config.resolution.width), ylim=(config.resolution.height, 0), aspect="equal")
            if column == 0:
                ax.set_ylabel(f"{name}\n{window['frame'] * window['frame_ms']:g} ms")
    fig.suptitle(f"Seed {seed}: green = TP/GT, red = FP, blue = FN; fixed GT-selected frames")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root / f"qualitative_seed{seed}.png", dpi=180)
    plt.close(fig)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))["aggregate"]
    ids = sorted(aggregate)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, ["score", "pd", "fa"], strict=True):
        mean = [aggregate[k][metric]["mean"] for k in ids]
        std = [aggregate[k][metric]["std"] or 0 for k in ids]
        ax.bar(np.arange(len(ids)), mean, yerr=std, capsize=3, color="#287bb5")
        ax.set_xticks(np.arange(len(ids)), ids)
        ax.set_title(metric)
        ax.set_ylabel("Mean across completed seeds")
    figure.tight_layout()
    figure.savefig(root / "ablation_metrics.png", dpi=180)
    plt.close(figure)
