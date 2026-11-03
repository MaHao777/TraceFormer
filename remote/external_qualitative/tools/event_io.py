"""核对原始事件与官方输入，并导出可逐行对齐的预测。"""

import hashlib
import json
from pathlib import Path

import numpy as np

THRESHOLD = 0.9


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_input(path):
    with np.load(path, allow_pickle=False) as data:
        raw, native, locations = data["ev"], data["evs_norm"], data["ev_loc"]
    count = len(raw)
    if not count or native.shape != (count, 6) or locations.shape != (count, 3):
        raise ValueError(f"{path}: require aligned ev / evs_norm(N,6) / ev_loc(N,3)")
    if raw.dtype.names:
        columns = [raw[key] for key in ("x", "y", "t", "p", "label")]
    elif raw.ndim == 2 and raw.shape[1] >= 5:
        columns = [raw[:, i] for i in range(5)]
    else:
        raise ValueError(f"{path}: unsupported raw ev representation")
    x, y, t, polarity, labels = columns
    if not np.isfinite(native).all() or not np.isfinite(np.column_stack(columns)).all():
        raise ValueError(f"{path}: nonfinite events")
    expected = np.column_stack((x, y, t.astype(np.int64)))
    if not np.array_equal(locations, expected):
        raise ValueError(f"{path}: ev_loc differs from raw event order / integer milliseconds")
    if not np.array_equal(native[:, 4], labels) or not np.array_equal(native[:, 3], polarity):
        raise ValueError(f"{path}: normalized labels or polarities differ from raw event order")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: nonbinary labels")
    record = dict(file=Path(path).name, events=count, sha256=sha256(path),
                  targets=int(np.sum(labels)), time_unit="ms",
                  time_range=[float(t.min()), float(t.max())])
    return record, raw, labels.astype(np.uint8)


def save_input(folder, path, record, raw, gt):
    target = Path(folder) / "inputs" / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with np.load(target, allow_pickle=False) as previous:
            if str(previous["source_sha256"].item()) != record["sha256"]:
                raise ValueError(f"Different source already exported: {target}")
            if not np.array_equal(previous["ev"], raw) or not np.array_equal(previous["gt"], gt):
                raise ValueError(f"Existing input export is corrupt: {target}")
        return
    # 原始坐标和时间只保存一份；所有模型结果按同一行号索引。
    np.savez_compressed(target, ev=raw, gt=gt, source_sha256=np.array(record["sha256"]),
                        time_unit=np.array("ms"))


def save_prediction(folder, name, probability, record):
    probability = np.asarray(probability, dtype=np.float32).reshape(-1)
    if probability.shape != (record["events"],) or not np.isfinite(probability).all():
        raise ValueError(f"{name}: invalid full-event probability shape / values")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"{name}: outputs are not probabilities")
    target = Path(folder) / "predictions" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, probability=probability,
                        prediction=(probability >= THRESHOLD).astype(np.uint8),
                        event_index=np.arange(len(probability), dtype=np.int64),
                        threshold=np.array(THRESHOLD), source_sha256=np.array(record["sha256"]))


def selected_files(root, split, sequences):
    folder = Path(root).resolve() / split
    files = sorted(folder.glob("*.npz"))
    if sequences:
        wanted = {Path(name).stem for name in sequences}
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            raise FileNotFoundError(f"Missing requested sequences in {folder}: {sorted(missing)}")
    if not files:
        raise FileNotFoundError(f"No NPZ sequences found in {folder}")
    return files
