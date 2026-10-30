"""后处理预测写出、独立评估和提交审计。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import sha256_file
from .metrics import BinaryCounts, score_binary


def load_prediction(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as saved:
            return saved["prediction"].astype(bool)
    return np.loadtxt(source, usecols=(4,), dtype=np.uint8).astype(bool)


def write_prediction_npz(path: str | Path, prediction: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, prediction=np.asarray(prediction, dtype=np.uint8))
    return output


def _source_arrays(
    path: Path, *, include_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if "ev" in saved:
            source = saved["ev"]
            locations = np.column_stack((source["x"], source["y"], source["t"]))
            raw = np.column_stack((source["x"], source["y"], source["t"], source["p"]))
            return (
                locations.astype(np.int64),
                source["label"].astype(bool) if include_labels else None,
                source["name"].astype(np.int64) if include_labels else None,
                raw,
            )
        locations = saved["ev_loc"].astype(np.int64)
        values = saved["evs_norm"]
        raw = np.column_stack((locations[:, 0], locations[:, 1], locations[:, 2], values[:, 3]))
        # 盲推导出不读取标签列；独立评分仍要求真实标注。
        return (locations, values[:, 4].astype(bool) if include_labels else None,
                values[:, 5].astype(np.int64) if include_labels else None, raw)


def _prediction_path(directory: Path, stem: str) -> Path:
    for suffix in (".npz", ".txt"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing prediction for {stem}")


def evaluate_predictions(
    prediction_dir: str | Path,
    dataset_root: str | Path,
    *,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """直接从原始数据独立复评，不读取训练缓存中的标签。"""

    prediction_dir = Path(prediction_dir)
    dataset_root = Path(dataset_root)
    source_paths = sorted(dataset_root.glob("*.npz"))
    if names is not None:
        selected = {Path(name).stem for name in names}
        source_paths = [path for path in source_paths if path.stem in selected]
    if not source_paths:
        raise FileNotFoundError(f"no source NPZ files found in {dataset_root}")
    aggregate = BinaryCounts()
    files: dict[str, Any] = {}
    for source_path in source_paths:
        locations, labels, target_ids, _ = _source_arrays(source_path)
        prediction_path = _prediction_path(prediction_dir, source_path.stem)
        prediction = load_prediction(prediction_path)
        if prediction.shape != labels.shape:
            raise ValueError(f"{source_path.name}: prediction row count mismatch")
        counts = score_binary(locations, labels, target_ids, prediction)
        aggregate += counts
        files[source_path.name] = counts.metrics()
    return {"aggregate": aggregate.metrics(), "files": files}


def write_submission_txt(
    source_path: str | Path, prediction: np.ndarray, output_path: str | Path
) -> dict[str, int]:
    """逐行保留源x y t p，时间用17位有效数字写出。"""

    source_path = Path(source_path)
    _, _, _, raw = _source_arrays(source_path, include_labels=False)
    prediction = np.asarray(prediction, dtype=bool)
    if len(raw) != len(prediction):
        raise ValueError(f"{source_path.name}: prediction row count mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as output:
        for (x, y, timestamp, polarity), label in zip(raw, prediction):
            output.write(
                f"{int(x)} {int(y)} {float(timestamp):.17g} {int(polarity)} {int(label)}\n"
            )
    return {"rows": int(len(prediction)), "positive": int(prediction.sum())}


def audit_submission(
    prediction_dir: str | Path,
    dataset_root: str | Path,
    zip_path: str | Path,
    *,
    expected_files: int = 24,
    expected_rows: int = 1_424_330,
) -> dict[str, Any]:
    """生成平铺ZIP，并逐值核对TXT的源事件顺序。"""

    prediction_dir = Path(prediction_dir)
    dataset_root = Path(dataset_root)
    zip_path = Path(zip_path)
    txt_root = zip_path.parent / "submission_txt"
    txt_root.mkdir(parents=True, exist_ok=False)
    sources = sorted(dataset_root.glob("*.npz"))
    if len(sources) != expected_files:
        raise ValueError(f"expected {expected_files} source files, got {len(sources)}")
    files: dict[str, Any] = {}
    total_rows = 0
    for source in sources:
        prediction = load_prediction(_prediction_path(prediction_dir, source.stem))
        txt_path = txt_root / f"{source.stem}.txt"
        record = write_submission_txt(source, prediction, txt_path)
        locations, _, _, raw = _source_arrays(source, include_labels=False)
        written = np.loadtxt(txt_path)
        if written.shape != (len(raw), 5):
            raise ValueError(f"{txt_path.name}: malformed submission rows")
        if not np.array_equal(written[:, :2].astype(np.int64), locations[:, :2]):
            raise ValueError(f"{txt_path.name}: x/y order mismatch")
        if not np.array_equal(written[:, 2], raw[:, 2].astype(np.float64)):
            raise ValueError(f"{txt_path.name}: timestamp mismatch")
        if not np.array_equal(written[:, 3].astype(np.int64), raw[:, 3].astype(np.int64)):
            raise ValueError(f"{txt_path.name}: polarity mismatch")
        if not np.all(np.isin(written[:, 4], (0.0, 1.0))):
            raise ValueError(f"{txt_path.name}: labels must be 0 or 1")
        record["sha256"] = sha256_file(txt_path)
        files[txt_path.name] = record
        total_rows += record["rows"]
    if total_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {total_rows}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for txt_path in sorted(txt_root.glob("*.txt")):
            archive.write(txt_path, arcname=txt_path.name)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
    if len(members) != expected_files or any("/" in name or "\\" in name for name in members):
        raise ValueError("submission ZIP must contain flat root-level TXT files")
    report = {
        "files": files,
        "file_count": len(files),
        "row_count": total_rows,
        "zip": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path),
    }
    (zip_path.parent / "submission_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
