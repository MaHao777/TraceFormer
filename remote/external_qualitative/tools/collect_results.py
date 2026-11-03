"""校验跨模型事件对齐，打包绘图所需的原始事件和逐事件预测。"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from event_io import THRESHOLD, sha256


def collect(root, archive_path):
    root, archive_path = root.resolve(), archive_path.resolve()
    reports = sorted(root.glob("*/report.json"))
    complete = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] == "complete":
            complete.append((path, report))
    if not complete:
        raise ValueError("No completed model predictions to collect")
    reference = {r["file"]: (r["sha256"], r["events"]) for r in complete[0][1]["sequences"]}
    selected = set()
    selected.update(root.glob("environment*.json"))
    for path, report in complete:
        current = {r["file"]: (r["sha256"], r["events"]) for r in report["sequences"]}
        if current != reference:
            raise ValueError("Models used different source files or sequence subsets")
        selected.add(path)
        selected.update(path.parent.rglob("*.log"))
        selected.update(path.parent.glob("native/config.py"))
        for name, (source_hash, count) in reference.items():
            prediction = path.parent / "predictions" / name
            original = root / "inputs" / name
            with np.load(original, allow_pickle=False) as inputs:
                if str(inputs["source_sha256"].item()) != source_hash or len(inputs["ev"]) != count:
                    raise ValueError(f"Source input mismatch: {original}")
                if inputs["gt"].shape != (count,) or not np.isin(inputs["gt"], [0, 1]).all():
                    raise ValueError(f"Invalid ground truth: {original}")
            with np.load(prediction, allow_pickle=False) as result:
                if str(result["source_sha256"].item()) != source_hash:
                    raise ValueError(f"Source hash mismatch: {prediction}")
                probability = result["probability"]
                if probability.shape != (count,) or not np.isfinite(probability).all():
                    raise ValueError(f"Invalid probabilities: {prediction}")
                if np.any((probability < 0) | (probability > 1)):
                    raise ValueError(f"Out-of-range probabilities: {prediction}")
                if not np.array_equal(result["event_index"], np.arange(count)):
                    raise ValueError(f"Event order mismatch: {prediction}")
                if float(result["threshold"].item()) != THRESHOLD:
                    raise ValueError(f"Unexpected threshold: {prediction}")
                if not np.array_equal(result["prediction"], probability >= THRESHOLD):
                    raise ValueError(f"Mask mismatch: {prediction}")
            selected.update([prediction, original])
    prefix = "external_qualitative_results/"
    sums = "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
                   for path in sorted(selected))
    summary = dict(models=[report["model"] for _, report in complete],
                   backends={report["model"]: report.get("backend", "unspecified")
                             for _, report in complete},
                   sequences=sorted(reference), threshold=THRESHOLD,
                   missing_models=sorted({"ev_spsegnet", "pact", "pointevent"}
                                         - {report["model"] for _, report in complete}))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path, prefix + path.relative_to(root).as_posix())
        archive.writestr(prefix + "SHA256SUMS", sums)
        archive.writestr(prefix + "summary.json", json.dumps(summary, indent=2) + "\n")
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Return this archive: {archive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    collect(args.results, args.archive)
