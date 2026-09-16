"""独立审计Track 2提交TXT及ZIP。"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from traceformer.postprocessing.contracts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="逐行审计Track 2提交格式")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--txt-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    sources = sorted(arguments.test_dir.glob("*.npz"))
    txt_files = sorted(arguments.txt_dir.glob("*.txt"))
    expected_names = [f"{path.stem}.txt" for path in sources]
    if [path.name for path in txt_files] != expected_names:
        raise ValueError("TXT names do not match Track 2 source files")

    total_rows = 0
    total_positives = 0
    records = []
    for source_path, txt_path in zip(sources, txt_files, strict=True):
        with np.load(source_path, allow_pickle=False) as archive:
            source = archive["ev"]
        row_count = 0
        positives = 0
        with txt_path.open("r", encoding="ascii") as stream:
            for row_count, line in enumerate(stream, start=1):
                columns = line.split()
                if len(columns) != 5:
                    raise ValueError(f"{txt_path.name}: row {row_count} is not 5 columns")
                event = source[row_count - 1]
                x, y, timestamp, polarity, label = columns
                if (
                    int(x) != int(event["x"])
                    or int(y) != int(event["y"])
                    or int(polarity) != int(event["p"])
                ):
                    raise ValueError(f"{txt_path.name}: source order mismatch")
                if timestamp != f"{float(event['t']):.9f}":
                    raise ValueError(f"{txt_path.name}: timestamp format mismatch")
                parsed_label = int(label)
                if parsed_label not in (0, 1):
                    raise ValueError(f"{txt_path.name}: label is not binary")
                positives += parsed_label
        if row_count != len(source):
            raise ValueError(f"{txt_path.name}: row count mismatch")
        total_rows += row_count
        total_positives += positives
        records.append(
            {
                "source": source_path.name,
                "txt": txt_path.name,
                "rows": row_count,
                "positives": positives,
                "txt_sha256": sha256_file(txt_path),
            }
        )

    with zipfile.ZipFile(arguments.zip) as archive:
        members = archive.namelist()
        if members != expected_names:
            raise ValueError("ZIP member names or order do not match Track 2 files")
        if any("/" in name or "\\" in name for name in members):
            raise ValueError("ZIP must contain root-level TXT files only")
        for txt_path in txt_files:
            if archive.read(txt_path.name) != txt_path.read_bytes():
                raise ValueError(f"{txt_path.name}: ZIP contents differ from TXT")

    report = {
        "passed": True,
        "source_file_count": len(sources),
        "txt_file_count": len(txt_files),
        "zip_member_count": len(members),
        "total_rows": total_rows,
        "total_positives": total_positives,
        "timestamp_format": "%.9f",
        "source_xy_p_order_exact": True,
        "timestamps_match_track2_formatter": True,
        "binary_labels_only": True,
        "zip_root_txt_only": True,
        "zip_matches_txt_files": True,
        "submission_zip": str(arguments.zip.resolve()),
        "submission_zip_size_bytes": arguments.zip.stat().st_size,
        "submission_zip_sha256": sha256_file(arguments.zip),
        "files": records,
    }
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
