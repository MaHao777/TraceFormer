"""Prediction and submission validation contracts, without artifact writers."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable, Sequence

import torch

from traceformer.data import EventSequence


def validate_prediction_logits(sequence: EventSequence, logits: torch.Tensor) -> torch.Tensor:
    """Validate logits in the exact original event order, one value per input event."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or not logits.is_floating_point():
        raise TypeError("prediction logits must be a one-dimensional floating torch Tensor")
    if len(logits) != sequence.length:
        raise ValueError("prediction logits must contain one logit per original input event")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("prediction logits must be finite")
    return logits


@dataclass(frozen=True)
class SubmissionRow:
    """One future Codebench row in the required ``x y t p label`` order."""

    x: int
    y: int
    t: float
    p: int
    label: int


@dataclass(frozen=True)
class SubmissionManifest:
    """Planned txt name and planned zip member names; no file is created here."""

    txt_name: str
    zip_members: tuple[str, ...]


def _integer_row_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"submission {name} must be a non-boolean integer")
    return int(value)


def _timestamp_row_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("submission t must be a finite non-boolean real value")
    return float(value)


def _validate_submission_row(row: SubmissionRow) -> None:
    _integer_row_value(row.x, "x")
    _integer_row_value(row.y, "y")
    _timestamp_row_value(row.t)
    p = _integer_row_value(row.p, "p")
    label = _integer_row_value(row.label, "label")
    if p not in (0, 1):
        raise ValueError("submission p must be 0 or 1")
    if label not in (0, 1):
        raise ValueError("submission labels must be 0 or 1")


def validate_zip_members(members: Iterable[str]) -> tuple[str, ...]:
    """Ensure a prospective zip has root-level txt files only and no directories."""
    if isinstance(members, (str, bytes)):
        raise ValueError("submission zip members must be a non-string iterable of strings")
    try:
        checked = tuple(members)
    except TypeError as error:
        raise ValueError("submission zip members must be an iterable of strings") from error
    if not checked:
        raise ValueError("submission zip must contain at least one txt file")
    for member in checked:
        if not isinstance(member, str):
            raise ValueError("submission zip members must be strings")
        if not member or any(ord(character) < 32 or ord(character) == 127 for character in member):
            raise ValueError("submission zip member names must be non-empty and contain no control characters")
        normalized = member.replace("\\", "/")
        if normalized.startswith("/") or Path(normalized).is_absolute() or normalized.endswith("/") or "/" in normalized or Path(normalized).suffix.lower() != ".txt":
            raise ValueError("submission zip may contain only root-level .txt files, without directories")
    if len(set(checked)) != len(checked):
        raise ValueError("submission zip members must be unique")
    return checked


def validate_submission_contract(
    sequence: EventSequence, rows: Sequence[SubmissionRow], manifest: SubmissionManifest
) -> None:
    """Validate the future five-column artifact without writing txt or zip files."""
    if not isinstance(manifest, SubmissionManifest):
        raise ValueError("manifest must be a SubmissionManifest")
    if not isinstance(manifest.txt_name, str):
        raise ValueError("manifest txt_name must be a string")
    if not isinstance(manifest.zip_members, tuple):
        raise ValueError("manifest zip_members must be a tuple")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("submission rows must be a sequence of SubmissionRow values")
    expected_name = f"{sequence.source_stem}.txt"
    if manifest.txt_name != expected_name:
        raise ValueError(f"txt stem must match source sequence: expected {expected_name}")
    members = validate_zip_members(manifest.zip_members)
    if manifest.txt_name not in members:
        raise ValueError("manifest txt_name must be included in zip members")
    if len(rows) != sequence.length:
        raise ValueError("submission row count must match source event count")
    for index, row in enumerate(rows):
        if not isinstance(row, SubmissionRow):
            raise ValueError("submission rows must be SubmissionRow values")
        _validate_submission_row(row)
        expected = (int(sequence.x[index]), int(sequence.y[index]), float(sequence.t[index]), int(sequence.p[index]))
        actual = (int(row.x), int(row.y), float(row.t), int(row.p))
        if actual != expected:
            raise ValueError("submission rows must preserve source x y t p event order")


def generate_submission_rows(
    sequence: EventSequence,
    logits: torch.Tensor,
    threshold: float,
) -> tuple[SubmissionRow, ...]:
    """Convert predicted logits to validated submission rows.

    Each logit is transformed via sigmoid; values >= *threshold* produce
    label=1, others label=0. The returned rows are validated against the
    source *sequence* before being returned.
    """
    validate_prediction_logits(sequence, logits)
    probabilities = torch.sigmoid(logits)
    predicted_labels = (probabilities >= threshold).to(torch.int64)
    rows: list[SubmissionRow] = []
    for i in range(sequence.length):
        rows.append(SubmissionRow(
            x=int(sequence.x[i]),
            y=int(sequence.y[i]),
            t=float(sequence.t[i]),
            p=int(sequence.p[i]),
            label=int(predicted_labels[i]),
        ))
    result = tuple(rows)
    txt_name = f"{sequence.source_stem}.txt"
    validate_submission_contract(
        sequence, result, SubmissionManifest(txt_name, (txt_name,))
    )
    return result


def write_prediction_txt(
    rows: Sequence[SubmissionRow],
    output_dir: str | Path,
    txt_name: str,
) -> Path:
    """Write one TXT file in the five-column competition submission format.

    Each line is ``x y t p label``, space-separated. Timestamps use enough
    significant digits to round-trip the source float64 value exactly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / txt_name
    lines = [
        f"{row.x} {row.y} {row.t:.17g} {row.p} {row.label}\n"
        for row in rows
    ]
    output_path.write_text("".join(lines), encoding="ascii")
    return output_path


def pack_submission_zip(
    txt_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Pack all root-level ``.txt`` files under *txt_dir* into a submission ZIP.

    Every member is validated via :func:`validate_zip_members` before
    writing, ensuring the ZIP contains only flat ``.txt`` entries with no
    directory structure or unsafe names.
    """
    txt_dir = Path(txt_dir)
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"no TXT files found in {txt_dir}")
    members = validate_zip_members(txt_file.name for txt_file in txt_files)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            zf.write(txt_dir / member, arcname=member)
    return output_path


__all__ = [
    "SubmissionManifest", "SubmissionRow",
    "generate_submission_rows", "write_prediction_txt", "pack_submission_zip",
    "validate_prediction_logits", "validate_submission_contract", "validate_zip_members",
]
