"""Prediction and submission validation contracts, without artifact writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable, Sequence

import torch

from huav.data import EventSequence


def validate_prediction_logits(sequence: EventSequence, logits: torch.Tensor) -> torch.Tensor:
    """Validate logits in the exact original event order, one value per input event."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or not logits.is_floating_point():
        raise TypeError("prediction logits must be a one-dimensional floating torch Tensor")
    if len(logits) != sequence.length:
        raise ValueError("prediction logits must contain one logit per original input event")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("prediction logits must be finite")
    return logits


class Predictor(ABC):
    """Future predictor interface. Implementations must preserve event count and order."""

    @abstractmethod
    def predict_sequence(self, sequence: EventSequence) -> torch.Tensor:
        """Return one floating logit per original input event, in original input order."""


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
    x = _integer_row_value(row.x, "x")
    y = _integer_row_value(row.y, "y")
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


class SubmissionWriter(ABC):
    """Future writer contract; Phase 1 intentionally implements validation only."""

    @abstractmethod
    def validate(self, sequence: EventSequence, rows: Sequence[SubmissionRow], manifest: SubmissionManifest) -> None:
        """Validate five-column rows and root-level txt-only zip manifest before writing."""


__all__ = [
    "Predictor", "SubmissionManifest", "SubmissionRow", "SubmissionWriter",
    "validate_prediction_logits", "validate_submission_contract", "validate_zip_members",
]
