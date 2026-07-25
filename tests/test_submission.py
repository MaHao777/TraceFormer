from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.data import EventSequence
from huav.inference import (
    SubmissionManifest,
    SubmissionRow,
    validate_submission_contract,
    validate_zip_members,
)


def raises(exc_type: type[Exception], fn: object) -> None:
    try:
        fn()  # type: ignore[operator]
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def source() -> EventSequence:
    return EventSequence(
        x=np.array([1, 2], dtype=np.int16), y=np.array([3, 4], dtype=np.int16),
        t=np.array([0.0, 0.01]), p=np.array([0, 1], dtype=np.uint8),
        width=346, height=260, source_name="sample_01.npz",
    )


def rows() -> tuple[SubmissionRow, ...]:
    return (SubmissionRow(1, 3, 0.0, 0, 0), SubmissionRow(2, 4, 0.01, 1, 1))


def test_submission_contract_accepts_valid_rows_and_root_txt_manifest() -> None:
    validate_submission_contract(source(), rows(), SubmissionManifest("sample_01.txt", ("sample_01.txt",)))


def test_submission_contract_rejects_count_label_stem_and_order_mismatches() -> None:
    raises(ValueError, lambda: validate_submission_contract(source(), rows()[:1], SubmissionManifest("sample_01.txt", ("sample_01.txt",))))
    raises(ValueError, lambda: validate_submission_contract(source(), (SubmissionRow(1, 3, 0.0, 0, 2), rows()[1]), SubmissionManifest("sample_01.txt", ("sample_01.txt",))))
    raises(ValueError, lambda: validate_submission_contract(source(), rows(), SubmissionManifest("wrong.txt", ("wrong.txt",))))
    raises(ValueError, lambda: validate_submission_contract(source(), (rows()[1], rows()[0]), SubmissionManifest("sample_01.txt", ("sample_01.txt",))))


def test_zip_members_reject_directories_nested_entries_and_non_txt_members() -> None:
    validate_zip_members(("a.txt", "b.txt"))
    assert validate_zip_members(["a.txt", "b.txt"]) == ("a.txt", "b.txt")
    assert validate_zip_members(member for member in ("a.txt", "b.txt")) == ("a.txt", "b.txt")
    raises(ValueError, lambda: validate_zip_members("a.txt"))
    raises(ValueError, lambda: validate_zip_members(("folder/",)))
    raises(ValueError, lambda: validate_zip_members(("folder/a.txt",)))
    raises(ValueError, lambda: validate_zip_members(("a.csv",)))


def test_submission_contract_rejects_coercible_or_nonfinite_row_values() -> None:
    manifest = SubmissionManifest("sample_01.txt", ("sample_01.txt",))
    invalid_rows = (
        (SubmissionRow(True, 3, 0.0, 0, 0), rows()[1]),
        (SubmissionRow(1.0, 3, 0.0, 0, 0), rows()[1]),
        (SubmissionRow(1, 3, "0.0", 0, 0), rows()[1]),  # type: ignore[arg-type]
        (SubmissionRow(1, 3, math.nan, 0, 0), rows()[1]),
        (SubmissionRow(1, 3, 0.0, True, 0), rows()[1]),
        (SubmissionRow(1, 3, 0.0, 2, 0), rows()[1]),
        (SubmissionRow(1, 3, 0.0, 0, 1.0), rows()[1]),
    )
    for candidate in invalid_rows:
        raises(ValueError, lambda candidate=candidate: validate_submission_contract(source(), candidate, manifest))


def test_submission_contract_rejects_runtime_types_and_unsafe_zip_member_names() -> None:
    source_item = source()
    valid_rows = rows()
    bad_manifests = (
        None,
        SubmissionManifest(1, ("sample_01.txt",)),  # type: ignore[arg-type]
        SubmissionManifest("sample_01.txt", None),  # type: ignore[arg-type]
        SubmissionManifest("sample_01.txt", ("",)),
        SubmissionManifest("sample_01.txt", ("/sample_01.txt",)),
        SubmissionManifest("sample_01.txt", ("C:\\sample_01.txt",)),
        SubmissionManifest("sample_01.txt", ("sample\x00_01.txt",)),
        SubmissionManifest("sample_01.txt", ("sample\n01.txt",)),
        SubmissionManifest("sample_01.txt", (42,)),  # type: ignore[arg-type]
    )
    for manifest in bad_manifests:
        raises(ValueError, lambda manifest=manifest: validate_submission_contract(source_item, valid_rows, manifest))  # type: ignore[arg-type]
    raises(ValueError, lambda: validate_submission_contract(source_item, (object(), object()), SubmissionManifest("sample_01.txt", ("sample_01.txt",))))
