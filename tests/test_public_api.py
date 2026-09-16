import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import traceformer
from traceformer import NPZEventDataset, TraceFormer
from traceformer.data import EventSequence
from traceformer.inference import (
    SubmissionManifest,
    SubmissionRow,
    generate_submission_rows,
    pack_submission_zip,
    validate_submission_contract,
    write_prediction_txt,
)
from traceformer.training import EvaluationResult, Evaluator, EventMetricAccumulator, Trainer


def test_public_contracts_are_importable() -> None:
    assert all(
        (
            EventSequence,
            NPZEventDataset,
            TraceFormer,
            Trainer,
            Evaluator,
            EvaluationResult,
            EventMetricAccumulator,
            SubmissionManifest,
            SubmissionRow,
            generate_submission_rows,
            pack_submission_zip,
            validate_submission_contract,
            write_prediction_txt,
        )
    )
    assert not hasattr(traceformer, "DualViewEventClassifier")
    assert not hasattr(traceformer, "SpatioTemporalWindowDetector")
    assert not hasattr(traceformer, "EventBatch")
    assert not hasattr(traceformer, "EventWindow")
    assert not hasattr(traceformer, "ModelOutput")
