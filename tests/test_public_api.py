import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import huav
from huav import NPZEventDataset, SpatioTemporalConeDetector
from huav.data import EventBatch, EventSequence, EventWindow
from huav.inference import (
    SubmissionManifest, SubmissionRow, generate_submission_rows,
    pack_submission_zip, validate_submission_contract, write_prediction_txt,
)
from huav.models import EventSegmenter, ModelOutput
from huav.training import EvaluationResult, Evaluator, EventMetricAccumulator, Trainer


def test_public_contracts_are_importable() -> None:
    assert all(
        (
            EventSequence,
            NPZEventDataset,
            EventWindow,
            EventBatch,
            EventSegmenter,
            SpatioTemporalConeDetector,
            ModelOutput,
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
    assert not hasattr(huav, "DualViewEventClassifier")
