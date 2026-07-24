import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav import DualViewEventClassifier, NPZEventDataset
from huav.data import EventBatch, EventSequence, EventWindow
from huav.inference import SubmissionManifest, SubmissionRow, validate_submission_contract
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
            DualViewEventClassifier,
            ModelOutput,
            Trainer,
            Evaluator,
            EvaluationResult,
            EventMetricAccumulator,
            SubmissionManifest,
            SubmissionRow,
            validate_submission_contract,
        )
    )
