from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from huav.experiments.lr_strategy import (
    _archive_incomplete_attempt,
    _completed_run_record,
    _state,
    assess_peak_history,
    choose_peak_candidate,
    choose_stop_rule,
    paired_verdict,
    peak_refinement_rates,
    transition_refinements,
)


def _history(scores: list[float], losses: list[float] | None = None) -> list[dict]:
    losses = losses or [1.0 - index * 0.05 for index in range(len(scores))]
    return [
        {
            "epoch": index + 1,
            "validation_loss": {"total": losses[index]},
            "validation": {"competition": {"score": score, "fa": 1e-5}},
            "optimization": {"gradient_clipped_fraction": 0.1},
        }
        for index, score in enumerate(scores)
    ]


def test_peak_assessment_and_selection_are_deterministic() -> None:
    slow = {
        "learning_rate": 0.00035,
        "assessment": assess_peak_history(_history([0.5, 0.6, 0.7, 0.75, 0.78, 0.8])),
    }
    fast = {
        "learning_rate": 0.0007,
        "assessment": assess_peak_history(_history([0.5, 0.65, 0.75, 0.8, 0.82, 0.84])),
    }

    assert fast["assessment"]["stable"]
    assert choose_peak_candidate([slow, fast]) is fast
    assert peak_refinement_rates([0.00035, 0.0007, 0.0014], 0.0007) == pytest.approx(
        [0.0004949747468, 0.0009899494937]
    )
    assert transition_refinements(12, [6, 12, 18, 24]) == [9, 15]


def test_stop_rule_must_preserve_every_seed_and_save_epochs() -> None:
    histories = [
        _history([0.5, 0.6, 0.7, 0.8, 0.85, 0.9] + [0.9] * 12)
        for _ in range(3)
    ]

    rule = choose_stop_rule(histories, fast_epochs=0)

    assert rule is not None
    assert rule["median_saved_epochs"] >= 3


def test_paired_verdict_prefers_score_then_efficiency() -> None:
    score = paired_verdict(
        [0.90, 0.91, 0.905],
        [0.902, 0.912, 0.904],
        saved_epochs=[0, 0, 0],
        time_savings=[0.0, 0.0, 0.0],
    )
    efficient = paired_verdict(
        [0.90, 0.91, 0.905],
        [0.9002, 0.9098, 0.9051],
        saved_epochs=[3, 4, 3],
        time_savings=[0.1, 0.12, 0.1],
    )

    assert score["verdict"] == "score_advantage"
    assert efficient["verdict"] == "efficiency_advantage"


def test_completed_run_is_reused_only_when_configuration_matches(tmp_path) -> None:
    config = {
        "dataset": {"root": str(tmp_path / "data")},
        "training": {
            "epochs": 6,
            "learning_rate": 0.0007,
            "output_root": str(tmp_path / "outputs"),
        },
    }
    requested = tmp_path / "requested.yaml"
    requested.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved_config = json.loads(json.dumps(config))
    saved_config["training"]["fixed_learning_rate_ratio"] = 0.2
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(saved_config), encoding="utf-8"
    )
    history = [
        {"epoch": 1, "validation": {"competition": {"score": 0.8}}},
        {"epoch": 2, "validation": {"competition": {"score": 0.9}}},
    ]
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"best_score": 0.9}), encoding="utf-8"
    )
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    record = _completed_run_record(
        name="candidate",
        requested_config=requested,
        run_dir=run_dir,
        log_path=tmp_path / "candidate.log",
    )

    assert record is not None
    assert record["reused"] is True
    assert record["best_epoch"] == 2
    requested.write_text(
        yaml.safe_dump(
            {
                "dataset": {"root": str(tmp_path / "data")},
                "training": {
                    "epochs": 7,
                    "learning_rate": 0.0007,
                    "output_root": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        _completed_run_record(
            name="candidate",
            requested_config=requested,
            run_dir=run_dir,
            log_path=tmp_path / "candidate.log",
        )


def test_incomplete_run_and_log_are_archived_without_deletion(tmp_path) -> None:
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    (run_dir / "partial.txt").write_text("preserve", encoding="utf-8")
    log_path = tmp_path / "candidate.log"
    log_path.write_text("failure", encoding="utf-8")

    archived = _archive_incomplete_attempt(run_dir, log_path)

    assert archived is not None
    archived_run = tmp_path / Path(archived["run_dir"]).name
    archived_log = tmp_path / Path(archived["log"]).name
    assert not run_dir.exists()
    assert not log_path.exists()
    assert (archived_run / "partial.txt").read_text(encoding="utf-8") == "preserve"
    assert archived_log.read_text(encoding="utf-8") == "failure"


def test_branch_state_falls_back_to_resumed_source_epoch(tmp_path) -> None:
    source = tmp_path / "source" / "epoch_006.pt"
    source.parent.mkdir()
    torch.save({"epoch": 6}, source)
    run_dir = tmp_path / "master"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"resumed_from_training_state": str(source)}),
        encoding="utf-8",
    )
    record = {"run_dir": str(run_dir)}

    assert _state(record, 6) == source
    with pytest.raises(FileNotFoundError, match="epoch 9"):
        _state(record, 9)
