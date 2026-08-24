"""Adaptive full-data learning-rate strategy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from huav.config import load_config


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def assess_peak_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if len(history) != 6:
        return {"stable": False, "reason": "incomplete_history"}
    losses = [float(item["validation_loss"]["total"]) for item in history]
    scores = [float(item["validation"]["competition"]["score"]) for item in history]
    clipping = [
        float(item.get("optimization", {}).get("gradient_clipped_fraction", 0.0))
        for item in history
    ]
    values = losses + scores + clipping
    if not all(math.isfinite(value) for value in values):
        return {"stable": False, "reason": "nonfinite_metric"}
    if losses[-1] >= losses[0]:
        return {"stable": False, "reason": "validation_loss_not_reduced"}
    severe_drops = [
        scores[index] < scores[index - 1] - 0.02 for index in range(1, len(scores))
    ]
    if any(left and right for left, right in zip(severe_drops, severe_drops[1:])):
        return {"stable": False, "reason": "consecutive_score_collapse"}
    for index in range(1, len(history)):
        if (
            clipping[index - 1] >= 1.0
            and clipping[index] >= 1.0
            and scores[index] < scores[index - 1] + 0.001
        ):
            return {"stable": False, "reason": "persistent_full_gradient_clipping"}
    return {
        "stable": True,
        "reason": None,
        "early_score": statistics.mean(scores[2:6]),
        "validation_loss_reduction": (losses[0] - losses[-1]) / losses[0],
        "mean_clipped_fraction": statistics.mean(clipping),
    }


def choose_peak_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stable = [item for item in candidates if item["assessment"]["stable"]]
    if not stable:
        raise RuntimeError("no stable peak learning-rate candidate")
    best_early = max(item["assessment"]["early_score"] for item in stable)
    tied = [
        item
        for item in stable
        if best_early - item["assessment"]["early_score"] < 0.001
    ]
    tied.sort(
        key=lambda item: (
            -item["assessment"]["validation_loss_reduction"],
            item["assessment"]["mean_clipped_fraction"],
            item["learning_rate"],
        )
    )
    return tied[0]


def peak_refinement_rates(
    ordered_rates: list[float], selected_rate: float
) -> list[float]:
    rates = sorted(set(float(value) for value in ordered_rates))
    index = rates.index(float(selected_rate))
    refinements: list[float] = []
    refinements.append(
        math.sqrt(rates[index - 1] * selected_rate)
        if index > 0
        else selected_rate / 2
    )
    refinements.append(
        math.sqrt(selected_rate * rates[index + 1])
        if index < len(rates) - 1
        else selected_rate * 2
    )
    return refinements


def _best_score_record(record: dict[str, Any]) -> tuple[float, int, float]:
    history = record["history"]
    scores = [float(item["validation"]["competition"]["score"]) for item in history]
    best = max(scores)
    first_near = next(
        int(item["epoch"])
        for item, score in zip(history, scores, strict=True)
        if best - score <= 0.001
    )
    best_item = max(
        history, key=lambda item: float(item["validation"]["competition"]["score"])
    )
    fa = float(best_item["validation"]["competition"]["fa"])
    return best, first_near, fa


def choose_score_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(candidates)
    best_score = max(_best_score_record(record)[0] for record in records)
    tied = [
        record
        for record in records
        if best_score - _best_score_record(record)[0] < 0.001
    ]
    tied.sort(key=lambda record: (_best_score_record(record)[1], _best_score_record(record)[2]))
    return tied[0]


def transition_refinements(selected: int, coarse: list[int]) -> list[int]:
    values = sorted(set(coarse))
    index = values.index(selected)
    result: list[int] = []
    if index > 0:
        result.append((values[index - 1] + selected) // 2)
    if index < len(values) - 1:
        result.append((selected + values[index + 1]) // 2)
    return [value for value in result if value >= 6 and value % 3 == 0]


def simulate_stop(
    history: list[dict[str, Any]],
    *,
    fast_epochs: int,
    min_delta: float,
    patience: int,
    minimum_fine_epochs: int = 6,
) -> dict[str, Any]:
    best = -math.inf
    bad_epochs = 0
    stop_epoch = int(history[-1]["epoch"])
    for item in history:
        epoch = int(item["epoch"])
        score = float(item["validation"]["competition"]["score"])
        if score >= best + min_delta:
            best = score
            bad_epochs = 0
        else:
            bad_epochs += 1
        fine_epochs = max(0, epoch - fast_epochs)
        if fine_epochs >= minimum_fine_epochs and bad_epochs >= patience:
            stop_epoch = epoch
            break
    stopped_scores = [
        float(item["validation"]["competition"]["score"])
        for item in history
        if int(item["epoch"]) <= stop_epoch
    ]
    return {"stop_epoch": stop_epoch, "best_score": max(stopped_scores)}


def choose_stop_rule(
    histories: list[list[dict[str, Any]]], *, fast_epochs: int
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for min_delta in (0.0005, 0.001):
        for patience in (3, 4, 5, 6):
            outcomes = [
                simulate_stop(
                    history,
                    fast_epochs=fast_epochs,
                    min_delta=min_delta,
                    patience=patience,
                )
                for history in histories
            ]
            full_bests = [
                max(
                    float(item["validation"]["competition"]["score"])
                    for item in history
                )
                for history in histories
            ]
            if any(
                full_best - outcome["best_score"] > 0.001
                for full_best, outcome in zip(full_bests, outcomes, strict=True)
            ):
                continue
            savings = [
                int(history[-1]["epoch"]) - outcome["stop_epoch"]
                for history, outcome in zip(histories, outcomes, strict=True)
            ]
            if statistics.median(savings) < 3:
                continue
            candidates.append(
                {
                    "min_delta": min_delta,
                    "patience": patience,
                    "minimum_fine_epochs": 6,
                    "outcomes": outcomes,
                    "median_saved_epochs": statistics.median(savings),
                    "median_stop_epoch": statistics.median(
                        outcome["stop_epoch"] for outcome in outcomes
                    ),
                }
            )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["median_stop_epoch"],
            -item["min_delta"],
            item["patience"],
        )
    )
    return candidates[0]


def paired_verdict(
    baseline_scores: list[float],
    candidate_scores: list[float],
    *,
    saved_epochs: list[int],
    time_savings: list[float],
) -> dict[str, Any]:
    differences = [
        candidate - baseline
        for baseline, candidate in zip(baseline_scores, candidate_scores, strict=True)
    ]
    median_difference = statistics.median(differences)
    wins = sum(value > 0 for value in differences)
    if median_difference >= 0.001 and wins >= 2:
        verdict = "score_advantage"
    elif (
        abs(median_difference) < 0.001
        and min(differences) >= -0.002
        and (
            statistics.median(saved_epochs) >= 3
            or statistics.median(time_savings) >= 0.10
        )
    ):
        verdict = "efficiency_advantage"
    else:
        verdict = "no_supported_replacement"
    return {
        "verdict": verdict,
        "paired_score_differences": differences,
        "median_score_difference": median_difference,
        "wins": wins,
    }


def _completed_run_record(
    *,
    name: str,
    requested_config: Path,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any] | None:
    required = (
        run_dir / "config.yaml",
        run_dir / "history.json",
        run_dir / "summary.json",
        run_dir / "best.pt",
        run_dir / "metrics.json",
    )
    if not all(path.is_file() for path in required):
        return None
    requested = asdict(load_config(requested_config))
    saved = asdict(load_config(run_dir / "config.yaml"))
    if saved != requested:
        raise RuntimeError(f"completed run configuration mismatch: {name}")
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "status": "completed",
        "reused": True,
        "run_dir": str(run_dir),
        "config": str(requested_config),
        "log": str(log_path),
        "history": history,
        "best_score": float(summary["best_score"]),
        "best_epoch": int(
            max(
                history,
                key=lambda item: float(item["validation"]["competition"]["score"]),
            )["epoch"]
        ),
    }


def _archive_incomplete_attempt(run_dir: Path, log_path: Path) -> dict[str, str] | None:
    if not run_dir.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_run = run_dir.with_name(f"{run_dir.name}_incomplete_{stamp}")
    run_dir.replace(archived_run)
    archived = {"run_dir": str(archived_run)}
    if log_path.exists():
        archived_log = log_path.with_name(
            f"{log_path.stem}_incomplete_{stamp}{log_path.suffix}"
        )
        log_path.replace(archived_log)
        archived["log"] = str(archived_log)
    return archived


class Experiment:
    def __init__(self, base_config: Path, root: Path) -> None:
        self.base_path = base_config.resolve()
        self.root = root.resolve()
        self.config_root = self.root / "configs"
        self.run_root = self.root / "runs"
        self.log_root = self.root / "logs"
        self.smoke_root = Path.cwd().resolve() / "smoke_outputs" / self.root.name
        self.smoke_run_root = self.smoke_root / "runs"
        self.smoke_log_root = self.smoke_root / "logs"
        for path in (
            self.config_root,
            self.run_root,
            self.log_root,
            self.smoke_run_root,
            self.smoke_log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.base = yaml.safe_load(self.base_path.read_text(encoding="utf-8"))
        dataset_root = Path(self.base["dataset"]["root"])
        if not dataset_root.is_absolute():
            dataset_root = (self.base_path.parent / dataset_root).resolve()
        self.base["dataset"]["root"] = str(dataset_root)
        self.records: list[dict[str, Any]] = []
        self.status_path = self.root / "status.json"
        self.leaderboard_path = self.root / "leaderboard.json"

    def status(self, **values: Any) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        _atomic_json(self.status_path, payload)

    def write_config(
        self,
        name: str,
        training: dict[str, Any],
        *,
        full_data: bool = True,
        output_root: Path | None = None,
    ) -> Path:
        payload = json.loads(json.dumps(self.base))
        payload["training"].update(training)
        payload["training"].update(
            {
                "device": "cuda",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "max_train_files": None if full_data else 2,
                "max_val_files": None if full_data else 2,
                "output_root": str(output_root or self.run_root),
            }
        )
        path = self.config_root / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def command(self, command: list[str], *, log_path: Path, run_name: str) -> None:
        self.status(stage="running", current_run=run_name, command=command)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if line.startswith("epoch "):
                    self.status(stage="running", current_run=run_name, last_line=line.strip())
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def run(
        self,
        name: str,
        training: dict[str, Any],
        *,
        resume_state: Path | None = None,
        allow_instability: bool = False,
        full_data: bool = True,
    ) -> dict[str, Any]:
        run_root = self.run_root if full_data else self.smoke_run_root
        log_root = self.log_root if full_data else self.smoke_log_root
        config = self.write_config(
            name, training, full_data=full_data, output_root=run_root
        )
        run_dir = run_root / name
        log_path = log_root / f"{name}.log"
        completed = _completed_run_record(
            name=name,
            requested_config=config,
            run_dir=run_dir,
            log_path=log_path,
        )
        if completed is not None:
            self.records.append(completed)
            self._leaderboard()
            return completed
        archived = _archive_incomplete_attempt(run_dir, log_path)
        if archived is not None:
            self.status(
                stage="recovering_incomplete_run",
                current_run=name,
                archived=archived,
            )
        command = [
            sys.executable,
            "-m",
            "huav.cli.train",
            "--config",
            str(config),
            "--device",
            "cuda",
            "--output-dir",
            str(run_dir),
            "--quiet",
            "--note",
            f"LR strategy experiment: {name}",
        ]
        if resume_state is not None:
            command.extend(["--resume-state", str(resume_state)])
        try:
            self.command(command, log_path=log_path, run_name=name)
        except subprocess.CalledProcessError:
            contents = log_path.read_text(encoding="utf-8", errors="replace").lower()
            if allow_instability and any(
                marker in contents
                for marker in ("non-finite", "nan", "inf", "overflow")
            ):
                record = {"name": name, "status": "rejected_unstable", "log": str(log_path)}
                self.records.append(record)
                self._leaderboard()
                return record
            raise
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        record = {
            "name": name,
            "status": "completed",
            "run_dir": str(run_dir),
            "config": str(config),
            "log": str(log_path),
            "history": history,
            "best_score": float(summary["best_score"]),
            "best_epoch": int(
                max(
                    history,
                    key=lambda item: float(item["validation"]["competition"]["score"]),
                )["epoch"]
            ),
        }
        self.records.append(record)
        self._leaderboard()
        return record

    def _leaderboard(self) -> None:
        payload = [
            {key: value for key, value in record.items() if key != "history"}
            for record in self.records
        ]
        _atomic_json(self.leaderboard_path, payload)

    def reevaluate(self, record: dict[str, Any]) -> None:
        run_dir = Path(record["run_dir"])
        self.command(
            [
                sys.executable,
                "-m",
                "huav.cli.evaluate",
                "--config",
                str(run_dir / "config.yaml"),
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--device",
                "cuda",
                "--quiet",
            ],
            log_path=self.log_root / f"{record['name']}_reevaluate.log",
            run_name=f"{record['name']}:reevaluate",
        )
        saved = hashlib.sha256((run_dir / "metrics.json").read_bytes()).hexdigest()
        reproduced = hashlib.sha256(
            (run_dir / "reproduced_metrics.json").read_bytes()
        ).hexdigest()
        if saved != reproduced:
            raise RuntimeError(f"independent reevaluation mismatch: {record['name']}")
        record["metrics_sha256"] = saved

    def audit_submission(self, record: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(record["run_dir"])
        self.command(
            [
                sys.executable,
                "-m",
                "huav.cli.predict",
                "--config",
                str(run_dir / "config.yaml"),
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--device",
                "cuda",
                "--quiet",
            ],
            log_path=self.log_root / f"{record['name']}_predict.log",
            run_name=f"{record['name']}:predict",
        )
        validation_root = Path(self.base["dataset"]["root"]) / "val"
        submission_root = run_dir / "submission"
        source_files = sorted(validation_root.glob("*.npz"))
        txt_files = sorted(submission_root.glob("*.txt"))
        if len(source_files) != 24 or len(txt_files) != 24:
            raise RuntimeError("submission must contain 24 NPZ/TXT pairs")
        total_rows = 0
        positive_rows = 0
        for source_path in source_files:
            txt_path = submission_root / f"{source_path.stem}.txt"
            with np.load(source_path, allow_pickle=False) as archive:
                events = archive["ev"]
            with txt_path.open("r", encoding="ascii") as stream:
                for line, event in zip_longest(stream, events):
                    if line is None or event is None:
                        raise RuntimeError(f"row count mismatch: {txt_path.name}")
                    fields = line.split()
                    if len(fields) != 5 or fields[4] not in {"0", "1"}:
                        raise RuntimeError(f"invalid prediction row: {txt_path.name}")
                    if (
                        int(fields[0]) != int(event["x"])
                        or int(fields[1]) != int(event["y"])
                        or float(fields[2]) != float(event["t"])
                        or int(fields[3]) != int(event["p"])
                    ):
                        raise RuntimeError(f"source order mismatch: {txt_path.name}")
                    total_rows += 1
                    positive_rows += int(fields[4])
        with zipfile.ZipFile(submission_root / "submission.zip") as archive:
            members = archive.namelist()
        expected = sorted(path.name for path in txt_files)
        if sorted(members) != expected or any("/" in member for member in members):
            raise RuntimeError("submission ZIP must contain root-level TXT files only")
        report = {
            "txt_files": 24,
            "total_rows": total_rows,
            "positive_rows": positive_rows,
            "source_order_and_values_match": True,
            "binary_labels_only": True,
            "zip_root_txt_only": True,
        }
        _atomic_json(run_dir / "submission_audit.json", report)
        return report


def _state(record: dict[str, Any], epoch: int) -> Path:
    run_dir = Path(record["run_dir"])
    candidate = run_dir / "training_states" / f"epoch_{epoch:03d}.pt"
    if candidate.is_file():
        return candidate
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_value = summary.get("resumed_from_training_state")
        if source_value:
            source = Path(source_value)
            if source.is_file():
                source_state = torch.load(
                    source, map_location="cpu", weights_only=False
                )
                if int(source_state["epoch"]) == epoch:
                    return source
    raise FileNotFoundError(
        f"training state for epoch {epoch} is unavailable in {run_dir}"
    )


def _run_experiment(base_config: Path, root: Path) -> dict[str, Any]:
    experiment = Experiment(base_config, root)
    experiment.status(stage="schedule_smoke")
    for schedule in (
        "legacy_cosine",
        "constant",
        "constant_then_cosine",
        "constant_then_exponential",
        "constant_then_fixed",
        "constant_then_plateau",
    ):
        smoke = experiment.run(
            f"smoke_{schedule}",
            {
                "epochs": 3,
                "seed": 37,
                "lr_schedule": schedule,
                "fast_learning_epochs": 1 if schedule.startswith("constant_then") else 0,
                "save_training_state_interval": 1,
            },
            full_data=False,
        )
        experiment.reevaluate(smoke)
    experiment.status(stage="peak_lr_screen")
    coarse_rates = [3.5e-4, 7e-4, 1.4e-3, 2.8e-3]
    peak_records: list[dict[str, Any]] = []
    for rate in coarse_rates:
        record = experiment.run(
            f"peak_{rate:.8f}".replace(".", "p"),
            {
                "epochs": 6,
                "seed": 37,
                "learning_rate": rate,
                "lr_schedule": "constant",
                "fast_learning_epochs": 0,
                "save_training_state_interval": 3,
            },
            allow_instability=True,
        )
        if record["status"] == "completed":
            record["learning_rate"] = rate
            record["assessment"] = assess_peak_history(record["history"])
            peak_records.append(record)
    first_peak = choose_peak_candidate(peak_records)
    for rate in peak_refinement_rates(coarse_rates, first_peak["learning_rate"]):
        record = experiment.run(
            f"peak_refine_{rate:.8f}".replace(".", "p"),
            {
                "epochs": 6,
                "seed": 37,
                "learning_rate": rate,
                "lr_schedule": "constant",
                "fast_learning_epochs": 0,
                "save_training_state_interval": 3,
            },
            allow_instability=True,
        )
        if record["status"] == "completed":
            record["learning_rate"] = rate
            record["assessment"] = assess_peak_history(record["history"])
            peak_records.append(record)
    peak = choose_peak_candidate(peak_records)
    peak_rate = float(peak["learning_rate"])

    experiment.status(stage="transition_screen", selected_peak_lr=peak_rate)
    master = experiment.run(
        "constant_master_seed37",
        {
            "epochs": 24,
            "seed": 37,
            "learning_rate": peak_rate,
            "lr_schedule": "constant",
            "fast_learning_epochs": 0,
            "save_training_state_interval": 3,
        },
        resume_state=_state(peak, 6),
    )
    baseline37 = experiment.run(
        "baseline_legacy_seed37",
        {
            "epochs": 24,
            "seed": 37,
            "lr_schedule": "legacy_cosine",
            "fast_learning_epochs": 0,
            "save_training_state_interval": 0,
        },
    )
    transition_records: list[dict[str, Any]] = []
    for fast in (6, 12, 18):
        record = experiment.run(
            f"transition_{fast:02d}_seed37",
            {
                "epochs": 24,
                "seed": 37,
                "learning_rate": peak_rate,
                "lr_schedule": "constant_then_cosine",
                "fast_learning_epochs": fast,
                "min_learning_rate_ratio": 0.1,
                "save_training_state_interval": 0,
            },
            resume_state=_state(master, fast),
        )
        record["fast_epochs"] = fast
        transition_records.append(record)
    master["fast_epochs"] = 24
    transition_records.append(master)
    coarse_transition = choose_score_candidate(transition_records)
    for fast in transition_refinements(
        int(coarse_transition["fast_epochs"]), [6, 12, 18, 24]
    ):
        record = experiment.run(
            f"transition_{fast:02d}_seed37",
            {
                "epochs": 24,
                "seed": 37,
                "learning_rate": peak_rate,
                "lr_schedule": "constant_then_cosine",
                "fast_learning_epochs": fast,
                "min_learning_rate_ratio": 0.1,
                "save_training_state_interval": 0,
            },
            resume_state=_state(master, fast),
        )
        record["fast_epochs"] = fast
        transition_records.append(record)
    transition = choose_score_candidate(transition_records)
    fast_epochs = int(transition["fast_epochs"])

    experiment.status(
        stage="fine_strategy_screen",
        selected_peak_lr=peak_rate,
        selected_fast_epochs=fast_epochs,
    )
    fine_specs = {
        "cosine": {"lr_schedule": "constant_then_cosine", "min_learning_rate_ratio": 0.1},
        "exponential": {
            "lr_schedule": "constant_then_exponential",
            "min_learning_rate_ratio": 0.1,
        },
        "fixed": {
            "lr_schedule": "constant_then_fixed",
            "fixed_learning_rate_ratio": 0.2,
        },
        "plateau": {
            "lr_schedule": "constant_then_plateau",
            "min_learning_rate_ratio": 0.05,
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_delta": 0.001,
        },
    }
    fine_records: list[dict[str, Any]] = []
    for fine_name, fields in fine_specs.items():
        record = experiment.run(
            f"fine_{fine_name}_seed37",
            {
                "epochs": fast_epochs + 18,
                "seed": 37,
                "learning_rate": peak_rate,
                "fast_learning_epochs": fast_epochs,
                "save_training_state_interval": 0,
                **fields,
            },
            resume_state=_state(master, fast_epochs),
        )
        record["fine_name"] = fine_name
        fine_records.append(record)
    fine = choose_score_candidate(fine_records)
    fine_name = str(fine["fine_name"])

    experiment.status(stage="paired_seed_final", selected_fine_strategy=fine_name)
    baselines = {37: baseline37}
    winners = {37: fine}
    for seed in (73, 101):
        baselines[seed] = experiment.run(
            f"baseline_legacy_seed{seed}",
            {
                "epochs": 24,
                "seed": seed,
                "lr_schedule": "legacy_cosine",
                "fast_learning_epochs": 0,
                "save_training_state_interval": 0,
            },
        )
        winners[seed] = experiment.run(
            f"winner_{fine_name}_seed{seed}",
            {
                "epochs": fast_epochs + 18,
                "seed": seed,
                "learning_rate": peak_rate,
                "fast_learning_epochs": fast_epochs,
                "save_training_state_interval": 0,
                **fine_specs[fine_name],
            },
        )

    finalists = [*baselines.values(), *winners.values()]
    for record in finalists:
        experiment.reevaluate(record)
    stop_rule = choose_stop_rule(
        [winners[seed]["history"] for seed in (37, 73, 101)],
        fast_epochs=fast_epochs,
    )
    baseline_scores = [baselines[seed]["best_score"] for seed in (37, 73, 101)]
    winner_scores = [winners[seed]["best_score"] for seed in (37, 73, 101)]
    if stop_rule is None:
        saved_epochs = [0, 0, 0]
    else:
        saved_epochs = [
            fast_epochs + 18 - int(outcome["stop_epoch"])
            for outcome in stop_rule["outcomes"]
        ]
    verdict = paired_verdict(
        baseline_scores,
        winner_scores,
        saved_epochs=saved_epochs,
        time_savings=[value / 24 for value in saved_epochs],
    )
    global_best = max(finalists, key=lambda record: record["best_score"])
    submission_audit = experiment.audit_submission(global_best)
    result = {
        "selected_peak_learning_rate": peak_rate,
        "selected_fast_learning_epochs": fast_epochs,
        "selected_fine_strategy": fine_name,
        "stop_rule": stop_rule,
        "baseline_scores": baseline_scores,
        "winner_scores": winner_scores,
        "verdict": verdict,
        "global_best_run": global_best["run_dir"],
        "submission_audit": submission_audit,
    }
    _atomic_json(root / "result.json", result)
    experiment._leaderboard()
    experiment.status(stage="completed", result=str(root / "result.json"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        _run_experiment(args.base_config, args.experiment_root)
    except BaseException as error:
        args.experiment_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            args.experiment_root / "failed.json",
            {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
