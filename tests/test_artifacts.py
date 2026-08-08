from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from huav.artifacts import (
    create_run_directory,
    rebuild_output_index,
    record_evaluation,
    record_run_finished,
    record_run_started,
    record_submission,
    resolve_evaluation_output,
    resolve_submission_output,
)


def test_auto_run_directory_uses_device_second_timestamp_and_collision_suffix(
    tmp_path: Path,
) -> None:
    started_at = datetime(2026, 8, 3, 21, 30, 15)

    cpu = create_run_directory(tmp_path, "cpu", started_at=started_at)
    cpu_collision = create_run_directory(tmp_path, "cpu", started_at=started_at)
    cuda = create_run_directory(tmp_path, "cuda", started_at=started_at)

    assert cpu.name == "cpu_20260803_213015"
    assert cpu_collision.name == "cpu_20260803_213015_02"
    assert cuda.name == "cuda_20260803_213015"


def test_explicit_run_directory_is_exact_and_must_be_empty(tmp_path: Path) -> None:
    exact = tmp_path / "my-run"
    created = create_run_directory(tmp_path / "ignored", "cpu", output_dir=exact)
    assert created == exact

    (exact / "occupied.txt").write_text("occupied", encoding="utf-8")
    try:
        create_run_directory(tmp_path / "ignored", "cuda", output_dir=exact)
    except FileExistsError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("expected a non-empty explicit directory to fail")


def test_experiment_updates_preserve_note_and_user_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "cpu_20260803_213015"
    run_dir.mkdir()
    record_run_started(
        run_dir,
        device="cpu",
        mode="resume",
        note="切换损失函数",
        config_source="configs/score.yaml",
        started_at=datetime(2026, 8, 3, 21, 30, 15),
        resumed_from="old/best.pt",
    )
    experiment = run_dir / "experiment.md"
    initial = experiment.read_text(encoding="utf-8")
    assert "切换损失函数" in initial
    assert "运行中" in initial
    assert "续训" in initial

    experiment.write_text(initial + "\n用户补充：保留这行。\n", encoding="utf-8")
    record_run_finished(
        run_dir,
        status="成功",
        ended_at=datetime(2026, 8, 3, 22, 0, 0),
    )
    updated = experiment.read_text(encoding="utf-8")
    assert "切换损失函数" in updated
    assert "用户补充：保留这行。" in updated
    assert "成功" in updated
    assert "2026-08-03 22:00:00" in updated


def test_manually_edited_note_is_preserved_and_used_by_index(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = output_root / "cpu_20260803_213015"
    run_dir.mkdir(parents=True)
    record_run_started(
        run_dir,
        device="cpu",
        mode="fresh",
        note="原始摘要",
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )
    experiment = run_dir / "experiment.md"
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("原始摘要", "人工修改摘要", 1),
        encoding="utf-8",
    )

    record_run_finished(run_dir, status="成功")
    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert "人工修改摘要" in experiment.read_text(encoding="utf-8")
    assert "人工修改摘要" in index
    assert "原始摘要" not in index


def test_missing_note_is_rendered_as_pending(tmp_path: Path) -> None:
    run_dir = tmp_path / "cpu_20260803_213015"
    run_dir.mkdir()
    record_run_started(
        run_dir,
        device="cpu",
        mode="fresh",
        note=None,
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )
    assert "待补充" in (run_dir / "experiment.md").read_text(encoding="utf-8")


def test_experiment_and_index_distinguish_stage_from_training_mode(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    smoke = output_root / "cpu_20260803_213015"
    full_resume = output_root / "cpu_20260803_213016"
    smoke.mkdir(parents=True)
    full_resume.mkdir()
    record_run_started(
        smoke,
        device="cpu",
        mode="fresh",
        stage="smoke",
        note=None,
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )
    record_run_started(
        full_resume,
        device="cpu",
        mode="resume",
        stage="full",
        note=None,
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 16),
    )

    smoke_text = (smoke / "experiment.md").read_text(encoding="utf-8")
    resume_text = (full_resume / "experiment.md").read_text(encoding="utf-8")
    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert "训练类型：普通训练" in smoke_text
    assert "训练阶段：smoke" in smoke_text
    assert "训练类型：续训" in resume_text
    assert "训练阶段：full" in resume_text
    assert "smoke/普通训练" in index
    assert "full/续训" in index


def test_stage_defaults_follow_checkpoint_and_explicit_paths_win(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cuda_20260803_213015" / "best.pt"

    assert resolve_evaluation_output(checkpoint, None) == checkpoint.parent / "reproduced_metrics.json"
    assert resolve_submission_output(checkpoint, None) == checkpoint.parent / "submission"
    assert resolve_evaluation_output(checkpoint, tmp_path / "custom.json") == tmp_path / "custom.json"
    assert resolve_submission_output(checkpoint, tmp_path / "custom") == tmp_path / "custom"


def test_index_sorts_new_and_legacy_runs_and_marks_unknown_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    older = output_root / "score_optimized_cpu_full_20260725_010203"
    newer = output_root / "cuda_20260803_213015"
    unknown = output_root / "old-results"
    for directory in (older, newer, unknown):
        directory.mkdir(parents=True)
    (older / "metrics.json").write_text(
        json.dumps({"threshold": 0.7, "competition": {"score": 0.8}}),
        encoding="utf-8",
    )
    record_run_started(
        newer,
        device="cuda",
        mode="fresh",
        note="新实验",
        config_source="configs/new.yaml",
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )
    (output_root / "loose.log").write_text("historical", encoding="utf-8")

    index_path = rebuild_output_index(output_root)
    index = index_path.read_text(encoding="utf-8")

    assert index.index("cuda_20260803_213015") < index.index("score_optimized_cpu_full_20260725_010203")
    assert "新实验" in index
    assert "历史产物" in index
    assert "未知" in index
    assert "loose.log" in index


def test_index_links_an_explicit_submission_outside_run_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = output_root / "cpu_20260803_213015"
    run_dir.mkdir(parents=True)
    record_run_started(
        run_dir,
        device="cpu",
        mode="fresh",
        note="外部提交目录",
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )
    submission_dir = tmp_path / "export"
    submission_dir.mkdir()
    zip_path = submission_dir / "submission.zip"
    zip_path.touch()
    record_submission(run_dir, zip_path)

    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert zip_path.as_posix() in index


def test_legacy_stage_updates_remain_legacy_unknown(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    evaluation_run = output_root / "old-evaluation"
    submission_run = output_root / "old-submission"
    evaluation_run.mkdir(parents=True)
    submission_run.mkdir()
    evaluation_output = evaluation_run / "reproduced_metrics.json"
    evaluation_output.write_text("{}\n", encoding="utf-8")
    record_evaluation(evaluation_run, evaluation_output)
    submission_dir = submission_run / "submission"
    submission_dir.mkdir()
    zip_path = submission_dir / "submission.zip"
    zip_path.touch()
    record_submission(submission_run, zip_path)

    evaluation_text = (evaluation_run / "experiment.md").read_text(encoding="utf-8")
    submission_text = (submission_run / "experiment.md").read_text(encoding="utf-8")
    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert "状态：历史产物" in evaluation_text
    assert "训练类型：未知" in evaluation_text
    assert "训练阶段：未知" in evaluation_text
    assert "状态：历史产物" in submission_text
    assert "训练类型：未知" in submission_text
    assert "普通训练" not in index
    assert index.count("历史产物") >= 2


def test_index_uses_mtime_for_unknown_sort_and_collision_suffix_for_ties(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    unknown_old = output_root / "unknown-old"
    unknown_new = output_root / "unknown-new"
    base = output_root / "cpu_20260803_213015"
    suffix_03 = output_root / "cpu_20260803_213015_03"
    suffix_02 = output_root / "cpu_20260803_213015_02"
    suffix_100 = output_root / "cpu_20260803_213015_100"
    for directory in (unknown_old, unknown_new, base, suffix_03, suffix_02, suffix_100):
        directory.mkdir(parents=True)
    os.utime(unknown_old, (100, 100))
    os.utime(unknown_new, (200, 200))

    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert index.index("unknown-new") < index.index("unknown-old")
    assert (
        index.index(f"[{suffix_100.name}](")
        < index.index(f"[{suffix_03.name}](")
        < index.index(f"[{suffix_02.name}](")
        < index.index(f"[{base.name}](")
    )
    unknown_new_row = next(line for line in index.splitlines() if "unknown-new" in line)
    suffix_100_row = next(line for line in index.splitlines() if suffix_100.name in line)
    assert "| 未知 | 历史产物 |" in unknown_new_row
    assert "| 20260803_213015 |" in suffix_100_row


def test_index_tolerates_malformed_experiment_and_invalid_timestamp(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    malformed = output_root / "malformed"
    invalid_time = output_root / "invalid-time"
    invalid_name = output_root / "cpu_99999999_999999"
    malformed.mkdir(parents=True)
    invalid_time.mkdir()
    invalid_name.mkdir()
    (malformed / "experiment.md").write_bytes(b"\xff\xfe\x80")
    (invalid_time / "experiment.md").write_text(
        '<!-- HUAV-EXPERIMENT-META: {"started_at":"not-a-time"} -->\n',
        encoding="utf-8",
    )

    index = rebuild_output_index(output_root).read_text(encoding="utf-8")

    assert "malformed" in index
    assert "invalid-time" in index
    assert "cpu_99999999_999999" in index
    assert index.count("历史产物") >= 2


def test_index_sanitizes_multiline_pipe_note_without_changing_columns(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = output_root / "cpu_20260803_213015"
    run_dir.mkdir(parents=True)
    record_run_started(
        run_dir,
        device="cpu",
        mode="fresh",
        stage="full",
        note="第一行 | 管道\n第二行",
        config_source=None,
        started_at=datetime(2026, 8, 3, 21, 30, 15),
    )

    index = rebuild_output_index(output_root).read_text(encoding="utf-8")
    row = next(line for line in index.splitlines() if run_dir.name in line)
    metadata_line = next(
        line
        for line in (run_dir / "experiment.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("<!-- HUAV-EXPERIMENT-META:")
    )

    assert "第一行 \\| 管道<br>第二行" in row
    assert row.count(" | ") == 11
    assert "第一行" not in metadata_line


def test_concurrent_index_rebuilds_leave_one_complete_index_and_hide_lock(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    for name in ("cpu_20260803_213015", "cuda_20260803_213016"):
        (output_root / name).mkdir(parents=True)
    (output_root / ".huav-index.lock").touch()

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(lambda _: rebuild_output_index(output_root), range(2)))

    assert paths == [output_root / "INDEX.md", output_root / "INDEX.md"]
    index = (output_root / "INDEX.md").read_text(encoding="utf-8")
    assert index.count("# HUAV 输出索引") == 1
    assert index.count("cpu_20260803_213015") == 2
    assert index.count("cuda_20260803_213016") == 2
    assert ".huav-index.lock" not in index


def test_submission_stats_stream_txt_files(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "run"
    submission_dir = run_dir / "submission"
    submission_dir.mkdir(parents=True)
    txt_path = submission_dir / "val_000.txt"
    txt_path.write_text("1 2 0.0 1 1\n2 3 0.1 0 0\n", encoding="ascii")
    zip_path = submission_dir / "submission.zip"
    zip_path.touch()
    original_read_text = Path.read_text

    def reject_whole_txt_reads(path: Path, *args, **kwargs):
        if path.suffix == ".txt":
            raise AssertionError("submission TXT must be streamed")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_whole_txt_reads)

    record_submission(run_dir, zip_path)

    experiment = (run_dir / "experiment.md").read_text(encoding="utf-8")
    assert "总行数：2" in experiment
    assert "预测为 1：1" in experiment
