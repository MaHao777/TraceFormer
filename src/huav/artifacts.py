"""Run-directory naming, experiment records, and output indexing."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

AUTO_START = "<!-- HUAV-EXPERIMENT-AUTO-START -->"
AUTO_END = "<!-- HUAV-EXPERIMENT-AUTO-END -->"
META_PREFIX = "<!-- HUAV-EXPERIMENT-META: "
_TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}_\d{6})(?:_\d{2,})?$")
_COLLISION_SUFFIX_RE = re.compile(r"\d{8}_\d{6}(?:_(?P<suffix>\d{2,}))?$")
_INDEX_LOCK_NAME = ".huav-index.lock"
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_root_lock(root: Path):
    lock_path = root / _INDEX_LOCK_NAME
    lock_key = str(root.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_key, threading.Lock())
    with thread_lock, lock_path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def create_run_directory(
    output_root: str | Path,
    device_name: str,
    *,
    started_at: datetime | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Create an exact or collision-safe ``<device>_<timestamp>`` run directory."""
    if output_dir is not None:
        run_dir = Path(output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = f"{device_name}_{timestamp}"
    suffix = 1
    while True:
        name = base_name if suffix == 1 else f"{base_name}_{suffix:02d}"
        run_dir = root / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return run_dir


def resolve_evaluation_output(
    checkpoint_path: str | Path, output_path: str | Path | None
) -> Path:
    if output_path is not None:
        return Path(output_path)
    return Path(checkpoint_path).parent / "reproduced_metrics.json"


def resolve_submission_output(
    checkpoint_path: str | Path, output_dir: str | Path | None
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(checkpoint_path).parent / "submission"


def _format_time(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else None


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "未知", "dirty": "未知", "changed_paths": []}
    changed_paths = [line[3:].strip() for line in changed if len(line) >= 4]
    return {
        "commit": commit,
        "dirty": bool(changed),
        "changed_paths": changed_paths,
    }


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    metadata: dict[str, Any] = {}
    for line in content.splitlines():
        if line.startswith(META_PREFIX) and line.endswith(" -->"):
            try:
                encoded = line[len(META_PREFIX) : -4]
                if encoded.startswith("b64:"):
                    payload = base64.b64decode(encoded[4:], validate=True).decode("utf-8")
                    metadata = json.loads(payload)
                else:
                    metadata = json.loads(encoded)
            except (UnicodeError, ValueError):
                return {}
            break
    note_match = re.search(
        rf"^## 本轮主要修改\s*\n+(.*?)(?=\n(?:## |{re.escape(AUTO_START)}))",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    if metadata and note_match:
        note = note_match.group(1).strip()
        if note:
            metadata["note"] = note
    return metadata


def _collect_artifacts(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    summary = _read_json(run_dir / "summary.json")
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_json(run_dir / "history.json")
    manifest = _read_json(run_dir / "dataset_manifest.json")
    if isinstance(summary, dict):
        result["summary"] = summary
        if "source_epoch" in summary:
            result["source_epoch"] = summary["source_epoch"]
    if isinstance(metrics, dict):
        result["metrics"] = metrics
    if isinstance(history, list) and history:
        records = [item for item in history if isinstance(item, dict)]
        if records:
            result["epochs_completed"] = len(records)
            scored = [
                item
                for item in records
                if isinstance(item.get("validation"), dict)
                and isinstance(item["validation"].get("competition"), dict)
                and isinstance(item["validation"]["competition"].get("score"), (int, float))
            ]
            if scored:
                best = max(scored, key=lambda item: item["validation"]["competition"]["score"])
                result["best_epoch"] = best.get("epoch")
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), list):
        train_count = 0
        validation_count = 0
        for item in manifest["files"]:
            split = str(item.get("split", "")).lower() if isinstance(item, dict) else ""
            if "train" in split:
                train_count += 1
            elif "val" in split:
                validation_count += 1
        result["train_files"] = train_count
        result["validation_files"] = validation_count
    saved = run_dir / "config.yaml"
    if saved.is_file():
        result["saved_config"] = str(saved)
    return result


def _show(value: Any, default: str = "未知") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _render_auto(metadata: dict[str, Any]) -> str:
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    competition = (
        metrics.get("competition") if isinstance(metrics.get("competition"), dict) else {}
    )
    event = metrics.get("event") if isinstance(metrics.get("event"), dict) else {}
    git = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
    changed_paths = git.get("changed_paths") if isinstance(git.get("changed_paths"), list) else []
    mode = {
        "fresh": "普通训练",
        "resume": "续训",
    }.get(metadata.get("mode"), "未知")
    lines = [
        AUTO_START,
        META_PREFIX
        + "b64:"
        + base64.b64encode(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        + " -->",
        "## 自动记录",
        "",
        f"- 状态：{_show(metadata.get('status'))}",
        f"- 开始时间：{_show(metadata.get('started_at'))}",
        f"- 结束时间：{_show(metadata.get('ended_at'), '未结束')}",
        f"- 设备：{_show(metadata.get('device'))}",
        f"- 训练类型：{mode}",
        f"- 训练阶段：{_show(metadata.get('stage'))}",
        f"- 配置来源：{_show(metadata.get('config_source'))}",
        f"- 保存配置：{_show(metadata.get('saved_config'))}",
        f"- 续训来源：{_show(metadata.get('resumed_from'), '不适用')}",
        f"- 续训源 epoch：{_show(metadata.get('source_epoch'), '不适用')}",
        f"- Git commit：{_show(git.get('commit'))}",
        f"- 工作区有未提交修改：{_show(git.get('dirty'))}",
        f"- 变更路径：{', '.join(map(str, changed_paths)) if changed_paths else '无'}",
        f"- 训练/验证文件数：{_show(metadata.get('train_files'))}/{_show(metadata.get('validation_files'))}",
        f"- 已完成 epoch：{_show(metadata.get('epochs_completed'))}",
        f"- 最佳 epoch：{_show(metadata.get('best_epoch'))}",
        f"- 最佳阈值：{_show(metrics.get('threshold'))}",
        "",
        "### Competition 指标",
        "",
        f"- Score：{_show(competition.get('score'))}",
        f"- IoU：{_show(competition.get('iou'))}",
        f"- SegAcc：{_show(competition.get('seg_acc'))}",
        f"- Pd：{_show(competition.get('pd'))}",
        f"- Fa：{_show(competition.get('fa'))}",
        "",
        "### 事件级指标",
        "",
        f"- Pd：{_show(event.get('pd'))}",
        f"- Fa：{_show(event.get('fa'))}",
        f"- IoU：{_show(event.get('iou'))}",
        f"- Acc：{_show(event.get('acc'))}",
        "",
        "### 独立复评与提交",
        "",
        f"- 独立复评：{_show(metadata.get('reevaluation'), '未运行')}",
        f"- Submission ZIP：{_show(metadata.get('submission_zip'), '未生成')}",
        f"- TXT 文件数：{_show(metadata.get('submission_txt_files'))}",
        f"- 总行数：{_show(metadata.get('submission_rows'))}",
        f"- 预测为 1：{_show(metadata.get('submission_positives'))}",
    ]
    if metadata.get("failure"):
        lines.extend(("", f"- 失败原因：{metadata['failure']}"))
    lines.extend(("", AUTO_END))
    return "\n".join(lines)


def _write_experiment(run_dir: Path, metadata: dict[str, Any], *, note: str | None = None) -> Path:
    path = run_dir / "experiment.md"
    auto = _render_auto(metadata)
    if path.is_file():
        current = path.read_text(encoding="utf-8")
        if AUTO_START in current and AUTO_END in current:
            prefix, remainder = current.split(AUTO_START, 1)
            _, suffix = remainder.split(AUTO_END, 1)
            content = prefix.rstrip() + "\n\n" + auto + suffix
        else:
            content = current.rstrip() + "\n\n" + auto + "\n"
    else:
        initial_note = note or metadata.get("note") or "待补充"
        content = (
            "# Experiment\n\n"
            "## 本轮主要修改\n\n"
            f"{initial_note}\n\n"
            f"{auto}\n\n"
            "## 补充说明\n\n"
            "可在此处添加人工记录；自动更新不会覆盖本区域。\n"
        )
    _atomic_write_text(path, content)
    return path


def record_run_started(
    run_dir: str | Path,
    *,
    device: str,
    mode: str,
    stage: str | None = None,
    note: str | None,
    config_source: str | Path | None,
    started_at: datetime | None = None,
    resumed_from: str | Path | None = None,
    source_epoch: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    metadata = {
        "status": "运行中",
        "started_at": _format_time(started_at or datetime.now()),
        "ended_at": None,
        "device": device,
        "mode": mode,
        "stage": stage,
        "note": note or "待补充",
        "config_source": str(config_source) if config_source is not None else None,
        "saved_config": str(run_dir / "config.yaml"),
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "source_epoch": source_epoch,
        "git": _git_state(),
    }
    return _write_experiment(run_dir, metadata, note=note)


def _legacy_metadata() -> dict[str, Any]:
    return {
        "status": "历史产物",
        "device": None,
        "mode": "unknown",
        "stage": None,
        "note": "未知",
        "legacy": True,
    }


def record_run_finished(
    run_dir: str | Path,
    *,
    status: str,
    ended_at: datetime | None = None,
    failure: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    metadata = _load_metadata(run_dir / "experiment.md")
    metadata.update(
        status=status,
        ended_at=_format_time(ended_at or datetime.now()),
    )
    if failure:
        metadata["failure"] = failure
    metadata = _collect_artifacts(run_dir, metadata)
    return _write_experiment(run_dir, metadata)


def record_evaluation(run_dir: str | Path, output_path: str | Path) -> Path:
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        state = "已完成（缺少训练指标，无法比较）"
    elif metrics_path.read_bytes() == output_path.read_bytes():
        state = "完全一致"
    else:
        state = "不一致"
    experiment_path = run_dir / "experiment.md"
    metadata = _load_metadata(experiment_path) if experiment_path.is_file() else _legacy_metadata()
    metadata = _collect_artifacts(run_dir, metadata)
    metadata["reevaluation"] = state
    metadata["reevaluation_output"] = str(output_path)
    return _write_experiment(run_dir, metadata)


def record_submission(run_dir: str | Path, zip_path: str | Path) -> Path:
    run_dir = Path(run_dir)
    zip_path = Path(zip_path)
    txt_files = sorted(path for path in zip_path.parent.glob("*.txt") if path.is_file())
    rows = 0
    positives = 0
    for path in txt_files:
        with path.open("r", encoding="ascii") as stream:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                if line.split()[-1] == "1":
                    positives += 1
    experiment_path = run_dir / "experiment.md"
    metadata = _load_metadata(experiment_path) if experiment_path.is_file() else _legacy_metadata()
    metadata = _collect_artifacts(run_dir, metadata)
    metadata.update(
        submission_zip=str(zip_path),
        submission_txt_files=len(txt_files),
        submission_rows=rows,
        submission_positives=positives,
    )
    return _write_experiment(run_dir, metadata)


def _directory_timestamp(path: Path, metadata: dict[str, Any]) -> str:
    started = metadata.get("started_at")
    if isinstance(started, str):
        try:
            return datetime.strptime(started, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S")
        except ValueError:
            pass
    match = _TIMESTAMP_RE.search(path.name)
    if not match:
        return ""
    stamp = match.group("stamp")
    try:
        datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return ""
    return stamp


def _legacy_device(name: str) -> str:
    lowered = name.lower()
    if "cuda" in lowered:
        return "cuda"
    if "cpu" in lowered:
        return "cpu"
    return "未知"


def _markdown_cell(value: Any) -> str:
    return _show(value).replace("\r\n", "\n").replace("\r", "\n").replace("|", "\\|").replace("\n", "<br>")


def _markdown_link(label: str, target: str) -> str:
    safe_label = _markdown_cell(label).replace("[", "\\[").replace("]", "\\]")
    safe_target = quote(target.replace("\\", "/"), safe="/:._-")
    return f"[{safe_label}]({safe_target})"


def _rebuild_output_index_unlocked(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[tuple[float, int], str]] = []
    for directory in (path for path in root.iterdir() if path.is_dir()):
        metadata = _load_metadata(directory / "experiment.md")
        is_managed = bool(metadata) and not metadata.get("legacy", False)
        metadata = _collect_artifacts(directory, metadata)
        metrics = _read_json(directory / "metrics.json")
        if not isinstance(metrics, dict):
            metrics = {}
        competition = metrics.get("competition") if isinstance(metrics.get("competition"), dict) else {}
        summary = _read_json(directory / "summary.json")
        if not isinstance(summary, dict):
            summary = {}
        device = metadata.get("device") or summary.get("device") or _legacy_device(directory.name)
        timestamp = _directory_timestamp(directory, metadata)
        mode_label = {
            "fresh": "普通训练",
            "resume": "续训",
        }.get(metadata.get("mode"), "未知")
        stage = metadata.get("stage") if is_managed else None
        mode = f"{stage}/{mode_label}" if stage else ("未知" if metadata else "历史产物")
        status = metadata.get("status") if is_managed else "历史产物"
        note = metadata.get("note") if is_managed else "未知"
        config = directory / "config.yaml"
        config_text = (
            _markdown_link(config.name, f"{directory.name}/config.yaml")
            if config.is_file()
            else "未知"
        )
        recorded_zip = metadata.get("submission_zip")
        zip_candidates: list[Path] = []
        if isinstance(recorded_zip, str):
            zip_candidates.append(Path(recorded_zip))
        zip_candidates.extend(
            (directory / "submission" / "submission.zip", directory / "submission.zip")
        )
        zip_path = next((path for path in zip_candidates if path.is_file()), None)
        if zip_path is None:
            submission = "未生成"
        else:
            try:
                zip_target = zip_path.relative_to(root).as_posix()
            except ValueError:
                zip_target = str(zip_path)
            submission = _markdown_link("submission.zip", zip_target)
        reevaluation = metadata.get("reevaluation", "未知" if not is_managed else "未运行")
        best_epoch = metadata.get("best_epoch", summary.get("best_epoch", "未知"))
        threshold = metrics.get("threshold", "未知")
        score = competition.get("score", summary.get("best_score", "未知"))
        row = (
            f"| {_markdown_link(directory.name, directory.name + '/')} | "
            f"{_markdown_cell(device)} | {_markdown_cell(timestamp or '未知')} | "
            f"{_markdown_cell(mode)} | {_markdown_cell(status)} | {_markdown_cell(note)} | "
            f"{config_text} | {_markdown_cell(best_epoch)} | {_markdown_cell(threshold)} | "
            f"{_markdown_cell(score)} | {_markdown_cell(reevaluation)} | {submission} |"
        )
        if timestamp:
            sort_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").timestamp()
        else:
            sort_time = directory.stat().st_mtime
        suffix_match = _COLLISION_SUFFIX_RE.search(directory.name)
        collision_suffix = (
            int(suffix_match.group("suffix") or 1) if suffix_match else 0
        )
        entries.append(((sort_time, collision_suffix), row))
    entries.sort(key=lambda item: item[0], reverse=True)
    lines = [
        "# HUAV 输出索引",
        "",
        "| 目录 | 设备 | 启动时间 | 类型 | 状态 | 主要修改 | 配置 | 最佳 epoch | 阈值 | Competition Score | 独立复评 | 提交包 |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
        *(row for _, row in entries),
    ]
    loose_files = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name not in {"INDEX.md", _INDEX_LOCK_NAME}
    )
    if loose_files:
        lines.extend(("", "## 根目录历史文件", ""))
        lines.extend(f"- {_markdown_link(name, name)}" for name in loose_files)
    index_path = root / "INDEX.md"
    _atomic_write_text(index_path, "\n".join(lines) + "\n")
    return index_path


def rebuild_output_index(output_root: str | Path = "outputs") -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with _output_root_lock(root):
        return _rebuild_output_index_unlocked(root)


__all__ = [
    "create_run_directory",
    "rebuild_output_index",
    "record_evaluation",
    "record_run_finished",
    "record_run_started",
    "record_submission",
    "resolve_evaluation_output",
    "resolve_submission_output",
]
