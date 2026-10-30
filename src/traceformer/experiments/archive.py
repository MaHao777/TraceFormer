"""历史实验登记与隔离执行；匹配源码缺失时拒绝使用当前模型替代。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def read_catalog(repo):
    return json.loads((repo / "archive/experiments/catalog.json").read_text(encoding="utf-8"))


def archive_inventory(repo):
    catalog = read_catalog(repo)
    result = {}
    for name, entry in catalog["experiments"].items():
        runtime = catalog["runtimes"][entry["runtime"]]
        paths = [entry["entry"], runtime["source"] + "/src/traceformer/training/engine.py",
                 runtime["source"] + "/src/traceformer/__init__.py"]
        if entry.get("frozen_entry"):
            paths.append(runtime["source"] + "/" + entry["frozen_entry"])
        paths += [entry[k] for k in ("checkpoint", "config", "report") if entry.get(k)]
        missing = [path for path in paths if not (repo / path).is_file()]
        dependencies = [name for name in entry["dependencies"] if importlib.util.find_spec(name) is None]
        ready = entry.get("validated", False) and not missing and not dependencies
        result[name] = {**entry, "source_root": runtime["source"], "missing": missing,
                        "missing_dependencies": dependencies,
                        "status": "runnable" if ready else "incomplete"}
    return result


def run_archive(repo, archive_id, source_root=None, arguments=()):
    catalog = read_catalog(repo)
    if archive_id not in catalog["experiments"]:
        raise ValueError(f"unknown archived experiment: {archive_id}")
    entry = catalog["experiments"][archive_id]
    runtime = catalog["runtimes"][entry["runtime"]]
    source = Path(source_root).resolve() if source_root else (repo / runtime["source"]).resolve()
    if not all((source / name).is_file() for name in
               ("src/traceformer/__init__.py", "src/traceformer/training/engine.py")):
        raise FileNotFoundError(f"matching historical source is missing: {source}")
    # 允许显式搬迁同字节快照，拒绝把另一个版本放到 PYTHONPATH 上凑合加载。
    for name, digest in runtime["sha256"].items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical source identity differs: {name}")
    script = source / entry["frozen_entry"] if entry.get("frozen_entry") else repo / entry["entry"]
    if not entry.get("frozen_entry") and entry.get("entry_sha256"):
        if hashlib.sha256(script.read_bytes()).hexdigest() != entry["entry_sha256"]:
            raise ValueError("archived adapter identity differs")
    arguments = list(arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if "--help" not in arguments and "-h" not in arguments:
        state = archive_inventory(repo)[archive_id]
        missing = [p for p in state["missing"] if not p.startswith(runtime["source"] + "/")]
        if missing or state["missing_dependencies"] or not entry.get("validated", False):
            raise ValueError(f"archive is incomplete: {missing or state['missing_dependencies'] or entry.get('validation_note')}")
    environment = dict(os.environ, PYTHONPATH=str(source / "src"),
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    # 归档脚本所在目录提供同系列适配器；模型库只能来自上面核验过的快照。
    result = subprocess.run([sys.executable, str(script), *arguments], cwd=repo, env=environment)
    return result.returncode
