"""正式训练尚未开始时，一次性修复上传包的大Token注意力上限。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TARGET = "src/traceformer/experiments/component_core.py"
OLD = """            output, _ = self.center_attention(tokens[:, None], summaries, summaries,
                                              key_padding_mask=mask, need_weights=False)
            output = output[:, 0]"""
NEW = """            # CUDA带Dropout的高效注意力限制批量维度；按独立中心分批，不截断任何邻居。
            chunk_size = 32768 if tokens.is_cuda else len(tokens)
            chunks = []
            for start in range(0, len(tokens), chunk_size):
                end = min(start + chunk_size, len(tokens))
                context = summaries[start:end]
                chunk, _ = self.center_attention(
                    tokens[start:end, None], context, context,
                    key_padding_mask=mask[start:end], need_weights=False,
                )
                chunks.append(chunk[:, 0])
            output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def stopped_experiment(root):
    # 与正式入口使用同一文件锁，拒绝在训练或预检进程活跃时改源码。
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def apply_fix(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "package_manifest.json"
    code_path = directory / TARGET
    if read(directory / "model_summary.json").get("experiment") != "M1_C4":
        raise ValueError("this fix only applies to the M1_C4 upload package")
    root = directory / "output/full"
    with stopped_experiment(root):
        # 本次报错位于正式预检阶段，没有可恢复的正式训练权重。
        if (root / "training.json").exists() or list((root / "training").rglob("*.pt")):
            raise ValueError("formal training already started; preserve its snapshot and do not hot-patch it")
        if (root / "preflight.json").exists():
            raise ValueError("formal preflight already passed; this pre-training fix does not apply")
        manifest = read(manifest_path)
        for name, digest in manifest["sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"source differs from original package manifest: {name}")
        code = code_path.read_text(encoding="utf-8")
        if NEW in code and OLD not in code:
            print("already fixed; run: python -u run.py --resume")
            return
        if code.count(OLD) != 1:
            raise ValueError("expected attention block not found exactly once; no files were changed")
        context_path = root / "experiment.json"
        context = read(context_path) if context_path.exists() else None
        old_manifest_digest = sha256(manifest_path)
        if context and context["package_sha256"] != old_manifest_digest:
            raise ValueError("existing experiment package identity differs; no files were changed")
        backup = directory / "output" / ("attention_fix_backup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        backup.mkdir()
        for source in (code_path, manifest_path, context_path, root / "package_manifest.json", root / "status.json"):
            if source.is_file():
                target = backup / source.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        # 同步校验记录而非关闭校验；原文件、旧身份和失败状态均有备份。
        old_code_digest = sha256(code_path)
        code_path.write_text(code.replace(OLD, NEW, 1), encoding="utf-8")
        compile(code_path.read_text(encoding="utf-8"), str(code_path), "exec")
        record = directory / "ATTENTION_BATCH_FIX.json"
        write(record, {"fix": "CUDA center attention batch <=32768", "changed_model_file": TARGET,
              "old_code_sha256": old_code_digest, "new_code_sha256": sha256(code_path),
              "original_package_sha256": old_manifest_digest,
              "backup": str(backup.relative_to(directory)), "formal_training_started": False,
              "note": "Existing LOCAL_CHECKS describes the original package. Real CUDA validation is rerun by formal preflight."})
        manifest["sha256"][TARGET] = sha256(code_path)
        manifest["sha256"][record.name] = sha256(record)
        script = Path(__file__).resolve()
        if script.is_relative_to(directory) and script.relative_to(directory).parts[0] != "output":
            manifest["sha256"][script.relative_to(directory).as_posix()] = sha256(script)
        write(manifest_path, manifest)
        if context is not None:
            context["package_sha256"] = sha256(manifest_path)
            write(context_path, context)
            write(root / "package_manifest.json", manifest)
        print(f"fixed one model file; original files backed up to {backup}")
        print("Next: python -u run.py --resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd(), help="upload package directory")
    apply_fix(parser.parse_args().directory)
