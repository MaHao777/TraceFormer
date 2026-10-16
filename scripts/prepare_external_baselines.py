"""下载同任务官方基线，冻结来源与哈希；不运行上游训练代码。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPOSITORIES = {"PACT": "fulongcai/PACT", "EV-SpSegNet": "ChenYichen9527/EV-UAV"}
PACT_CHECKPOINT = (
    "https://github.com/fulongcai/PACT/releases/download/v1.0/checkpoint_improved.pt"
)
CHECKPOINTS = {
    "PACT": PACT_CHECKPOINT,
    "EV-SpSegNet": "https://drive.usercontent.google.com/download?"
    "id=1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq&export=download&confirm=t",
}


def download(url: str, target: Path) -> str:
    """流式落盘并记录 SHA256，失败的临时文件不冒充完整权重。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TraceFormer-reproduction"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            digest.update(block)
            stream.write(block)
    temporary.replace(target)
    return digest.hexdigest()


def extract_source(archive: Path, destination: Path) -> None:
    # 只允许归档文件写入指定源码目录，拒绝穿越路径及符号链接。
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        prefix = bundle.namelist()[0].split("/", 1)[0]
        for entry in bundle.infolist():
            relative = Path(entry.filename).relative_to(prefix)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("archive path escapes source directory")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("source archive contains a symlink")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def prepare(root: Path, names: list[str]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "upstream_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name in names:
        slug = REPOSITORIES[name]
        record = manifest.setdefault(name, {"repository": f"https://github.com/{slug}"})
        try:
            api = f"https://api.github.com/repos/{slug}/commits?per_page=1"
            metadata = root / "upstream" / f"{name}_commit.json"
            if not metadata.exists():
                download(api, metadata)
            commit = json.loads(metadata.read_text(encoding="utf-8"))[0]["sha"]
            archive = root / "upstream" / f"{name}-{commit[:12]}.zip"
            digest = (hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists()
                      else download(f"https://codeload.github.com/{slug}/zip/{commit}", archive))
            folder = root / "upstream" / f"{name}-{commit[:12]}"
            extract_source(archive, folder)
            record.update(commit=commit, archive_sha256=digest, source=str(folder.resolve()),
                          source_status="downloaded", evaluation_status="not_run")
            record.pop("source_error", None)
            record["requirements"] = "CUDA, spconv, compiled HAIS; native CPU runtime unavailable"
            print(f"{name}: source {commit} downloaded", flush=True)
        except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
            record.update(source_status="failed", source_error=str(error))
            print(f"{name}: source download failed: {error}", flush=True)
        # 下载官方权重与源码独立记录；缺失权重不能填写复现指标。
        if name in CHECKPOINTS:
            try:
                checkpoint = root / "weights" / f"{name}_checkpoint_improved.pt"
                digest = (hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists()
                          else download(CHECKPOINTS[name], checkpoint))
                with checkpoint.open("rb") as stream:
                    valid_header = stream.read(4).startswith((b"PK", b"\x80"))
                if not valid_header:
                    # 登录网页保留作失败证据，不能继续占用 .pt 权重路径。
                    rejected = checkpoint.with_suffix(".download.html")
                    checkpoint.replace(rejected)
                    record["rejected_download"] = str(rejected.resolve())
                    raise ValueError("download is not a PyTorch archive; login/HTML is not a weight")
                record.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                              checkpoint_url=CHECKPOINTS[name], checkpoint_status="downloaded")
                record.pop("checkpoint_error", None)
                print(f"{name}: checkpoint downloaded", flush=True)
            except (OSError, ValueError, urllib.error.URLError) as error:
                record.pop("checkpoint", None)
                record.pop("checkpoint_sha256", None)
                record.update(checkpoint_status="failed", checkpoint_error=str(error),
                              checkpoint_url=CHECKPOINTS[name])
                print(f"{name}: checkpoint download failed: {error}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--models", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    arguments = parser.parse_args()
    prepare(arguments.root.resolve(), arguments.models)
