"""用标准库验证上传文件哈希，不要求 NumPy、CUDA 或 Linux 命令。"""

import hashlib
from pathlib import Path


def verify(root):
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Invalid manifest path: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1048576), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"File hash mismatch: {relative}")
        count += 1
    print(f"Verified {count} files")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
