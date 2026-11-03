"""从冻结的 PointEvent 官方文件提取模型权重，不反序列化训练对象。"""

import argparse
import hashlib
import pickletools
import tempfile
import zipfile
from pathlib import Path

OFFICIAL_SHA256 = "21e958c4225d4821577ffefea3cd277d6538bc418cf4fc6164211ce77ba1ca7f"


def extract(source, destination):
    import torch

    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Only the audited official PointEvent checkpoint is supported")
    with zipfile.ZipFile(source) as archive:
        key = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        data = archive.read(key)
        operations = list(pickletools.genops(data))
        # 此文件顶层依次为 epoch、state_dict、optimizer；只保留前两项。
        cut = next(pos for op, arg, pos in operations
                   if op.name == "BINUNICODE" and arg == "optimizer")
        assert data[cut - 1:cut] == b"b"  # state_dict 的 BUILD 已结束。
        prefix = data[:cut] + b"u."  # 关闭顶层字典并结束 pickle。
        allowed = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2",
                   "torch FloatStorage", "torch LongStorage"}
        globals_used = {arg for op, arg, _ in pickletools.genops(prefix) if op.name == "GLOBAL"}
        if not globals_used <= allowed:
            raise ValueError(f"Unexpected model globals: {globals_used - allowed}")
        with tempfile.TemporaryDirectory() as folder:
            stripped = Path(folder) / "state_only.pt"
            with zipfile.ZipFile(stripped, "w") as output:
                for info in archive.infolist():
                    output.writestr(info, prefix if info.filename == key else archive.read(info))
            # 张量存储原样复制；weights_only 拒绝任意训练对象。
            checkpoint = torch.load(stripped, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Invalid model state dictionary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)["state_dict"]
    if list(restored) != list(state) or not all(torch.equal(state[k], restored[k]) for k in state):
        raise ValueError("Export changed model tensors")
    print(f"Verified {len(state)} identical tensors -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract(args.source, args.destination)
