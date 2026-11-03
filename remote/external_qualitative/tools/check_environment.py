"""检查 Windows GPU 运行时、稀疏卷积前向和三份推理权重。"""

import argparse
import importlib.metadata
import platform
import sys
import traceback
from pathlib import Path

from event_io import sha256, write_json


def require_cuda(torch):
    # 区分 CPU 轮子和驱动初始化失败，保留真正的底层原因。
    if torch.version.cuda is None:
        raise RuntimeError(f"CPU-only PyTorch installed: {torch.__version__}. "
                           "Install the pinned +cu118 build; see README.md.")
    if not torch.cuda.is_available():
        try:
            torch.cuda.init()
        except Exception as error:
            raise RuntimeError(f"CUDA {torch.version.cuda} initialization failed: {error}") from error
        raise RuntimeError("CUDA GPU is unavailable; check driver and GPU visibility")


def check(output):
    root = Path(__file__).resolve().parents[1]
    report = dict(platform=platform.platform(), python=sys.version, status="checking")
    try:
        import torch

        report["torch"], report["cuda"] = torch.__version__, torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        for package in ["spconv-cu118", "cumm-cu118", "spconv-cu126", "cumm-cu126", "timm", "numpy"]:
            try:
                report[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        require_cuda(torch)
        from reference_backend import prepare_cuda_dlls

        prepare_cuda_dlls()
        import spconv.pytorch as spconv

        report["gpu"] = torch.cuda.get_device_name()
        report["capability"] = list(torch.cuda.get_device_capability())
        # 用真实稀疏卷积检查 DLL、GPU 内核和运行时，不以 import 成功代替前向。
        locations = torch.tensor([[0, 1, 1, 1], [0, 2, 1, 1]], dtype=torch.int32, device="cuda")
        sparse = spconv.SparseConvTensor(torch.ones(2, 4, device="cuda"), locations, [4, 4, 4], 1)
        layer = spconv.SubMConv3d(4, 4, 3, padding=1, bias=False).cuda().eval()
        with torch.no_grad():
            result = layer(sparse).features
        torch.cuda.synchronize()
        if result.shape != (2, 4) or not bool(torch.isfinite(result).all()):
            raise ValueError("Invalid sparse-convolution GPU smoke output")
        report["sparse_gpu_forward"] = "passed"
        weights = {}
        from predict import CHECKPOINTS

        for model, relative in CHECKPOINTS.items():
            path = root / relative
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint)
            weights[model] = dict(sha256=sha256(path), tensor_entries=len(state))
        report["weights"] = weights
        report["status"] = "passed"
    except Exception:
        report["status"], report["error"] = "failed", traceback.format_exc()
        raise
    finally:
        write_json(output, report)
    print(f"GPU environment check passed: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/environment_windows.json")
    args = parser.parse_args()
    check(args.output)
