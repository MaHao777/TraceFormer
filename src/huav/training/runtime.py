"""Runtime device preparation and CUDA compatibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

CUDA_INSTALL_HINT = (
    "RTX 50-series GPUs require a PyTorch build with CUDA 12.8 or newer; "
    "install a current cu128/cu130 wheel from https://pytorch.org/get-started/locally/."
)


@dataclass(frozen=True)
class PreparedDevice:
    device: torch.device
    metadata: dict[str, Any]


def _verify_cuda_kernel(device: torch.device) -> None:
    """Launch a real kernel so architecture mismatches fail before a long run."""
    values = torch.arange(4, dtype=torch.float32, device=device)
    result = torch.sum(values * values)
    torch.cuda.synchronize(device)
    if float(result.item()) != 14.0:
        raise RuntimeError("CUDA preflight kernel returned an unexpected result")


def prepare_device(name: str, *, num_threads: int) -> PreparedDevice:
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    torch.set_num_threads(num_threads)
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "device": name,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_kernel_verified": False,
    }
    device = torch.device(name)
    if name == "cpu":
        return PreparedDevice(device=device, metadata=metadata)
    if not cuda_available:
        raise RuntimeError(f"CUDA was requested but is not available. {CUDA_INSTALL_HINT}")

    try:
        gpu_name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        properties = torch.cuda.get_device_properties(device)
        architectures = list(torch.cuda.get_arch_list())
        _verify_cuda_kernel(device)
    except RuntimeError as error:
        capability = ""
        try:
            major, minor = torch.cuda.get_device_capability(device)
            capability = f" (sm_{major}{minor})"
        except RuntimeError:
            pass
        raise RuntimeError(
            f"CUDA kernel preflight failed{capability}: {error}. {CUDA_INSTALL_HINT}"
        ) from error

    metadata.update(
        {
            "gpu_name": gpu_name,
            "compute_capability": f"{major}.{minor}",
            "supported_architectures": architectures,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_kernel_verified": True,
        }
    )
    return PreparedDevice(device=device, metadata=metadata)


__all__ = ["CUDA_INSTALL_HINT", "PreparedDevice", "prepare_device"]
