from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_prepare_cpu_device_reports_reproducible_runtime_metadata() -> None:
    from huav.training.runtime import prepare_device

    runtime = prepare_device("cpu", num_threads=1)

    assert runtime.device == torch.device("cpu")
    assert runtime.metadata["device"] == "cpu"
    assert runtime.metadata["torch_version"] == torch.__version__
    assert runtime.metadata["cuda_available"] == torch.cuda.is_available()
    assert runtime.metadata["cuda_kernel_verified"] is False


def test_prepare_cuda_device_rejects_an_unavailable_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huav.training.runtime import prepare_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA 12.8"):
        prepare_device("cuda", num_threads=1)


def test_prepare_cuda_device_reports_blackwell_and_verifies_a_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huav.training import runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_120"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=32_607 * 1024 * 1024),
    )
    verified: list[torch.device] = []
    monkeypatch.setattr(runtime, "_verify_cuda_kernel", verified.append)

    prepared = runtime.prepare_device("cuda", num_threads=1)

    assert prepared.device == torch.device("cuda")
    assert verified == [torch.device("cuda")]
    assert prepared.metadata["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert prepared.metadata["compute_capability"] == "12.0"
    assert prepared.metadata["supported_architectures"] == ["sm_80", "sm_120"]
    assert prepared.metadata["cuda_kernel_verified"] is True


def test_prepare_cuda_device_translates_kernel_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huav.training import runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=32_607 * 1024 * 1024),
    )

    def fail(_: torch.device) -> None:
        raise RuntimeError("no kernel image is available for execution")

    monkeypatch.setattr(runtime, "_verify_cuda_kernel", fail)

    with pytest.raises(RuntimeError, match=r"sm_120.*CUDA 12\.8"):
        runtime.prepare_device("cuda", num_threads=1)
