"""与 PyTorch 共享 CUDA 内存及 stream，不建立独立设备缓存。"""

from __future__ import annotations

import numpy as np
import torch


def add_backend_arguments(parser, *, trees: bool = False):
    """各入口共用后端参数，默认 CPU 保持既有开发流程。"""
    if trees:
        parser.add_argument("--prediction-backend", choices=("cpu", "cuda"), default="cpu",
                            help="事件树推理后端")
        parser.add_argument("--feature-backend", choices=("cpu", "cuda"), default="cpu",
                            help="事件上下文和特征展开后端；cuda 要求 CUDA 树")


def validate_backends(*, prediction_backend: str = "cpu", feature_backend: str = "cpu"):
    for backend in (prediction_backend, feature_backend):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("inference backend must be cpu or cuda")
    if feature_backend == "cuda" and prediction_backend != "cuda":
        raise ValueError("CUDA features require --prediction-backend cuda")
    if "cuda" in (prediction_backend, feature_backend):
        require_cuda()


def require_cuda(device: str | torch.device = "cuda"):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires a CUDA-enabled PyTorch and NVIDIA GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        from numba import cuda
    except ImportError as error:
        raise RuntimeError('GPU inference requires: pip install "numba-cuda[cu12]"') from error
    if not cuda.is_available():
        raise RuntimeError('Numba CUDA unavailable; install "numba-cuda[cu12]" and check the driver')
    return device, cuda


class _TensorView:
    """零拷贝导出 Torch Tensor；由调用方保证 Tensor 生命周期。"""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        types = {torch.int64: "<i8", torch.int32: "<i4", torch.float32: "<f4",
                 torch.float64: "<f8", torch.uint8: "|u1"}
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape), "strides": tuple(s * tensor.element_size() for s in tensor.stride()),
            "typestr": types[tensor.dtype], "data": (tensor.data_ptr(), False), "version": 3,
        }


def launch(kernel, count: int, device: torch.device, *arguments, threads: int = 128):
    """内核使用 Torch 当前 stream；结果保持在 Torch 分配的显存。"""
    if not count:
        return
    _, cuda = require_cuda(device)
    with torch.cuda.device(device), cuda.gpus[torch.cuda.current_device()]:
        stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        views = [cuda.as_cuda_array(_TensorView(value), sync=False)
                 if isinstance(value, torch.Tensor) else value for value in arguments]
        kernel[(count + threads - 1) // threads, threads, stream](*views)
        # 防止异步内核完成前，输入临时数组被其他 Torch stream 回收。
        for value in arguments:
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream(device))


def upload(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values)).to(device)
