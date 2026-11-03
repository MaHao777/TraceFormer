"""按已验证的 PACT 构建方式编译当前模型的 HAIS 算子。"""

import hashlib
import json
import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path.cwd()
if root.name != "hais_ops" or not (root / "src/hais_ops_api.cpp").is_file():
    raise RuntimeError("Run from models/<model>/lib/hais_ops")

# EV-SpSegNet 这两个头文件与 PACT 仅差废弃的 THC 引用；不改算子实现。
changes = []
for relative in ["src/bfs_cluster/bfs_cluster.h",
                 "src/hierarchical_aggregation/hierarchical_aggregation.h"]:
    path = root / relative
    content = path.read_bytes()
    revised = content.replace(b"#include <THC/THC.h>", b"")
    if revised != content:
        path.write_bytes(revised)
        changes.append(dict(file=relative, before=hashlib.sha256(content).hexdigest(),
                            after=hashlib.sha256(revised).hexdigest()))
if changes:
    (root / "compatibility_patch.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")

# ABI 必须输出 0/1；上游 PACT 把 Python bool 直接传入宏。
abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
arches = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")
nvcc = ["-O0", "-std=c++17", abi, "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
        "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__"]
for arch in arches.split(";"):
    code = arch.replace(".", "")
    nvcc += [f"-gencode=arch=compute_{code},code=sm_{code}"]
setup(
    name="HAIS_OP",
    ext_modules=[CUDAExtension(
        name="HAIS_OP",
        sources=["src/hais_ops_api.cpp", "src/hais_ops.cpp", "src/cuda.cu"],
        extra_compile_args={"cxx": ["-O0", "-std=c++17", abi], "nvcc": nvcc},
    )],
    cmdclass={"build_ext": BuildExtension},
)
