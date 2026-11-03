"""按限定的官方文件安装内存导入适配，保留源码、网络参数及权重键。"""

import ast
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

DLL_HANDLES = []


def prepare_cuda_dlls():
    if os.name != "nt":
        return
    import torch

    directories = [Path(torch.__file__).resolve().parent / "lib"]
    for package in ["nvidia.cuda_runtime", "nvidia.cuda_nvrtc"]:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            continue
        if spec and spec.submodule_search_locations:
            directories.extend(Path(path) / "bin" for path in spec.submodule_search_locations)
    # 保留目录句柄，避免 CUDA DLL 搜索路径随临时对象释放。
    for directory in directories:
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))


class ReferenceLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        from windows_ops import selective_scan

        tree = ast.parse(self.get_data(self.path).decode("utf-8"), filename=self.path)
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                # 稀疏网络中的 HAIS import 未被 forward 使用；体素化由包装器完成。
                aliases = [alias for alias in node.names
                           if alias.name not in {"HAIS_OP", "selective_scan_cuda"}]
                for alias in aliases:
                    if alias.name == "torch_scatter":
                        alias.name, alias.asname = "windows_ops", "torch_scatter"
                if not aliases:
                    continue
                node.names = aliases
            if isinstance(node, ast.ImportFrom) and node.module in {
                "causal_conv1d", "mamba_ssm.modules.mamba_simple"
            }:
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                "SelectiveScanFn", "selective_scan_fn", "MambaLayer"
            }:
                continue
            nodes.append(node)
        tree.body = nodes
        # 只替换扩展边界；SimpleAwareStateSpaceModel 本体仍执行官方定义。
        module.__dict__["selective_scan_fn"] = selective_scan
        exec(compile(ast.fix_missing_locations(tree), self.path, "exec"), module.__dict__)


class ReferenceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source, model):
        if model == "pointevent":
            self.files = {"models.PointEvent.PointEvent": source / "models/PointEvent/PointEvent.py",
                          "models.PointEvent.mssm": source / "models/PointEvent/mssm.py"}
        elif model == "ev_spsegnet":
            self.files = {"model.evspsegnet": source / "model/evspsegnet.py",
                          "model.basemodel": source / "model/basemodel.py"}
        else:
            self.files = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.files:
            source = str(self.files[fullname])
            return importlib.util.spec_from_file_location(
                fullname, source, loader=ReferenceLoader(fullname, source))
        return None


def install(source, model):
    prepare_cuda_dlls()
    finder = ReferenceFinder(source, model)
    sys.meta_path.insert(0, finder)
    return finder
