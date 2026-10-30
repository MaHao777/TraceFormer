"""归档适配器记录实际执行的源码，避免把当前模型误记为旧模型来源。"""

from pathlib import Path

import traceformer


def source_file(name):
    if name.startswith("src/traceformer/"):
        return Path(traceformer.__file__).resolve().parent / name.removeprefix("src/traceformer/")
    if name.startswith("scripts/"):
        return Path(__file__).resolve().parent / Path(name).name
    return Path(__file__).resolve().parents[3] / name
