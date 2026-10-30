"""两套实验共用的源码冻结与独立上传包生成。"""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from traceformer.config import load_config
from traceformer.experiments.ablation import ComponentCoreSpec, LocalPoolingSpec
from traceformer.experiments.component_suite import read_json
from traceformer.training.engine import (
    _file_sha256,
    _serializable,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
)

REPO = Path(__file__).resolve().parents[3]
VARIANTS = ("S0", "S1", "S2", "L0", "L1", "L2")
BASELINE_FILES = ("best.pt", "config.yaml", "metrics.json", "history.json", "summary.json")
OPTIONAL_BASELINE_FILES = ("training_curves.png", "training_state.pt")

# 唯一上传启动文件在导入模型前设置线程与路径，之后执行包内同一 CLI。
BOOTSTRAP = '''"""冻结上传包的唯一启动入口。"""
import os
import sys
from pathlib import Path

bundle = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(bundle / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(bundle / "output/.cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from traceformer.cli.experiment import bundled_main

if __name__ == "__main__":
    bundled_main(bundle)
'''

REQUIREMENTS = """# 保留主机已有 CUDA PyTorch，不由此清单替换。
numpy>=1.24
scipy>=1.10
PyYAML>=6
matplotlib>=3.8
scikit-learn>=1.3
psutil>=5.9
"""

def data_manifest(data_root, test_root):
    identities = {}
    for split, parent, count in (("train", data_root, 99), ("val", data_root, 24), ("test", test_root, 24)):
        paths = sorted((Path(parent) / split).glob("*.npz"))
        if len(paths) != count:
            raise ValueError(f"{split}: expected {count} NPZ files, found {len(paths)}")
        identities[split] = {path.name: {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
                             for path in paths}
    return identities

def validate_baseline(root, config, identities):
    if root is None:
        return None
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"baseline folder missing: {root}")
    runs = [root] if (root / "best.pt").exists() else sorted((root / "training").glob("attempt_*"))
    runs = [run for run in runs if (run / "best.pt").is_file()]
    if not runs:
        raise ValueError("baseline folder contains no completed checkpoint")
    run = runs[-1]
    for name in BASELINE_FILES:
        if not (run / name).is_file():
            raise ValueError(f"baseline artifact missing: {name}")
    identity_file = root / "dataset_identity.json"
    if not identity_file.exists() or read_json(identity_file) != identities:
        raise ValueError("baseline data identity differs or lacks its full dataset identity")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(checkpoint, config, ablation=ComponentCoreSpec("M1_C4"))
    if checkpoint.get("training_seed") != 37:
        raise ValueError("baseline training seed differs from seed37")
    summary = read_json(run / "summary.json")
    if summary.get("mixed_precision") or summary.get("device") != "cuda" or summary.get("epochs") != 24:
        raise ValueError("baseline must have completed 24 CUDA FP32 epochs")
    history = read_json(run / "history.json")
    if [item.get("epoch") for item in history] != list(range(1, 25)):
        raise ValueError("baseline lacks all 24 completed epoch records")
    previous = load_config(run / "config.yaml")
    # FP32协议下AMP类型字段不参与计算；文件上限仍必须与完整训练一致。
    ignored = {"output_root", "mixed_precision_dtype"}
    old_training = {key: value for key, value in asdict(previous.training).items() if key not in ignored}
    new_training = {key: value for key, value in asdict(config.training).items() if key not in ignored}
    if old_training != new_training or previous.dataset.duration_ms != config.dataset.duration_ms:
        raise ValueError("baseline training protocol differs from this experiment")
    for section in ("loss", "evaluation", "frame_refinement", "inference"):
        if asdict(getattr(previous, section)) != asdict(getattr(config, section)):
            raise ValueError(f"baseline {section} protocol differs from this experiment")
    source = read_json(root / "training.json")
    digest = _file_sha256(run / "best.pt")
    if source.get("sha256") != digest:
        raise ValueError("baseline checkpoint differs from original training pointer")
    metrics = read_json(run / "metrics.json")
    if float(checkpoint["decision_threshold"]) != float(metrics["threshold"]):
        raise ValueError("baseline saved decision threshold differs from metrics")
    preserved = (*BASELINE_FILES, *(name for name in OPTIONAL_BASELINE_FILES if (run / name).is_file()))
    return root, run, {"checkpoint_sha256": digest, "canonical_variant": "S0",
                       "ablation": checkpoint["ablation"], "source_training": source,
                       "source_files": {name: _file_sha256(run / name) for name in preserved}}


def source_manifest(destination, experiment):
    """输出与解释器缓存不进入冻结源码身份。"""
    hashes = {p.relative_to(destination).as_posix(): _file_sha256(p)
              for p in sorted(destination.rglob("*")) if p.is_file()
              and p.relative_to(destination).parts[0] != "output"
              and "__pycache__" not in p.parts and p.suffix != ".pyc"
              and p.name != "package_manifest.json"}
    _write_json(destination / "package_manifest.json", {"schema": 1, "sha256": hashes,
                "experiment": experiment, "contains_training_data": False,
                "contains_trained_checkpoint": (destination / "baseline/best.pt").is_file()})


def build_package(experiment, destination, config_source, data_root, test_root, *,
                  baseline_root=None, references=None):
    """先核验全部输入，再创建目标；已有目录永不覆盖。"""
    if experiment not in {"mean_past_gated", "local_pooling_six"}:
        raise ValueError(f"unknown maintained experiment: {experiment}")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite upload directory: {destination}")
    identities = data_manifest(Path(data_root), Path(test_root))
    base = load_config(config_source)
    config = replace(base, dataset=replace(base.dataset, root="data"), training=replace(
        base.training, device="cuda", mixed_precision=False, seed=37, epochs=24, num_threads=8,
        output_root="output", max_train_files=None, max_val_files=None,
        save_training_state_interval=1,
    )).validate()
    baseline = validate_baseline(baseline_root, config, identities) if experiment == "local_pooling_six" else None
    if baseline_root is not None and experiment != "local_pooling_six":
        raise ValueError("only local_pooling_six accepts a reusable baseline")
    destination.mkdir(parents=True)
    shutil.copytree(REPO / "src/traceformer", destination / "src/traceformer",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "run.py").write_text(BOOTSTRAP, encoding="utf-8")
    (destination / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (destination / "config.yaml").write_text(yaml.safe_dump(_serializable(asdict(config)),
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_json(destination / "dataset_manifest.json", identities)
    _write_json(destination / "creation.json", {"config_source_sha256": _file_sha256(config_source)})
    (destination / "output").mkdir()
    (destination / "README.md").write_text(
        f"# {experiment}\n\n安装 requirements.txt 后执行：\n\n"
        "```bash\npython -u run.py --data-root /path/to/data --test-root /path/to/data\n"
        "python -u run.py --resume\n```\n\n"
        "阶段：all/preflight/train/predict；--smoke --device cpu 仅做微型诊断。\n"
        "结果在 output/full，诊断在 output/smoke；回传 output/results.zip。\n"
        "源码与数据身份冻结；失败现场保留，恢复必须使用原包和原配置。\n"
        + ("六组固定 CUDA FP32；耗时与显存仅记录，不设置相对预算门槛。\n"
           "查询分块默认8192、空间分块4096；实际值冻结后不能在恢复时修改。\n"
           if experiment == "local_pooling_six" else
           "M1+C4 默认 CUDA FP32；可显式选择 --precision bf16。\n"), encoding="utf-8")
    if experiment == "mean_past_gated":
        torch.manual_seed(config.training.seed)
        model = build_model(config, ablation=ComponentCoreSpec("M1_C4"))
        _write_json(destination / "ablation.json", ComponentCoreSpec("M1_C4").to_dict())
        _write_json(destination / "references.json", references or {})
        _write_json(destination / "model_summary.json", {"experiment": "M1_C4",
                    "tokenizer": type(model.tokenizer).__name__, "pooling_queries": 0,
                    "statistics": "independent concatenation", "layers": len(model.encoder.layers),
                    "directions": [-1], "learned_pool_gate": True,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "initialization": "random from seed37; no existing trained weights reused"})
    else:
        rows = []
        for name in VARIANTS:
            torch.manual_seed(37)
            spec = LocalPoolingSpec(variant=name)
            model = build_model(config, ablation=spec)
            rows.append({"variant": name, "ablation": spec.to_dict(),
                         "checkpoint_contract": (ComponentCoreSpec("M1_C4").to_dict()
                                                 if name == "S0" and baseline else spec.to_dict()),
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "tokenizer": type(model.tokenizer).__name__, "layers": len(model.encoder.layers)})
        _write_json(destination / "suite.json", {"variants": rows, "seed": 37, "epochs": 24,
                    "precision": "fp32", "resource_policy": "timing_only", "budget_enforced": False,
                    "timing_split": "test", "baseline_reused": baseline is not None})
    if baseline is not None:
        source_root, run, identity = baseline
        target = destination / "baseline"
        target.mkdir()
        for name in identity["source_files"]:
            shutil.copy2(run / name, target / name)
        for name in ("package_manifest.json", "environment.json", "experiment.json", "dataset_identity.json"):
            if (source_root / name).exists():
                shutil.copy2(source_root / name, target / ("source_" + name))
        _write_json(target / "identity.json", identity)
    source_manifest(destination, experiment)
    return destination
