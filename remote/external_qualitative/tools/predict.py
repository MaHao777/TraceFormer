"""使用官方网络与预处理，固定阈值 0.9 导出完整逐事件预测。"""

import argparse
import importlib
import json
import os
import random
import runpy
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from event_io import (
    THRESHOLD,
    audit_input,
    save_input,
    save_prediction,
    selected_files,
    sha256,
    write_json,
)

BUNDLE = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "ev_spsegnet": "models/ev_spsegnet/checkpoints/best_iou_seed37.pt",
    "pact": "models/pact/checkpoints/checkpoint_improved.pt",
    "pointevent": "models/pointevent/exp/EV-UAV/model/model_inference.pth",
}


def run_sparse(args, source, checkpoint, output, records):
    import torch

    # 上游配置在导入时解析 argv；只把官方 YAML 交给它。
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "lib/hais_ops"))
    sys.argv = [str(source / "test.py"), "--config", str(source / "configs/evisseg_evuav.yaml")]
    cfg = importlib.import_module("configs.configs").cfg
    cfg.root, cfg.model_path, cfg.batch_size = str(args.data_root.resolve()), str(checkpoint), 1
    reference = getattr(args, "backend", "official") == "torch-reference"
    if not reference:
        dataset = importlib.import_module("dataset.ev_uav").EvUAV(cfg, mode=args.split)
        dataset.file_list = [item["file"] for item in records]
    module, constructor = (("model.PACT", "PACT") if args.model == "pact"
                           else ("model.evspsegnet", "evspsegnet"))
    model = getattr(importlib.import_module(module), constructor)(cfg).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        for index, record in enumerate(records):
            if reference:
                from windows_ops import sparse_batch

                batch = sparse_batch(args.data_root.resolve() / args.split / record["file"])
            else:
                native = dataset[index]
                batch = dataset.custom_collate([native])
                if not torch.equal(batch["locs"][:, 1:], torch.as_tensor(native["ev_loc"]).long()):
                    raise ValueError("Official collate changed event order")
            prediction, voxels = model(batch["voxel_ev"])[:2]
            if not torch.equal(voxels.indices, batch["voxel_ev"].indices):
                raise ValueError("Decoder voxel order changed; explicit remapping is required")
            probability = prediction[batch["p2v_map"].long().cuda()].reshape(-1).cpu().numpy()
            save_prediction(output, record["file"], probability, record)
            print(f"{args.model}: {record['file']} ({record['events']} events)", flush=True)


def stage_input(path, target):
    # Windows 硬链接无需管理员权限；跨盘时复制，Linux 优先符号链接。
    try:
        if os.name == "nt":
            os.link(path, target)
        else:
            target.symlink_to(path)
    except OSError:
        shutil.copy2(path, target)


def run_point(args, source, checkpoint, output, records, files):
    # 临时目录仅链接选中的完整序列；不裁剪上下文、不重新归一化。
    with tempfile.TemporaryDirectory(prefix="pointevent_input_") as folder:
        # 官方结果查找只剥离 test+ 前缀，验证集也用临时 test 路径承载。
        subset = Path(folder) / "test"
        subset.mkdir()
        for path in files:
            stage_input(path, subset / path.name)
        native_output = output / "native"
        worker_count = 0 if os.name == "nt" else 1
        options = [f"data_root={folder}", "data.test.split=test",
                   f"weight={checkpoint}", f"save_path={native_output}",
                   "test_save_txt=False", "test_save_pic=False", "test_save_npz=True",
                   f"num_worker={worker_count}", "seed=1024"]
        sys.path.insert(0, str(source))
        # 原生 Windows 不派生数据进程；修正上游 seed*worker_count 的零种子。
        if worker_count == 0 and getattr(args, "backend", "official") == "torch-reference":
            defaults = importlib.import_module("engines.defaults")
            original_setup = defaults.default_setup

            def setup_with_fixed_seed(cfg):
                cfg = original_setup(cfg)
                importlib.import_module("utils.env").set_seed(1024)
                return cfg

            defaults.default_setup = setup_with_fixed_seed
        sys.argv = [str(source / "tools/test.py"), "--config-file",
                    str(source / "configs/EVUAV/evuav.py"), "--options", *options]
        os.chdir(source)
        runpy.run_path(str(source / "tools/test.py"), run_name="__main__")
        for path, record in zip(files, records):
            with np.load(native_output / "npz" / path.name, allow_pickle=False) as result:
                with np.load(path, allow_pickle=False) as original:
                    if not np.array_equal(result["ev"], original["ev"]):
                        raise ValueError("PointEvent raw-event order differs from original input")
                    if not np.array_equal(result["gt"], original["evs_norm"][:, 4]):
                        raise ValueError("PointEvent inverse mapping did not restore original labels")
                expected = (result["target_scores"] >= THRESHOLD).astype(np.uint8)
                if not np.array_equal(result["pred"], expected):
                    raise ValueError("PointEvent prediction differs from official threshold")
                save_prediction(output, path.name, result["target_scores"], record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="含 test/ 或 val/ 的目录")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--sequences", nargs="+", help="序列名；省略则处理该 split 全部文件")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=BUNDLE / "results")
    parser.add_argument("--audit-only", action="store_true", help="仅核对输入，不要求 GPU/权重")
    parser.add_argument("--backend", choices=["official", "torch-reference"],
                        default="torch-reference" if os.name == "nt" else "official",
                        help="Windows 默认使用 PyTorch 参考算子，无需 HAIS/Mamba 编译")
    args = parser.parse_args()
    files = selected_files(args.data_root, args.split, args.sequences)
    output_root = args.output_root.resolve()
    output = output_root / args.model
    if (output / "report.json").exists():
        raise FileExistsError(f"Existing run: {output}; choose a new --output-root")
    checkpoint = (args.checkpoint or BUNDLE / CHECKPOINTS[args.model]).resolve()
    records = []
    for path in files:
        record, raw, gt = audit_input(path)
        records.append(record)
        if not args.audit_only:
            save_input(output_root, path, record, raw, gt)
    report = dict(model=args.model, status="audit_only" if args.audit_only else "running",
                  split=args.split, threshold=THRESHOLD, seed=1024, batch_size=1,
                  backend=args.backend,
                  sequences=records, checkpoint=str(checkpoint),
                  checkpoint_available=checkpoint.is_file())
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    report["upstream"] = manifest["models"][args.model]
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    write_json(report_path, report)
    if args.audit_only:
        print(f"Audited {len(records)} sequences; checkpoint_available={checkpoint.is_file()}")
        return
    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Place the official checkpoint at {checkpoint}")
        report["checkpoint_sha256"] = sha256(checkpoint)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch and an NVIDIA GPU are required; see README.md")
        random.seed(1024)
        np.random.seed(1024)
        torch.manual_seed(1024)
        torch.cuda.manual_seed_all(1024)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["environment"] = dict(python=sys.version, torch=torch.__version__,
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        write_json(report_path, report)
        source = BUNDLE / "models" / args.model
        if args.backend == "torch-reference":
            from reference_backend import install

            install(source, args.model)
            report["operator_note"] = (
                "PyTorch reference voxel/scatter/scan; strict official weights; "
                "floating-point operation order differs from fused CUDA kernels; "
                "GPU parity with official extensions has not been verified")
            write_json(report_path, report)
        if args.model == "pointevent":
            run_point(args, source, checkpoint, output, records, files)
        else:
            run_sparse(args, source, checkpoint, output, records)
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(report_path, report)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
