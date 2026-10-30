"""打包已通过独立验证的当前模型，供后续云端执行。"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.release import event_validation_gate, load_event_release

CLOUD_README = """# 当前三锥模型的云端预测与计时

本包只包含代码、冻结模型、规则和本地验证证据，不含原始数据。
保留 artifacts/config.yaml 原始字节；用 --dataset-root 指定云端数据目录。
正式模式：GPU网络、GPU特征/树、CPU轨迹，FP32，TF32关闭。

在已配置CUDA PyTorch的Python环境安装并检查（先确认 torch.cuda.is_available()）：

```bash
python -m pip install -e '.[gpu,dev,experiments]'
python -m pytest tests/test_gpu_inference.py -q
export EVENT_DATASET=/root/autodl-tmp/ev-uav
```

先核对完整验证集；入口会自动调用已有数值检查器，检查重复前向、112维特征和活动树分数。
本地规则已冻结，禁止在云端调阈值或按测试表现选模型。
若数值或最终掩码不一致，保留差异报告并检查运算路径；不得直接跳过验收。

```bash
python scripts/benchmark_event_system.py --config artifacts/config.yaml --checkpoint artifacts/best.pt --event-release-dir artifacts/event_release --validation-report validation/independent_validation.json --dataset-root "$EVENT_DATASET" --split val --reference-dir validation/event_release --output results/val_cuda.json --profile
```

验证通过后，测试预测、三轮计时和TXT/ZIP导出共用已保存掩码，不另跑网络：

```bash
python scripts/benchmark_event_system.py --config artifacts/config.yaml --checkpoint artifacts/best.pt --event-release-dir artifacts/event_release --validation-report validation/independent_validation.json --dataset-root "$EVENT_DATASET" --split test --validation-evidence results/val_cuda.json --output results/test_cuda.json --profile --export-track2
python -m traceformer.cli.postprocess evaluate --prediction-dir results/test_cuda_predictions --dataset-root "$EVENT_DATASET/test" --output results/test_independent_metrics.json
python scripts/audit_track2_submission.py --test-dir "$EVENT_DATASET/test" --txt-dir results/test_cuda_predictions_txt --zip results/test_cuda_predictions.zip --report results/submission_audit.json
```

正式总耗时从内存归一化输入到最终掩码，包括Sparse、传输、量化、特征、树和轨迹；
排除加载、磁盘IO、评分、导出与审计。预热最大序列1次，全集测3轮。
--profile 额外测1轮阶段同步耗时，仅用于诊断，不与正式耗时拼接或相加。
network_forward_seconds 为网络forward，嵌套于含Sparse准备和缓存转换的阶段。
报告含逐轮原始值、每序列平均/中位/P95/最大值、吞吐率、显存峰值和硬件版本。
把 results/ 完整回传本地，保留原始耗时、掩码、独立评分和审计证据。
"""


def package(run):
    run = Path(run).resolve()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    validation = json.loads((run / "validation/independent_validation.json").read_text(encoding="utf-8"))
    if status["status"] != "validation_accepted" or not event_validation_gate(
            validation["event_release"], validation["original"])["passed"]:
        raise ValueError("only an independently accepted release can be packaged")
    release = run / "training/event_release"
    load_event_release(release, checkpoint=run / "best.pt", config=run / "config.yaml")
    if validation["protocol"]["event_release_report_sha256"] != sha256_file(release / "release_report.json"):
        raise ValueError("validation release differs")
    repo = Path(__file__).resolve().parents[1]
    destination = run / "cloud_bundle"
    destination.mkdir(exist_ok=False)
    # 仅复制普通源码文件，排除缓存、运行结果和本机的原始数据。
    sources = list((repo / "src").rglob("*.py"))
    sources += [repo / "scripts" / name for name in (
        "benchmark_event_system.py", "check_gpu_inference.py", "predict_postprocess_track2.py", "audit_track2_submission.py")]
    sources += [repo / "tests" / name for name in (
        "conftest.py", "test_gpu_inference.py", "gpu_kernel_cases.py")]
    for path in sources:
        target = destination / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(repo / "pyproject.toml", destination / "pyproject.toml")
    artifacts = destination / "artifacts"
    artifacts.mkdir()
    for name in ("best.pt", "config.yaml"):
        shutil.copy2(run / name, artifacts / name)
    shutil.copytree(release, artifacts / "event_release")
    shutil.copytree(run / "validation", destination / "validation")
    (destination / "README.md").write_text(CLOUD_README, encoding="utf-8")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {str(path.relative_to(destination)).replace("\\", "/"): sha256_file(path) for path in files}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive_path = run / "cloud_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(destination)).replace("\\", "/"))
    report = {"files": len(manifest), "zip_sha256": sha256_file(archive_path),
              "zip_size_bytes": archive_path.stat().st_size, "test_inference": False}
    (run / "cloud_bundle_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    package(parser.parse_args().run_dir)
