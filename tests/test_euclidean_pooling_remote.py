"""四组独立上传包的完整闭环、强制预检与冻结恢复。"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from test_local_pooling_remote import bundle, packager, read, run_bundle

from traceformer.experiments.euclidean_pooling import VARIANTS

# 共用既有147个微型NPZ文件的打包fixture，只替换实验名称。
__all__ = ["bundle", "packager"]
pytestmark = pytest.mark.parametrize("bundle", ["euclidean_pooling_four"], indirect=True)


def test_four_group_bundle_contains_exact_structure_without_historical_weights(bundle, packager):
    directory, data, _ = bundle
    suite = read(directory / "suite.json")
    assert suite["baseline_reused"] is False and suite["mandatory_preflight"] is True
    assert [row["variant"] for row in suite["variants"]] == list(VARIANTS)
    assert suite["variants"][0]["windows_per_layer"] == [[[2, 2]]] * 3
    assert suite["variants"][2]["windows_per_layer"] == [[[8, 2], [4, 4], [2, 8]]] * 3
    assert read(directory / "initialization_audit.json")["common_parameters_bitwise_equal"]
    assert not (directory / "baseline").exists() and not list(directory.rglob("*.pt"))
    assert not list(directory.rglob("*.npz"))
    with pytest.raises(ValueError, match="only local_pooling_six"):
        packager.build_package("euclidean_pooling_four", directory.parent / "bad baseline",
                               directory / "config.yaml", data, data, baseline_root=directory)
    with pytest.raises(ValueError, match="historical reference"):
        packager.build_package("euclidean_pooling_four", directory.parent / "bad reference",
                               directory / "config.yaml", data, data, references={})


def test_isolated_four_group_smoke_preflights_trains_scores_archives_and_resumes(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--smoke", "--device", "cpu")
    root = directory / "output/smoke"
    assert read(root / "status.json")["status"] == "complete"
    assert set(read(root / "preflight_timing.json")["reports"]) == set(VARIANTS)
    result = read(root / "results.json")
    assert result["complete"] and result["diagnostic_only"]
    assert set(result["comparisons"]["val"]) == {
        "E1-E0", "E3-E2", "E2-E0", "E3-E1", "E3-E0", "single_kernel_interaction"}
    assert read(root / "completion_audit.json")["partitions"] == {"val": 2, "test": 2}
    before = {}
    for name in VARIANTS:
        pointer = read(root / name / "training.json")
        assert pointer["reused"] is False
        before[name] = Path(pointer["checkpoint"]).read_bytes()
        for split in ("val", "test"):
            report = read(root / name / split / "independent_report.json")
            assert report["sequences"] == 2 and report["test_threshold_selection"] is False
            assert (root / name / split / "raw_independent_report.json").is_file()
    with zipfile.ZipFile(directory / "output/smoke_results.zip") as archive:
        assert archive.testzip() is None
        assert "smoke/initialization_audit.json" in archive.namelist()
        assert all(f"smoke/{v}/training/attempt_001/best.pt" in archive.namelist() for v in VARIANTS)
    archive_path = directory / "output/smoke_results.zip"
    checksum = archive_path.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()[0]
    assert checksum == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    run_bundle(bundle, "--smoke", "--device", "cpu", "--resume")
    for name in VARIANTS:
        pointer = read(root / name / "training.json")
        assert Path(pointer["checkpoint"]).read_bytes() == before[name]
        assert len(list((root / name / "training").glob("attempt_*"))) == 1
    changed = run_bundle(bundle, "--smoke", "--device", "cpu", "--resume", check=False,
                         env_override={"TRACEFORMER_LOCAL_QUERY_CHUNK": "17"})
    assert changed.returncode != 0 and "settings/source changed" in changed.stderr


@pytest.mark.parametrize("tamper", ["data", "source"])
def test_four_group_bundle_refuses_changed_inputs_before_training(bundle, tamper):
    directory, data, _ = bundle
    path = (data / "train/train_000.npz" if tamper == "data"
            else directory / "src/traceformer/models/cone_geometry.py")
    with path.open("ab") as stream:
        stream.write(b"\n# changed\n")
    result = run_bundle(bundle, "--smoke", "--device", "cpu", check=False)
    assert result.returncode != 0
    assert ("data identity differs" if tamper == "data" else "package file changed") in result.stderr
    assert not list((directory / "output/smoke").glob("*/training/attempt_*"))


def test_formal_four_group_bundle_rejects_cpu_and_bf16(bundle):
    cpu = run_bundle(bundle, "--device", "cpu", check=False)
    assert cpu.returncode != 0 and "require CUDA FP32" in cpu.stderr
    bf16 = run_bundle(bundle, "--precision", "bf16", check=False)
    assert bf16.returncode != 0 and "requires CUDA FP32" in bf16.stderr


def test_explicit_train_performs_preflight_without_exporting_test_predictions(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--smoke", "--device", "cpu", "--stage", "train")
    root = directory / "output/smoke"
    assert read(root / "status.json")["status"] == "training_complete"
    assert (root / "preflight_timing.json").is_file()
    assert all((root / name / "training.json").is_file() for name in VARIANTS)
    assert not any((root / name / "test").exists() for name in VARIANTS)
    assert not (directory / "output/smoke_results.zip").exists()
    assert all((root / name / "structure.json").is_file() for name in VARIANTS)
    # 预测阶段也不能换主机或GPU后沿用旧队列。
    path = root / "execution_environment.json"
    environment = read(path)
    environment["host"] = "different execution host"
    path.write_text(json.dumps(environment), encoding="utf-8")
    changed = run_bundle(bundle, "--smoke", "--device", "cpu", "--stage", "predict", "--resume", check=False)
    assert changed.returncode != 0 and "execution environment changed" in changed.stderr
    assert not any((root / name / "test").exists() for name in VARIANTS)
