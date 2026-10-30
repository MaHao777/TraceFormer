"""核验远端一次性修复只在正式训练前生效，并保留源码校验。"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def fixer():
    path = Path(__file__).resolve().parents[1] / "archive/experiments/scripts/fix_mean_past_attention.py"
    spec = importlib.util.spec_from_file_location("mean_past_attention_fix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def original_package(tmp_path, fixer):
    package = tmp_path / "old upload"
    source = package / fixer.TARGET
    source.parent.mkdir(parents=True)
    source.write_text(fixer.OLD, encoding="utf-8")
    fixer.write(package / "model_summary.json", {"experiment": "M1_C4"})
    manifest = {"sha256": {fixer.TARGET: fixer.sha256(source),
                          "model_summary.json": fixer.sha256(package / "model_summary.json")}}
    fixer.write(package / "package_manifest.json", manifest)
    root = package / "output/full"
    root.mkdir(parents=True)
    fixer.write(root / "experiment.json", {"package_sha256": fixer.sha256(package / "package_manifest.json"),
                                          "config": {"seed": 37}})
    fixer.write(root / "status.json", {"status": "failed", "error": "preflight failed"})
    return package


def test_hotfix_backs_up_and_updates_source_identity(original_package, fixer, monkeypatch):
    package = original_package
    # 使用真实源码的旧版本块，保证替换后是可以编译的完整模块。
    source = package / fixer.TARGET
    current = Path(__file__).resolve().parents[1] / fixer.TARGET
    source.write_text(current.read_text(encoding="utf-8").replace(fixer.NEW, fixer.OLD), encoding="utf-8")
    manifest = fixer.read(package / "package_manifest.json")
    manifest["sha256"][fixer.TARGET] = fixer.sha256(source)
    fixer.write(package / "package_manifest.json", manifest)
    original_digest = fixer.sha256(package / "package_manifest.json")
    fixer.write(package / "output/full/experiment.json", {"package_sha256": original_digest, "config": {"seed": 37}})
    before = source.read_bytes()
    fixer.apply_fix(package)
    assert fixer.NEW in source.read_text(encoding="utf-8")
    record = fixer.read(package / "ATTENTION_BATCH_FIX.json")
    assert record["original_package_sha256"] == original_digest
    assert (package / record["backup"] / fixer.TARGET).read_bytes() == before
    manifest = fixer.read(package / "package_manifest.json")
    assert all(fixer.sha256(package / key) == value for key, value in manifest["sha256"].items())
    context = fixer.read(package / "output/full/experiment.json")
    assert context["package_sha256"] == fixer.sha256(package / "package_manifest.json")
    assert context["config"] == {"seed": 37}
    assert fixer.read(package / "output/full/status.json")["status"] == "failed"
    fixer.apply_fix(package)
    assert len(list((package / "output").glob("attention_fix_backup_*"))) == 1


def test_hotfix_refuses_training_already_started(original_package, fixer):
    root = original_package / "output/full/training/attempt_001"
    root.mkdir(parents=True)
    (root / "training_state.pt").write_bytes(b"existing state")
    before = (original_package / fixer.TARGET).read_bytes()
    with pytest.raises(ValueError, match="already started"):
        fixer.apply_fix(original_package)
    assert (original_package / fixer.TARGET).read_bytes() == before


def test_hotfix_refuses_unexpected_source_change(original_package, fixer):
    source = original_package / fixer.TARGET
    source.write_text("unexpected code", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        fixer.apply_fix(original_package)
    assert not list((original_package / "output").glob("attention_fix_backup_*"))


def test_script_matches_exact_repository_replacement(fixer):
    current = Path(__file__).resolve().parents[1] / fixer.TARGET
    code = current.read_text(encoding="utf-8")
    assert code.count(fixer.NEW) == 1 and fixer.OLD not in code
