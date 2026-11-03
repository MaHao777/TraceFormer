"""九组独立包的完整微型闭环与原契约基线复用。"""

import hashlib
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_local_pooling_remote import bundle, packager, read, run_bundle

from traceformer.experiments.ablation import ComponentCoreSpec, LocalPoolingSpec
from traceformer.experiments.s1_paper import MECHANISM_ROWS, STRUCTURE_ROWS, VARIANTS
from traceformer.experiments.workflows.s1_paper import S1PaperWorkflow
from traceformer.training.engine import _file_sha256, _serializable, run_training

__all__ = ["bundle", "packager"]
pytestmark = pytest.mark.parametrize("bundle", ["s1_paper_ablation"], indirect=True)


def test_s1_bundle_is_screening_only_and_records_actual_structures(bundle, packager):
    directory, data, _ = bundle
    suite = read(directory / "suite.json")
    assert suite["profile"] == "screen" and suite["seed"] == 37 and suite["epochs"] == 24
    assert suite["references"] == {}
    rows = {row["variant"]: row for row in suite["variants"]}
    assert list(rows) == list(VARIANTS) and rows["T1"]["direct_union_candidates"] == 552
    assert rows["P1"]["pool_kernel_sizes"] == [[1, 1]]
    assert not list(directory.rglob("*.npz"))
    assert (directory / "EXPERIMENT.md").is_file()
    with pytest.raises(ValueError, match="training pointer missing"):
        packager.build_package("s1_paper_ablation", directory.parent / "bad refs", directory / "config.yaml",
                               data, data, baseline_root=directory.parent / "absent refs")


def test_s1_isolated_cpu_smoke_scores_two_tables_and_preserves_weights_on_resume(bundle):
    directory, _, _ = bundle
    run_bundle(bundle, "--smoke", "--device", "cpu", timeout=900)
    root = directory / "output/smoke"
    result = read(root / "results.json")
    assert read(root / "status.json")["status"] == "complete"
    assert result["ablation_tables"] == {"structure": list(STRUCTURE_ROWS), "mechanism": list(MECHANISM_ROWS)}
    assert result["seed"] == 37 and result["profile"] == "screen" and result["diagnostic_only"]
    assert "single_kernel_interaction" not in result["comparisons"]["test"]
    assert len(read(root / "preflight_timing.json")["reports"]) == 9
    support = read(root / "support_geometry.json")
    assert support["direct_union_candidates"] == 552 and len(support["files"]) == 2
    before = {}
    for variant in VARIANTS:
        checkpoint = Path(read(root / variant / "training.json")["checkpoint"])
        before[variant] = checkpoint.read_bytes()
        for split in ("val", "test"):
            assert read(root / variant / split / "independent_report.json")["sequences"] == 2
            assert (root / variant / split / "raw_independent_report.json").is_file()
    archive = directory / "output/smoke_results.zip"
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert "smoke/support_geometry.json" in zipped.namelist()
    assert archive.with_suffix(".zip.sha256").read_text().split()[0] == hashlib.sha256(archive.read_bytes()).hexdigest()
    run_bundle(bundle, "--smoke", "--device", "cpu", "--resume", timeout=900)
    for variant in VARIANTS:
        assert Path(read(root / variant / "training.json")["checkpoint"]).read_bytes() == before[variant]
        assert len(list((root / variant / "training").glob("attempt_*"))) == 1
    refused = run_bundle(bundle, "--smoke", "--device", "cpu", "--resume", check=False,
                         env_override={"TRACEFORMER_S1_DIRECT_QUERY_CHUNK": "17"})
    assert refused.returncode != 0 and "settings/source changed" in refused.stderr


def test_s1_formal_cpu_bf16_and_tampered_source_are_rejected(bundle):
    cpu = run_bundle(bundle, "--device", "cpu", check=False)
    assert cpu.returncode != 0 and "require CUDA FP32" in cpu.stderr
    bf16 = run_bundle(bundle, "--precision", "bf16", check=False)
    assert bf16.returncode != 0 and "requires CUDA FP32" in bf16.stderr
    directory, _, _ = bundle
    with (directory / "src/traceformer/experiments/s1_paper.py").open("ab") as stream:
        stream.write(b"\n# changed\n")
    refused = run_bundle(bundle, "--smoke", "--device", "cpu", check=False)
    assert refused.returncode != 0 and "package file changed" in refused.stderr
    assert not list((directory / "output/smoke").glob("*/training/attempt_*"))


@pytest.mark.parametrize("fail_validation", [False, True])
def test_s1_and_s0_reuse_preserves_original_contract_bytes_and_requires_both_validations(
    bundle, tmp_path, monkeypatch, fail_validation,
):
    directory, data, config = bundle
    config = replace(config, training=replace(config.training, device="cpu", epochs=1, num_threads=1,
                     max_train_files=2, max_val_files=2))
    original, contracts = {}, {"S1": LocalPoolingSpec("S1"), "S0": ComponentCoreSpec("M1_C4")}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for variant, spec in contracts.items():
            source = directory / "baseline" / variant
            run_training(config, output_dir=source, ablation=spec)
            original[variant] = (source / "best.pt").read_bytes()
            files = ("best.pt", "config.yaml", "metrics.json", "history.json", "summary.json")
            identity = {"ablation": spec.to_dict(), "checkpoint_sha256": _file_sha256(source / "best.pt"),
                        "source_files": {name: _file_sha256(source / name) for name in files}}
            (source / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    finally:
        torch.set_num_threads(previous_threads)
    root = tmp_path / "reuse unit"
    for variant, spec in contracts.items():
        folder = root / variant
        folder.mkdir(parents=True)
        (folder / "ablation.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
        (folder / "config.yaml").write_text(yaml.safe_dump(_serializable(asdict(config))), encoding="utf-8")
    calls = []
    def prediction(folder, *args, **kwargs):
        calls.append((folder.name, kwargs["splits"]))
        if fail_validation and folder.name == "S0":
            raise ValueError("independent validation mismatch")
    workflow = S1PaperWorkflow(directory)
    monkeypatch.setattr(workflow, "predict_variant", prediction)
    if fail_validation:
        with pytest.raises(ValueError, match="validation mismatch"):
            workflow.baseline(root, config, data, SimpleNamespace(smoke=False))
        assert not (root / "baseline_audit.json").exists()
        assert not (root / "S0/training.json").exists()
    else:
        workflow.baseline(root, config, data, SimpleNamespace(smoke=False))
        assert read(root / "baseline_audit.json")["passed"]
        for variant, spec in contracts.items():
            pointer = read(root / variant / "training.json")
            assert pointer["reused"] and pointer["ablation"] == spec.to_dict()
            assert Path(pointer["checkpoint"]).read_bytes() == original[variant]
    assert calls == [("S1", ("val",)), ("S0", ("val",))]
