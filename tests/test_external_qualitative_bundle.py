"""检查外部定性包的事件行序与回传契约，不模拟 GPU 推理。"""

import importlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "remote/external_qualitative/tools"


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("event_io"), importlib.import_module("collect_results")


@pytest.fixture
def source(tmp_path):
    raw = np.zeros(4, dtype=[("x", "i2"), ("y", "i2"), ("t", "f8"),
                             ("p", "i1"), ("label", "i1"), ("name", "i1")])
    raw["x"], raw["y"], raw["t"] = [4, 8, 9, 12], [1, 2, 3, 4], [0, 0.7, 1.1, 2.4]
    raw["p"], raw["label"], raw["name"] = [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]
    native = np.column_stack([raw[k] for k in ("x", "y", "t", "p", "label", "name")])
    native[:, :3] /= [352, 288, 8000]
    locations = np.column_stack((raw["x"], raw["y"], raw["t"].astype(np.int64)))
    path = tmp_path / "test" / "test_000.npz"
    path.parent.mkdir()
    np.savez(path, ev=raw, evs_norm=native.astype(np.float32), ev_loc=locations)
    return path


def test_audit_retains_raw_submillisecond_time(helpers, source):
    io, _ = helpers
    record, raw, gt = io.audit_input(source)
    assert record["events"] == 4 and record["targets"] == 2
    assert raw["t"].tolist() == [0, 0.7, 1.1, 2.4]
    assert gt.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("field", ["ev_loc", "evs_norm"])
def test_audit_rejects_changed_row_order(helpers, source, field):
    io, _ = helpers
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files}
    arrays[field] = arrays[field][::-1]
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="event order"):
        io.audit_input(source)


def test_export_requires_all_events(helpers, source, tmp_path):
    io, _ = helpers
    record, _, _ = io.audit_input(source)
    with pytest.raises(ValueError, match="full-event"):
        io.save_prediction(tmp_path, source.name, [0.1, 0.2, 0.3], record)


def test_return_archive_keeps_common_input_and_threshold_boundary(helpers, source, tmp_path):
    io, collector = helpers
    record, raw, gt = io.audit_input(source)
    root = tmp_path / "results"
    io.save_input(root, source, record, raw, gt)
    probability = np.array([0.0, 0.899, 0.9, 1.0], dtype=np.float32)
    for model in ["pact", "pointevent"]:
        io.save_prediction(root / model, source.name, probability, record)
        io.write_json(root / model / "report.json",
                      dict(model=model, status="complete", sequences=[record]))
    archive_path = tmp_path / "return.zip"
    collector.collect(root, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert sum("inputs/test_000.npz" in name for name in names) == 1
        summary = json.loads(archive.read("external_qualitative_results/summary.json"))
        assert summary["missing_models"] == ["ev_spsegnet"]
    with np.load(root / "pact/predictions" / source.name) as result:
        assert result["prediction"].tolist() == [0, 0, 1, 1]
        assert result["event_index"].tolist() == [0, 1, 2, 3]


def test_collector_rejects_different_source_hashes(helpers, source, tmp_path):
    io, collector = helpers
    record, raw, gt = io.audit_input(source)
    io.save_input(tmp_path, source, record, raw, gt)
    for model, checksum in [("pact", record["sha256"]), ("pointevent", "different")]:
        changed = dict(record, sha256=checksum)
        io.save_prediction(tmp_path / model, source.name, np.zeros(4), changed)
        io.write_json(tmp_path / model / "report.json",
                      dict(model=model, status="complete", sequences=[changed]))
    with pytest.raises(ValueError, match="different source"):
        collector.collect(tmp_path, tmp_path / "return.zip")


def test_selection_requires_all_requested_sequences(helpers, source):
    io, _ = helpers
    with pytest.raises(FileNotFoundError, match="test_023"):
        io.selected_files(source.parent.parent, "test", ["test_000", "test_023"])


def test_point_validation_input_uses_official_test_name_mapping(helpers, source, tmp_path, monkeypatch):
    io, _ = helpers
    predictor = importlib.import_module("predict")
    source = source.rename(source.with_name("val_000.npz"))
    record, _, _ = io.audit_input(source)
    model_source = tmp_path / "official_point"
    model_source.mkdir()
    output = tmp_path / "results/pointevent"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["predict"])
    # 测试包装器的路径和结果映射；不模拟或宣称神经网络推理。
    monkeypatch.setattr(Path, "symlink_to", lambda link, target: shutil.copy2(target, link))

    def official_export_stub(*args, **kwargs):
        options = dict(item.split("=", 1) for item in sys.argv[sys.argv.index("--options") + 1:])
        assert options["data.test.split"] == "test"
        linked = Path(options["data_root"]) / "test" / source.name
        with np.load(linked) as data:
            raw, gt = data["ev"], data["evs_norm"][:, 4]
        native = Path(options["save_path"]) / "npz"
        native.mkdir(parents=True)
        probability = np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32)
        np.savez(native / source.name, ev=raw, gt=gt, target_scores=probability,
                 pred=probability >= 0.9)

    monkeypatch.setattr(predictor.runpy, "run_path", official_export_stub)
    predictor.run_point(SimpleNamespace(split="val"), model_source,
                        tmp_path / "model_inference.pth", output, [record], [source])
    with np.load(output / "predictions" / source.name) as result:
        assert result["event_index"].tolist() == [0, 1, 2, 3]
        assert str(result["source_sha256"].item()) == record["sha256"]


def test_staging_copies_when_link_creation_is_unavailable(helpers, source, tmp_path, monkeypatch):
    predictor = importlib.import_module("predict")

    def unavailable(*args, **kwargs):
        raise OSError("cross-device link or missing link privilege")

    if os.name == "nt":
        monkeypatch.setattr(predictor.os, "link", unavailable)
    else:
        monkeypatch.setattr(Path, "symlink_to", unavailable)
    destination = tmp_path / "staged.npz"
    predictor.stage_input(source, destination)
    assert destination.read_bytes() == source.read_bytes()
