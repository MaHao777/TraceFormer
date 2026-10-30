"""默认主模型必须使用完整后处理，并保持产物和真实验证结论绑定。"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traceformer.cli import predict
from traceformer.postprocessing.contracts import sha256_file


def test_default_predict_dispatches_complete_primary_model(tmp_path, monkeypatch):
    observed = {}

    def generate(output_dir, **kwargs):
        observed.update(kwargs)
        observed['output_dir'] = output_dir
        return {'output_dir': str(tmp_path), 'files': {}}

    monkeypatch.setattr(predict, 'predict_primary_model', generate)
    monkeypatch.setattr(sys, 'argv', ['traceformer-predict', '--input-dir', str(tmp_path),
                                     '--predictions-only', '--quiet'])
    predict.main()
    assert observed['input_dir'] == tmp_path
    assert observed['export_submission'] is False


def test_primary_manifest_retains_failed_gate_and_rejects_tampering(tmp_path, monkeypatch):
    from traceformer import primary

    metrics = {'aggregate': {'score': 0.95, 'fa': 7e-6},
               'files': {'val_023.npz': {'pd': 0.9}}}
    original = {'aggregate': {'score': 0.9}, 'files': {'val_023.npz': {'pd': 0.8}}}
    for name in ['best.pt', 'config.yaml', 'event_release/release_report.json']:
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'fixed')
    validation = {'event_release': metrics, 'original': original, 'protocol': {
        'event_release_report_sha256': sha256_file(tmp_path/'event_release/release_report.json')}}
    (tmp_path/'validation.json').write_text(json.dumps(validation))
    manifest = {'schema_version': 1, 'artifact_root': '.', 'files': {
        name: sha256_file(tmp_path/name) for name in ['best.pt', 'config.yaml',
        'event_release/release_report.json', 'validation.json']}, 'selection': {
        'method': 'user_requested', 'validation_gate': primary.event_validation_gate(metrics, original)}}
    path = tmp_path/'main.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(primary, 'load_event_release', lambda *a, **k: ({}, {}))
    _, loaded = primary.load_primary_manifest(path)
    assert loaded['selection']['validation_gate']['passed'] is False
    (tmp_path/'best.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='hash differs'):
        primary.load_primary_manifest(path)


@pytest.mark.parametrize('export_submission', [False, True])
def test_primary_prediction_uses_both_event_branches_and_no_labels(
    tmp_path, monkeypatch, export_submission,
):
    from traceformer import primary

    raw = np.zeros(3, dtype=[('x', 'i4'), ('y', 'i4'), ('t', 'f8'), ('p', 'i4')])
    raw['x'], raw['y'], raw['t'] = [2, 3, 4], 1, [0, 25, 50]
    np.savez(tmp_path/'tiny.npz', ev=raw, ev_loc=np.column_stack((raw['x'], raw['y'], raw['t'])),
             evs_norm=np.zeros((3, 4), np.float32))
    selection = {'method': 'user_requested', 'validation_gate': {'passed': False}}
    monkeypatch.setattr(primary, 'load_primary_manifest', lambda *a: (
        tmp_path, {'selection': selection, 'runtime': {'num_threads': 2}}))
    monkeypatch.setattr(primary, 'load_config', lambda *a: SimpleNamespace(
        training=SimpleNamespace(num_threads=1), dataset=SimpleNamespace(duration_ms=8000),
        resolution=SimpleNamespace(width=346, height=260)))
    monkeypatch.setattr(primary, 'load_inference_model', lambda *a, **k:
                        (torch.nn.Linear(1, 1), {'decision_threshold': 0.5}))
    monkeypatch.setattr(primary, 'load_event_release', lambda *a, **k: ({}, {'event_contract': {}}))

    def infer(model, config, sequence, threshold, *, source):
        assert not sequence.labels.any() and not sequence.object_ids.any()
        assert 'labels' not in source and 'target_ids' not in source
        return SimpleNamespace(name='tiny.npz', locations=source['locations'])

    monkeypatch.setattr(primary, 'infer_event_cache', infer)
    monkeypatch.setattr(primary, 'predict_event_scores', lambda *a, **k: {})
    monkeypatch.setattr(primary, 'event_branches', lambda *a: (
        None, None, np.array([True, False, False]), np.array([False, False, True])))
    previous_threads = torch.get_num_threads()
    manifest_path = tmp_path / 'main.json'
    manifest_path.write_text(json.dumps({'selection': selection}))
    try:
        report = primary.predict_primary_model(tmp_path/'out', input_dir=tmp_path,
                                              manifest_path=manifest_path,
                                              export_submission=export_submission, progress=lambda _: None)
    finally:
        torch.set_num_threads(previous_threads)
    with np.load(tmp_path/'out/predictions/tiny.npz') as saved:
        assert saved['prediction'].tolist() == [1, 0, 1]
    assert report['selection'] == selection
    assert report['torch_threads'] == 2
    assert report['validation_labels_accessed_during_inference'] is False
    if export_submission:
        assert (tmp_path/'out/submission.zip').is_file()
