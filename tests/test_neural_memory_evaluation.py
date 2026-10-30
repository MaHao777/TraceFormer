"""回归：评估不能在inference_mode内构造模型参数。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive/experiments/scripts"))
import diagnose_memory_alignment as alignment
import finish_neural_event_memory as finish


class FactoryReached(Exception):
    pass


@pytest.mark.parametrize("module", [finish, alignment])
def test_evaluation_parameter_factory_preserves_normal_tensor_contract(module, monkeypatch, tmp_path):
    """旧外层inference上下文会在这里创建inference参数，改变真实CPU前向。"""
    monkeypatch.setattr(module, "validate_lock", lambda root: ({}, "identity", {}))
    if module is alignment:
        monkeypatch.setattr(module, "read", lambda path: {"status": "complete", "lock_sha256": "identity"})

    def factory():
        parameter = torch.nn.Linear(2, 2).weight
        assert not parameter.is_inference()
        assert not torch.is_grad_enabled()
        raise FactoryReached

    monkeypatch.setattr(module, "source_model", factory)
    with pytest.raises(FactoryReached):
        if module is finish:
            module.evaluate(tmp_path, "val", atol=1e-6)
        else:
            module.run(tmp_path)
