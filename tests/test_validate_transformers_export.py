from pathlib import Path

import torch

from experiments.pretrain.scripts import validate_transformers_export as validator


def test_custom_export_loader_explicitly_trusts_local_code(monkeypatch) -> None:
    captured = {}

    class DummyModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(validator.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    validator.load_exported("artifact", torch.device("cpu"), torch.float16)

    assert captured["path"] == validator.resolve(Path("artifact"))
    assert captured["trust_remote_code"] is True
    assert captured["dtype"] == torch.float16
