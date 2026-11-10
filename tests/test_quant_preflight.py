import sys

import pytest


@pytest.mark.parametrize("mode,device,message", [
    ("gptq", "cuda:0", "requires --gptq-dir"),
    ("hqq", "cpu", "requires a GPU"),
])
def test_invalid_quantization_before_model_loading(sls, monkeypatch, mode, device, message):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit, match=message):
        sls.load_base("unavailable", mode, device)
