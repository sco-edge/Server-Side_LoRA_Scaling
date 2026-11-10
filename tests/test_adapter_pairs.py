import pytest


@pytest.mark.parametrize("suffix", ["A", "B"])
def test_unmatched_tensor_among_valid_pairs(sls, monkeypatch, suffix):
    state = dict.fromkeys(["a.lora_A", "a.lora_B", "b.lora_" + suffix])
    monkeypatch.setattr(sls.torch, "load", lambda *args, **kwargs: state, raising=False)
    with pytest.raises(SystemExit, match="incomplete LoRA pairs for b"):
        sls.attach_adapter(None, "adapter.pt")
