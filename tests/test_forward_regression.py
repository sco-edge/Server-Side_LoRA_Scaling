from types import SimpleNamespace

import pytest


class QuantLinear:
    def forward(self, value):
        return value


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_forward_restored_after_jvp(sls, override, fail):
    layer = QuantLinear()
    if override:
        layer.forward = lambda value: value + 1
    original = layer.forward
    model = SimpleNamespace(modules=lambda: [layer])

    def run():
        with sls._jvp_ready(model):
            assert layer.forward != original
            if fail:
                raise RuntimeError("interrupted")

    if fail:
        with pytest.raises(RuntimeError, match="interrupted"):
            run()
    else:
        run()
    assert layer.forward == original
    assert ("forward" in vars(layer)) == override
