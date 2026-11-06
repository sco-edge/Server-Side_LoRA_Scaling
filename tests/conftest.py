import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ZO-FL Server-Side LoRA Scaling.py"


def _install_torch_stub():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    autograd = types.ModuleType("torch.autograd")
    forward_ad = types.ModuleType("torch.autograd.forward_ad")
    nn.Module = object
    nn.LayerNorm = object
    nn.Linear = object
    nn.functional = functional
    autograd.forward_ad = forward_ad
    torch.nn = nn
    torch.autograd = autograd
    torch.no_grad = lambda: (lambda f: f)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.update({
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torch.autograd": autograd,
        "torch.autograd.forward_ad": forward_ad,
    })


@pytest.fixture(scope="session")
def sls():
    if importlib.util.find_spec("torch") is None:
        _install_torch_stub()
    spec = importlib.util.spec_from_file_location("server_lora_scaling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
