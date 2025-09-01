import importlib
import sys
import types

import pytest


def test_import_error_when_stable_baselines3_missing(monkeypatch):
    """Policies module should raise ImportError when stable-baselines3 is absent."""
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("nn")
    fake_nn.Module = object
    fake_nn.Linear = object
    fake_nn.BatchNorm1d = object
    fake_nn.Dropout = object
    fake_nn.ReLU = object
    fake_nn.Tanh = object
    fake_nn.ELU = object
    fake_nn.LeakyReLU = object
    fake_nn.SiLU = object
    fake_nn.GELU = object
    fake_nn.functional = types.ModuleType("functional")
    fake_torch.nn = fake_nn
    fake_torch.Tensor = object
    fake_torch.distributions = types.ModuleType("distributions")
    fake_torch.distributions.Normal = object
    fake_torch.optim = types.ModuleType("optim")
    fake_torch.optim.Optimizer = object
    fake_torch.optim.Adam = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_nn.functional)
    monkeypatch.setitem(sys.modules, "torch.distributions", fake_torch.distributions)
    monkeypatch.setitem(sys.modules, "torch.optim", fake_torch.optim)

    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    sys.modules.pop("odor_plume_nav.rl.policies", None)

    with pytest.raises(ImportError):
        importlib.import_module("odor_plume_nav.rl.policies")
