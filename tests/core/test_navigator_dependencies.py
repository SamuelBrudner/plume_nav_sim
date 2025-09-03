import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


def _install_dummy_controllers(monkeypatch):
    dummy = types.ModuleType("plume_nav_sim.core.controllers")
    dummy.SingleAgentController = object
    dummy.MultiAgentController = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.controllers", dummy)


def _load_navigator(monkeypatch, missing_modules):
    _install_dummy_controllers(monkeypatch)
    for name in missing_modules:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    loader = SourceFileLoader("navigator", str(Path("src/plume_nav_sim/core/navigator.py")))
    module = types.ModuleType("navigator")
    loader.exec_module(module)
    return module


def test_import_error_when_hydra_missing(monkeypatch):
    with pytest.raises(ImportError):
        _load_navigator(monkeypatch, ["omegaconf"])


def test_import_error_when_gymnasium_missing(monkeypatch):
    with pytest.raises(ImportError):
        _load_navigator(monkeypatch, ["gymnasium", "gym"])
