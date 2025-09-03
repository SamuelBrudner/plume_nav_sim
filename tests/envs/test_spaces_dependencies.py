import importlib
import sys
import types
from pathlib import Path

import pytest


def _import_spaces(monkeypatch):
    package = types.ModuleType("plume_nav_sim")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", package)
    sys.modules.pop("plume_nav_sim.envs.spaces", None)
    return importlib.import_module("plume_nav_sim.envs.spaces")


def test_import_fails_without_gymnasium(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    with pytest.raises(ImportError):
        _import_spaces(monkeypatch)


def test_import_fails_without_sensor_implementations(monkeypatch):
    modules = [
        "plume_nav_sim.core.sensors.binary_sensor",
        "plume_nav_sim.core.sensors.concentration_sensor",
        "plume_nav_sim.core.sensors.gradient_sensor",
    ]
    for mod in modules:
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(ImportError):
        _import_spaces(monkeypatch)
