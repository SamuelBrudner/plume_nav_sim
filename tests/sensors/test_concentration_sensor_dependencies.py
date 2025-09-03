import builtins
import importlib.util
import pathlib
import sys
import types

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "src/plume_nav_sim/core/sensors/concentration_sensor.py"


def _load_module(monkeypatch):
    # Stub required package structure
    pkg = types.ModuleType("plume_nav_sim")
    pkg.__path__ = [str(MODULE_PATH.parents[2])]
    sys.modules.setdefault("plume_nav_sim", pkg)

    core_pkg = types.ModuleType("plume_nav_sim.core")
    core_pkg.__path__ = []
    sys.modules.setdefault("plume_nav_sim.core", core_pkg)

    sensors_pkg = types.ModuleType("plume_nav_sim.core.sensors")
    sensors_pkg.__path__ = []
    sys.modules.setdefault("plume_nav_sim.core.sensors", sensors_pkg)

    protocol_module = types.ModuleType("plume_nav_sim.protocols.sensor")
    class DummySensorProtocol: ...
    protocol_module.SensorProtocol = DummySensorProtocol
    sys.modules.setdefault("plume_nav_sim.protocols.sensor", protocol_module)

    base_sensor_module = types.ModuleType("plume_nav_sim.core.sensors.base_sensor")
    class DummyBaseSensor: ...
    base_sensor_module.BaseSensor = DummyBaseSensor
    sys.modules.setdefault("plume_nav_sim.core.sensors.base_sensor", base_sensor_module)

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.core.sensors.concentration_sensor", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_error_when_loguru_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "loguru":
            raise ImportError("loguru missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        _load_module(monkeypatch)


def test_import_error_when_omegaconf_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf":
            raise ImportError("omegaconf missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        _load_module(monkeypatch)
