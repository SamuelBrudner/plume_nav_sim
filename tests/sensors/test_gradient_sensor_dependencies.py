import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
GRADIENT_SENSOR_PATH = (
    SRC_PATH / "plume_nav_sim" / "core" / "sensors" / "gradient_sensor.py"
)
MODULE_NAME = "plume_nav_sim.core.sensors.gradient_sensor"


def load_gradient_sensor(monkeypatch):
    packages = [
        "plume_nav_sim",
        "plume_nav_sim.core",
        "plume_nav_sim.core.sensors",
        "plume_nav_sim.protocols",
        "plume_nav_sim.config",
    ]
    for pkg in packages:
        mod = ModuleType(pkg)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, mod)

    base_sensor = ModuleType("plume_nav_sim.core.sensors.base_sensor")
    base_sensor.BaseSensor = object
    monkeypatch.setitem(
        sys.modules, "plume_nav_sim.core.sensors.base_sensor", base_sensor
    )

    sensor_protocol = ModuleType("plume_nav_sim.protocols.sensor")
    sensor_protocol.SensorProtocol = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.sensor", sensor_protocol)

    schemas = ModuleType("plume_nav_sim.config.schemas")
    schemas.SensorConfig = object
    schemas.BinarySensorConfig = object
    schemas.ConcentrationSensorConfig = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.config.schemas", schemas)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, GRADIENT_SENSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def test_gradient_sensor_import_requires_loguru(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "loguru":
            raise ImportError("No module named 'loguru'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "loguru", raising=False)

    with pytest.raises(ImportError):
        load_gradient_sensor(monkeypatch)


def test_gradient_sensor_import_requires_hydra(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf":
            raise ImportError("No module named 'omegaconf'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "omegaconf", raising=False)

    with pytest.raises(ImportError):
        load_gradient_sensor(monkeypatch)
