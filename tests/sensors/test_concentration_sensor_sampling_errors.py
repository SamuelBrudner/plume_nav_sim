import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "core" / "sensors" / "concentration_sensor.py"


def load_concentration_sensor(monkeypatch):
    module_name = "plume_nav_sim.core.sensors.concentration_sensor"
    # Stub package structure
    packages = [
        "plume_nav_sim",
        "plume_nav_sim.core",
        "plume_nav_sim.core.sensors",
        "plume_nav_sim.protocols",
    ]
    for pkg in packages:
        mod = ModuleType(pkg)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, mod)

    sensor_protocol = ModuleType("plume_nav_sim.protocols.sensor")
    class SensorProtocol: ...
    sensor_protocol.SensorProtocol = SensorProtocol
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.sensor", sensor_protocol)

    base_sensor_module = ModuleType("plume_nav_sim.core.sensors.base_sensor")
    class BaseSensor:
        def __init__(self, *args, **kwargs):
            self._enable_logging = kwargs.get("enable_logging", True)
            self._sensor_id = kwargs.get("sensor_id", "sensor")
            self._performance_metrics = {}
    base_sensor_module.BaseSensor = BaseSensor
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.sensors.base_sensor", base_sensor_module)

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ConcentrationSensor


def test_sample_raw_concentrations_invalid_plume_state(monkeypatch):
    ConcentrationSensor = load_concentration_sensor(monkeypatch)
    sensor = ConcentrationSensor()
    positions = np.array([[0.0, 0.0]])
    with pytest.raises(TypeError):
        sensor._sample_raw_concentrations(object(), positions)


def test_sample_raw_concentrations_sampling_failure(monkeypatch):
    ConcentrationSensor = load_concentration_sensor(monkeypatch)
    sensor = ConcentrationSensor()
    positions = np.array([[0.0, 0.0]])

    class FailingPlume:
        def concentration_at(self, pos):
            raise ValueError("bad")

    with pytest.raises(RuntimeError):
        sensor._sample_raw_concentrations(FailingPlume(), positions)


def test_sample_from_array_invalid_array(monkeypatch):
    ConcentrationSensor = load_concentration_sensor(monkeypatch)
    sensor = ConcentrationSensor()
    positions = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        sensor._sample_from_array(np.array([1, 2, 3]), positions)
