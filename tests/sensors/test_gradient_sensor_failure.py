import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_gradient_sensor(monkeypatch):
    module_name = "plume_nav_sim.core.sensors.gradient_sensor"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "plume_nav_sim"
        / "core"
        / "sensors"
        / "gradient_sensor.py"
    )

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
    class SensorProtocol:
        pass
    sensor_protocol.SensorProtocol = SensorProtocol
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.sensor", sensor_protocol)

    base_sensor_module = ModuleType("plume_nav_sim.core.sensors.base_sensor")
    class BaseSensor:
        def __init__(self, *args, **kwargs):
            self._logger = None
            self._performance_metrics = {}
            self._enable_performance_monitoring = False
        def _execute_with_monitoring(self, func, label, plume_state, positions):
            return func(plume_state, positions)
    base_sensor_module.BaseSensor = BaseSensor
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.sensors.base_sensor", base_sensor_module)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.GradientSensor


def test_gradient_sensor_raises_on_gradient_failure(monkeypatch):
    GradientSensor = load_gradient_sensor(monkeypatch)
    sensor = GradientSensor()
    positions = np.array([[0.0, 1.0]])  # Valid shape
    plume_state = object()  # Missing required interface
    with pytest.raises(Exception):
        sensor.compute_gradient(plume_state, positions)
