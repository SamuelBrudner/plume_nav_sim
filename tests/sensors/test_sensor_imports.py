import importlib
import importlib.machinery
import sys

import pytest

SENSOR_MODULES = [
    ("base_sensor", "BaseSensor"),
    ("binary_sensor", "BinarySensor"),
    ("concentration_sensor", "ConcentrationSensor"),
    ("gradient_sensor", "GradientSensor"),
    ("historical_sensor", "HistoricalSensor"),
]


@pytest.mark.parametrize("module_name,class_name", SENSOR_MODULES)
def test_sensor_import_raises(monkeypatch, module_name, class_name):
    """Importing sensors should raise ImportError when modules are missing."""
    real_find_spec = importlib.machinery.PathFinder.find_spec

    def fake_find_spec(name, path=None, target=None):
        if name == f"plume_nav_sim.core.sensors.{module_name}":
            return None
        return real_find_spec(name, path, target)

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", fake_find_spec)

    for mod in list(sys.modules):
        if mod.startswith("plume_nav_sim.core.sensors"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError):
        __import__("plume_nav_sim.core.sensors", fromlist=[class_name])
