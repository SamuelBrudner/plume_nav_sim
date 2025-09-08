import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "core" / "sensors" / "__init__.py"


def test_sensors_import_requires_hydra(monkeypatch):
    """Sensors module should error when Hydra is missing."""
    pkg_plume = types.ModuleType("plume_nav_sim"); pkg_plume.__path__ = []
    pkg_core = types.ModuleType("plume_nav_sim.core"); pkg_core.__path__ = []
    sensors_pkg = types.ModuleType("plume_nav_sim.core.sensors"); sensors_pkg.__path__ = []
    protocols_pkg = types.ModuleType("plume_nav_sim.protocols"); protocols_pkg.__path__ = []
    sensor_protocol_pkg = types.ModuleType("plume_nav_sim.protocols.sensor")
    class SensorProtocol: ...
    sensor_protocol_pkg.SensorProtocol = SensorProtocol

    sys.modules.update({
        "plume_nav_sim": pkg_plume,
        "plume_nav_sim.core": pkg_core,
        "plume_nav_sim.core.sensors": sensors_pkg,
        "plume_nav_sim.protocols": protocols_pkg,
        "plume_nav_sim.protocols.sensor": sensor_protocol_pkg,
    })

    for name in ["base_sensor", "binary_sensor", "concentration_sensor", "gradient_sensor", "historical_sensor"]:
        mod = types.ModuleType(f"plume_nav_sim.core.sensors.{name}")
        cls_name = "".join(part.capitalize() for part in name.split("_"))
        setattr(mod, cls_name, type(cls_name, (), {}))
        sys.modules[f"plume_nav_sim.core.sensors.{name}"] = mod

    for mod in list(sys.modules):
        if mod.startswith("hydra") or mod.startswith("omegaconf"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    real_find_spec = importlib.machinery.PathFinder.find_spec

    def fake_find_spec(fullname, path=None, target=None):
        if fullname.startswith("hydra") or fullname.startswith("omegaconf"):
            return None
        return real_find_spec(fullname, path, target)

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", fake_find_spec)

    spec = importlib.util.spec_from_file_location("plume_nav_sim.core.sensors", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError):
        spec.loader.exec_module(module)
