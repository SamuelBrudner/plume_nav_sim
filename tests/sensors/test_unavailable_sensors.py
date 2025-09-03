import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "core" / "sensors" / "__init__.py"


def load_sensors_module():
    pkg = types.ModuleType("plume_nav_sim")
    sys.modules.setdefault("plume_nav_sim", pkg)
    core_pkg = types.ModuleType("plume_nav_sim.core")
    sys.modules.setdefault("plume_nav_sim.core", core_pkg)
    protocols_pkg = types.ModuleType("plume_nav_sim.protocols")
    sensor_protocol_pkg = types.ModuleType("plume_nav_sim.protocols.sensor")

    class SensorProtocol: ...

    sensor_protocol_pkg.SensorProtocol = SensorProtocol
    protocols_pkg.sensor = sensor_protocol_pkg

    sys.modules.setdefault("plume_nav_sim.protocols", protocols_pkg)
    sys.modules.setdefault("plume_nav_sim.protocols.sensor", sensor_protocol_pkg)

    # minimal sensor implementations to satisfy imports
    for name in ["base_sensor", "binary_sensor", "concentration_sensor", "gradient_sensor", "historical_sensor"]:
        mod = types.ModuleType(f"plume_nav_sim.core.sensors.{name}")
        class _Base(SensorProtocol):
            def __init__(self, **kwargs):
                pass
            def configure(self, **kwargs):
                pass
            def get_metadata(self):
                return {}
        cls_name = "".join(part.capitalize() for part in name.split("_"))
        setattr(mod, cls_name, _Base)
        sys.modules[f"plume_nav_sim.core.sensors.{name}"] = mod

    spec = importlib.util.spec_from_file_location("plume_nav_sim.core.sensors", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plume_nav_sim.core.sensors"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sensors_module():
    return load_sensors_module()


@pytest.mark.parametrize(
    "sensor_type,flag",
    [
        ("BinarySensor", "BINARY_SENSOR_AVAILABLE"),
        ("ConcentrationSensor", "CONCENTRATION_SENSOR_AVAILABLE"),
        ("GradientSensor", "GRADIENT_SENSOR_AVAILABLE"),
    ],
)
def test_create_sensor_raises_import_error_when_unavailable(sensors_module, monkeypatch, sensor_type, flag):
    monkeypatch.setattr(sensors_module, flag, False)
    with pytest.raises(ImportError):
        sensors_module.create_sensor_from_config({"type": sensor_type})


def test_create_historical_sensor_raises_import_error_when_unavailable(sensors_module, monkeypatch):
    monkeypatch.setattr(sensors_module, "HISTORICAL_SENSOR_AVAILABLE", False)
    base_config = {"type": "BinarySensor"}
    with pytest.raises(ImportError):
        sensors_module.create_historical_sensor(base_config)
