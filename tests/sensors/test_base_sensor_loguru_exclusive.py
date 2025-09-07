from loguru import logger as loguru_logger
import importlib.util
import pathlib
import sys
import types


def test_base_sensor_uses_loguru_exclusively():
    module_path = pathlib.Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'core' / 'sensors' / 'base_sensor.py'

    # Stub package hierarchy to satisfy relative imports without executing package __init__
    pkg_plume = types.ModuleType('plume_nav_sim'); pkg_plume.__path__ = []
    pkg_core = types.ModuleType('plume_nav_sim.core'); pkg_core.__path__ = []
    pkg_core_sensors = types.ModuleType('plume_nav_sim.core.sensors'); pkg_core_sensors.__path__ = []
    pkg_config = types.ModuleType('plume_nav_sim.config'); pkg_config.__path__ = []
    pkg_schemas = types.ModuleType('plume_nav_sim.config.schemas'); pkg_schemas.__path__ = []
    class SensorConfig: ...
    pkg_schemas.SensorConfig = SensorConfig

    pkg_protocols = types.ModuleType('plume_nav_sim.protocols'); pkg_protocols.__path__ = []
    pkg_protocols_sensor = types.ModuleType('plume_nav_sim.protocols.sensor')
    class SensorProtocol: ...
    pkg_protocols_sensor.SensorProtocol = SensorProtocol

    sys.modules.update({
        'plume_nav_sim': pkg_plume,
        'plume_nav_sim.core': pkg_core,
        'plume_nav_sim.core.sensors': pkg_core_sensors,
        'plume_nav_sim.config': pkg_config,
        'plume_nav_sim.config.schemas': pkg_schemas,
        'plume_nav_sim.protocols': pkg_protocols,
        'plume_nav_sim.protocols.sensor': pkg_protocols_sensor,
    })

    spec = importlib.util.spec_from_file_location('plume_nav_sim.core.sensors.base_sensor', module_path)
    base_sensor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base_sensor
    spec.loader.exec_module(base_sensor)

    assert base_sensor.logger is loguru_logger
    assert not hasattr(base_sensor, 'LOGURU_AVAILABLE')
