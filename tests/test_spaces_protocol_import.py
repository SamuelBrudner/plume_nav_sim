import importlib
import sys
from io import StringIO

from loguru import logger as loguru_logger


def test_sensor_protocol_import_logs_success():
    """Spaces module logs successful SensorProtocol import."""
    sys.modules.pop("plume_nav_sim.envs.spaces", None)
    buffer = StringIO()
    handler_id = loguru_logger.add(buffer, level="INFO")
    importlib.import_module("plume_nav_sim.envs.spaces")
    loguru_logger.remove(handler_id)
    assert "SensorProtocol import successful" in buffer.getvalue()
