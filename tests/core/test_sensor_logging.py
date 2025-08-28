import pytest
from unittest.mock import MagicMock

from src.plume_nav_sim.core import sensors as sensors_module
from src.plume_nav_sim.core.sensors.binary_sensor import BinarySensor
from src.plume_nav_sim.core.sensors import historical_sensor as hs
from src.plume_nav_sim.core.sensors.historical_sensor import HistoricalSensorConfig


def test_create_sensor_from_config_logs_protocol_validation(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(sensors_module, "logger", mock_logger)

    sensors_module.create_sensor_from_config({"type": "BinarySensor"})

    assert mock_logger.debug.called, "Expected debug log for protocol validation"
    kwargs = mock_logger.debug.call_args.kwargs
    assert "metadata" in kwargs
    assert "observation_shape" in kwargs


def test_historical_sensor_logs_base_sensor_validation(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(hs, "logger", mock_logger)

    base_sensor = BinarySensor()
    config = HistoricalSensorConfig(history_length=5)
    hs.HistoricalSensor(base_sensor=base_sensor, config=config)

    assert mock_logger.debug.called, "Expected debug log for base sensor validation"
    kwargs = mock_logger.debug.call_args.kwargs
    assert "metadata" in kwargs
    assert "observation_shape" in kwargs
