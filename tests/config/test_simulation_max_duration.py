from loguru import logger
import pytest
from pydantic import ValidationError
from src.plume_nav_sim.config.schemas import SimulationConfig


def test_max_duration_negative_logs_and_raises(caplog):
    """SimulationConfig should log and raise on non-positive max_duration."""
    with caplog.at_level(logger.DEBUG):
        with pytest.raises(ValidationError, match="ensure this value is greater than 0"):
            SimulationConfig(max_duration=-1)
    assert "max_duration is not positive" in caplog.text
