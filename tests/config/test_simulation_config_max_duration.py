from loguru import logger
import pytest
from pydantic import ValidationError

from plume_nav_sim.config import SimulationConfig


def test_max_duration_must_be_positive_logs_error(caplog):
    caplog.set_level(logger.ERROR, logger="plume_nav_sim.config.schemas")
    with pytest.raises(ValidationError, match="ensure this value is greater than 0"):
        SimulationConfig(max_duration=0)
    assert any("max_duration" in record.getMessage() for record in caplog.records)
