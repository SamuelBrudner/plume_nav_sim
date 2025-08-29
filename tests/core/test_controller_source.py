import pytest
from unittest.mock import MagicMock
from plume_nav_sim.core.controllers import SingleAgentController
from plume_nav_sim.core.protocols import SourceProtocol


def test_controller_accepts_source():
    mock_source = MagicMock(spec=SourceProtocol)
    controller = SingleAgentController(source=mock_source)
    assert controller.source is mock_source


def test_sampling_without_source_or_plume_state_raises():
    controller = SingleAgentController()
    with pytest.raises(ValueError):
        controller.sample_odor(None)
