import pytest
from src.plume_nav_sim.config.schemas import SingleAgentConfig, NavigatorConfig, MultiAgentConfig


@pytest.mark.parametrize("cls, kwargs", [
    (SingleAgentConfig, {"orientation": -1}),
    (NavigatorConfig, {"orientation": -1}),
])
def test_orientation_lower_bound(cls, kwargs):
    with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
        cls(**kwargs)


@pytest.mark.parametrize("cls, kwargs", [
    (SingleAgentConfig, {"orientation": 361}),
    (NavigatorConfig, {"orientation": 361}),
])
def test_orientation_upper_bound(cls, kwargs):
    with pytest.raises(ValueError, match="ensure this value is less than or equal to 360"):
        cls(**kwargs)


@pytest.mark.parametrize("orientations, msg", [
    ([-1.0], "ensure this value is greater than or equal to 0"),
    ([361.0], "ensure this value is less than or equal to 360"),
])
def test_multi_agent_orientation_bounds(orientations, msg):
    positions = [[0.0, 0.0] for _ in orientations]
    with pytest.raises(ValueError, match=msg):
        MultiAgentConfig(positions=positions, orientations=orientations)
