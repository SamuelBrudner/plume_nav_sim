import numpy as np
import pytest

from plume_nav_sim.core.controllers import SingleAgentController, MultiAgentController

# Deterministic tests for environments without Hypothesis

def test_multi_agent_permissive_speed_clamp_deterministic():
    positions = np.zeros((2, 2))
    speeds = [2.0, 2.0]
    max_speeds = [1.0, 1.0]
    controller = MultiAgentController(positions=positions, speeds=speeds, max_speeds=max_speeds)
    controller.step(np.zeros((1, 1)))
    assert np.allclose(controller.speeds, controller.max_speeds)


def test_multi_agent_strict_speed_validation_raises_deterministic():
    positions = np.zeros((2, 2))
    speeds = [2.0, 2.0]
    max_speeds = [1.0, 1.0]
    with pytest.raises(ValueError):
        MultiAgentController(
            positions=positions,
            speeds=speeds,
            max_speeds=max_speeds,
            strict_validation=True,
        )


# Property-based tests using Hypothesis
try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:  # pragma: no cover - hypothesis is optional
    HYPOTHESIS_AVAILABLE = False

if HYPOTHESIS_AVAILABLE:

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        max_speed=st.floats(min_value=0.1, max_value=10.0),
        excess=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_single_agent_permissive_speed_clamp(max_speed, excess):
        controller = SingleAgentController(speed=max_speed + excess, max_speed=max_speed)
        controller.step(np.zeros((1, 1)))
        assert controller.speeds[0] == pytest.approx(controller.max_speeds[0])

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        max_speed=st.floats(min_value=0.1, max_value=10.0),
        excess=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_single_agent_strict_speed_validation_raises(max_speed, excess):
        with pytest.raises(ValueError):
            SingleAgentController(
                speed=max_speed + excess, max_speed=max_speed, strict_validation=True
            )

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        max_speed=st.floats(min_value=0.1, max_value=10.0),
        excess=st.floats(min_value=0.1, max_value=5.0),
        num_agents=st.integers(min_value=1, max_value=5),
    )
    def test_multi_agent_permissive_speed_clamp(max_speed, excess, num_agents):
        speeds = [max_speed + excess] * num_agents
        max_speeds = [max_speed] * num_agents
        positions = np.zeros((num_agents, 2))
        controller = MultiAgentController(
            positions=positions, speeds=speeds, max_speeds=max_speeds
        )
        controller.step(np.zeros((1, 1)))
        assert np.allclose(controller.speeds, controller.max_speeds)

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        max_speed=st.floats(min_value=0.1, max_value=10.0),
        excess=st.floats(min_value=0.1, max_value=5.0),
        num_agents=st.integers(min_value=1, max_value=5),
    )
    def test_multi_agent_strict_speed_validation_raises(max_speed, excess, num_agents):
        speeds = [max_speed + excess] * num_agents
        max_speeds = [max_speed] * num_agents
        positions = np.zeros((num_agents, 2))
        with pytest.raises(ValueError):
            MultiAgentController(
                positions=positions,
                speeds=speeds,
                max_speeds=max_speeds,
                strict_validation=True,
            )
else:  # pragma: no cover - executed only when Hypothesis missing

    def test_hypothesis_not_installed():
        pytest.skip("Hypothesis not available")
