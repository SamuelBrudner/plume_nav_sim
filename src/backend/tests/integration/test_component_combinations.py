"""Integration tests covering all component combinations.

Ensures compatibility across action processors, observation models, and
reward functions in the component-based environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from plume_nav_sim.envs import create_component_environment

ACTION_TYPES = ["discrete", "oriented"]
OBSERVATION_TYPES = ["concentration", "antennae"]
REWARD_TYPES = ["sparse", "step_penalty"]


@pytest.mark.parametrize("action_type", ACTION_TYPES)
@pytest.mark.parametrize("observation_type", OBSERVATION_TYPES)
@pytest.mark.parametrize("reward_type", REWARD_TYPES)
def test_environment_component_combinations(action_type, observation_type, reward_type):
    """All component combinations should initialize and run a single step."""
    env = create_component_environment(
        action_type=action_type,
        observation_type=observation_type,
        reward_type=reward_type,
        grid_size=(32, 32),
        goal_location=(16, 16),
        max_steps=25,
    )

    try:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        if reward_type == "sparse":
            assert reward in (0.0, 1.0)
        elif reward_type == "step_penalty":
            # Expected to be goal reward or negative penalty
            assert reward <= 10.0
    finally:
        env.close()
