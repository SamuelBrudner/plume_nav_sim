from __future__ import annotations

import pytest

from plume_nav_sim.envs.factory import create_component_environment
from plume_nav_sim.policies import TemporalDerivativePolicy


def test_temporal_derivative_requires_sensor_index_for_antennae():
    env = create_component_environment(
        action_type="oriented",
        observation_type="antennae",
        reward_type="sparse",
    )
    try:
        policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
        obs, _ = env.reset(seed=0)

        with pytest.raises(ValueError, match="sensor_index"):
            policy.select_action(obs, explore=False)
    finally:
        env.close()


def test_temporal_derivative_can_select_antenna_with_sensor_index():
    env = create_component_environment(
        action_type="oriented",
        observation_type="antennae",
        reward_type="sparse",
    )
    try:
        policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0, sensor_index=0)
        policy.reset(seed=1)
        obs, _ = env.reset(seed=1)

        action = policy.select_action(obs, explore=False)
        assert env.action_space.contains(action)
    finally:
        env.close()


def test_temporal_derivative_errors_on_wind_vector_observation():
    env = create_component_environment(
        action_type="oriented",
        observation_type="wind_vector",
        reward_type="sparse",
    )
    try:
        policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
        obs, _ = env.reset(seed=2)

        with pytest.raises(ValueError, match="scalar concentration"):
            policy.select_action(obs, explore=False)
    finally:
        env.close()
