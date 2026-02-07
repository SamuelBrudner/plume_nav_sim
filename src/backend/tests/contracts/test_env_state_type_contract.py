"""Contract tests for the typed EnvState structure."""

from __future__ import annotations

import numpy as np

from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.types import Coordinates, EnvState, GridSize
from plume_nav_sim.envs import ComponentBasedEnvironment
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.gaussian import GaussianPlume
from plume_nav_sim.rewards import SparseGoalReward

EXPECTED_ENV_STATE_KEYS = {
    "agent_state",
    "plume_field",
    "concentration_field",
    "wind_field",
    "goal_location",
    "grid_size",
    "step_count",
    "max_steps",
    "rng",
}


def _make_component_env() -> ComponentBasedEnvironment:
    grid_size = GridSize(width=16, height=12)
    goal_location = Coordinates(8, 6)
    return ComponentBasedEnvironment(
        action_processor=DiscreteGridActions(step_size=1),
        observation_model=ConcentrationSensor(),
        reward_function=SparseGoalReward(goal_position=goal_location, goal_radius=2.0),
        concentration_field=GaussianPlume(
            grid_size=grid_size,
            source_location=goal_location,
            sigma=4.0,
        ),
        wind_field=None,
        grid_size=grid_size,
        max_steps=20,
        goal_location=goal_location,
        goal_radius=2.0,
        start_location=Coordinates(1, 1),
        _warn_deprecated=False,
    )


def test_env_state_typed_dict_required_keys_match_contract():
    assert EnvState.__required_keys__ == EXPECTED_ENV_STATE_KEYS


def test_component_env_build_env_state_dict_matches_typed_structure():
    env = _make_component_env()
    env.reset(seed=7)

    env_state: EnvState = env._build_env_state_dict()

    assert set(env_state.keys()) == EXPECTED_ENV_STATE_KEYS
    assert env_state["agent_state"] is env._agent_state
    assert env_state["plume_field"] is env._concentration_field.field_array
    assert isinstance(env_state["plume_field"], np.ndarray)
    assert env_state["concentration_field"] is env._concentration_field
    assert env_state["wind_field"] is None
    assert env_state["goal_location"] == env.goal_location
    assert env_state["grid_size"] == env.grid_size
    assert env_state["step_count"] == env._step_count
    assert env_state["max_steps"] == env.max_steps
    assert env_state["rng"] is env._rng
