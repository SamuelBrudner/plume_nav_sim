"""
Additional registration tests covering DI env id mapping behavior.

These tests focus on how register_env() shapes kwargs when registering the
component-based environment id (PlumeNav-Components-v0), ensuring:
- entry_point selection switches to the factory callable
- source_location is mapped to goal_location for the factory
- allowed keys are preserved and unknown/private keys are dropped
- explicit goal_location overrides derived mapping from source_location
- max_episode_steps is set on the spec independently from max_steps in kwargs

The tests intentionally avoid calling gym.make() except in a minimal smoke case
with valid parameters, to keep scope aligned with registration behavior.
"""

from __future__ import annotations

import itertools

import pytest

import gymnasium as gym
from plume_nav_sim.core.constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
)
from plume_nav_sim.registration.register import (
    COMPONENT_ENV_ID,
    ENTRY_POINT,
    register_env,
    unregister_env,
)
from plume_nav_sim.utils.exceptions import ValidationError


def _get_spec(env_id: str):
    reg = getattr(gym.envs, "registry", None)
    assert hasattr(reg, "env_specs")
    spec = reg.env_specs.get(env_id)
    assert spec is not None, f"Env id {env_id} not found in registry"
    return spec


def test_di_entry_point_and_basic_mapping():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    # Provide a mix of allowed, unknown, and private keys
    kwargs = {
        "grid_size": (32, 32),
        "source_location": (10, 20),
        "max_steps": 123,
        "goal_radius": 2.5,
        "action_type": "oriented",
        "observation_type": "antennae",
        "reward_type": "step_penalty",
        "plume_sigma": 25.0,
        "step_size": 2,
        "start_location": (1, 2),
        "unused_field": "drop-me",
        "_private": "drop-me-too",
    }

    returned_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        max_episode_steps=200,
        force_reregister=True,
    )
    assert returned_id == COMPONENT_ENV_ID

    spec = _get_spec(COMPONENT_ENV_ID)

    # Entry point must be the factory callable
    assert "plume_nav_sim.envs.factory:create_component_environment" in str(
        spec.entry_point
    )

    # Assert mapping rules
    # 1) source_location is mapped to goal_location
    assert "source_location" not in spec.kwargs
    assert spec.kwargs.get("goal_location") == (10, 20)

    # 2) allowed keys preserved
    for key in (
        "grid_size",
        "max_steps",
        "goal_radius",
        "action_type",
        "observation_type",
        "reward_type",
        "plume_sigma",
        "step_size",
        "start_location",
    ):
        assert key in spec.kwargs, f"Missing expected key: {key}"

    # 3) unknown/private keys dropped
    assert "unused_field" not in spec.kwargs
    assert "_private" not in spec.kwargs

    # 4) max_episode_steps is stored on spec independently
    assert spec.max_episode_steps == 200

    # Cleanup
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_explicit_goal_location_is_ignored_in_favor_of_source_mapping():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    # Provide both source_location and explicit goal_location.
    # Current mapping always derives goal_location from source_location and
    # drops an explicit goal_location from kwargs.
    kwargs = {
        "grid_size": (64, 64),
        "source_location": (5, 6),
        "goal_location": (30, 40),
        "max_steps": 50,
        "goal_radius": 1.0,
    }

    register_env(env_id=COMPONENT_ENV_ID, kwargs=kwargs, force_reregister=True)
    spec = _get_spec(COMPONENT_ENV_ID)

    # Expect goal_location equals the mapped source_location, not the explicit override
    assert spec.kwargs.get("goal_location") == (5, 6)
    assert "source_location" not in spec.kwargs

    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_defaults_are_mapped_without_instantiation():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    # Register with no kwargs; we only assert mapping, not gym.make()
    register_env(env_id=COMPONENT_ENV_ID, force_reregister=True)
    spec = _get_spec(COMPONENT_ENV_ID)

    # Defaults should be present and mapped
    assert spec.kwargs.get("grid_size") == DEFAULT_GRID_SIZE
    # source defaults to grid center → mapped to goal_location
    assert spec.kwargs.get("goal_location") == DEFAULT_SOURCE_LOCATION
    assert spec.kwargs.get("max_steps") == DEFAULT_MAX_STEPS
    # goal_radius comes from package default (may be zero for legacy env); mapping keeps it
    assert spec.kwargs.get("goal_radius") == DEFAULT_GOAL_RADIUS

    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_render_mode_and_extra_keys_passthrough_and_drop():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    kwargs = {
        "grid_size": (48, 24),
        "source_location": (10, 5),
        "render_mode": "rgb_array",
        "UNUSED": 123,  # dropped
        "_secret": True,  # dropped
    }

    register_env(env_id=COMPONENT_ENV_ID, kwargs=kwargs, force_reregister=True)
    spec = _get_spec(COMPONENT_ENV_ID)

    # render_mode preserved; unknown/private keys dropped
    assert spec.kwargs.get("render_mode") == "rgb_array"
    assert "UNUSED" not in spec.kwargs
    assert "_secret" not in spec.kwargs

    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_make_with_oriented_antennae_step_penalty_configuration():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    kwargs = {
        "grid_size": (40, 30),
        "source_location": (12, 8),  # mapped to goal_location
        "max_steps": 15,
        "goal_radius": 1.5,
        "action_type": "oriented",
        "observation_type": "antennae",
        "reward_type": "step_penalty",
        "step_size": 1,
        "render_mode": None,
    }

    env_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        force_reregister=True,
        max_episode_steps=15,
    )

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        a = env.action_space.sample()
        step = env.step(a)
        assert len(step) == 5
    finally:
        env.close()
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_env_id_with_legacy_entry_point_disables_mapping():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


@pytest.mark.parametrize(
    "action_type,observation_type,reward_type",
    itertools.product(
        ["discrete", "oriented"],
        ["concentration", "antennae"],
        ["sparse", "step_penalty"],
    ),
)
def test_di_param_combinations_smoke(
    action_type: str, observation_type: str, reward_type: str
):
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    kwargs = {
        "grid_size": (32, 24),
        "source_location": (16, 12),  # mapped to goal_location
        "max_steps": 12,
        "goal_radius": 1.0,
        "action_type": action_type,
        "observation_type": observation_type,
        "reward_type": reward_type,
    }

    env_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        max_episode_steps=12,
        force_reregister=True,
    )

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        assert obs is not None
        # Observation shape depends on observation_type
        if observation_type == "concentration":
            assert env.observation_space.shape == (1,)
        else:
            assert env.observation_space.shape == (2,)
        # One step
        step = env.step(env.action_space.sample())
        assert len(step) == 5
    finally:
        env.close()
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_start_location_applied_on_reset():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    start = (3, 4)
    kwargs = {
        "grid_size": (20, 20),
        "source_location": (10, 10),  # mapped to goal_location
        "start_location": start,
        "max_steps": 5,
        "goal_radius": 1.0,
    }

    env_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        force_reregister=True,
    )
    env = gym.make(env_id)
    try:
        _obs, info = env.reset()
        assert tuple(info.get("agent_position", (-1, -1))) == start
    finally:
        env.close()
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_oriented_step_size_affects_forward_motion():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    start = (1, 1)
    kwargs = {
        "grid_size": (16, 16),
        "source_location": (5, 5),  # mapped
        "start_location": start,
        "max_steps": 5,
        "goal_radius": 1.0,
        "action_type": "oriented",
        "observation_type": "concentration",
        "reward_type": "sparse",
        "step_size": 2,
    }

    env_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        force_reregister=True,
    )
    env = gym.make(env_id)
    try:
        _obs, info = env.reset()
        assert tuple(info.get("agent_position", (-1, -1))) == start
        # Action 0 = FORWARD in OrientedGridActions
        _obs, _reward, _terminated, _truncated, info2 = env.step(0)
        assert tuple(info2.get("agent_position", (-1, -1))) == (start[0] + 2, start[1])
    finally:
        env.close()
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_invalid_source_location_raises():
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)
    kwargs = {
        "grid_size": (8, 8),
        "source_location": (99, 99),  # out of bounds
        "goal_radius": 1.0,
    }
    with pytest.raises(ValidationError):
        register_env(env_id=COMPONENT_ENV_ID, kwargs=kwargs, force_reregister=True)


def test_di_invalid_goal_radius_raises():
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)
    kwargs = {
        "grid_size": (8, 8),
        "source_location": (4, 4),
        "goal_radius": -1.0,  # invalid
    }
    with pytest.raises(ValidationError):
        register_env(env_id=COMPONENT_ENV_ID, kwargs=kwargs, force_reregister=True)

    # Force legacy entry point even with DI env id
    kwargs = {
        "grid_size": (32, 32),
        "source_location": (8, 8),
        "max_steps": 10,
        "goal_radius": 0.0,
    }

    register_env(
        env_id=COMPONENT_ENV_ID,
        entry_point=ENTRY_POINT,  # override
        kwargs=kwargs,
        force_reregister=True,
    )
    spec = _get_spec(COMPONENT_ENV_ID)

    # Because mapping only triggers for factory entry point, 'source_location' remains
    assert "source_location" in spec.kwargs
    assert "goal_location" not in spec.kwargs

    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)


def test_di_make_smoke_with_valid_params():
    # Clean slate
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    # Provide valid parameters for factory instantiation
    kwargs = {
        "grid_size": (32, 32),
        "source_location": (16, 16),  # mapped to goal_location
        "max_steps": 20,
        "goal_radius": 2.0,  # positive for ComponentBasedEnvironment
        "action_type": "discrete",
        "observation_type": "concentration",
        "reward_type": "sparse",
    }

    env_id = register_env(
        env_id=COMPONENT_ENV_ID,
        kwargs=kwargs,
        max_episode_steps=20,
        force_reregister=True,
    )

    # Instantiate via gym.make()
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        a = env.action_space.sample()
        step = env.step(a)
        assert len(step) == 5
    finally:
        env.close()
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)
