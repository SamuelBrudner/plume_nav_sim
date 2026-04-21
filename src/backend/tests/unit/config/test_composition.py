from __future__ import annotations

import warnings

import plume_nav_sim as pns
from plume_nav_sim.config import SimulationSpec, build_env


def test_build_env_bypasses_public_make_env_router(monkeypatch) -> None:
    def _unexpected_make_env(**kwargs):
        raise AssertionError("build_env should construct envs directly from the factory")

    monkeypatch.setattr(pns, "make_env", _unexpected_make_env)

    env = build_env(
        SimulationSpec(
            grid_size=(16, 16),
            source_location=(8, 8),
            max_steps=5,
            render=False,
        )
    )

    try:
        obs, info = env.reset(seed=0)

        assert env.goal_location.x == 8
        assert env.goal_location.y == 8
        assert obs is not None
        assert info["goal_location"] == (8, 8)
    finally:
        env.close()


def test_build_env_suppresses_component_deprecation_warning() -> None:
    spec = SimulationSpec(
        grid_size=(16, 16),
        source_location=(8, 8),
        max_steps=5,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = build_env(spec)

    try:
        dep_messages = [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]
        assert dep_messages == []
    finally:
        env.close()


def test_simulation_spec_accepts_zero_based_coordinates() -> None:
    env = build_env(
        SimulationSpec(
            grid_size=(16, 16),
            source_location=(0, 0),
            start_location=(0, 0),
            max_steps=5,
            render=False,
        )
    )

    try:
        obs, info = env.reset(seed=0)

        assert obs is not None
        assert env.goal_location.x == 0
        assert env.goal_location.y == 0
        assert info["agent_position"] == (0, 0)
    finally:
        env.close()


def test_simulation_spec_preserves_step_size_wind_and_render_mode() -> None:
    env = build_env(
        SimulationSpec(
            grid_size=(12, 12),
            source_location=(4, 5),
            max_steps=9,
            action_type="oriented",
            step_size=2,
            observation_type="wind_vector",
            wind_noise_std=0.25,
            enable_wind=True,
            wind_direction_deg=45.0,
            wind_speed=1.5,
            render=False,
            render_mode="human",
        )
    )

    try:
        assert env.render_mode == "human"
        assert getattr(env.action_model, "step_size", None) == 2
        assert getattr(env.sensor_model, "noise_std", None) == 0.25
        assert env.wind_field is not None
    finally:
        env.close()
