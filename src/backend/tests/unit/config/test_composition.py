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
