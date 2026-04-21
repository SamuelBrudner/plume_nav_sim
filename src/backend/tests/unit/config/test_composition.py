from __future__ import annotations

import sys
import warnings
from pathlib import Path

import plume_nav_sim as pns
import pytest

from plume_nav_sim.config import SimulationSpec, build_env, load_policy


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


def test_build_env_has_no_deprecation_warnings() -> None:
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
        assert caught == []
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


def test_load_policy_preserves_nested_missing_dependency(tmp_path: Path) -> None:
    module_dir = tmp_path / "mods"
    module_dir.mkdir()
    module_path = module_dir / "badpolicy.py"
    module_path.write_text(
        "import definitely_missing_dependency\n"
        "class Policy:\n"
        "    pass\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(module_dir))
    try:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            load_policy("badpolicy.Policy")
    finally:
        sys.path.remove(str(module_dir))

    assert exc_info.value.name == "definitely_missing_dependency"
