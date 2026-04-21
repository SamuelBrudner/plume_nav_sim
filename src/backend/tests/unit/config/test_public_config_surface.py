from __future__ import annotations

import plume_nav_sim as pns
import plume_nav_sim.config as config_api
import pytest

from plume_nav_sim.config import SimulationSpec, build_env, create_simulation_spec
from plume_nav_sim.envs.config_types import EnvironmentConfig as CanonicalEnvironmentConfig


def test_config_public_surface_exports_canonical_environment_config() -> None:
    assert pns.EnvironmentConfig is CanonicalEnvironmentConfig
    assert config_api.EnvironmentConfig is CanonicalEnvironmentConfig
    assert isinstance(config_api.get_default_environment_config(), CanonicalEnvironmentConfig)


def test_removed_component_config_surfaces_are_absent() -> None:
    removed_symbols = [
        "ComponentEnvironmentConfig",
        "component_environment_config_to_spec",
        "component_environment_config_to_kwargs",
        "create_component_environment_from_config",
        "create_environment_from_config",
        "get_default_component_environment_config",
    ]

    for name in removed_symbols:
        assert not hasattr(config_api, name), f"{name} should not be exported"


def test_create_simulation_spec_accepts_canonical_environment_config() -> None:
    config = CanonicalEnvironmentConfig(
        grid_size=(8, 10),
        source_location=(3, 4),
        max_steps=12,
        goal_radius=1.5,
        plume_params={"sigma": 6.0},
        enable_rendering=False,
    )

    spec = create_simulation_spec(config)

    assert spec.grid_size == (8, 10)
    assert spec.source_location == (3, 4)
    assert spec.max_steps == 12
    assert spec.goal_radius == 1.5
    assert spec.plume_sigma == 6.0
    assert spec.render is False


def test_create_simulation_spec_accepts_canonical_mapping() -> None:
    spec = create_simulation_spec(
        {
            "grid_size": [16, 12],
            "source_location": [5, 6],
            "max_steps": 40,
            "goal_radius": 2.5,
            "plume_params": {"sigma": 9.0},
            "enable_rendering": False,
            "action_type": "run_tumble",
            "step_size": 2,
        }
    )

    assert spec.grid_size == (16, 12)
    assert spec.source_location == (5, 6)
    assert spec.max_steps == 40
    assert spec.goal_radius == 2.5
    assert spec.plume_sigma == 9.0
    assert spec.render is False
    assert spec.action_type == "run_tumble"
    assert spec.step_size == 2


def test_create_simulation_spec_rejects_component_style_mapping() -> None:
    with pytest.raises(ValueError, match="component-style config mappings"):
        create_simulation_spec(
            {
                "grid_size": [64, 48],
                "goal_location": [9, 8],
                "action": {"type": "run_tumble", "step_size": 2},
            }
        )


def test_build_env_still_constructs_from_canonical_simulation_spec() -> None:
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
