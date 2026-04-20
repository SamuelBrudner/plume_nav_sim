from __future__ import annotations

import pytest

import plume_nav_sim as pns
import plume_nav_sim.config as config_api
from plume_nav_sim.config import (
    ActionConfig,
    ComponentEnvironmentConfig,
    ObservationConfig,
    PlumeConfig,
    RewardConfig,
    WindConfig,
    component_environment_config_to_kwargs,
    create_environment_from_config,
    get_default_component_environment_config,
    get_default_environment_config,
)
from plume_nav_sim.envs.config_types import EnvironmentConfig as CanonicalEnvironmentConfig


def test_canonical_and_component_config_surfaces_are_distinct() -> None:
    assert pns.EnvironmentConfig is CanonicalEnvironmentConfig
    assert config_api.ComponentEnvironmentConfig is not CanonicalEnvironmentConfig
    assert issubclass(config_api.EnvironmentConfig, config_api.ComponentEnvironmentConfig)


def test_component_environment_config_alias_warns_on_instantiation() -> None:
    with pytest.warns(DeprecationWarning, match="plume_nav_sim\\.config\\.EnvironmentConfig is deprecated"):
        config = config_api.EnvironmentConfig()

    assert isinstance(config, config_api.ComponentEnvironmentConfig)
    assert not isinstance(config, CanonicalEnvironmentConfig)


def test_default_config_helpers_split_canonical_and_component_roles() -> None:
    canonical = get_default_environment_config()
    component = get_default_component_environment_config()

    assert isinstance(canonical, CanonicalEnvironmentConfig)
    assert isinstance(component, ComponentEnvironmentConfig)


def test_component_config_adapter_normalizes_explicit_env_kwargs() -> None:
    config = ComponentEnvironmentConfig(
        grid_size=(32, 24),
        goal_location=(5, 6),
        start_location=(1, 2),
        max_steps=77,
        action=ActionConfig(type="discrete", step_size=3),
        observation=ObservationConfig(type="wind_vector", noise_std=0.15),
        reward=RewardConfig(
            type="step_penalty",
            goal_radius=2.5,
            goal_reward=3.0,
            step_penalty=0.2,
        ),
        plume=PlumeConfig(sigma=11.0),
        wind=WindConfig(direction_deg=90.0, speed=2.0),
    )

    kwargs = component_environment_config_to_kwargs(config)

    assert kwargs["grid_size"] == (32, 24)
    assert kwargs["goal_location"] == (5, 6)
    assert kwargs["start_location"] == (1, 2)
    assert kwargs["max_steps"] == 77
    assert kwargs["render_mode"] is None
    assert kwargs["action_type"] == "discrete"
    assert kwargs["step_size"] == 3
    assert kwargs["observation_type"] == "wind_vector"
    assert kwargs["reward_type"] == "step_penalty"
    assert kwargs["plume_sigma"] == 11.0
    assert kwargs["wind_direction_deg"] == 90.0
    assert kwargs["wind_speed"] == 2.0
    assert kwargs["enable_wind"] is True


def test_deprecated_factory_alias_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ComponentEnvironmentConfig()
    seen: dict[str, object] = {}

    def _fake_create_component_environment_from_config(config_obj: object) -> str:
        seen["config"] = config_obj
        return "sentinel"

    monkeypatch.setattr(
        "plume_nav_sim.config.factories.create_component_environment_from_config",
        _fake_create_component_environment_from_config,
    )

    with pytest.warns(DeprecationWarning, match="create_environment_from_config is deprecated"):
        result = create_environment_from_config(config)

    assert result == "sentinel"
    assert seen["config"] is config
