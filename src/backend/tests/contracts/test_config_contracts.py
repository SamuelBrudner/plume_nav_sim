"""Contract enforcement tests for configuration dataclasses.

Tests that config interfaces match CONTRACTS.md specification.
Ensures required vs optional parameters are correctly defined.

Reference: ../../CONTRACTS.md Section "Configuration Dataclasses"
"""

from dataclasses import MISSING, fields

import pytest

from plume_nav_sim.core.episode_manager import EpisodeManagerConfig
from plume_nav_sim.core.reward_calculator import RewardCalculatorConfig
from plume_nav_sim.core.state_manager import StateManagerConfig
from plume_nav_sim.core.types import EnvironmentConfig


class TestRewardCalculatorConfigContract:
    """Enforce RewardCalculatorConfig contract from CONTRACTS.md."""

    def test_required_parameters_have_no_defaults(self):
        """Required params must not have defaults per CONTRACTS.md."""
        config_fields = {f.name: f for f in fields(RewardCalculatorConfig)}

        # Per CONTRACTS.md, these are REQUIRED (no defaults)
        required = ["goal_radius", "reward_goal_reached", "reward_default"]

        for param_name in required:
            field = config_fields[param_name]
            assert (
                field.default == MISSING and field.default_factory == MISSING
            ), f"{param_name} must be REQUIRED (no default) per CONTRACTS.md"

    def test_optional_parameters_have_defaults(self):
        """Optional params must have defaults per CONTRACTS.md."""
        config_fields = {f.name: f for f in fields(RewardCalculatorConfig)}

        # Per CONTRACTS.md, these are OPTIONAL (with defaults)
        optional_with_defaults = {
            "distance_calculation_method": "euclidean",
            "distance_precision": 1e-12,
            "enable_performance_monitoring": True,
            "enable_caching": True,
        }

        for param_name, expected_default in optional_with_defaults.items():
            field = config_fields[param_name]
            has_default = (
                field.default != MISSING  # type: ignore
                or field.default_factory != MISSING  # type: ignore
            )
            assert (
                has_default
            ), f"{param_name} must have a default value per CONTRACTS.md"

    def test_validation_rules_from_contracts(self):
        """Test validation rules documented in CONTRACTS.md."""
        # goal_radius >= 0 and finite
        with pytest.raises(Exception):  # ValidationError
            RewardCalculatorConfig(
                goal_radius=-1.0,  # Invalid: negative
                reward_goal_reached=1.0,
                reward_default=0.0,
            )

        # reward values must be finite
        with pytest.raises(Exception):
            RewardCalculatorConfig(
                goal_radius=5.0,
                reward_goal_reached=float("nan"),  # Invalid: NaN
                reward_default=0.0,
            )

    def test_can_instantiate_with_required_only(self):
        """Config should work with only required parameters."""
        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0,
        )

        assert config.goal_radius == 5.0
        assert config.distance_calculation_method == "euclidean"  # default
        assert config.enable_caching is True  # default


class TestEpisodeManagerConfigContract:
    """Enforce EpisodeManagerConfig contract from CONTRACTS.md."""

    def test_required_parameters(self):
        """env_config is REQUIRED per CONTRACTS.md."""
        config_fields = {f.name: f for f in fields(EpisodeManagerConfig)}

        # Only env_config is required
        env_config_field = config_fields["env_config"]
        assert env_config_field.default == MISSING  # type: ignore

    def test_optional_parameters_have_defaults(self):
        """enable_performance_monitoring and enable_state_validation optional."""
        config_fields = {f.name: f for f in fields(EpisodeManagerConfig)}

        # These were made optional in Phase 2 for backward compatibility
        optional_params = [
            "enable_performance_monitoring",
            "enable_state_validation",
        ]

        for param_name in optional_params:
            field = config_fields[param_name]
            assert field.default is not MISSING, (  # type: ignore
                f"{param_name} should have default per backward compatibility fix"
            )

    def test_can_instantiate_with_env_config_only(self):
        """Should work with just env_config (backward compatibility)."""
        env_config = EnvironmentConfig()

        config = EpisodeManagerConfig(env_config=env_config)

        assert config.env_config is env_config
        assert config.enable_performance_monitoring is True  # default
        assert config.enable_state_validation is True  # default


class TestStateManagerConfigContract:
    """Enforce StateManagerConfig contract from CONTRACTS.md."""

    def test_required_parameters(self):
        """grid_size and max_steps are REQUIRED."""
        config_fields = {f.name: f for f in fields(StateManagerConfig)}

        required = ["grid_size", "max_steps"]

        for param_name in required:
            field = config_fields[param_name]
            assert field.default == MISSING, (  # type: ignore
                f"{param_name} must be REQUIRED per CONTRACTS.md"
            )

    def test_optional_parameters_have_defaults(self):
        """Optional params have defaults."""
        from plume_nav_sim.core.geometry import GridSize

        config_fields = {f.name: f for f in fields(StateManagerConfig)}

        optional = [
            "enable_boundary_enforcement",
            "enable_state_validation",
        ]

        for param_name in optional:
            field = config_fields[param_name]
            has_default = (
                field.default != MISSING  # type: ignore
                or field.default_factory != MISSING  # type: ignore
            )
            assert has_default, f"{param_name} should have default"


class TestConfigGeneralContract:
    """Test general contract properties that apply to ALL configs."""

    def test_all_configs_use_dataclass_decorator(self):
        """All config classes must be dataclasses per CONTRACTS.md."""
        from dataclasses import is_dataclass

        configs = [
            RewardCalculatorConfig,
            EpisodeManagerConfig,
            StateManagerConfig,
            EnvironmentConfig,
        ]

        for config_class in configs:
            assert is_dataclass(
                config_class
            ), f"{config_class.__name__} must be a dataclass"

    def test_all_configs_have_post_init(self):
        """Configs should have __post_init__ for validation."""
        configs = [
            RewardCalculatorConfig,
            EpisodeManagerConfig,
            StateManagerConfig,
        ]

        for config_class in configs:
            assert hasattr(
                config_class, "__post_init__"
            ), f"{config_class.__name__} should have __post_init__ for validation"

    def test_all_configs_have_validate_method(self):
        """Configs should have validate() method per CONTRACTS.md."""
        configs = [
            RewardCalculatorConfig,
            EpisodeManagerConfig,
            StateManagerConfig,
        ]

        for config_class in configs:
            assert hasattr(
                config_class, "validate"
            ), f"{config_class.__name__} should have validate() method"

    def test_required_params_before_optional(self):
        """Required parameters must come before optional ones."""
        configs = [
            RewardCalculatorConfig,
            EpisodeManagerConfig,
            StateManagerConfig,
        ]

        for config_class in configs:
            config_fields_list = fields(config_class)

            # Find first optional param (one with default)
            first_optional_idx = None
            for i, field in enumerate(config_fields_list):
                has_default = (
                    field.default != MISSING  # type: ignore
                    or field.default_factory != MISSING  # type: ignore
                )
                if has_default:
                    first_optional_idx = i
                    break

            if first_optional_idx is None:
                continue  # All required

            # All params after first optional must also be optional
            for i in range(first_optional_idx, len(config_fields_list)):
                field = config_fields_list[i]
                has_default = (
                    field.default != MISSING  # type: ignore
                    or field.default_factory != MISSING  # type: ignore
                )
                assert has_default, (
                    f"{config_class.__name__}.{field.name} is required but comes "
                    f"after optional parameters - violates CONTRACTS.md"
                )


class TestConfigImmutability:
    """Test config immutability requirements from CONTRACTS.md."""

    def test_environment_config_is_frozen(self):
        """EnvironmentConfig should be frozen (immutable)."""
        from dataclasses import fields as get_fields

        # Check if frozen
        env_config = EnvironmentConfig()

        # Attempt to modify should fail
        with pytest.raises((AttributeError, Exception)):
            env_config.max_steps = 9999  # type: ignore

    def test_configs_validate_on_construction(self):
        """Configs must validate immediately in __post_init__."""
        # This should raise during construction, not later
        with pytest.raises(Exception):  # ValidationError
            RewardCalculatorConfig(
                goal_radius=-5.0,  # Invalid
                reward_goal_reached=1.0,
                reward_default=0.0,
            )
