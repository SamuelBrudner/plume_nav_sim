"""Configuration package for environment and component setup.

This package provides:
- Pydantic models for component configuration
- Factory functions for config-driven creation
- Test configuration utilities
"""

from .component_configs import (
    ActionConfig,
    EnvironmentConfig,
    ObservationConfig,
    PlumeConfig,
    RewardConfig,
    WindConfig,
)
from .composition import (
    BuiltinPolicyName,
    LoadedPolicy,
    PolicySpec,
    SimulationSpec,
    build_env,
    build_policy,
    load_policy,
    prepare,
    reset_policy_if_possible,
)
from .defaults import get_complete_default_config, get_default_environment_config
from .factories import (
    create_action_processor,
    create_concentration_field,
    create_environment_from_config,
    create_observation_model,
    create_reward_function,
    create_wind_field,
)
from .registration import (
    COMPONENT_ENV_ID,
    ENV_ID,
    ensure_registered,
    is_registered,
    register_env,
    unregister_env,
)
from .test_configs import (
    REPRODUCIBILITY_SEEDS,
    TestConfigFactory,
    create_edge_case_test_config,
    create_integration_test_config,
    create_performance_test_config,
    create_reproducibility_test_config,
    create_unit_test_config,
)

__all__ = [
    # Component configs
    "ActionConfig",
    "ObservationConfig",
    "RewardConfig",
    "PlumeConfig",
    "WindConfig",
    "EnvironmentConfig",
    # Factory functions
    "create_action_processor",
    "create_observation_model",
    "create_reward_function",
    "create_concentration_field",
    "create_wind_field",
    "create_environment_from_config",
    # Defaults
    "get_complete_default_config",
    "get_default_environment_config",
    # Composition helpers and specs
    "BuiltinPolicyName",
    "LoadedPolicy",
    "PolicySpec",
    "SimulationSpec",
    "load_policy",
    "reset_policy_if_possible",
    "build_env",
    "build_policy",
    "prepare",
    # Registration (unified surface)
    "register_env",
    "unregister_env",
    "is_registered",
    "ensure_registered",
    "ENV_ID",
    "COMPONENT_ENV_ID",
    # Test configs
    "REPRODUCIBILITY_SEEDS",
    "TestConfigFactory",
    "create_edge_case_test_config",
    "create_integration_test_config",
    "create_performance_test_config",
    "create_reproducibility_test_config",
    "create_unit_test_config",
]
