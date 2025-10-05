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
)
from .factories import (
    create_action_processor,
    create_concentration_field,
    create_environment_from_config,
    create_observation_model,
    create_reward_function,
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
    "EnvironmentConfig",
    # Factory functions
    "create_action_processor",
    "create_observation_model",
    "create_reward_function",
    "create_concentration_field",
    "create_environment_from_config",
    # Test configs
    "REPRODUCIBILITY_SEEDS",
    "TestConfigFactory",
    "create_edge_case_test_config",
    "create_integration_test_config",
    "create_performance_test_config",
    "create_reproducibility_test_config",
    "create_unit_test_config",
]
