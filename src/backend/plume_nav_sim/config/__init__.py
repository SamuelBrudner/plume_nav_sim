"""Canonical config helpers for plume_nav_sim."""

from ..envs.config_types import EnvironmentConfig, create_environment_config
from .composition import (
    BuiltinPolicyName,
    LoadedPolicy,
    PolicySpec,
    SimulationSpec,
    build_env,
    build_policy,
    create_simulation_spec,
    load_policy,
    prepare,
    reset_policy_if_possible,
)
from .defaults import (
    get_complete_default_config,
    get_default_environment_config,
)
from .registration import (
    ENV_ID,
    ensure_registered,
    is_registered,
    register_env,
    unregister_env,
)
from .testing_configs import (
    REPRODUCIBILITY_SEEDS,
    TestConfigFactory,
    create_edge_case_test_config,
    create_integration_test_config,
    create_performance_test_config,
    create_reproducibility_test_config,
    create_unit_test_config,
)

__all__ = [
    "EnvironmentConfig",
    "create_environment_config",
    # Defaults
    "get_complete_default_config",
    "get_default_environment_config",
    # Composition helpers and specs
    "BuiltinPolicyName",
    "LoadedPolicy",
    "PolicySpec",
    "SimulationSpec",
    "create_simulation_spec",
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
    # Test configs
    "REPRODUCIBILITY_SEEDS",
    "TestConfigFactory",
    "create_edge_case_test_config",
    "create_integration_test_config",
    "create_performance_test_config",
    "create_reproducibility_test_config",
    "create_unit_test_config",
]
