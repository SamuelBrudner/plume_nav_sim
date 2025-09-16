"""
Minimal stubs for core test utilities expected by the test harness. These are
lightweight to allow import without pulling in the entire infra.
"""
from typing import Optional, Dict, Any
from ...plume_nav_sim.envs.plume_search_env import create_plume_search_env, PlumeSearchEnv

class CoreComponentTestFixtures:
    pass

class StateManagementTestUtilities:
    pass

class CorePerformanceBenchmark:
    pass

def create_core_test_environment(config: Optional[Dict[str, Any]] = None) -> PlumeSearchEnv:
    return create_plume_search_env(**(config or {}))
