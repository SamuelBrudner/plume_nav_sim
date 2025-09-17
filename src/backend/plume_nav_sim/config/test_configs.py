"""
Targeted re-export of top-level config.test_configs so tests can import via
plume_nav_sim.config.test_configs without relying on that module's __all__.
"""
import importlib
_mod = importlib.import_module('config.test_configs')

# Explicitly export only the symbols used by the test suite
create_unit_test_config = getattr(_mod, 'create_unit_test_config')
create_integration_test_config = getattr(_mod, 'create_integration_test_config')
create_performance_test_config = getattr(_mod, 'create_performance_test_config')
create_reproducibility_test_config = getattr(_mod, 'create_reproducibility_test_config')
create_edge_case_test_config = getattr(_mod, 'create_edge_case_test_config')
TestConfigFactory = getattr(_mod, 'TestConfigFactory')
REPRODUCIBILITY_SEEDS = getattr(_mod, 'REPRODUCIBILITY_SEEDS')

__all__ = [
    'create_unit_test_config', 'create_integration_test_config', 'create_performance_test_config',
    'create_reproducibility_test_config', 'create_edge_case_test_config', 'TestConfigFactory',
    'REPRODUCIBILITY_SEEDS'
]
