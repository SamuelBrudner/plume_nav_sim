"""
Test package for plume_nav_sim library.

This package initializer establishes the test namespace for the plume navigation 
simulation library and enables pytest test discovery for the comprehensive test suite. 
This structure supports the Gymnasium 0.29 migration while maintaining compatibility 
with legacy Gym patterns through the shim layer.

Test Organization:
    - Unit tests: Isolated component testing with comprehensive mock coverage
    - Integration tests: Cross-component validation and Gymnasium API compliance
    - Configuration tests: Hydra configuration composition and frame cache validation
    - Performance tests: Sub-10ms step execution and >90% cache efficiency validation
    - Gymnasium compliance tests: Full environment API standard compliance
    - Shim compatibility tests: Legacy gym.make() compatibility and deprecation handling
    - Cross-repository tests: Integration with place_mem_rl and other RL frameworks

Pytest Discovery:
    This module enables automatic test discovery by pytest across all test modules
    within the plume_nav_sim package structure. Tests are organized by component domain
    following the modernized gymnasium-compliant architecture requirements.

Migration-Specific Testing:
    - Gymnasium API compliance validation using gymnasium.utils.env_checker
    - Legacy Gym compatibility through shim layer testing
    - 4-tuple ⇄ 5-tuple return format conversion validation
    - Deprecation warning emission and migration guidance testing
    - Performance regression detection for frame cache and environment operations

Coverage Requirements:
    - Overall target: >70% line coverage across all library modules
    - Critical path coverage: >90% for core navigation and environment components
    - Frame cache system: >85% coverage for LRU and memory management
    - Shim compatibility layer: >90% coverage for backward compatibility
    - Gymnasium integration: >85% coverage for API compliance

Fixture Sharing:
    Test fixtures are shared across modules through conftest.py configuration,
    enabling consistent mock behavior and test data management across the
    library test suite. This includes enhanced fixtures for:
    - Gymnasium environment creation and API compliance testing
    - Legacy gym.make() shim testing with deprecation warning validation
    - Frame cache performance testing with LRU eviction and memory monitoring
    - Configuration composition testing for new frame_cache and environment settings
    - Performance benchmarking with sub-10ms step execution validation

Testing Best Practices:
    - Deterministic test execution through controlled randomization
    - Isolated test environments with comprehensive fixture teardown
    - Scientific accuracy validation using numpy.testing assertions
    - Performance validation against specified SLA requirements (sub-10ms steps)
    - Backward compatibility validation with existing RL training workflows

Integration Points:
    - gymnasium.utils.env_checker for comprehensive API compliance validation
    - pytest-benchmark for performance regression detection
    - place_mem_rl compatibility testing for cross-repository validation
    - numpy.testing for numerical precision validation (±1e-6 tolerance)
    - psutil for memory pressure monitoring during frame cache testing

Example Usage:
    Run all library tests:
        pytest src/plume_nav_sim/tests/

    Run specific test categories:
        pytest src/plume_nav_sim/tests/test_core.py
        pytest src/plume_nav_sim/tests/test_gymnasium_compliance.py
        pytest src/plume_nav_sim/tests/test_shim_compatibility.py
        pytest -m performance src/plume_nav_sim/tests/

    Run with coverage analysis:
        pytest --cov=plume_nav_sim src/plume_nav_sim/tests/

    Run performance benchmarks:
        pytest --benchmark-only src/plume_nav_sim/tests/test_performance.py

See Also:
    - tests/conftest.py: Shared fixture definitions and test configuration
    - Section 6.6 of Technical Specification: Comprehensive testing strategy
    - docs/migration_guide.md: Gymnasium migration documentation
    - tests/test_gymnasium_compliance.py: API compliance validation
"""

# Package version for test compatibility tracking - aligned with library v0.3.0
__version__ = "0.3.0"

# Test category markers for pytest organization including gymnasium-specific markers
__test_categories__ = [
    "unit",
    "integration", 
    "config",
    "performance",
    "gymnasium_compliance",
    "shim_compatibility",
    "cross_repository",
    "frame_cache",
    "deprecation",
    "backward_compatibility"
]

# Pytest discovery support - ensure this package is recognized
__all__ = [
    "__version__",
    "__test_categories__",
    "get_test_data_path",
    "create_temp_config_file", 
    "check_test_dependencies",
    "validate_gymnasium_environment",
    "assert_performance_requirements",
    "create_mock_gymnasium_env",
    "validate_legacy_compatibility"
]


def pytest_configure(config):
    """
    Pytest configuration hook for plume_nav_sim test customization.
    
    This function is automatically called by pytest during test discovery
    and allows for test-specific configuration including gymnasium-specific
    markers and performance testing setup.
    
    Args:
        config: pytest configuration object
        
    Note:
        This hook is optional and only executed when pytest discovers
        this package during test collection.
    """
    # Register custom markers for test categorization
    for category in __test_categories__:
        config.addinivalue_line(
            "markers", 
            f"{category}: mark test as {category} test category"
        )
    
    # Register gymnasium-specific markers
    config.addinivalue_line(
        "markers",
        "requires_gymnasium: mark test as requiring Gymnasium 0.29.x"
    )
    
    config.addinivalue_line(
        "markers", 
        "legacy_gym_compat: mark test as validating legacy Gym compatibility"
    )
    
    config.addinivalue_line(
        "markers",
        "performance_critical: mark test as enforcing performance SLAs (sub-10ms steps)"
    )


def get_test_data_path():
    """
    Utility function to resolve test data directory path.
    
    Returns:
        pathlib.Path: Absolute path to test data directory
        
    Example:
        >>> from plume_nav_sim.tests import get_test_data_path
        >>> test_data_dir = get_test_data_path()
        >>> config_file = test_data_dir / "sample_config.yaml"
    """
    import pathlib
    return pathlib.Path(__file__).parent / "data"


def create_temp_config_file(config_dict, suffix=".yaml"):
    """
    Utility function for creating temporary configuration files in tests.
    
    Enhanced for gymnasium migration testing with frame_cache and environment
    configuration support.
    
    Args:
        config_dict (dict): Configuration data to write
        suffix (str): File extension for the temporary file
        
    Returns:
        pathlib.Path: Path to the created temporary file
        
    Example:
        >>> config = {
        ...     "frame_cache": {"mode": "lru", "memory_limit_mb": 1024},
        ...     "environment": {"id": "PlumeNavSim-v0"}
        ... }
        >>> temp_file = create_temp_config_file(config)
        >>> # Use temp_file in test, automatic cleanup via pytest tmp_path
    """
    import tempfile
    import yaml
    import json
    import pathlib
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        if suffix.endswith('.yaml') or suffix.endswith('.yml'):
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        elif suffix.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        else:
            f.write(str(config_dict))
    
    return pathlib.Path(f.name)


def check_test_dependencies():
    """
    Verify that required testing dependencies are available for gymnasium migration.
    
    Enhanced to check gymnasium-specific dependencies and frame cache requirements.
    
    Returns:
        dict: Status of each required testing dependency
        
    Raises:
        ImportError: If critical testing dependencies are missing
    """
    dependencies = {}
    
    # Critical testing dependencies
    try:
        import pytest
        dependencies['pytest'] = pytest.__version__
    except ImportError:
        raise ImportError("pytest is required for test execution")
    
    try:
        import numpy
        import numpy.testing
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        raise ImportError("numpy is required for numerical test assertions")
    
    # Gymnasium migration dependencies
    try:
        import gymnasium
        dependencies['gymnasium'] = gymnasium.__version__
        
        # Check for env_checker utility
        try:
            from gymnasium.utils.env_checker import check_env
            dependencies['gymnasium.utils.env_checker'] = 'available'
        except ImportError:
            dependencies['gymnasium.utils.env_checker'] = 'not available - API compliance testing limited'
            
    except ImportError:
        raise ImportError("gymnasium>=0.29.0 is required for environment testing")
    
    # Performance testing dependencies
    try:
        import psutil
        dependencies['psutil'] = psutil.__version__
    except ImportError:
        dependencies['psutil'] = 'not available - memory monitoring tests will be skipped'
    
    # Optional enhanced testing dependencies
    optional_deps = {
        'pytest_benchmark': 'Performance regression testing',
        'hypothesis': 'Property-based testing for invariants',
        'pytest_cov': 'Coverage measurement and reporting',
        'pytest_mock': 'Mock object utilities',
        'hydra_core': 'Configuration composition testing',
        'cv2': 'Frame cache and video processing testing'
    }
    
    for dep_name, description in optional_deps.items():
        try:
            if dep_name == 'cv2':
                import cv2 as dep_module
            elif dep_name == 'hydra_core':
                import hydra
                dep_module = hydra
            else:
                dep_module = __import__(dep_name)
            dependencies[dep_name] = getattr(dep_module, '__version__', 'available')
        except ImportError:
            dependencies[dep_name] = f'not available - {description}'
    
    return dependencies


def validate_gymnasium_environment(env):
    """
    Utility function to validate Gymnasium environment compliance.
    
    Args:
        env: Gymnasium environment instance to validate
        
    Returns:
        dict: Validation results including API compliance and performance metrics
        
    Raises:
        AssertionError: If environment fails critical compliance checks
    """
    import time
    import numpy as np
    
    validation_results = {
        'api_compliance': False,
        'reset_signature': False,
        'step_signature': False,
        'performance_compliant': False,
        'step_latency_ms': None,
        'observation_space_valid': False,
        'action_space_valid': False
    }
    
    try:
        # Check basic API compliance
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        assert hasattr(env, 'observation_space'), "Environment missing observation_space"
        assert hasattr(env, 'action_space'), "Environment missing action_space"
        
        # Validate reset signature (should accept seed and options)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, (np.ndarray, dict)), "Reset observation must be ndarray or dict"
        assert isinstance(info, dict), "Reset info must be dict"
        validation_results['reset_signature'] = True
        
        # Validate step signature (should return 5-tuple)
        action = env.action_space.sample()
        step_start = time.perf_counter()
        step_result = env.step(action)
        step_latency_ms = (time.perf_counter() - step_start) * 1000
        
        assert len(step_result) == 5, f"Step must return 5-tuple, got {len(step_result)}"
        obs, reward, terminated, truncated, info = step_result
        
        assert isinstance(obs, (np.ndarray, dict)), "Step observation must be ndarray or dict"
        assert isinstance(reward, (int, float, np.number)), "Reward must be numeric"
        assert isinstance(terminated, bool), "Terminated must be boolean"
        assert isinstance(truncated, bool), "Truncated must be boolean"
        assert isinstance(info, dict), "Info must be dict"
        
        validation_results['step_signature'] = True
        validation_results['step_latency_ms'] = step_latency_ms
        validation_results['performance_compliant'] = step_latency_ms < 10.0  # Sub-10ms requirement
        
        # Validate spaces
        validation_results['observation_space_valid'] = env.observation_space is not None
        validation_results['action_space_valid'] = env.action_space is not None
        
        validation_results['api_compliance'] = True
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results


def assert_performance_requirements(step_latencies_ms, cache_hit_rate=None, memory_usage_mb=None):
    """
    Assert that performance measurements meet SLA requirements.
    
    Args:
        step_latencies_ms (list): List of step execution times in milliseconds
        cache_hit_rate (float, optional): Frame cache hit rate (0.0-1.0)
        memory_usage_mb (float, optional): Memory usage in megabytes
        
    Raises:
        AssertionError: If performance requirements are not met
    """
    import numpy as np
    
    if step_latencies_ms:
        p95_latency = np.percentile(step_latencies_ms, 95)
        mean_latency = np.mean(step_latencies_ms)
        
        assert p95_latency < 10.0, f"P95 step latency {p95_latency:.2f}ms exceeds 10ms target"
        assert mean_latency < 8.0, f"Mean step latency {mean_latency:.2f}ms exceeds 8ms target"
    
    if cache_hit_rate is not None:
        assert cache_hit_rate >= 0.90, f"Cache hit rate {cache_hit_rate:.1%} below 90% target"
    
    if memory_usage_mb is not None:
        assert memory_usage_mb <= 2048, f"Memory usage {memory_usage_mb:.1f}MB exceeds 2GB limit"


def create_mock_gymnasium_env():
    """
    Create a mock Gymnasium environment for testing purposes.
    
    Returns:
        Mock environment instance compatible with Gymnasium API
    """
    import numpy as np
    from unittest.mock import Mock
    
    mock_env = Mock()
    
    # Mock spaces
    mock_env.observation_space = Mock()
    mock_env.observation_space.sample.return_value = np.array([0.0, 0.0, 0.0])
    mock_env.action_space = Mock()
    mock_env.action_space.sample.return_value = np.array([0.0, 0.0])
    
    # Mock reset method (returns observation, info)
    def mock_reset(seed=None, options=None):
        return np.array([0.0, 0.0, 0.0]), {}
    mock_env.reset = mock_reset
    
    # Mock step method (returns obs, reward, terminated, truncated, info)
    def mock_step(action):
        return np.array([0.0, 0.0, 0.0]), 0.0, False, False, {}
    mock_env.step = mock_step
    
    return mock_env


def validate_legacy_compatibility(gym_make_func, env_id="PlumeNavSim-v0"):
    """
    Validate legacy gym.make() compatibility through shim layer.
    
    Args:
        gym_make_func: The gym_make shim function to test
        env_id (str): Environment ID to test with
        
    Returns:
        dict: Compatibility validation results
    """
    import warnings
    
    validation_results = {
        'deprecation_warning_emitted': False,
        'returns_valid_env': False,
        'supports_legacy_api': False,
        'error': None
    }
    
    try:
        # Capture deprecation warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            env = gym_make_func(env_id)
            
            # Check if deprecation warning was emitted
            deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
            validation_results['deprecation_warning_emitted'] = len(deprecation_warnings) > 0
            
            # Validate returned environment
            validation_results['returns_valid_env'] = env is not None
            
            if env is not None:
                # Test if environment supports both legacy and modern APIs
                try:
                    # Try reset
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple) and len(reset_result) == 2:
                        # Modern API (observation, info)
                        obs, info = reset_result
                        validation_results['supports_modern_api'] = True
                    else:
                        # Legacy API (just observation)
                        validation_results['supports_legacy_api'] = True
                    
                    # Try step
                    action = env.action_space.sample() if hasattr(env, 'action_space') else None
                    if action is not None:
                        step_result = env.step(action)
                        if len(step_result) == 5:
                            validation_results['supports_modern_api'] = True
                        elif len(step_result) == 4:
                            validation_results['supports_legacy_api'] = True
                            
                except Exception as e:
                    validation_results['api_test_error'] = str(e)
                    
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results


# Initialize test environment check on import
if __name__ != "__main__":
    # Only run dependency check during normal import, not direct execution
    try:
        _test_deps = check_test_dependencies()
    except ImportError as e:
        import warnings
        warnings.warn(f"Test dependency issue: {e}", UserWarning)