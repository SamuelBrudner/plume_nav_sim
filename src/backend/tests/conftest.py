"""
Pytest configuration and fixture management module providing comprehensive test fixtures, utilities, and 
setup/teardown functionality for the plume_nav_sim test suite. Includes fixtures for environment instances, 
test configurations, performance monitoring, data validation, reproducibility testing, and error handling 
scenarios with intelligent test optimization and resource management.

This module serves as the centralized testing infrastructure for plume_nav_sim, providing comprehensive
fixture management, performance monitoring, and test environment lifecycle management with automatic
cleanup and resource validation.
"""

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for fixture definitions, parametrization, test configuration, and comprehensive test execution support with session and module scoped fixtures
import numpy as np  # >=2.1.0 - Array operations for test data creation, mathematical validation, performance benchmarking, and observation/rendering validation utilities
import time  # >=3.10 - High-precision timing utilities for performance measurement fixtures, benchmark validation, and latency testing infrastructure
import gc  # >=3.10 - Garbage collection utilities for memory management in test fixtures, resource cleanup validation, and memory leak detection
import contextlib  # >=3.10 - Context manager utilities for test resource management, exception handling, and cleanup automation in fixtures
import warnings  # >=3.10 - Warning management for performance testing fixtures, deprecation handling, and system compatibility validation
import tempfile  # >=3.10 - Temporary file management for test data persistence, configuration storage, and test isolation utilities
import copy  # >=3.10 - Deep copying utilities for test data isolation, configuration cloning, and state management in fixtures
import threading  # >=3.10 - Threading utilities for concurrent testing fixtures, thread safety validation, and performance testing under concurrency
import psutil  # >=5.0.0 - System monitoring utilities for memory usage tracking, CPU utilization monitoring, and resource constraint validation in performance fixtures
from typing import Dict, List, Tuple, Any, Optional, Generator, Union
from dataclasses import dataclass, field

# Internal imports from plume_nav_sim core modules
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv, create_plume_search_env
from plume_nav_sim.config.test_configs import (
    create_unit_test_config, create_integration_test_config, create_performance_test_config,
    create_reproducibility_test_config, create_edge_case_test_config, TestConfigFactory,
    REPRODUCIBILITY_SEEDS
)
from plume_nav_sim.core.types import (
    Action, Coordinates, GridSize, create_coordinates, create_grid_size
)
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, 
    PERFORMANCE_TARGET_STEP_LATENCY_MS, PERFORMANCE_TARGET_RGB_RENDER_MS
)


# ===== GLOBAL REGISTRY AND STATE MANAGEMENT =====

# Global registry for managing test environment instances and cleanup
_test_environments: Dict[str, PlumeSearchEnv] = {}

# Global storage for performance metrics collection and analysis
_performance_data: Dict[str, Dict[str, Any]] = {}

# Global TestConfigFactory instance for intelligent configuration management
_config_factory: Optional[TestConfigFactory] = None

# Configuration flags for test behavior and monitoring
TEST_ISOLATION_ENABLED = True  # Flag enabling test isolation and independent fixture management
PERFORMANCE_MONITORING_ENABLED = True  # Flag enabling comprehensive performance monitoring during tests
MEMORY_LEAK_DETECTION_ENABLED = True  # Flag enabling memory leak detection and resource monitoring
CLEANUP_VALIDATION_ENABLED = True  # Flag enabling cleanup validation and resource deallocation verification


# ===== PYTEST HOOKS FOR SESSION AND TEST MANAGEMENT =====

def pytest_configure(config: pytest.Config) -> None:
    """Pytest hook for test session configuration including global setup, system capability detection, 
    performance monitoring initialization, and test infrastructure preparation.
    
    Args:
        config: Pytest configuration object for session setup
    """
    global _config_factory, _performance_data, _test_environments
    
    # Initialize global TestConfigFactory with system capability detection and optimization
    _config_factory = TestConfigFactory(auto_optimize=True)
    try:
        _config_factory.detect_system_capabilities(force_refresh=True)
        print(f"System capabilities detected: {_config_factory._system_capabilities}")
    except Exception as e:
        warnings.warn(f"Failed to detect system capabilities: {e}", UserWarning)
    
    # Set up global performance monitoring infrastructure and metrics collection
    _performance_data.clear()
    _performance_data['session'] = {
        'start_time': time.time(),
        'test_count': 0,
        'fixture_creation_times': {},
        'memory_baselines': {},
        'performance_violations': []
    }
    
    # Initialize memory leak detection and resource tracking systems
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            initial_memory = psutil.Process().memory_info()
            _performance_data['session']['initial_memory_mb'] = initial_memory.rss / (1024 * 1024)
        except Exception as e:
            warnings.warn(f"Memory monitoring initialization failed: {e}", UserWarning)
    
    # Configure test environment registry for instance management and cleanup
    _test_environments.clear()
    
    # Set up warning filters and performance monitoring configuration
    if PERFORMANCE_MONITORING_ENABLED:
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='gymnasium')
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Initialize temporary directory management for test data isolation
    temp_dir = tempfile.mkdtemp(prefix='plume_nav_sim_tests_')
    _performance_data['session']['temp_directory'] = temp_dir
    
    # Configure logging and debugging infrastructure for comprehensive test analysis
    import logging
    logging.getLogger('plume_nav_sim').setLevel(logging.WARNING)
    
    # Validate system capabilities and set performance testing thresholds
    if _config_factory and _config_factory._system_capabilities:
        caps = _config_factory._system_capabilities
        memory_gb = caps.get('memory_gb', 8)
        cpu_count = caps.get('cpu_count', 4)
        
        # Adjust performance thresholds based on system capabilities
        if memory_gb < 8 or cpu_count < 4:
            _performance_data['session']['performance_multiplier'] = 2.0
        else:
            _performance_data['session']['performance_multiplier'] = 1.0
    
    # Initialize concurrent testing infrastructure and thread safety validation
    _performance_data['session']['thread_safety_enabled'] = threading.active_count() == 1


def pytest_unconfigure(config: pytest.Config) -> None:
    """Pytest hook for test session cleanup including global resource deallocation, performance report 
    generation, memory leak analysis, and infrastructure shutdown.
    
    Args:
        config: Pytest configuration object for session cleanup
    """
    global _test_environments, _performance_data, _config_factory
    
    # Generate comprehensive performance report with session statistics and analysis
    if PERFORMANCE_MONITORING_ENABLED and _performance_data.get('session'):
        session_data = _performance_data['session']
        session_duration = time.time() - session_data['start_time']
        
        print(f"\n=== Test Session Performance Report ===")
        print(f"Total session duration: {session_duration:.2f} seconds")
        print(f"Total tests executed: {session_data.get('test_count', 0)}")
        print(f"Average test duration: {session_duration / max(session_data.get('test_count', 1), 1):.3f} seconds")
        
        if session_data.get('fixture_creation_times'):
            avg_fixture_time = np.mean(list(session_data['fixture_creation_times'].values()))
            print(f"Average fixture creation time: {avg_fixture_time:.3f} seconds")
        
        if session_data.get('performance_violations'):
            print(f"Performance violations: {len(session_data['performance_violations'])}")
            for violation in session_data['performance_violations'][:5]:  # Show first 5
                print(f"  - {violation}")
    
    # Perform memory leak detection analysis and resource usage validation
    if MEMORY_LEAK_DETECTION_ENABLED and _performance_data.get('session', {}).get('initial_memory_mb'):
        try:
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            initial_memory = _performance_data['session']['initial_memory_mb']
            memory_delta = final_memory - initial_memory
            
            print(f"Memory usage change: {memory_delta:+.1f} MB")
            if memory_delta > 100:  # More than 100MB increase
                warnings.warn(f"Potential memory leak detected: {memory_delta:.1f} MB increase", UserWarning)
        except Exception as e:
            print(f"Memory leak analysis failed: {e}")
    
    # Clean up all registered test environment instances and release resources
    cleanup_report = _cleanup_test_environments(validate_cleanup=True)
    if cleanup_report.get('failed_cleanups'):
        warnings.warn(f"Failed to clean up {len(cleanup_report['failed_cleanups'])} environments", UserWarning)
    
    # Validate complete resource deallocation and cleanup effectiveness
    if CLEANUP_VALIDATION_ENABLED:
        # Force garbage collection to validate cleanup
        gc.collect()
        remaining_environments = len(_test_environments)
        if remaining_environments > 0:
            warnings.warn(f"Cleanup validation failed: {remaining_environments} environments remain", UserWarning)
    
    # Generate test infrastructure summary with optimization recommendations
    if _config_factory and hasattr(_config_factory, '_system_capabilities'):
        print(f"System optimization enabled: {_config_factory._auto_optimize}")
        print(f"Configuration cache size: {len(_config_factory._configuration_cache)}")
    
    # Clean up temporary directories and test data files
    if _performance_data.get('session', {}).get('temp_directory'):
        import shutil
        try:
            shutil.rmtree(_performance_data['session']['temp_directory'])
        except Exception as e:
            warnings.warn(f"Failed to cleanup temporary directory: {e}", UserWarning)
    
    # Shutdown performance monitoring and metrics collection systems
    _performance_data.clear()
    _test_environments.clear()
    
    # Validate system state restoration and resource cleanup completion
    print("=== Test session cleanup completed ===")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Pytest hook for individual test setup including test isolation preparation, resource allocation, 
    and performance monitoring initialization for each test.
    
    Args:
        item: Pytest test item for individual test setup
    """
    global _performance_data
    
    # Initialize test-specific resource tracking and performance monitoring
    test_id = f"{item.module.__name__}::{item.name}"
    if 'tests' not in _performance_data:
        _performance_data['tests'] = {}
    
    _performance_data['tests'][test_id] = {
        'start_time': time.time(),
        'setup_time': None,
        'memory_baseline': None,
        'resource_warnings': []
    }
    
    # Set up test isolation environment and resource boundaries
    if TEST_ISOLATION_ENABLED:
        # Clear any existing warnings for this test
        warnings.resetwarnings()
        
        # Set up test-specific warning handling
        if hasattr(item, 'config') and item.config.getoption('--disable-warnings', default=False):
            warnings.filterwarnings('ignore')
    
    # Initialize memory usage baseline and resource monitoring for test execution
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            _performance_data['tests'][test_id]['memory_baseline'] = baseline_memory
        except Exception:
            pass
    
    # Configure test-specific warning handling and error reporting
    if PERFORMANCE_MONITORING_ENABLED:
        # Set performance warning thresholds for individual tests
        _performance_data['tests'][test_id]['performance_threshold'] = PERFORMANCE_TARGET_STEP_LATENCY_MS
    
    # Set up concurrent testing infrastructure if required by test markers
    if hasattr(item, 'get_closest_marker') and item.get_closest_marker('concurrent'):
        _performance_data['tests'][test_id]['concurrent_test'] = True
        _performance_data['tests'][test_id]['thread_count'] = threading.active_count()
    
    # Initialize test data isolation and temporary resource allocation
    if hasattr(item, 'get_closest_marker') and item.get_closest_marker('tempdir'):
        test_temp_dir = tempfile.mkdtemp(prefix=f'test_{item.name}_')
        _performance_data['tests'][test_id]['temp_directory'] = test_temp_dir
    
    # Configure performance measurement infrastructure for test-specific metrics
    _performance_data['tests'][test_id]['setup_time'] = time.time() - _performance_data['tests'][test_id]['start_time']
    
    # Increment session test counter
    if 'session' in _performance_data:
        _performance_data['session']['test_count'] += 1


def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> None:
    """Pytest hook for individual test cleanup including resource deallocation, performance data collection, 
    memory validation, and isolation cleanup.
    
    Args:
        item: Pytest test item for individual test cleanup
        nextitem: Next test item in sequence for optimization
    """
    global _performance_data
    
    test_id = f"{item.module.__name__}::{item.name}"
    
    # Collect test-specific performance metrics and timing data
    if test_id in _performance_data.get('tests', {}):
        test_data = _performance_data['tests'][test_id]
        test_data['end_time'] = time.time()
        test_data['total_duration'] = test_data['end_time'] - test_data['start_time']
        
        # Check performance against targets
        if PERFORMANCE_MONITORING_ENABLED and test_data['total_duration'] > 10.0:  # Tests over 10 seconds
            violation = f"Test {test_id} exceeded 10s duration: {test_data['total_duration']:.2f}s"
            _performance_data['session']['performance_violations'].append(violation)
    
    # Validate memory usage and detect potential memory leaks from test execution
    if MEMORY_LEAK_DETECTION_ENABLED and test_id in _performance_data.get('tests', {}):
        test_data = _performance_data['tests'][test_id]
        if test_data.get('memory_baseline') is not None:
            try:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_delta = current_memory - test_data['memory_baseline']
                test_data['memory_delta_mb'] = memory_delta
                
                # Flag significant memory increases
                if memory_delta > 50:  # More than 50MB increase
                    test_data['resource_warnings'].append(f"High memory usage: +{memory_delta:.1f} MB")
            except Exception:
                pass
    
    # Clean up test-specific resources and validate complete deallocation
    test_env_count = 0
    cleanup_failures = []
    
    # Clean up test-specific environments
    test_envs_to_cleanup = [env_id for env_id in _test_environments.keys() if test_id in env_id]
    for env_id in test_envs_to_cleanup:
        try:
            env = _test_environments.pop(env_id)
            env.close()
            test_env_count += 1
        except Exception as e:
            cleanup_failures.append(f"Failed to cleanup {env_id}: {e}")
    
    # Store test performance data for session analysis and optimization
    if test_id in _performance_data.get('tests', {}):
        test_data = _performance_data['tests'][test_id]
        test_data['environments_cleaned'] = test_env_count
        test_data['cleanup_failures'] = cleanup_failures
    
    # Validate test isolation effectiveness and resource boundary integrity
    if TEST_ISOLATION_ENABLED and cleanup_failures:
        warnings.warn(f"Test isolation compromised for {test_id}: {len(cleanup_failures)} cleanup failures", UserWarning)
    
    # Clean up test-specific temporary resources and data files
    if test_id in _performance_data.get('tests', {}) and 'temp_directory' in _performance_data['tests'][test_id]:
        import shutil
        try:
            shutil.rmtree(_performance_data['tests'][test_id]['temp_directory'])
        except Exception as e:
            _performance_data['tests'][test_id]['resource_warnings'].append(f"Temp cleanup failed: {e}")
    
    # Generate test-specific cleanup report and resource validation summary
    if CLEANUP_VALIDATION_ENABLED and (test_env_count > 0 or cleanup_failures):
        gc.collect()  # Force garbage collection after test cleanup


# ===== INTERNAL UTILITY FUNCTIONS =====

def _register_test_environment(env: PlumeSearchEnv, test_id: str, cleanup_metadata: Dict[str, Any]) -> None:
    """Internal utility for registering test environment instances for automatic cleanup and resource 
    management with comprehensive tracking.
    
    Args:
        env: Environment instance to register for cleanup
        test_id: Test identifier for tracking and cleanup association
        cleanup_metadata: Metadata for cleanup validation and resource management
    """
    global _test_environments
    
    # Generate unique environment registration ID for tracking and cleanup
    import uuid
    env_id = f"{test_id}_{uuid.uuid4().hex[:8]}"
    
    # Store environment instance in global registry with test association
    _test_environments[env_id] = env
    
    # Initialize resource monitoring for environment instance
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            env_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            cleanup_metadata['registration_memory_mb'] = env_memory
        except Exception:
            pass
    
    # Set up automatic cleanup scheduling and resource management
    cleanup_metadata.update({
        'registration_time': time.time(),
        'env_id': env_id,
        'test_association': test_id
    })
    
    # Record environment configuration and resource requirements
    if hasattr(env, 'get_environment_metrics'):
        try:
            metrics = env.get_environment_metrics()
            cleanup_metadata['environment_metrics'] = metrics
        except Exception:
            pass
    
    # Initialize performance tracking for environment operations
    cleanup_metadata['operation_count'] = 0
    cleanup_metadata['total_step_time'] = 0.0
    
    # Configure memory monitoring and resource usage validation
    cleanup_metadata['resource_tracking_enabled'] = MEMORY_LEAK_DETECTION_ENABLED


def _cleanup_test_environments(test_filter: Optional[str] = None, 
                              validate_cleanup: bool = True) -> Dict[str, Any]:
    """Internal utility for cleaning up registered test environments with comprehensive resource validation 
    and memory leak detection.
    
    Args:
        test_filter: Optional filter for specific test environments
        validate_cleanup: Whether to perform comprehensive cleanup validation
        
    Returns:
        dict: Cleanup report with resource deallocation status and validation results
    """
    global _test_environments
    
    cleanup_report = {
        'cleaned_environments': 0,
        'failed_cleanups': [],
        'memory_recovered_mb': 0.0,
        'validation_errors': [],
        'total_cleanup_time': 0.0
    }
    
    start_cleanup_time = time.time()
    initial_memory = None
    
    # Record initial memory usage for leak detection
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    
    # Iterate through registered environment instances matching test_filter
    envs_to_cleanup = list(_test_environments.items())
    if test_filter:
        envs_to_cleanup = [(env_id, env) for env_id, env in envs_to_cleanup if test_filter in env_id]
    
    # Close each environment instance and validate resource deallocation
    for env_id, env in envs_to_cleanup:
        try:
            # Monitor memory usage during cleanup and detect memory leaks
            pre_cleanup_memory = None
            if MEMORY_LEAK_DETECTION_ENABLED:
                try:
                    pre_cleanup_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
            
            # Close environment and validate resource release
            env.close()
            cleanup_report['cleaned_environments'] += 1
            
            # Validate complete resource release including rendering and computation resources
            if validate_cleanup and hasattr(env, 'get_environment_metrics'):
                try:
                    # Attempt to get metrics after close - should fail or return empty
                    metrics = env.get_environment_metrics()
                    if metrics and any(metrics.values()):
                        cleanup_report['validation_errors'].append(f"Environment {env_id} still active after close")
                except Exception:
                    # Expected - environment should be closed
                    pass
            
            # Remove environment instances from global registry after successful cleanup
            _test_environments.pop(env_id, None)
            
            # Monitor memory recovery after cleanup
            if pre_cleanup_memory is not None and MEMORY_LEAK_DETECTION_ENABLED:
                try:
                    post_cleanup_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_recovered = pre_cleanup_memory - post_cleanup_memory
                    cleanup_report['memory_recovered_mb'] += max(0, memory_recovered)
                except Exception:
                    pass
            
        except Exception as e:
            cleanup_report['failed_cleanups'].append({
                'env_id': env_id,
                'error': str(e),
                'environment_type': type(env).__name__
            })
    
    # Perform garbage collection and validate memory recovery effectiveness
    if cleanup_report['cleaned_environments'] > 0:
        gc.collect()
        
        # Validate memory recovery after garbage collection
        if initial_memory is not None and MEMORY_LEAK_DETECTION_ENABLED:
            try:
                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                total_memory_change = initial_memory - final_memory
                cleanup_report['memory_recovered_mb'] = max(0, total_memory_change)
            except Exception:
                pass
    
    # Record total cleanup duration
    cleanup_report['total_cleanup_time'] = time.time() - start_cleanup_time
    
    # Return comprehensive cleanup analysis with success status and recommendations
    cleanup_report['cleanup_success'] = len(cleanup_report['failed_cleanups']) == 0
    cleanup_report['recommendations'] = []
    
    if cleanup_report['failed_cleanups']:
        cleanup_report['recommendations'].append("Investigate environment cleanup failures")
    if cleanup_report['memory_recovered_mb'] < 1.0 and cleanup_report['cleaned_environments'] > 0:
        cleanup_report['recommendations'].append("Monitor for potential memory leaks")
    
    return cleanup_report


def _create_test_data(data_type: str, data_config: Dict[str, Any], 
                     include_edge_cases: bool = True) -> Dict[str, Any]:
    """Internal utility for creating standardized test data including coordinates, actions, expected outcomes, 
    and validation datasets with comprehensive coverage.
    
    Args:
        data_type: Type of test data to create
        data_config: Configuration for test data creation
        include_edge_cases: Whether to include edge case scenarios
        
    Returns:
        dict: Standardized test data with validation information and expected outcomes
    """
    # Validate data_type against supported test data categories
    supported_types = ['coordinates', 'actions', 'grid_sizes', 'seeds', 'configurations', 'trajectories']
    if data_type not in supported_types:
        raise ValueError(f"Unsupported data type: {data_type}. Supported: {supported_types}")
    
    test_data = {
        'data_type': data_type,
        'config': data_config.copy(),
        'include_edge_cases': include_edge_cases,
        'created_at': time.time(),
        'validation_criteria': {},
        'expected_outcomes': {},
        'test_scenarios': []
    }
    
    # Generate base test data according to data_config specifications
    if data_type == 'coordinates':
        # Create coordinate sequences for movement testing and trajectory validation
        grid_size = data_config.get('grid_size', DEFAULT_GRID_SIZE)
        
        # Generate valid coordinates within grid bounds
        valid_coords = []
        for x in range(0, grid_size[0], max(1, grid_size[0] // 8)):
            for y in range(0, grid_size[1], max(1, grid_size[1] // 8)):
                valid_coords.append(create_coordinates(x, y))
        
        test_data['valid_coordinates'] = valid_coords
        test_data['center_coordinate'] = create_coordinates(grid_size[0] // 2, grid_size[1] // 2)
        
        # Include edge case scenarios if include_edge_cases is enabled
        if include_edge_cases:
            edge_coords = [
                create_coordinates(0, 0),  # Top-left corner
                create_coordinates(grid_size[0] - 1, 0),  # Top-right corner
                create_coordinates(0, grid_size[1] - 1),  # Bottom-left corner
                create_coordinates(grid_size[0] - 1, grid_size[1] - 1)  # Bottom-right corner
            ]
            test_data['edge_coordinates'] = edge_coords
            
            # Invalid coordinates for boundary testing
            invalid_coords = [
                (-1, -1), (-1, 0), (0, -1),  # Negative coordinates
                (grid_size[0], grid_size[1]),  # Out of bounds
                (grid_size[0] + 1, grid_size[1] + 1)  # Far out of bounds
            ]
            test_data['invalid_coordinates'] = invalid_coords
        
        # Create expected outcomes and validation criteria for test data
        test_data['validation_criteria'] = {
            'coordinate_bounds': (0, 0, grid_size[0] - 1, grid_size[1] - 1),
            'distance_calculation': True,
            'coordinate_arithmetic': True
        }
        
    elif data_type == 'actions':
        # Create action sequences for comprehensive behavior testing
        all_actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        test_data['valid_actions'] = all_actions
        
        # Generate action sequences for movement testing
        sequence_length = data_config.get('sequence_length', 10)
        random_sequence = np.random.choice(all_actions, size=sequence_length).tolist()
        test_data['random_action_sequence'] = random_sequence
        
        # Deterministic action patterns
        circular_pattern = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT] * (sequence_length // 4)
        test_data['circular_pattern'] = circular_pattern[:sequence_length]
        
        if include_edge_cases:
            # Invalid action values for boundary testing
            test_data['invalid_actions'] = [-1, 4, 5, 100, -100]
            
            # Single action repeated
            test_data['single_action_sequence'] = [Action.UP] * sequence_length
        
        test_data['validation_criteria'] = {
            'action_range': (0, 3),
            'sequence_validity': True,
            'movement_consistency': True
        }
        
    elif data_type == 'seeds':
        # Generate seed values for reproducibility testing
        test_data['reproducibility_seeds'] = REPRODUCIBILITY_SEEDS.copy()
        test_data['random_seeds'] = [np.random.randint(0, 2**31 - 1) for _ in range(5)]
        
        if include_edge_cases:
            test_data['edge_case_seeds'] = [0, 1, 2**31 - 1, 42, 12345]
        
        test_data['validation_criteria'] = {
            'seed_consistency': True,
            'reproducibility_check': True,
            'cross_platform_consistency': True
        }
    
    # Include performance validation data and timing expectations
    test_data['performance_expectations'] = {
        'creation_time_ms': (time.time() - test_data['created_at']) * 1000,
        'memory_usage_estimate': len(str(test_data)) / 1024,  # Rough KB estimate
        'validation_time_target_ms': 1.0
    }
    
    # Generate comprehensive test data package with metadata and validation criteria
    test_data['metadata'] = {
        'total_items': sum(len(v) if isinstance(v, list) else 1 for k, v in test_data.items() if k.endswith('_coordinates') or k.endswith('_actions') or k.endswith('_seeds')),
        'edge_cases_included': include_edge_cases,
        'validation_ready': True
    }
    
    return test_data


def _measure_performance(operation_name: str, operation_func: callable, 
                        performance_context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Internal utility for measuring and recording performance metrics during test execution with 
    comprehensive analysis and validation.
    
    Args:
        operation_name: Name of operation being measured
        operation_func: Function to execute and measure
        performance_context: Context information for performance analysis
        
    Returns:
        tuple: Tuple of (operation_result, performance_metrics) with timing and resource analysis
    """
    # Initialize performance measurement infrastructure with high-precision timing
    performance_metrics = {
        'operation_name': operation_name,
        'start_time': time.perf_counter(),
        'start_memory_mb': None,
        'end_time': None,
        'end_memory_mb': None,
        'duration_ms': None,
        'memory_delta_mb': None,
        'context': performance_context.copy()
    }
    
    # Record baseline memory usage and system resource state
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            performance_metrics['start_memory_mb'] = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    
    # Execute operation_func with comprehensive monitoring and timing measurement
    try:
        operation_result = operation_func()
        performance_metrics['execution_status'] = 'success'
    except Exception as e:
        operation_result = None
        performance_metrics['execution_status'] = 'failed'
        performance_metrics['error'] = str(e)
        raise
    finally:
        # Record operation completion timing and resource usage changes
        performance_metrics['end_time'] = time.perf_counter()
        performance_metrics['duration_ms'] = (performance_metrics['end_time'] - performance_metrics['start_time']) * 1000
        
        # Monitor memory usage after operation completion
        if MEMORY_LEAK_DETECTION_ENABLED and performance_metrics['start_memory_mb'] is not None:
            try:
                performance_metrics['end_memory_mb'] = psutil.Process().memory_info().rss / (1024 * 1024)
                performance_metrics['memory_delta_mb'] = performance_metrics['end_memory_mb'] - performance_metrics['start_memory_mb']
            except Exception:
                pass
    
    # Calculate performance metrics including latency, memory delta, and resource utilization
    performance_analysis = {
        'latency_ms': performance_metrics['duration_ms'],
        'memory_usage_mb': performance_metrics.get('memory_delta_mb', 0),
        'performance_tier': 'fast' if performance_metrics['duration_ms'] < 1.0 else 'medium' if performance_metrics['duration_ms'] < 10.0 else 'slow'
    }
    
    # Compare performance against targets from performance_context
    targets = performance_context.get('targets', {})
    if 'latency_target_ms' in targets:
        target_latency = targets['latency_target_ms']
        performance_analysis['latency_meets_target'] = performance_metrics['duration_ms'] <= target_latency
        performance_analysis['latency_ratio'] = performance_metrics['duration_ms'] / target_latency
    
    if 'memory_target_mb' in targets and performance_metrics.get('memory_delta_mb') is not None:
        target_memory = targets['memory_target_mb']
        performance_analysis['memory_meets_target'] = performance_metrics['memory_delta_mb'] <= target_memory
        performance_analysis['memory_ratio'] = performance_metrics['memory_delta_mb'] / target_memory
    
    # Generate performance analysis with optimization recommendations
    recommendations = []
    if performance_metrics['duration_ms'] > 100:  # Over 100ms
        recommendations.append("Consider optimization for operations over 100ms")
    if performance_metrics.get('memory_delta_mb', 0) > 10:  # Over 10MB
        recommendations.append("Monitor memory usage for operations using >10MB")
    
    performance_analysis['recommendations'] = recommendations
    performance_metrics['analysis'] = performance_analysis
    
    # Store performance data globally for session analysis
    if 'performance_measurements' not in _performance_data:
        _performance_data['performance_measurements'] = []
    _performance_data['performance_measurements'].append(performance_metrics)
    
    # Return operation result with comprehensive performance metrics and analysis
    return operation_result, performance_metrics


# ===== CONTEXT MANAGER CLASSES =====

class TestEnvironmentManager:
    """Context manager class for test environment lifecycle management including creation, monitoring, 
    cleanup, and resource validation with comprehensive tracking and optimization.
    
    This class provides comprehensive environment lifecycle management with automatic cleanup,
    performance monitoring, and resource validation for test execution.
    """
    
    def __init__(self, test_category: str, env_config: Dict[str, Any], 
                 enable_monitoring: bool = True):
        """Initialize test environment manager with category-specific configuration and monitoring setup.
        
        Args:
            test_category: Test category for configuration optimization
            env_config: Environment configuration dictionary
            enable_monitoring: Whether to enable performance monitoring and resource tracking
        """
        # Store test category for configuration optimization and resource management
        self.test_category = test_category
        
        # Store environment configuration with validation and consistency checking
        self.env_config = env_config.copy()
        
        # Set monitoring enabled flag for performance tracking and resource monitoring
        self.monitoring_enabled = enable_monitoring
        
        # Initialize environment instance to None pending context manager entry
        self.environment: Optional[PlumeSearchEnv] = None
        
        # Initialize empty performance metrics dictionary for tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize resource tracking dictionary for memory and CPU monitoring
        self.resource_tracking: Dict[str, Any] = {
            'creation_time': None,
            'cleanup_time': None,
            'memory_usage': {},
            'operation_count': 0
        }
    
    def __enter__(self) -> PlumeSearchEnv:
        """Context manager entry creating environment instance with monitoring and resource tracking setup.
        
        Returns:
            PlumeSearchEnv: Fully configured and monitored environment instance ready for testing
        """
        start_time = time.perf_counter()
        
        # Create environment instance using env_config with comprehensive validation
        try:
            self.environment = create_plume_search_env(**self.env_config)
            self.resource_tracking['creation_time'] = time.perf_counter() - start_time
        except Exception as e:
            raise RuntimeError(f"Failed to create test environment: {e}") from e
        
        # Register environment instance for automatic cleanup and resource management
        test_id = f"test_env_manager_{self.test_category}"
        cleanup_metadata = {
            'manager_created': True,
            'category': self.test_category,
            'monitoring_enabled': self.monitoring_enabled
        }
        _register_test_environment(self.environment, test_id, cleanup_metadata)
        
        # Initialize performance monitoring if monitoring_enabled is True
        if self.monitoring_enabled:
            self.performance_metrics.update({
                'creation_duration_ms': self.resource_tracking['creation_time'] * 1000,
                'monitoring_start_time': time.perf_counter(),
                'step_times': [],
                'reset_times': []
            })
        
        # Set up resource tracking including memory usage baseline and CPU monitoring
        if MEMORY_LEAK_DETECTION_ENABLED:
            try:
                baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                self.resource_tracking['memory_usage']['baseline_mb'] = baseline_memory
            except Exception:
                pass
        
        # Validate environment initialization success and component readiness
        try:
            # Test basic environment functionality
            obs, info = self.environment.reset()
            self.resource_tracking['initialization_validated'] = True
            
            # Store initial observation for validation
            self.resource_tracking['initial_observation'] = {
                'shape': obs.shape if hasattr(obs, 'shape') else None,
                'dtype': str(obs.dtype) if hasattr(obs, 'dtype') else None
            }
            
        except Exception as e:
            # Clean up on initialization failure
            if self.environment:
                self.environment.close()
            raise RuntimeError(f"Environment initialization validation failed: {e}") from e
        
        # Return environment instance ready for test execution
        return self.environment
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], 
                exc_tb: Optional[object]) -> bool:
        """Context manager exit performing comprehensive cleanup, resource validation, and performance analysis.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            bool: False to propagate exceptions, with comprehensive cleanup regardless of exception status
        """
        cleanup_start_time = time.perf_counter()
        
        # Collect final performance metrics and resource usage analysis
        if self.monitoring_enabled and self.performance_metrics:
            self.performance_metrics['total_duration_ms'] = (
                time.perf_counter() - self.performance_metrics['monitoring_start_time']
            ) * 1000
            
            # Calculate average step time if steps were monitored
            if self.performance_metrics['step_times']:
                self.performance_metrics['avg_step_time_ms'] = np.mean(self.performance_metrics['step_times'])
                self.performance_metrics['max_step_time_ms'] = np.max(self.performance_metrics['step_times'])
        
        # Close environment instance and validate resource deallocation
        if self.environment:
            try:
                self.environment.close()
                self.resource_tracking['environment_closed'] = True
            except Exception as e:
                self.resource_tracking['cleanup_errors'] = [f"Environment close failed: {e}"]
                warnings.warn(f"Failed to close environment: {e}", UserWarning)
        
        # Perform memory usage validation and memory leak detection
        if MEMORY_LEAK_DETECTION_ENABLED and self.resource_tracking['memory_usage'].get('baseline_mb'):
            try:
                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                baseline_memory = self.resource_tracking['memory_usage']['baseline_mb']
                memory_delta = final_memory - baseline_memory
                
                self.resource_tracking['memory_usage']['final_mb'] = final_memory
                self.resource_tracking['memory_usage']['delta_mb'] = memory_delta
                
                # Flag potential memory leaks
                if memory_delta > 10:  # More than 10MB increase
                    warning_msg = f"Potential memory leak in {self.test_category}: +{memory_delta:.1f}MB"
                    warnings.warn(warning_msg, UserWarning)
                    
            except Exception as e:
                self.resource_tracking['memory_validation_error'] = str(e)
        
        # Validate complete resource cleanup including rendering and computation resources
        if CLEANUP_VALIDATION_ENABLED:
            gc.collect()  # Force garbage collection
            
            # Validate environment is properly closed
            if self.environment and hasattr(self.environment, 'get_environment_metrics'):
                try:
                    metrics = self.environment.get_environment_metrics()
                    if metrics and any(metrics.values()):
                        self.resource_tracking['cleanup_warnings'] = ["Environment still reports active metrics after close"]
                except Exception:
                    # Expected - environment should be closed
                    self.resource_tracking['environment_fully_closed'] = True
        
        # Store performance data in global performance registry for analysis
        self.resource_tracking['cleanup_time'] = time.perf_counter() - cleanup_start_time
        
        # Generate cleanup report with resource validation and optimization recommendations
        cleanup_report = {
            'category': self.test_category,
            'success': self.resource_tracking.get('environment_closed', False),
            'performance_metrics': self.performance_metrics,
            'resource_tracking': self.resource_tracking,
            'exception_during_test': exc_type is not None
        }
        
        # Store in global performance data
        if 'environment_managers' not in _performance_data:
            _performance_data['environment_managers'] = []
        _performance_data['environment_managers'].append(cleanup_report)
        
        # Return False to propagate any exceptions while ensuring complete cleanup
        return False
    
    def get_performance_report(self, include_resource_analysis: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report with metrics analysis and optimization recommendations.
        
        Args:
            include_resource_analysis: Whether to include detailed resource usage analysis
            
        Returns:
            dict: Comprehensive performance report with metrics, analysis, and recommendations
        """
        # Compile performance metrics including timing data and latency analysis
        report = {
            'test_category': self.test_category,
            'monitoring_enabled': self.monitoring_enabled,
            'performance_summary': {},
            'recommendations': []
        }
        
        # Include performance metrics if monitoring was enabled
        if self.monitoring_enabled and self.performance_metrics:
            report['performance_summary'] = {
                'creation_time_ms': self.performance_metrics.get('creation_duration_ms', 0),
                'total_duration_ms': self.performance_metrics.get('total_duration_ms', 0),
                'operation_count': self.resource_tracking.get('operation_count', 0)
            }
            
            # Include step timing analysis if available
            if self.performance_metrics.get('step_times'):
                step_times = self.performance_metrics['step_times']
                report['performance_summary']['step_analysis'] = {
                    'avg_step_time_ms': np.mean(step_times),
                    'min_step_time_ms': np.min(step_times),
                    'max_step_time_ms': np.max(step_times),
                    'step_count': len(step_times)
                }
                
                # Performance target comparison
                if np.mean(step_times) > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                    report['recommendations'].append("Step latency exceeds target - consider optimization")
        
        # Include resource analysis with memory usage and CPU utilization if requested
        if include_resource_analysis and self.resource_tracking:
            report['resource_analysis'] = {
                'creation_time_ms': (self.resource_tracking.get('creation_time', 0)) * 1000,
                'cleanup_time_ms': (self.resource_tracking.get('cleanup_time', 0)) * 1000,
                'memory_usage': self.resource_tracking.get('memory_usage', {})
            }
            
            # Memory usage analysis
            memory_usage = self.resource_tracking.get('memory_usage', {})
            if memory_usage.get('delta_mb'):
                delta = memory_usage['delta_mb']
                if delta > 5:  # More than 5MB
                    report['recommendations'].append(f"High memory usage detected: +{delta:.1f}MB")
                elif delta < -1:  # Memory recovery
                    report['recommendations'].append("Good memory management - memory recovered after test")
        
        # Generate performance comparison against category-specific targets
        category_targets = {
            'unit': {'max_duration_ms': 1000, 'max_memory_mb': 10},
            'integration': {'max_duration_ms': 5000, 'max_memory_mb': 50},
            'performance': {'max_duration_ms': 30000, 'max_memory_mb': 100}
        }
        
        targets = category_targets.get(self.test_category, {})
        if targets and self.performance_metrics:
            target_comparison = {}
            
            total_duration = self.performance_metrics.get('total_duration_ms', 0)
            if 'max_duration_ms' in targets:
                target_comparison['duration_within_target'] = total_duration <= targets['max_duration_ms']
                target_comparison['duration_ratio'] = total_duration / targets['max_duration_ms']
            
            memory_delta = self.resource_tracking.get('memory_usage', {}).get('delta_mb', 0)
            if 'max_memory_mb' in targets and memory_delta > 0:
                target_comparison['memory_within_target'] = memory_delta <= targets['max_memory_mb']
                target_comparison['memory_ratio'] = memory_delta / targets['max_memory_mb']
            
            report['target_comparison'] = target_comparison
        
        # Create optimization recommendations based on performance patterns
        if not report['recommendations']:
            report['recommendations'].append("Performance within acceptable ranges")
        
        # Include test category context and configuration impact analysis
        report['configuration_analysis'] = {
            'category_appropriate': True,
            'monitoring_overhead_ms': 0.1 if self.monitoring_enabled else 0,
            'optimization_opportunities': []
        }
        
        # Return comprehensive performance report with actionable insights
        return report


class PerformanceTracker:
    """Performance monitoring and analysis class for comprehensive test performance tracking including timing 
    measurement, resource monitoring, and optimization analysis.
    
    This class provides detailed performance monitoring capabilities with statistical analysis
    and target validation for comprehensive test performance evaluation.
    """
    
    def __init__(self, tracker_name: str, performance_targets: Dict[str, float]):
        """Initialize performance tracker with target specifications and monitoring infrastructure.
        
        Args:
            tracker_name: Name identifier for the performance tracker
            performance_targets: Dictionary of performance targets for validation
        """
        # Store tracker name for identification and reporting purposes
        self.tracker_name = tracker_name
        
        # Store performance targets for validation and comparison analysis
        self.performance_targets = performance_targets.copy()
        
        # Initialize empty timing data dictionary for operation measurements
        self.timing_data: Dict[str, List[float]] = {}
        
        # Initialize empty resource data dictionary for memory and CPU tracking
        self.resource_data: Dict[str, List[float]] = {}
        
        # Initialize empty performance history list for trend analysis
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize measurement tracking
        self._active_measurements: Dict[str, Dict[str, Any]] = {}
    
    def start_measurement(self, operation_name: str) -> str:
        """Start performance measurement for specific operation with comprehensive monitoring setup.
        
        Args:
            operation_name: Name of operation to measure
            
        Returns:
            str: Measurement ID for tracking and completion correlation
        """
        # Generate unique measurement ID for operation tracking
        import uuid
        measurement_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        # Record operation start time with high-precision timing
        start_time = time.perf_counter()
        
        # Capture baseline memory usage and system resource state
        baseline_memory = None
        if MEMORY_LEAK_DETECTION_ENABLED:
            try:
                baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        
        # Initialize operation-specific monitoring and resource tracking
        measurement_context = {
            'operation_name': operation_name,
            'start_time': start_time,
            'baseline_memory_mb': baseline_memory,
            'measurement_id': measurement_id,
            'tracker_name': self.tracker_name
        }
        
        # Store measurement context for completion analysis and validation
        self._active_measurements[measurement_id] = measurement_context
        
        # Initialize timing data for operation if not exists
        if operation_name not in self.timing_data:
            self.timing_data[operation_name] = []
        
        # Initialize resource data for operation if not exists
        if operation_name not in self.resource_data:
            self.resource_data[operation_name] = []
        
        # Return measurement ID for operation completion tracking
        return measurement_id
    
    def end_measurement(self, measurement_id: str, 
                       additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Complete performance measurement with analysis and target validation.
        
        Args:
            measurement_id: Measurement ID from start_measurement
            additional_context: Optional additional context for analysis
            
        Returns:
            dict: Performance measurement results with analysis and target comparison
        """
        # Retrieve measurement context using measurement_id
        if measurement_id not in self._active_measurements:
            raise ValueError(f"No active measurement found for ID: {measurement_id}")
        
        context = self._active_measurements.pop(measurement_id)
        
        # Calculate operation duration with high-precision timing
        end_time = time.perf_counter()
        duration_ms = (end_time - context['start_time']) * 1000
        
        operation_name = context['operation_name']
        
        # Measure resource usage changes including memory and CPU utilization
        memory_delta = None
        if context['baseline_memory_mb'] is not None and MEMORY_LEAK_DETECTION_ENABLED:
            try:
                final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_delta = final_memory - context['baseline_memory_mb']
            except Exception:
                pass
        
        # Store measurement results in timing_data and resource_data
        self.timing_data[operation_name].append(duration_ms)
        if memory_delta is not None:
            self.resource_data[operation_name].append(memory_delta)
        
        # Compare performance against targets from performance_targets
        target_analysis = {}
        operation_target = self.performance_targets.get(f"{operation_name}_latency_ms")
        if operation_target:
            target_analysis['latency_target_ms'] = operation_target
            target_analysis['latency_meets_target'] = duration_ms <= operation_target
            target_analysis['latency_ratio'] = duration_ms / operation_target
        
        # Memory target analysis
        memory_target = self.performance_targets.get(f"{operation_name}_memory_mb")
        if memory_target and memory_delta is not None:
            target_analysis['memory_target_mb'] = memory_target
            target_analysis['memory_meets_target'] = memory_delta <= memory_target
            target_analysis['memory_ratio'] = memory_delta / memory_target
        
        # Generate performance analysis with optimization recommendations
        analysis_results = {
            'measurement_id': measurement_id,
            'operation_name': operation_name,
            'duration_ms': duration_ms,
            'memory_delta_mb': memory_delta,
            'target_analysis': target_analysis,
            'timestamp': time.time(),
            'additional_context': additional_context or {}
        }
        
        # Performance classification
        if duration_ms < 1.0:
            analysis_results['performance_tier'] = 'excellent'
        elif duration_ms < 10.0:
            analysis_results['performance_tier'] = 'good'
        elif duration_ms < 100.0:
            analysis_results['performance_tier'] = 'acceptable'
        else:
            analysis_results['performance_tier'] = 'needs_optimization'
        
        # Generate optimization recommendations
        recommendations = []
        if duration_ms > 100:
            recommendations.append(f"Operation {operation_name} exceeds 100ms - consider optimization")
        if memory_delta and memory_delta > 10:
            recommendations.append(f"High memory usage: +{memory_delta:.1f}MB")
        if target_analysis.get('latency_ratio', 0) > 1.5:
            recommendations.append(f"Operation significantly exceeds target latency")
        
        analysis_results['recommendations'] = recommendations
        
        # Add measurement to performance_history for trend analysis
        self.performance_history.append(analysis_results)
        
        # Return comprehensive performance measurement results with validation
        return analysis_results
    
    def get_summary_report(self, include_trend_analysis: bool = True, 
                          include_optimization_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance summary report with statistical analysis and recommendations.
        
        Args:
            include_trend_analysis: Whether to include trend analysis from performance history
            include_optimization_recommendations: Whether to include optimization recommendations
            
        Returns:
            dict: Comprehensive performance summary with statistics, trends, and recommendations
        """
        # Compile performance statistics including averages, percentiles, and distributions
        summary_report = {
            'tracker_name': self.tracker_name,
            'generation_time': time.time(),
            'measurement_count': len(self.performance_history),
            'operation_statistics': {},
            'overall_statistics': {},
            'target_compliance': {}
        }
        
        # Generate statistics for each measured operation
        for operation_name, times in self.timing_data.items():
            if not times:
                continue
                
            operation_stats = {
                'count': len(times),
                'mean_ms': np.mean(times),
                'median_ms': np.median(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'percentiles': {
                    'p50': np.percentile(times, 50),
                    'p90': np.percentile(times, 90),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99)
                }
            }
            
            # Add resource statistics if available
            if operation_name in self.resource_data and self.resource_data[operation_name]:
                memory_data = self.resource_data[operation_name]
                operation_stats['memory_stats'] = {
                    'mean_mb': np.mean(memory_data),
                    'max_mb': np.max(memory_data),
                    'total_mb': np.sum([max(0, x) for x in memory_data])  # Only positive deltas
                }
            
            summary_report['operation_statistics'][operation_name] = operation_stats
        
        # Generate overall statistics across all operations
        all_times = [t for times in self.timing_data.values() for t in times]
        if all_times:
            summary_report['overall_statistics'] = {
                'total_measurements': len(all_times),
                'overall_mean_ms': np.mean(all_times),
                'overall_median_ms': np.median(all_times),
                'overall_p95_ms': np.percentile(all_times, 95),
                'operations_measured': len(self.timing_data)
            }
        
        # Include target comparison analysis with pass/fail status for each metric
        target_compliance = {}
        for operation_name, times in self.timing_data.items():
            target_key = f"{operation_name}_latency_ms"
            if target_key in self.performance_targets and times:
                target_value = self.performance_targets[target_key]
                mean_time = np.mean(times)
                p95_time = np.percentile(times, 95)
                
                compliance = {
                    'target_ms': target_value,
                    'mean_compliant': mean_time <= target_value,
                    'p95_compliant': p95_time <= target_value,
                    'mean_ratio': mean_time / target_value,
                    'p95_ratio': p95_time / target_value,
                    'compliance_rate': sum(1 for t in times if t <= target_value) / len(times)
                }
                target_compliance[operation_name] = compliance
        
        summary_report['target_compliance'] = target_compliance
        
        # Generate trend analysis from performance_history if include_trend_analysis enabled
        if include_trend_analysis and len(self.performance_history) > 1:
            trend_analysis = {}
            
            # Analyze performance trends over time
            recent_measurements = self.performance_history[-10:]  # Last 10 measurements
            if len(self.performance_history) > 10:
                earlier_measurements = self.performance_history[-20:-10]  # Previous 10
                
                # Compare recent vs earlier performance
                recent_times = [m['duration_ms'] for m in recent_measurements]
                earlier_times = [m['duration_ms'] for m in earlier_measurements]
                
                trend_analysis['performance_trend'] = {
                    'recent_mean_ms': np.mean(recent_times),
                    'earlier_mean_ms': np.mean(earlier_times),
                    'trend_direction': 'improving' if np.mean(recent_times) < np.mean(earlier_times) else 'degrading',
                    'trend_magnitude_ms': np.mean(recent_times) - np.mean(earlier_times)
                }
            
            summary_report['trend_analysis'] = trend_analysis
        
        # Create optimization recommendations based on performance patterns if enabled
        if include_optimization_recommendations:
            recommendations = []
            
            # Check for consistently slow operations
            for operation_name, stats in summary_report['operation_statistics'].items():
                if stats['mean_ms'] > 50:  # Over 50ms average
                    recommendations.append(f"Operation {operation_name} averaging {stats['mean_ms']:.1f}ms - consider optimization")
                
                if stats['std_ms'] > stats['mean_ms']:  # High variability
                    recommendations.append(f"Operation {operation_name} has high timing variability - investigate inconsistency")
            
            # Check target compliance
            for operation_name, compliance in target_compliance.items():
                if compliance['compliance_rate'] < 0.9:  # Less than 90% compliance
                    recommendations.append(f"Operation {operation_name} target compliance only {compliance['compliance_rate']:.1%}")
            
            # Memory usage recommendations
            all_memory = [m for mem_list in self.resource_data.values() for m in mem_list if m > 0]
            if all_memory and np.mean(all_memory) > 5:  # Average over 5MB
                recommendations.append(f"High average memory usage: {np.mean(all_memory):.1f}MB")
            
            if not recommendations:
                recommendations.append("Performance within acceptable ranges")
            
            summary_report['optimization_recommendations'] = recommendations
        
        # Include statistical analysis with confidence intervals and variability metrics
        summary_report['statistical_analysis'] = {
            'measurement_reliability': 'high' if len(all_times) > 30 else 'medium' if len(all_times) > 10 else 'low',
            'data_quality': 'good' if not any(t < 0 for t in all_times) else 'needs_review'
        }
        
        # Return comprehensive summary report with actionable performance insights
        return summary_report
    
    def validate_targets(self, tolerance_factor: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
        """Validate current performance measurements against defined targets with detailed analysis.
        
        Args:
            tolerance_factor: Tolerance factor for flexible target validation (0.1 = 10% tolerance)
            
        Returns:
            tuple: Tuple of (targets_met: bool, validation_report: dict) with detailed target analysis
        """
        validation_report = {
            'validation_time': time.time(),
            'tolerance_factor': tolerance_factor,
            'target_results': {},
            'overall_compliance': True,
            'critical_violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        targets_met = True
        
        # Compare each performance metric against corresponding target values
        for target_name, target_value in self.performance_targets.items():
            # Parse target name to extract operation and metric type
            if '_latency_ms' in target_name:
                operation_name = target_name.replace('_latency_ms', '')
                metric_type = 'latency'
                
                if operation_name in self.timing_data and self.timing_data[operation_name]:
                    times = self.timing_data[operation_name]
                    
                    # Apply tolerance_factor for flexible target validation
                    adjusted_target = target_value * (1 + tolerance_factor)
                    
                    # Calculate compliance metrics
                    mean_time = np.mean(times)
                    p95_time = np.percentile(times, 95)
                    compliance_rate = sum(1 for t in times if t <= adjusted_target) / len(times)
                    
                    target_result = {
                        'operation': operation_name,
                        'metric_type': metric_type,
                        'target_value': target_value,
                        'adjusted_target': adjusted_target,
                        'measured_mean': mean_time,
                        'measured_p95': p95_time,
                        'compliance_rate': compliance_rate,
                        'mean_meets_target': mean_time <= adjusted_target,
                        'p95_meets_target': p95_time <= adjusted_target,
                        'overall_meets_target': compliance_rate >= 0.9  # 90% of measurements must pass
                    }
                    
                    # Include performance margin analysis and optimization opportunities
                    target_result['performance_margin'] = {
                        'mean_margin_ms': adjusted_target - mean_time,
                        'p95_margin_ms': adjusted_target - p95_time,
                        'margin_percentage': ((adjusted_target - mean_time) / adjusted_target) * 100
                    }
                    
                    # Check if target is met
                    if not target_result['overall_meets_target']:
                        targets_met = False
                        validation_report['overall_compliance'] = False
                        
                        if compliance_rate < 0.5:  # Less than 50% compliance is critical
                            validation_report['critical_violations'].append(
                                f"{operation_name} latency critical failure: {compliance_rate:.1%} compliance"
                            )
                        else:
                            validation_report['warnings'].append(
                                f"{operation_name} latency below target: {compliance_rate:.1%} compliance"
                            )
                    
                    validation_report['target_results'][target_name] = target_result
            
            elif '_memory_mb' in target_name:
                operation_name = target_name.replace('_memory_mb', '')
                metric_type = 'memory'
                
                if operation_name in self.resource_data and self.resource_data[operation_name]:
                    memory_data = self.resource_data[operation_name]
                    positive_memory = [m for m in memory_data if m > 0]  # Only positive deltas
                    
                    if positive_memory:
                        adjusted_target = target_value * (1 + tolerance_factor)
                        mean_memory = np.mean(positive_memory)
                        max_memory = np.max(positive_memory)
                        compliance_rate = sum(1 for m in positive_memory if m <= adjusted_target) / len(positive_memory)
                        
                        target_result = {
                            'operation': operation_name,
                            'metric_type': metric_type,
                            'target_value': target_value,
                            'adjusted_target': adjusted_target,
                            'measured_mean': mean_memory,
                            'measured_max': max_memory,
                            'compliance_rate': compliance_rate,
                            'mean_meets_target': mean_memory <= adjusted_target,
                            'max_meets_target': max_memory <= adjusted_target,
                            'overall_meets_target': compliance_rate >= 0.9
                        }
                        
                        if not target_result['overall_meets_target']:
                            targets_met = False
                            validation_report['overall_compliance'] = False
                            validation_report['warnings'].append(
                                f"{operation_name} memory usage exceeds target: {mean_memory:.1f}MB avg"
                            )
                        
                        validation_report['target_results'][target_name] = target_result
        
        # Identify critical performance bottlenecks and improvement areas
        if validation_report['critical_violations']:
            validation_report['recommendations'].extend([
                "Address critical performance violations immediately",
                "Consider system resource constraints and optimization strategies"
            ])
        
        if validation_report['warnings']:
            validation_report['recommendations'].extend([
                "Monitor performance trends for warning conditions",
                "Consider performance tuning for operations approaching limits"
            ])
        
        # Generate recommendations for performance optimization and tuning
        if targets_met:
            validation_report['recommendations'].append("All performance targets met within tolerance")
        else:
            validation_report['recommendations'].extend([
                "Review performance optimization strategies",
                f"Consider adjusting tolerance factor from {tolerance_factor:.1%}",
                "Analyze system capabilities and resource constraints"
            ])
        
        # Return validation status with comprehensive analysis and actionable feedback
        return targets_met, validation_report


# ===== CORE PYTEST FIXTURES =====

@pytest.fixture(scope='function')
def unit_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Creates PlumeSearchEnv instance optimized for unit testing with minimal parameters and fast execution 
    configuration for rapid test feedback.
    
    Yields:
        PlumeSearchEnv: Unit test environment instance ready for component testing
    """
    # Create unit test configuration using create_unit_test_config with minimal parameters
    config = create_unit_test_config()
    
    # Initialize PlumeSearchEnv using create_plume_search_env with unit test configuration
    env = create_plume_search_env(
        grid_size=config.grid_size,
        source_location=config.source_location,
        plume_sigma=config.plume_sigma,
        max_episode_steps=config.max_steps,
        render_mode=config.render_mode,
        random_seed=config.random_seed
    )
    
    # Register environment instance for automatic cleanup and resource management
    test_id = f"unit_test_env_{int(time.time() * 1000)}"
    cleanup_metadata = {'fixture_type': 'unit_test', 'category': 'unit'}
    _register_test_environment(env, test_id, cleanup_metadata)
    
    # Validate environment initialization and component readiness for testing
    try:
        obs, info = env.reset()
        assert obs is not None, "Environment reset failed to return observation"
        assert isinstance(info, dict), "Environment reset failed to return info dict"
    except Exception as e:
        env.close()
        raise RuntimeError(f"Unit test environment validation failed: {e}")
    
    try:
        # Yield environment instance to test function for usage
        yield env
    finally:
        # Perform cleanup and resource validation after test completion
        try:
            env.close()
        except Exception as e:
            warnings.warn(f"Unit test environment cleanup failed: {e}", UserWarning)


@pytest.fixture(scope='function')
def integration_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Creates PlumeSearchEnv instance optimized for integration testing with realistic parameters and 
    full component integration validation.
    
    Yields:
        PlumeSearchEnv: Integration test environment instance ready for cross-component testing
    """
    # Create integration test configuration using create_integration_test_config with realistic parameters
    config = create_integration_test_config(enable_performance_monitoring=True)
    
    # Initialize PlumeSearchEnv with integration configuration and cross-component validation
    env = create_plume_search_env(
        grid_size=config.grid_size,
        source_location=config.source_location,
        plume_sigma=config.plume_sigma,
        max_episode_steps=config.max_steps,
        render_mode=config.render_mode,
        random_seed=config.random_seed
    )
    
    # Enable performance monitoring and component interaction tracking
    test_id = f"integration_test_env_{int(time.time() * 1000)}"
    cleanup_metadata = {
        'fixture_type': 'integration_test',
        'category': 'integration',
        'performance_monitoring': True
    }
    
    # Register environment for cleanup with integration-specific resource management
    _register_test_environment(env, test_id, cleanup_metadata)
    
    # Validate comprehensive integration functionality
    try:
        # Test basic functionality
        obs, info = env.reset()
        
        # Test step functionality
        action = Action.UP
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Validate integration components
        assert obs is not None and next_obs is not None, "Observation integration failed"
        assert isinstance(reward, (int, float)), "Reward system integration failed"
        assert isinstance(terminated, bool) and isinstance(truncated, bool), "Termination logic integration failed"
        
    except Exception as e:
        env.close()
        raise RuntimeError(f"Integration test environment validation failed: {e}")
    
    try:
        # Yield fully configured environment ready for cross-component testing
        yield env
    finally:
        # Perform comprehensive cleanup and integration validation after test completion
        try:
            env.close()
        except Exception as e:
            warnings.warn(f"Integration test environment cleanup failed: {e}", UserWarning)


@pytest.fixture(scope='function')
def performance_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Creates PlumeSearchEnv instance optimized for performance testing with benchmark parameters 
    and comprehensive performance monitoring.
    
    Yields:
        PlumeSearchEnv: Performance test environment instance ready for benchmark validation
    """
    # Create performance test configuration using create_performance_test_config with benchmark parameters
    config = create_performance_test_config(strict_timing=True)
    
    # Initialize PlumeSearchEnv with performance optimization and monitoring enabled
    env = create_plume_search_env(
        grid_size=config.grid_size,
        source_location=config.source_location,
        plume_sigma=config.plume_sigma,
        max_episode_steps=config.max_steps,
        render_mode=config.render_mode,
        random_seed=config.random_seed
    )
    
    # Set up comprehensive performance tracking including timing and resource monitoring
    test_id = f"performance_test_env_{int(time.time() * 1000)}"
    cleanup_metadata = {
        'fixture_type': 'performance_test',
        'category': 'performance',
        'performance_monitoring': True,
        'benchmark_mode': True,
        'target_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS
    }
    
    # Register environment with performance-specific cleanup and validation
    _register_test_environment(env, test_id, cleanup_metadata)
    
    # Performance validation
    try:
        # Warm-up phase
        env.reset()
        
        # Performance validation test
        start_time = time.perf_counter()
        for _ in range(10):  # Quick performance test
            env.step(Action.UP)
        duration_ms = (time.perf_counter() - start_time) * 1000 / 10  # Average per step
        
        # Validate performance meets targets
        if duration_ms > PERFORMANCE_TARGET_STEP_LATENCY_MS * 2:  # 2x tolerance for setup
            warnings.warn(f"Performance test environment step latency {duration_ms:.2f}ms exceeds 2x target", UserWarning)
        
    except Exception as e:
        env.close()
        raise RuntimeError(f"Performance test environment validation failed: {e}")
    
    try:
        # Yield environment configured for performance benchmarking and analysis
        yield env
    finally:
        # Collect performance metrics and generate analysis report after test completion
        try:
            env.close()
        except Exception as e:
            warnings.warn(f"Performance test environment cleanup failed: {e}", UserWarning)


@pytest.fixture(scope='function')
def reproducibility_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Creates PlumeSearchEnv instance optimized for reproducibility testing with fixed seeding and 
    deterministic parameters.
    
    Yields:
        PlumeSearchEnv: Reproducibility test environment instance ready for deterministic validation
    """
    # Create reproducibility configuration using create_reproducibility_test_config with fixed seeding
    config = create_reproducibility_test_config(seed=REPRODUCIBILITY_SEEDS[0], strict_determinism=True)
    
    # Initialize PlumeSearchEnv with deterministic parameters and reproducibility validation
    env = create_plume_search_env(
        grid_size=config.grid_size,
        source_location=config.source_location,
        plume_sigma=config.plume_sigma,
        max_episode_steps=config.max_steps,
        render_mode=config.render_mode,
        random_seed=config.random_seed
    )
    
    # Set up reproducibility tracking and cross-session consistency monitoring
    test_id = f"reproducibility_test_env_{int(time.time() * 1000)}"
    cleanup_metadata = {
        'fixture_type': 'reproducibility_test',
        'category': 'reproducibility',
        'fixed_seed': config.random_seed,
        'deterministic_mode': True
    }
    
    # Register environment with reproducibility-specific validation and cleanup
    _register_test_environment(env, test_id, cleanup_metadata)
    
    # Reproducibility validation
    try:
        # Test deterministic behavior
        obs1, info1 = env.reset()
        action_sequence = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        
        results1 = []
        for action in action_sequence:
            result = env.step(action)
            results1.append(result[0])  # Store observations
        
        # Reset and repeat
        obs2, info2 = env.reset()
        results2 = []
        for action in action_sequence:
            result = env.step(action)
            results2.append(result[0])
        
        # Validate reproducibility
        if not np.array_equal(obs1, obs2):
            warnings.warn("Reproducibility test environment may not be deterministic", UserWarning)
        
    except Exception as e:
        env.close()
        raise RuntimeError(f"Reproducibility test environment validation failed: {e}")
    
    try:
        # Yield environment configured for deterministic behavior testing and validation
        yield env
    finally:
        # Validate reproducibility and generate consistency analysis after test completion
        try:
            env.close()
        except Exception as e:
            warnings.warn(f"Reproducibility test environment cleanup failed: {e}", UserWarning)


@pytest.fixture(scope='function')
def edge_case_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Creates PlumeSearchEnv instance optimized for edge case testing with extreme parameters 
    and boundary conditions.
    
    Yields:
        PlumeSearchEnv: Edge case test environment instance ready for robustness validation
    """
    # Create edge case configuration using create_edge_case_test_config with extreme parameters
    config = create_edge_case_test_config('boundary_conditions', enable_error_monitoring=True)
    
    # Initialize PlumeSearchEnv with boundary conditions and stress testing setup
    env = create_plume_search_env(
        grid_size=config.grid_size,
        source_location=config.source_location,
        plume_sigma=config.plume_sigma,
        max_episode_steps=config.max_steps,
        render_mode=config.render_mode,
        random_seed=config.random_seed
    )
    
    # Set up error monitoring and exception tracking for edge case validation
    test_id = f"edge_case_test_env_{int(time.time() * 1000)}"
    cleanup_metadata = {
        'fixture_type': 'edge_case_test',
        'category': 'edge_case',
        'error_monitoring': True,
        'boundary_testing': True
    }
    
    # Register environment with edge-case-specific cleanup and error handling
    _register_test_environment(env, test_id, cleanup_metadata)
    
    # Edge case validation with error handling
    try:
        # Test basic functionality under edge conditions
        obs, info = env.reset()
        
        # Test boundary actions
        for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
            try:
                result = env.step(action)
                assert len(result) == 5, "Step result should be 5-tuple"
            except Exception as e:
                # Log edge case behavior but don't fail fixture
                warnings.warn(f"Edge case behavior detected for action {action}: {e}", UserWarning)
        
    except Exception as e:
        # For edge cases, log errors but continue
        warnings.warn(f"Edge case environment validation encountered: {e}", UserWarning)
    
    try:
        # Yield environment configured for robustness testing and boundary validation
        yield env
    finally:
        # Perform error analysis and robustness validation after test completion
        try:
            env.close()
        except Exception as e:
            warnings.warn(f"Edge case test environment cleanup failed: {e}", UserWarning)


# ===== TEST DATA FIXTURES =====

@pytest.fixture(scope='module')
def test_coordinates() -> Dict[str, Any]:
    """Provides comprehensive coordinate test data including valid positions, boundary cases, 
    and edge scenarios.
    
    Returns:
        dict: Comprehensive coordinate test data with validation criteria
    """
    # Generate valid coordinate sets for different grid sizes using create_coordinates
    coordinate_data = _create_test_data('coordinates', {
        'grid_size': DEFAULT_GRID_SIZE,
        'include_center': True,
        'include_corners': True
    }, include_edge_cases=True)
    
    return coordinate_data


@pytest.fixture(scope='module')
def test_actions() -> Dict[str, Any]:
    """Provides action sequence test data including valid movements, invalid actions, 
    and boundary cases.
    
    Returns:
        dict: Comprehensive action test data with movement validation
    """
    # Generate action sequences using Action enum values
    action_data = _create_test_data('actions', {
        'sequence_length': 20,
        'include_patterns': True,
        'include_random': True
    }, include_edge_cases=True)
    
    return action_data


@pytest.fixture(scope='session')
def test_seeds() -> Dict[str, Any]:
    """Provides reproducibility seed values for deterministic testing and consistency validation.
    
    Returns:
        dict: Comprehensive seed collection with reproducibility metadata
    """
    # Load REPRODUCIBILITY_SEEDS and generate additional test seeds
    seed_data = _create_test_data('seeds', {
        'reproducibility_seeds': REPRODUCIBILITY_SEEDS,
        'additional_count': 10
    }, include_edge_cases=True)
    
    return seed_data


# ===== UTILITY FIXTURES =====

@pytest.fixture(scope='function')
def performance_tracker() -> Generator[PerformanceTracker, None, None]:
    """Provides PerformanceTracker instance for comprehensive performance monitoring during tests.
    
    Yields:
        PerformanceTracker: Performance tracker ready for measurement and analysis
    """
    # Create performance targets dictionary with system-specific targets
    performance_targets = {
        'step_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS,
        'reset_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS * 10,
        'render_latency_ms': PERFORMANCE_TARGET_RGB_RENDER_MS,
        'memory_mb': 10.0
    }
    
    # Initialize PerformanceTracker with test-specific configuration
    tracker_name = f"test_tracker_{int(time.time() * 1000)}"
    tracker = PerformanceTracker(tracker_name, performance_targets)
    
    try:
        # Yield PerformanceTracker ready for comprehensive performance measurement and analysis
        yield tracker
    finally:
        # Generate final performance report and validation after test completion
        try:
            if tracker.performance_history:
                summary = tracker.get_summary_report(include_optimization_recommendations=True)
                if summary.get('optimization_recommendations'):
                    print(f"\nPerformance Recommendations for {tracker_name}:")
                    for rec in summary['optimization_recommendations'][:3]:  # Top 3 recommendations
                        print(f"  - {rec}")
        except Exception as e:
            warnings.warn(f"Performance tracker summary generation failed: {e}", UserWarning)


@pytest.fixture(scope='session')
def test_config_factory() -> Generator[TestConfigFactory, None, None]:
    """Provides TestConfigFactory instance for intelligent test configuration creation.
    
    Yields:
        TestConfigFactory: Configuration factory ready for intelligent test setup
    """
    # Use global config factory or create new one
    global _config_factory
    
    if _config_factory is None:
        _config_factory = TestConfigFactory(auto_optimize=True)
        _config_factory.detect_system_capabilities(force_refresh=True)
    
    try:
        # Return TestConfigFactory ready for intelligent test configuration creation and system optimization
        yield _config_factory
    finally:
        # No cleanup required as this is a session-scoped utility
        pass


@pytest.fixture(scope='function')
def memory_monitor() -> Generator[Dict[str, Any], None, None]:
    """Provides memory monitoring utilities for leak detection and resource validation.
    
    Yields:
        dict: Memory monitoring context with baseline and tracking utilities
    """
    # Initialize memory monitoring with baseline measurement
    memory_context = {
        'baseline_mb': None,
        'measurements': [],
        'warnings': [],
        'leak_detected': False
    }
    
    if MEMORY_LEAK_DETECTION_ENABLED:
        try:
            memory_context['baseline_mb'] = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            memory_context['warnings'].append("Failed to establish memory baseline")
    
    def measure_memory():
        """Take a memory measurement and store it."""
        if MEMORY_LEAK_DETECTION_ENABLED:
            try:
                current_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_context['measurements'].append({
                    'timestamp': time.time(),
                    'memory_mb': current_mb,
                    'delta_mb': current_mb - memory_context.get('baseline_mb', current_mb)
                })
                return current_mb
            except Exception:
                memory_context['warnings'].append("Failed to measure memory")
        return None
    
    def check_for_leaks(threshold_mb: float = 5.0):
        """Check for potential memory leaks."""
        if memory_context['measurements'] and len(memory_context['measurements']) > 1:
            latest = memory_context['measurements'][-1]
            if latest['delta_mb'] > threshold_mb:
                memory_context['leak_detected'] = True
                memory_context['warnings'].append(f"Potential memory leak: +{latest['delta_mb']:.1f}MB")
    
    memory_context['measure'] = measure_memory
    memory_context['check_leaks'] = check_for_leaks
    
    try:
        # Yield memory monitor ready for comprehensive memory analysis
        yield memory_context
    finally:
        # Generate memory analysis report after test completion
        if memory_context['measurements']:
            final_measurement = memory_context['measurements'][-1]
            if final_measurement['delta_mb'] > 1.0:  # More than 1MB increase
                warnings.warn(f"Test memory usage: +{final_measurement['delta_mb']:.1f}MB", UserWarning)


@pytest.fixture(scope='function')
def cleanup_validator() -> Generator[Dict[str, Any], None, None]:
    """Provides cleanup validation utilities for resource deallocation verification.
    
    Yields:
        dict: Cleanup validation context with verification utilities
    """
    # Initialize cleanup validation context
    validation_context = {
        'resources_registered': [],
        'cleanup_results': [],
        'validation_errors': [],
        'start_time': time.time()
    }
    
    def register_resource(resource_id: str, resource_type: str, cleanup_func: callable = None):
        """Register a resource for cleanup validation."""
        resource_info = {
            'id': resource_id,
            'type': resource_type,
            'cleanup_func': cleanup_func,
            'registered_time': time.time(),
            'cleaned': False
        }
        validation_context['resources_registered'].append(resource_info)
    
    def validate_cleanup():
        """Validate that all registered resources were properly cleaned up."""
        for resource in validation_context['resources_registered']:
            if resource['cleanup_func']:
                try:
                    result = resource['cleanup_func']()
                    validation_context['cleanup_results'].append({
                        'resource_id': resource['id'],
                        'success': True,
                        'result': result
                    })
                    resource['cleaned'] = True
                except Exception as e:
                    validation_context['validation_errors'].append({
                        'resource_id': resource['id'],
                        'error': str(e)
                    })
        
        # Force garbage collection
        if CLEANUP_VALIDATION_ENABLED:
            gc.collect()
    
    validation_context['register'] = register_resource
    validation_context['validate'] = validate_cleanup
    
    try:
        # Yield cleanup validator ready for resource management validation
        yield validation_context
    finally:
        # Perform final cleanup validation and report generation
        if validation_context['resources_registered']:
            validation_context['validate']()
            
            uncleaned = [r for r in validation_context['resources_registered'] if not r['cleaned']]
            if uncleaned:
                warnings.warn(f"Cleanup validation failed: {len(uncleaned)} resources not cleaned", UserWarning)


@pytest.fixture(scope='module')
def test_data_factory() -> Dict[str, callable]:
    """Provides standardized test data creation utilities for comprehensive testing scenarios.
    
    Returns:
        dict: Test data factory functions for various data types
    """
    # Initialize test data factory with comprehensive data generation capabilities
    factory_functions = {
        'create_coordinates': lambda grid_size=DEFAULT_GRID_SIZE, count=10: _create_test_data(
            'coordinates', {'grid_size': grid_size, 'count': count}
        ),
        'create_actions': lambda length=10, patterns=True: _create_test_data(
            'actions', {'sequence_length': length, 'include_patterns': patterns}
        ),
        'create_seeds': lambda count=5: _create_test_data(
            'seeds', {'count': count}
        ),
        'create_trajectories': lambda steps=20, grid_size=DEFAULT_GRID_SIZE: _create_test_data(
            'trajectories', {'steps': steps, 'grid_size': grid_size}
        )
    }
    
    # Return test data factory ready for comprehensive test scenario creation
    return factory_functions


@pytest.fixture(scope='function')
def test_environment_manager(request) -> Generator[TestEnvironmentManager, None, None]:
    """Provides TestEnvironmentManager context manager for comprehensive environment lifecycle management.
    
    Yields:
        TestEnvironmentManager: Environment manager ready for comprehensive lifecycle management
    """
    # Detect test category from test name or markers
    test_category = 'unit'  # Default
    if hasattr(request, 'node'):
        if 'integration' in request.node.name:
            test_category = 'integration'
        elif 'performance' in request.node.name:
            test_category = 'performance'
        elif 'edge' in request.node.name:
            test_category = 'edge_case'
        elif hasattr(request.node, 'get_closest_marker'):
            if request.node.get_closest_marker('performance'):
                test_category = 'performance'
            elif request.node.get_closest_marker('integration'):
                test_category = 'integration'
    
    # Create appropriate configuration for test category
    env_config = {
        'grid_size': DEFAULT_GRID_SIZE,
        'source_location': DEFAULT_SOURCE_LOCATION,
        'max_episode_steps': 100,
        'render_mode': 'rgb_array'
    }
    
    # Adjust configuration based on category
    if test_category == 'unit':
        env_config['grid_size'] = (32, 32)
        env_config['max_episode_steps'] = 50
    elif test_category == 'performance':
        env_config['grid_size'] = (128, 128)
        env_config['max_episode_steps'] = 1000
    
    # Create TestEnvironmentManager
    manager = TestEnvironmentManager(
        test_category=test_category,
        env_config=env_config,
        enable_monitoring=PERFORMANCE_MONITORING_ENABLED
    )
    
    try:
        # Yield TestEnvironmentManager ready for comprehensive environment management
        yield manager
    finally:
        # TestEnvironmentManager handles its own cleanup through context manager
        pass


# ===== MODULE EXPORTS =====

__all__ = [
    # Pytest hooks
    'pytest_configure', 'pytest_unconfigure', 'pytest_runtest_setup', 'pytest_runtest_teardown',
    
    # Core environment fixtures
    'unit_test_env', 'integration_test_env', 'performance_test_env', 'reproducibility_test_env', 'edge_case_test_env',
    
    # Test data fixtures
    'test_coordinates', 'test_actions', 'test_seeds',
    
    # Utility fixtures
    'performance_tracker', 'test_config_factory', 'memory_monitor', 'cleanup_validator', 
    'test_data_factory', 'test_environment_manager',
    
    # Context manager classes
    'TestEnvironmentManager', 'PerformanceTracker'
]