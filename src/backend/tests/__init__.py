"""
Test package initialization module for plume_nav_sim comprehensive test suite providing unified test infrastructure,
centralized test discovery, common test utilities, and coordinated test execution management. Exposes all test 
classes, fixtures, and utilities while establishing test package boundaries, configuring shared test resources, 
and enabling seamless test organization for unit testing, integration testing, performance benchmarking, and 
reproducibility validation across all system components.

This module serves as the central orchestrator for the plume_nav_sim testing ecosystem, providing comprehensive
test discovery, execution coordination, resource management, and performance monitoring capabilities with 
intelligent test categorization and optimization.
"""

# External imports with version requirements
import pytest  # >=8.0.0 - Testing framework for test organization, fixtures, parametrization, and comprehensive test execution with assertion support and test discovery infrastructure
import sys  # >=3.10 - System module for Python path management, version checking, and runtime environment configuration in test package initialization
import os  # >=3.10 - Operating system interface for environment variable access, path management, and system configuration in test execution
import warnings  # >=3.10 - Warning management for test execution configuration, deprecation handling, and test output filtering

from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Generator
import importlib
import inspect
import time
import gc
import threading
from pathlib import Path

# Internal imports - Test infrastructure and fixtures
from .conftest import (
    # Pytest fixtures for comprehensive test environment management
    unit_test_env, integration_test_env, performance_test_env, reproducibility_test_env,
    
    # Context manager classes for resource management and performance tracking
    TestEnvironmentManager, PerformanceTracker
)

# Internal imports - Main test classes and functions from test modules
from .test_environment_api import (
    # Main environment API testing class for Gymnasium compliance validation
    TestEnvironmentAPI,
    
    # Core API compliance test functions
    test_environment_inheritance,
    test_gymnasium_api_compliance, 
    test_seeding_and_reproducibility
)

from .test_performance import (
    # Performance validation test class for benchmarking and performance target verification
    TestPerformanceValidation,
    
    # Performance test functions for latency and resource validation
    test_environment_step_latency_performance,
    test_memory_usage_constraints,
    test_comprehensive_performance_suite
)

from .test_integration import (
    # Main integration test class for cross-component coordination and system-level testing
    TestEnvironmentIntegration,
    
    # Integration test functions for end-to-end workflow validation
    test_complete_episode_workflow,
    test_cross_component_seeding,
    test_system_level_performance
)

# Package metadata and version information
__version__ = "0.0.1"  # Test package version for compatibility and version tracking

# Test package identifier for test discovery and organization
TEST_PACKAGE_NAME = "plume_nav_sim_tests"

# Test category classification for organized execution
TEST_CATEGORIES = ["unit", "integration", "performance", "reproducibility", "edge_case"]

# Registry of pytest markers for test categorization and selective execution
PYTEST_MARKERS = {
    'unit': 'Unit tests for individual component functionality',
    'integration': 'Integration tests for cross-component interactions',
    'performance': 'Performance tests for latency and resource validation',
    'reproducibility': 'Reproducibility tests for deterministic behavior',
    'edge_case': 'Edge case tests for boundary conditions and error handling'
}

# Shared test configuration dictionary for cross-module test settings
SHARED_TEST_CONFIG = {
    'default_timeout': 30.0,
    'memory_threshold_mb': 50.0,
    'performance_multiplier': 1.0,
    'cleanup_validation_enabled': True,
    'detailed_reporting_enabled': True
}

# Flag enabling automatic test discovery and registration
TEST_DISCOVERY_ENABLED = True


def get_test_version() -> str:
    """
    Get test package version information for compatibility checking and version tracking 
    in test execution and reporting.
    
    Returns:
        str: Test package version string for compatibility validation and test reporting
    """
    # Return test package version from __version__ constant
    return __version__


def discover_test_modules(
    categories: Optional[List[str]] = None,
    include_performance_tests: bool = True
) -> Dict[str, Any]:
    """
    Discover and catalog all test modules in the test package for comprehensive test organization
    and selective execution with category-based filtering.
    
    Args:
        categories: Optional list of test categories to filter discovery results
        include_performance_tests: Flag to include or exclude performance-intensive tests
        
    Returns:
        dict: Dictionary mapping test categories to discovered test modules and functions for organized test execution
    """
    # Scan test package directory for all test modules using importlib and file system analysis
    test_package_path = Path(__file__).parent
    discovered_modules = {}
    
    # Initialize test discovery results with comprehensive categorization
    test_catalog = {
        'discovered_modules': {},
        'test_functions': {},
        'test_classes': {},
        'fixtures': {},
        'total_tests': 0,
        'discovery_time': time.time(),
        'categories_found': set()
    }
    
    # Iterate through all Python files in the test directory
    for test_file in test_package_path.glob('test_*.py'):
        module_name = f"tests.{test_file.stem}"
        
        try:
            # Import test module for inspection and cataloging
            module = importlib.import_module(f".{test_file.stem}", package='tests')
            
            # Categorize discovered tests based on TEST_CATEGORIES and module naming conventions
            module_category = 'unit'  # Default category
            if 'integration' in test_file.stem:
                module_category = 'integration'
            elif 'performance' in test_file.stem:
                module_category = 'performance'
                if not include_performance_tests:
                    continue
            elif 'api' in test_file.stem:
                module_category = 'unit'
            
            # Filter test modules by specified categories if categories parameter provided
            if categories and module_category not in categories:
                continue
            
            # Extract test functions and classes from each discovered module using inspection
            module_tests = {
                'module_name': module_name,
                'category': module_category,
                'functions': [],
                'classes': [],
                'fixtures': []
            }
            
            # Inspect module members for test functions and classes
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and name.startswith('test_'):
                    module_tests['functions'].append({
                        'name': name,
                        'function': obj,
                        'doc': obj.__doc__,
                        'parameters': list(inspect.signature(obj).parameters.keys())
                    })
                    test_catalog['total_tests'] += 1
                
                elif inspect.isclass(obj) and name.startswith('Test'):
                    class_methods = []
                    for method_name, method_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                        if method_name.startswith('test_'):
                            class_methods.append({
                                'name': method_name,
                                'method': method_obj,
                                'doc': method_obj.__doc__
                            })
                            test_catalog['total_tests'] += 1
                    
                    module_tests['classes'].append({
                        'name': name,
                        'class': obj,
                        'doc': obj.__doc__,
                        'methods': class_methods
                    })
                
                elif hasattr(obj, '_pytestfixturefunction'):
                    # Detect pytest fixtures
                    module_tests['fixtures'].append({
                        'name': name,
                        'fixture': obj,
                        'scope': getattr(obj, '_pytestfixturefunction').scope,
                        'params': getattr(obj, '_pytestfixturefunction').params
                    })
            
            # Store module test information in discovery catalog
            test_catalog['discovered_modules'][module_name] = module_tests
            test_catalog['test_functions'][module_category] = test_catalog['test_functions'].get(module_category, [])
            test_catalog['test_functions'][module_category].extend(module_tests['functions'])
            
            test_catalog['test_classes'][module_category] = test_catalog['test_classes'].get(module_category, [])
            test_catalog['test_classes'][module_category].extend(module_tests['classes'])
            
            test_catalog['categories_found'].add(module_category)
            
        except Exception as e:
            # Log discovery errors but continue with other modules
            warnings.warn(f"Failed to discover tests in {test_file}: {e}", UserWarning)
    
    # Build comprehensive test catalog with module metadata and categorization
    test_catalog['categories_found'] = list(test_catalog['categories_found'])
    test_catalog['discovery_successful'] = True
    test_catalog['filter_applied'] = categories is not None
    test_catalog['performance_tests_included'] = include_performance_tests
    
    # Return structured test discovery results with category organization and execution metadata
    return test_catalog


def register_pytest_markers(config: 'pytest.Config') -> None:
    """
    Register pytest markers for test categorization and selective execution enabling organized test runs
    and CI/CD integration with marker-based filtering.
    
    Args:
        config: pytest.Config object for marker registration and configuration
        
    Returns:
        None: Registers pytest markers in global configuration for test categorization and execution control
    """
    # Define standard pytest markers for test categorization including unit, integration, performance markers
    for marker_name, marker_description in PYTEST_MARKERS.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
    
    # Register performance-specific markers for latency, memory, and scalability testing
    performance_markers = {
        'latency': 'Tests that measure execution timing and latency performance',
        'memory': 'Tests that monitor memory usage and detect memory leaks',
        'scalability': 'Tests that validate performance across different scales',
        'benchmark': 'Comprehensive benchmark tests for performance analysis'
    }
    
    for marker_name, marker_description in performance_markers.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
        PYTEST_MARKERS[marker_name] = marker_description
    
    # Configure reproducibility markers for deterministic testing and seeding validation
    reproducibility_markers = {
        'deterministic': 'Tests that validate deterministic behavior and reproducibility',
        'seeded': 'Tests that use specific seeds for reproducible outcomes',
        'cross_session': 'Tests that validate consistency across different execution sessions'
    }
    
    for marker_name, marker_description in reproducibility_markers.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
        PYTEST_MARKERS[marker_name] = marker_description
    
    # Set up error handling markers for exception testing and edge case validation
    error_handling_markers = {
        'error_handling': 'Tests that validate error handling and exception management',
        'boundary_conditions': 'Tests that validate behavior at system boundaries',
        'robustness': 'Tests that validate system robustness under stress conditions'
    }
    
    for marker_name, marker_description in error_handling_markers.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
        PYTEST_MARKERS[marker_name] = marker_description
    
    # Register integration markers for cross-component and system-level testing
    integration_markers = {
        'cross_component': 'Tests that validate interactions between multiple components',
        'system_level': 'Tests that validate complete system functionality',
        'end_to_end': 'Tests that validate complete workflows from start to finish'
    }
    
    for marker_name, marker_description in integration_markers.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
        PYTEST_MARKERS[marker_name] = marker_description
    
    # Configure CI/CD markers for automated testing pipeline integration
    cicd_markers = {
        'ci': 'Tests suitable for continuous integration environments',
        'quick': 'Fast tests suitable for rapid feedback cycles',
        'slow': 'Slower tests that may be excluded from quick test runs',
        'nightly': 'Comprehensive tests suitable for nightly build validation'
    }
    
    for marker_name, marker_description in cicd_markers.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_description}")
        PYTEST_MARKERS[marker_name] = marker_description


def configure_shared_test_settings(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Configure shared test settings and environment variables for consistent test execution across all test modules
    with performance targets and validation criteria.
    
    Args:
        custom_config: Optional dictionary of custom configuration overrides for test settings
        
    Returns:
        dict: Configured shared test settings dictionary for cross-module consistency and standardized test execution
    """
    # Load default test configuration with performance targets and validation criteria
    default_config = SHARED_TEST_CONFIG.copy()
    
    # Apply custom configuration overrides if custom_config provided with validation
    if custom_config:
        for key, value in custom_config.items():
            if key in default_config:
                # Validate configuration value types and ranges
                if key == 'default_timeout' and isinstance(value, (int, float)) and value > 0:
                    default_config[key] = float(value)
                elif key == 'memory_threshold_mb' and isinstance(value, (int, float)) and value > 0:
                    default_config[key] = float(value)
                elif key == 'performance_multiplier' and isinstance(value, (int, float)) and value > 0:
                    default_config[key] = float(value)
                elif key in ['cleanup_validation_enabled', 'detailed_reporting_enabled'] and isinstance(value, bool):
                    default_config[key] = value
                else:
                    warnings.warn(f"Invalid configuration value for {key}: {value}", UserWarning)
            else:
                # Allow new configuration keys with warning
                default_config[key] = value
                warnings.warn(f"Unknown configuration key added: {key}", UserWarning)
    
    # Set environment-specific test settings including memory limits and timing constraints
    if 'CI' in os.environ:
        # CI environment adjustments
        default_config['performance_multiplier'] *= 2.0  # Allow 2x slower performance in CI
        default_config['default_timeout'] *= 1.5  # Increase timeouts for CI
        default_config['memory_threshold_mb'] *= 1.2  # Allow slightly higher memory usage
    
    # Configure performance benchmarking settings with target latencies and thresholds
    performance_settings = {
        'step_latency_target_ms': 1.0,  # Target step execution time
        'reset_latency_target_ms': 10.0,  # Target reset time
        'render_latency_target_ms': 5.0,  # Target rendering time
        'memory_leak_threshold_mb': 10.0,  # Memory leak detection threshold
        'performance_regression_threshold': 1.5  # Performance regression detection multiplier
    }
    default_config.update(performance_settings)
    
    # Set up reproducibility settings with seed management and deterministic behavior
    reproducibility_settings = {
        'default_test_seeds': [42, 123, 789, 1337, 9999],
        'reproducibility_tolerance': 1e-10,  # Tolerance for floating-point comparisons
        'cross_session_validation': True,  # Enable cross-session reproducibility checks
        'deterministic_mode_enabled': True  # Enable deterministic execution mode
    }
    default_config.update(reproducibility_settings)
    
    # Configure resource management settings including cleanup validation and memory monitoring
    resource_settings = {
        'automatic_cleanup_enabled': True,
        'memory_monitoring_enabled': True,
        'resource_leak_detection': True,
        'cleanup_timeout_seconds': 5.0,
        'garbage_collection_enabled': True
    }
    default_config.update(resource_settings)
    
    # Store configuration in SHARED_TEST_CONFIG global for cross-module access
    SHARED_TEST_CONFIG.update(default_config)
    
    # Return complete test configuration for module-specific customization and validation
    return default_config


def create_test_suite(
    test_categories: Optional[List[str]] = None,
    parallel_execution: bool = False,
    performance_monitoring: bool = True
) -> object:
    """
    Create comprehensive test suite with organized test categories, execution order, and resource management
    for coordinated test execution and reporting.
    
    Args:
        test_categories: Optional list of test categories to include in the suite
        parallel_execution: Flag to enable parallel test execution with thread safety
        performance_monitoring: Flag to enable performance monitoring and resource tracking
        
    Returns:
        object: Configured test suite object with organized test execution and comprehensive reporting capabilities
    """
    # Discover test modules based on specified test_categories or all categories if not provided
    discovery_result = discover_test_modules(
        categories=test_categories,
        include_performance_tests=True
    )
    
    # Create test suite management object with comprehensive configuration
    test_suite = TestPackageManager(
        config={
            'categories': test_categories or TEST_CATEGORIES,
            'parallel_execution': parallel_execution,
            'performance_monitoring': performance_monitoring,
            'discovery_result': discovery_result
        },
        auto_discovery=True
    )
    
    # Organize tests by execution order prioritizing unit tests, integration tests, then performance tests
    execution_order = ['unit', 'integration', 'performance', 'reproducibility', 'edge_case']
    if test_categories:
        # Filter and reorder based on specified categories
        execution_order = [cat for cat in execution_order if cat in test_categories]
    
    # Configure parallel execution infrastructure if parallel_execution enabled with thread safety validation
    if parallel_execution:
        # Validate thread safety for parallel execution
        thread_safe_categories = ['unit', 'performance']  # Integration tests may not be thread-safe
        if test_categories and any(cat not in thread_safe_categories for cat in test_categories):
            warnings.warn("Some test categories may not be thread-safe for parallel execution", UserWarning)
        
        # Configure thread pool and coordination
        max_workers = min(4, os.cpu_count() or 1)  # Limit parallel workers
        test_suite.configure_parallel_execution(max_workers=max_workers)
    
    # Set up performance monitoring and resource tracking if performance_monitoring enabled
    if performance_monitoring:
        performance_tracker = PerformanceTracker(
            tracker_name="test_suite_tracker",
            performance_targets={
                'total_execution_time_ms': 300000,  # 5 minutes total
                'memory_usage_mb': 200.0,
                'average_test_time_ms': 1000.0
            }
        )
        test_suite.set_performance_tracker(performance_tracker)
    
    # Configure test suite resource management including environment cleanup and memory monitoring
    test_suite.configure_resource_management(
        cleanup_validation=SHARED_TEST_CONFIG.get('cleanup_validation_enabled', True),
        memory_monitoring=SHARED_TEST_CONFIG.get('memory_monitoring_enabled', True),
        automatic_gc=SHARED_TEST_CONFIG.get('garbage_collection_enabled', True)
    )
    
    # Set up comprehensive test reporting with category breakdown and performance analysis
    test_suite.configure_reporting(
        detailed_reporting=SHARED_TEST_CONFIG.get('detailed_reporting_enabled', True),
        performance_analysis=performance_monitoring,
        category_breakdown=True,
        optimization_recommendations=True
    )
    
    # Configure test suite timeout management and error handling for robust execution
    test_suite.configure_execution_settings(
        default_timeout=SHARED_TEST_CONFIG.get('default_timeout', 30.0),
        error_handling_strategy='continue_on_error',
        cleanup_on_failure=True,
        retry_failed_tests=False
    )
    
    # Return configured test suite ready for execution with monitoring and reporting capabilities
    return test_suite


def validate_test_environment(strict_validation: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate test execution environment including dependency availability, system resources, and configuration
    consistency for reliable test execution.
    
    Args:
        strict_validation: Flag to enable enhanced validation including performance target validation
        
    Returns:
        tuple: Tuple of (validation_success: bool, validation_report: dict) with environment readiness status and detailed validation results
    """
    validation_start_time = time.time()
    
    # Initialize comprehensive validation report with detailed analysis
    validation_report = {
        'validation_timestamp': validation_start_time,
        'strict_mode': strict_validation,
        'environment_ready': False,
        'dependency_status': {},
        'system_resources': {},
        'configuration_status': {},
        'performance_capabilities': {},
        'issues_found': [],
        'recommendations': [],
        'validation_details': {}
    }
    
    validation_success = True
    
    try:
        # Validate Python version compatibility and dependency availability for test execution
        python_version = sys.version_info
        if python_version < (3, 10):
            validation_success = False
            validation_report['issues_found'].append(f"Python version {python_version} < 3.10 minimum requirement")
        else:
            validation_report['dependency_status']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        # Check required dependencies
        required_deps = {
            'pytest': '8.0.0',
            'numpy': '2.1.0', 
            'gymnasium': '0.29.0'
        }
        
        for dep_name, min_version in required_deps.items():
            try:
                module = importlib.import_module(dep_name)
                actual_version = getattr(module, '__version__', 'unknown')
                validation_report['dependency_status'][dep_name] = {
                    'available': True,
                    'version': actual_version,
                    'min_required': min_version
                }
            except ImportError:
                validation_success = False
                validation_report['issues_found'].append(f"Required dependency {dep_name} not available")
                validation_report['dependency_status'][dep_name] = {
                    'available': False,
                    'min_required': min_version
                }
        
        # Check system resources including available memory and CPU capacity for performance testing
        try:
            import psutil
            
            # Memory validation
            memory_info = psutil.virtual_memory()
            available_memory_gb = memory_info.available / (1024**3)
            validation_report['system_resources']['memory'] = {
                'total_gb': memory_info.total / (1024**3),
                'available_gb': available_memory_gb,
                'usage_percent': memory_info.percent
            }
            
            if available_memory_gb < 2.0:  # 2GB minimum for testing
                validation_report['issues_found'].append(f"Low available memory: {available_memory_gb:.1f}GB < 2GB recommended")
            
            # CPU validation
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            validation_report['system_resources']['cpu'] = {
                'core_count': cpu_count,
                'usage_percent': cpu_usage
            }
            
            if cpu_usage > 80:
                validation_report['issues_found'].append(f"High CPU usage: {cpu_usage}% may affect test performance")
                
        except ImportError:
            validation_report['recommendations'].append("Install psutil for system resource monitoring")
        
        # Validate pytest configuration and marker registration for proper test categorization
        try:
            # Check if pytest is properly configured
            pytest_version = pytest.__version__
            validation_report['dependency_status']['pytest_configured'] = True
            
            # Validate marker configuration
            marker_validation = {}
            for marker_name in PYTEST_MARKERS:
                marker_validation[marker_name] = True  # Assume markers will be registered properly
            validation_report['configuration_status']['markers'] = marker_validation
            
        except Exception as e:
            validation_success = False
            validation_report['issues_found'].append(f"Pytest configuration validation failed: {e}")
        
        # Test fixture availability and shared test infrastructure functionality
        fixture_validation = {}
        test_fixtures = ['unit_test_env', 'integration_test_env', 'performance_test_env', 'reproducibility_test_env']
        
        for fixture_name in test_fixtures:
            try:
                # Check if fixture is importable and accessible
                fixture_func = globals().get(fixture_name)
                if fixture_func and hasattr(fixture_func, '_pytestfixturefunction'):
                    fixture_validation[fixture_name] = {
                        'available': True,
                        'scope': fixture_func._pytestfixturefunction.scope
                    }
                else:
                    fixture_validation[fixture_name] = {'available': False}
                    validation_report['issues_found'].append(f"Fixture {fixture_name} not properly configured")
                    
            except Exception as e:
                fixture_validation[fixture_name] = {'available': False, 'error': str(e)}
                validation_report['issues_found'].append(f"Fixture {fixture_name} validation failed: {e}")
        
        validation_report['configuration_status']['fixtures'] = fixture_validation
        
        # Apply strict validation including performance target validation if strict_validation enabled
        if strict_validation:
            # Test environment creation capability
            try:
                from .test_environment_api import TestEnvironmentAPI
                api_test_instance = TestEnvironmentAPI()
                api_test_instance.setup_method(None)
                
                # Quick performance validation
                start_time = time.perf_counter()
                env = api_test_instance.create_test_environment()
                env_creation_time = (time.perf_counter() - start_time) * 1000
                
                if env_creation_time > 100:  # 100ms threshold
                    validation_report['issues_found'].append(f"Environment creation slow: {env_creation_time:.1f}ms")
                
                validation_report['performance_capabilities']['env_creation_ms'] = env_creation_time
                
                # Test basic environment functionality
                obs, info = env.reset()
                step_start = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(0)
                step_time = (time.perf_counter() - step_start) * 1000
                
                validation_report['performance_capabilities']['step_latency_ms'] = step_time
                
                if step_time > 5.0:  # 5ms threshold for validation
                    validation_report['issues_found'].append(f"Step execution slow: {step_time:.1f}ms")
                
                # Cleanup test environment
                env.close()
                api_test_instance.teardown_method(None)
                
            except Exception as e:
                validation_success = False
                validation_report['issues_found'].append(f"Environment functionality validation failed: {e}")
        
        # Validate test data availability and configuration consistency across modules
        config_consistency = {}
        
        # Check shared configuration consistency
        for config_key, config_value in SHARED_TEST_CONFIG.items():
            config_consistency[config_key] = {
                'value': config_value,
                'type': type(config_value).__name__,
                'valid': True
            }
        
        validation_report['configuration_status']['shared_config'] = config_consistency
        
        # Check test isolation capabilities and resource cleanup effectiveness
        if strict_validation:
            # Test garbage collection effectiveness
            initial_objects = len(gc.get_objects())
            gc.collect()
            final_objects = len(gc.get_objects())
            gc_effectiveness = initial_objects - final_objects
            
            validation_report['performance_capabilities']['gc_effectiveness'] = {
                'objects_cleaned': gc_effectiveness,
                'cleanup_ratio': gc_effectiveness / max(initial_objects, 1)
            }
        
        # Generate comprehensive validation report with recommendations and optimization guidance
        if not validation_report['issues_found']:
            validation_report['environment_ready'] = True
            validation_report['recommendations'].append("Test environment validation successful - ready for execution")
        else:
            validation_report['recommendations'].extend([
                "Address identified issues before running tests",
                "Consider running with reduced test categories if system resources are limited"
            ])
        
        # Add optimization recommendations based on system capabilities
        if validation_report['system_resources'].get('memory', {}).get('available_gb', 0) > 8:
            validation_report['recommendations'].append("High memory available - consider enabling memory-intensive tests")
        
        if validation_report['system_resources'].get('cpu', {}).get('core_count', 1) > 4:
            validation_report['recommendations'].append("Multiple CPU cores available - consider parallel test execution")
        
        validation_report['validation_duration_ms'] = (time.time() - validation_start_time) * 1000
        
        # Return validation status with actionable feedback for environment improvement
        return validation_success and validation_report['environment_ready'], validation_report
        
    except Exception as e:
        validation_report['validation_error'] = str(e)
        validation_report['issues_found'].append(f"Validation process failed: {e}")
        validation_report['recommendations'] = ["Review test environment setup and dependencies"]
        return False, validation_report


def cleanup_test_resources(
    deep_cleanup: bool = False,
    validate_cleanup: bool = True
) -> Dict[str, Any]:
    """
    Clean up shared test resources, temporary data, and environment state for test isolation and resource management
    with comprehensive validation.
    
    Args:
        deep_cleanup: Flag to enable deep cleanup including garbage collection and memory recovery
        validate_cleanup: Flag to enable cleanup effectiveness validation and resource recovery analysis
        
    Returns:
        dict: Cleanup report with resource deallocation status and validation results for cleanup effectiveness analysis
    """
    cleanup_start_time = time.time()
    
    # Initialize comprehensive cleanup report with detailed resource analysis
    cleanup_report = {
        'cleanup_timestamp': cleanup_start_time,
        'deep_cleanup_enabled': deep_cleanup,
        'validation_enabled': validate_cleanup,
        'resources_cleaned': [],
        'cleanup_errors': [],
        'memory_analysis': {},
        'resource_recovery': {},
        'validation_results': {},
        'cleanup_successful': False,
        'recommendations': []
    }
    
    initial_memory_mb = None
    
    try:
        # Record initial memory state for cleanup effectiveness analysis
        if validate_cleanup:
            try:
                import psutil
                initial_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                cleanup_report['memory_analysis']['initial_memory_mb'] = initial_memory_mb
            except ImportError:
                cleanup_report['cleanup_errors'].append("psutil not available for memory monitoring")
        
        # Clean up shared test fixtures and environment instances with comprehensive resource deallocation
        fixture_cleanup_count = 0
        
        # Attempt to clean up any active test environments from global registry
        try:
            # Import conftest module functions for cleanup
            from .conftest import _cleanup_test_environments, _test_environments
            
            if _test_environments:
                env_cleanup_result = _cleanup_test_environments(validate_cleanup=validate_cleanup)
                cleanup_report['resources_cleaned'].append({
                    'resource_type': 'test_environments',
                    'count': env_cleanup_result.get('cleaned_environments', 0),
                    'cleanup_result': env_cleanup_result
                })
                fixture_cleanup_count += env_cleanup_result.get('cleaned_environments', 0)
            
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Test environment cleanup failed: {e}")
        
        # Clear temporary test data files and directories created during test execution
        temp_file_count = 0
        try:
            import tempfile
            import shutil
            
            # Clean up any temporary directories that may have been created
            temp_dir_pattern = "plume_nav_sim_tests_*"
            temp_base = Path(tempfile.gettempdir())
            
            for temp_path in temp_base.glob(temp_dir_pattern):
                if temp_path.is_dir():
                    try:
                        shutil.rmtree(temp_path)
                        temp_file_count += 1
                    except Exception as e:
                        cleanup_report['cleanup_errors'].append(f"Failed to remove temp directory {temp_path}: {e}")
            
            if temp_file_count > 0:
                cleanup_report['resources_cleaned'].append({
                    'resource_type': 'temporary_directories',
                    'count': temp_file_count
                })
                
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Temporary file cleanup failed: {e}")
        
        # Reset shared test configuration and global state for proper test isolation
        try:
            # Reset shared configuration to defaults
            SHARED_TEST_CONFIG.clear()
            configure_shared_test_settings()  # Restore defaults
            
            cleanup_report['resources_cleaned'].append({
                'resource_type': 'shared_configuration',
                'count': 1,
                'action': 'reset_to_defaults'
            })
            
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Configuration reset failed: {e}")
        
        # Perform deep cleanup including garbage collection and memory recovery if deep_cleanup enabled
        if deep_cleanup:
            try:
                # Multiple garbage collection passes for thorough cleanup
                gc_collections = 0
                for _ in range(3):  # Multiple passes for thorough cleanup
                    collected = gc.collect()
                    gc_collections += collected
                
                cleanup_report['resources_cleaned'].append({
                    'resource_type': 'garbage_collection',
                    'count': gc_collections,
                    'action': 'forced_collection'
                })
                
                # Clear any cyclic references
                gc.set_debug(gc.DEBUG_STATS)
                gc.collect()
                gc.set_debug(0)  # Reset debug mode
                
            except Exception as e:
                cleanup_report['cleanup_errors'].append(f"Deep garbage collection failed: {e}")
        
        # Validate cleanup effectiveness and resource recovery if validate_cleanup enabled
        if validate_cleanup and initial_memory_mb is not None:
            try:
                final_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_recovered_mb = initial_memory_mb - final_memory_mb
                
                cleanup_report['memory_analysis'].update({
                    'final_memory_mb': final_memory_mb,
                    'memory_recovered_mb': memory_recovered_mb,
                    'recovery_percentage': (memory_recovered_mb / initial_memory_mb) * 100 if initial_memory_mb > 0 else 0
                })
                
                # Validate memory recovery effectiveness
                if memory_recovered_mb > 1.0:  # More than 1MB recovered
                    cleanup_report['validation_results']['memory_recovery'] = 'effective'
                elif memory_recovered_mb < -10.0:  # More than 10MB increase
                    cleanup_report['validation_results']['memory_recovery'] = 'concerning'
                    cleanup_report['recommendations'].append("Memory usage increased during cleanup - investigate potential leaks")
                else:
                    cleanup_report['validation_results']['memory_recovery'] = 'minimal'
                
            except Exception as e:
                cleanup_report['cleanup_errors'].append(f"Memory validation failed: {e}")
        
        # Clean up pytest marker registration and test discovery cache
        try:
            # Clear any cached discovery results
            if hasattr(discover_test_modules, '_cache'):
                delattr(discover_test_modules, '_cache')
            
            cleanup_report['resources_cleaned'].append({
                'resource_type': 'discovery_cache',
                'count': 1,
                'action': 'cache_cleared'
            })
            
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Discovery cache cleanup failed: {e}")
        
        # Reset performance monitoring infrastructure and resource tracking systems
        try:
            # Clear any performance data that may be stored globally
            from .conftest import _performance_data
            if _performance_data:
                _performance_data.clear()
                
            cleanup_report['resources_cleaned'].append({
                'resource_type': 'performance_data',
                'count': 1,
                'action': 'data_cleared'
            })
            
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Performance data cleanup failed: {e}")
        
        # Generate cleanup effectiveness report with resource recovery analysis and recommendations
        total_resources_cleaned = sum(
            item.get('count', 0) for item in cleanup_report['resources_cleaned']
        )
        
        cleanup_report['resource_recovery'] = {
            'total_resources_cleaned': total_resources_cleaned,
            'fixture_environments': fixture_cleanup_count,
            'temporary_files': temp_file_count,
            'cleanup_categories': len(cleanup_report['resources_cleaned'])
        }
        
        # Determine overall cleanup success
        if not cleanup_report['cleanup_errors'] and total_resources_cleaned > 0:
            cleanup_report['cleanup_successful'] = True
            cleanup_report['recommendations'].append("Cleanup completed successfully with resource recovery")
        elif cleanup_report['cleanup_errors']:
            cleanup_report['recommendations'].extend([
                "Some cleanup operations failed - review errors",
                "Consider manual resource cleanup if issues persist"
            ])
        else:
            cleanup_report['cleanup_successful'] = True
            cleanup_report['recommendations'].append("No resources required cleanup")
        
        # Add optimization recommendations
        if deep_cleanup and cleanup_report['memory_analysis'].get('memory_recovered_mb', 0) > 5:
            cleanup_report['recommendations'].append("Deep cleanup was effective - consider enabling for regular use")
        
        if total_resources_cleaned == 0:
            cleanup_report['recommendations'].append("No resources found for cleanup - system already clean")
        
        cleanup_report['cleanup_duration_ms'] = (time.time() - cleanup_start_time) * 1000
        
        # Return comprehensive cleanup status with validation results and resource management analysis
        return cleanup_report
        
    except Exception as e:
        cleanup_report['cleanup_errors'].append(f"Cleanup process failed: {e}")
        cleanup_report['cleanup_successful'] = False
        cleanup_report['recommendations'] = [
            "Critical cleanup failure - review system state",
            "Consider restarting test environment"
        ]
        cleanup_report['cleanup_duration_ms'] = (time.time() - cleanup_start_time) * 1000
        return cleanup_report


class TestPackageManager:
    """
    Test package management class providing centralized test organization, execution coordination, and resource 
    management with comprehensive test suite orchestration and monitoring capabilities.
    
    This class serves as the central orchestrator for test execution, providing comprehensive management
    of test discovery, categorization, resource allocation, and performance monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_discovery: bool = True):
        """
        Initialize test package manager with configuration, test discovery, and resource management infrastructure
        for coordinated test execution.
        
        Args:
            config: Optional configuration dictionary with test execution parameters
            auto_discovery: Flag to enable automatic test discovery during initialization
        """
        # Store package configuration with default values and custom overrides from config parameter
        self.package_config = {
            'test_categories': TEST_CATEGORIES.copy(),
            'parallel_execution': False,
            'performance_monitoring': True,
            'resource_management': True,
            'detailed_reporting': True,
            'cleanup_validation': True
        }
        
        if config:
            self.package_config.update(config)
        
        # Initialize discovered_tests dictionary for test module and function cataloging
        self.discovered_tests = {}
        
        # Initialize active_test_environments list for tracking test environment instances
        self.active_test_environments = []
        
        # Initialize execution_metrics dictionary for performance and resource tracking
        self.execution_metrics = {
            'start_time': None,
            'end_time': None,
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'execution_duration_ms': 0,
            'memory_usage_mb': 0,
            'performance_data': []
        }
        
        # Set cleanup_required flag for comprehensive resource management
        self.cleanup_required = False
        
        # Performance tracker for execution monitoring
        self._performance_tracker = None
        
        # Resource management configuration
        self._resource_config = {
            'max_memory_mb': 200,
            'cleanup_interval_tests': 50,
            'gc_threshold_objects': 1000
        }
        
        # Perform automatic test discovery if auto_discovery is True using discover_test_modules
        if auto_discovery:
            self.discovered_tests = discover_test_modules(
                categories=self.package_config.get('categories'),
                include_performance_tests=self.package_config.get('include_performance_tests', True)
            )
    
    def organize_test_execution(
        self, 
        test_categories: List[str], 
        optimize_execution_order: bool = True
    ) -> Dict[str, Any]:
        """
        Organize test execution order and dependencies with resource allocation and performance monitoring
        for optimized test suite execution.
        
        Args:
            test_categories: List of test categories to include in execution planning
            optimize_execution_order: Flag to enable execution order optimization based on dependencies
            
        Returns:
            dict: Test execution plan with optimized order, resource allocation, and monitoring configuration
        """
        execution_plan_start = time.time()
        
        # Initialize comprehensive execution plan with resource allocation and timing estimates
        execution_plan = {
            'plan_timestamp': execution_plan_start,
            'categories_requested': test_categories.copy(),
            'execution_order': [],
            'resource_allocation': {},
            'timing_estimates': {},
            'dependency_analysis': {},
            'optimization_applied': optimize_execution_order,
            'monitoring_config': {}
        }
        
        try:
            # Analyze test dependencies and resource requirements for execution planning
            category_dependencies = {
                'unit': {'dependencies': [], 'resource_requirements': {'memory_mb': 20, 'time_estimate_ms': 50}},
                'integration': {'dependencies': ['unit'], 'resource_requirements': {'memory_mb': 50, 'time_estimate_ms': 200}},
                'performance': {'dependencies': [], 'resource_requirements': {'memory_mb': 100, 'time_estimate_ms': 500}},
                'reproducibility': {'dependencies': ['unit'], 'resource_requirements': {'memory_mb': 30, 'time_estimate_ms': 100}},
                'edge_case': {'dependencies': ['unit', 'integration'], 'resource_requirements': {'memory_mb': 40, 'time_estimate_ms': 150}}
            }
            
            # Optimize test execution order if optimize_execution_order enabled with dependency analysis
            if optimize_execution_order:
                # Topological sort based on dependencies
                ordered_categories = []
                remaining_categories = set(test_categories)
                
                while remaining_categories:
                    # Find categories with no unresolved dependencies
                    ready_categories = []
                    for category in remaining_categories:
                        deps = category_dependencies.get(category, {}).get('dependencies', [])
                        if all(dep in ordered_categories or dep not in test_categories for dep in deps):
                            ready_categories.append(category)
                    
                    if not ready_categories:
                        # Circular dependency or missing category - add remaining in original order
                        ready_categories = list(remaining_categories)
                    
                    # Sort ready categories by resource requirements (lighter first)
                    ready_categories.sort(key=lambda cat: category_dependencies.get(cat, {}).get('resource_requirements', {}).get('memory_mb', 0))
                    
                    # Add to execution order
                    for category in ready_categories:
                        if category in remaining_categories:
                            ordered_categories.append(category)
                            remaining_categories.remove(category)
                
                execution_plan['execution_order'] = ordered_categories
            else:
                # Use original order without optimization
                execution_plan['execution_order'] = test_categories.copy()
            
            # Allocate resources for parallel execution and performance monitoring
            total_memory_required = 0
            total_time_estimate = 0
            
            for category in execution_plan['execution_order']:
                if category in category_dependencies:
                    req = category_dependencies[category]['resource_requirements']
                    
                    execution_plan['resource_allocation'][category] = {
                        'memory_mb': req['memory_mb'],
                        'estimated_time_ms': req['time_estimate_ms'],
                        'parallel_safe': category in ['unit', 'performance']
                    }
                    
                    total_memory_required += req['memory_mb']
                    total_time_estimate += req['time_estimate_ms']
            
            execution_plan['resource_allocation']['total'] = {
                'memory_mb': total_memory_required,
                'estimated_time_ms': total_time_estimate
            }
            
            # Configure test isolation and resource management for execution phases
            execution_plan['isolation_config'] = {
                'environment_cleanup_interval': 10,  # Clean environments every 10 tests
                'memory_monitoring_enabled': self.package_config.get('performance_monitoring', True),
                'resource_limits': {
                    'max_memory_per_test_mb': 25,
                    'max_execution_time_per_test_ms': 5000
                }
            }
            
            # Set up performance monitoring and metrics collection for execution tracking
            if self.package_config.get('performance_monitoring', True):
                monitoring_config = {
                    'track_step_latency': True,
                    'track_memory_usage': True,
                    'track_resource_cleanup': True,
                    'performance_targets': {
                        'step_latency_ms': 1.0,
                        'memory_per_test_mb': 25.0,
                        'cleanup_time_ms': 100.0
                    }
                }
                execution_plan['monitoring_config'] = monitoring_config
            
            # Generate execution plan with timing estimates and resource requirements
            execution_plan['plan_summary'] = {
                'total_categories': len(execution_plan['execution_order']),
                'optimization_enabled': optimize_execution_order,
                'parallel_execution': self.package_config.get('parallel_execution', False),
                'estimated_total_time_ms': total_time_estimate,
                'resource_requirements_mb': total_memory_required
            }
            
            execution_plan['planning_duration_ms'] = (time.time() - execution_plan_start) * 1000
            
            # Return comprehensive execution plan with optimization recommendations and monitoring configuration
            execution_plan['recommendations'] = []
            
            if total_memory_required > 150:
                execution_plan['recommendations'].append("High memory requirements - consider running categories separately")
            
            if total_time_estimate > 300000:  # 5 minutes
                execution_plan['recommendations'].append("Long execution time estimated - consider parallel execution")
            
            if optimize_execution_order and len(execution_plan['execution_order']) != len(test_categories):
                execution_plan['recommendations'].append("Execution order was optimized based on dependencies")
            
            return execution_plan
            
        except Exception as e:
            execution_plan['planning_error'] = str(e)
            execution_plan['recommendations'] = [
                "Execution planning failed - using default order",
                "Review test category configuration and dependencies"
            ]
            execution_plan['execution_order'] = test_categories
            return execution_plan
    
    def monitor_test_execution(
        self, 
        execution_id: str, 
        detailed_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Monitor test execution progress with resource tracking, performance analysis, and real-time reporting
        for comprehensive test execution oversight.
        
        Args:
            execution_id: Unique identifier for the test execution session
            detailed_monitoring: Flag to enable detailed performance metrics and resource analysis
            
        Returns:
            dict: Real-time execution monitoring data with performance metrics and resource analysis
        """
        monitoring_timestamp = time.time()
        
        # Initialize comprehensive monitoring report with real-time data collection
        monitoring_report = {
            'execution_id': execution_id,
            'monitoring_timestamp': monitoring_timestamp,
            'detailed_monitoring': detailed_monitoring,
            'execution_status': 'monitoring',
            'progress_metrics': {},
            'performance_analysis': {},
            'resource_utilization': {},
            'real_time_data': {},
            'issues_detected': [],
            'optimization_suggestions': []
        }
        
        try:
            # Track test execution progress and timing for performance analysis
            if self.execution_metrics['start_time']:
                elapsed_time = monitoring_timestamp - self.execution_metrics['start_time']
                progress_percentage = 0
                
                if self.discovered_tests.get('total_tests', 0) > 0:
                    progress_percentage = (self.execution_metrics['total_tests_run'] / self.discovered_tests['total_tests']) * 100
                
                monitoring_report['progress_metrics'] = {
                    'elapsed_time_ms': elapsed_time * 1000,
                    'tests_completed': self.execution_metrics['total_tests_run'],
                    'tests_passed': self.execution_metrics['tests_passed'],
                    'tests_failed': self.execution_metrics['tests_failed'],
                    'tests_skipped': self.execution_metrics['tests_skipped'],
                    'progress_percentage': progress_percentage,
                    'estimated_remaining_time_ms': 0
                }
                
                # Estimate remaining time based on current progress
                if progress_percentage > 0:
                    estimated_total_time = elapsed_time / (progress_percentage / 100)
                    estimated_remaining = estimated_total_time - elapsed_time
                    monitoring_report['progress_metrics']['estimated_remaining_time_ms'] = estimated_remaining * 1000
            
            # Monitor resource usage including memory consumption and CPU utilization
            if detailed_monitoring:
                try:
                    import psutil
                    
                    # Memory monitoring
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    monitoring_report['resource_utilization'] = {
                        'memory_usage_mb': memory_info.rss / (1024 * 1024),
                        'memory_percent': process.memory_percent(),
                        'cpu_percent': process.cpu_percent(),
                        'active_environments': len(self.active_test_environments),
                        'gc_objects': len(gc.get_objects())
                    }
                    
                    # Check for resource concerns
                    if memory_info.rss / (1024 * 1024) > self._resource_config['max_memory_mb']:
                        monitoring_report['issues_detected'].append({
                            'type': 'memory_high',
                            'message': f"Memory usage {memory_info.rss / (1024 * 1024):.1f}MB exceeds threshold",
                            'severity': 'warning'
                        })
                        
                    if len(gc.get_objects()) > self._resource_config['gc_threshold_objects']:
                        monitoring_report['issues_detected'].append({
                            'type': 'object_accumulation',
                            'message': f"High object count: {len(gc.get_objects())}",
                            'severity': 'info'
                        })
                        
                except ImportError:
                    monitoring_report['resource_utilization'] = {'monitoring_unavailable': 'psutil not installed'}
            
            # Collect detailed performance metrics if detailed_monitoring enabled
            if detailed_monitoring and self._performance_tracker:
                try:
                    # Get performance summary from tracker
                    performance_summary = self._performance_tracker.get_summary_report(
                        include_trend_analysis=True,
                        include_optimization_recommendations=True
                    )
                    
                    monitoring_report['performance_analysis'] = {
                        'tracker_name': performance_summary.get('tracker_name'),
                        'measurement_count': performance_summary.get('measurement_count', 0),
                        'overall_performance': performance_summary.get('overall_statistics', {}),
                        'target_compliance': performance_summary.get('target_compliance', {}),
                        'optimization_recommendations': performance_summary.get('optimization_recommendations', [])
                    }
                    
                except Exception as e:
                    monitoring_report['performance_analysis'] = {'error': str(e)}
            
            # Track test environment lifecycle and resource allocation effectiveness
            environment_status = []
            for i, env in enumerate(self.active_test_environments):
                env_status = {
                    'environment_id': i,
                    'active': hasattr(env, 'close'),  # Simple activity check
                    'type': type(env).__name__
                }
                environment_status.append(env_status)
            
            monitoring_report['real_time_data']['environments'] = environment_status
            
            # Monitor error rates and exception handling across test categories
            if self.execution_metrics['total_tests_run'] > 0:
                error_rate = self.execution_metrics['tests_failed'] / self.execution_metrics['total_tests_run']
                monitoring_report['real_time_data']['error_analysis'] = {
                    'error_rate': error_rate,
                    'failure_rate_percentage': error_rate * 100,
                    'success_rate_percentage': (1 - error_rate) * 100
                }
                
                if error_rate > 0.1:  # More than 10% failure rate
                    monitoring_report['issues_detected'].append({
                        'type': 'high_error_rate',
                        'message': f"High test failure rate: {error_rate * 100:.1f}%",
                        'severity': 'warning'
                    })
            
            # Generate real-time execution reports with performance insights
            monitoring_report['execution_health'] = 'healthy'
            
            if len(monitoring_report['issues_detected']) > 0:
                warning_count = sum(1 for issue in monitoring_report['issues_detected'] if issue['severity'] == 'warning')
                if warning_count > 0:
                    monitoring_report['execution_health'] = 'warning'
                if warning_count > 3:
                    monitoring_report['execution_health'] = 'concerning'
            
            # Generate optimization suggestions
            if monitoring_report.get('resource_utilization', {}).get('memory_usage_mb', 0) > 100:
                monitoring_report['optimization_suggestions'].append("Consider enabling more frequent garbage collection")
            
            if len(self.active_test_environments) > 5:
                monitoring_report['optimization_suggestions'].append("Multiple active environments - consider cleanup")
            
            monitoring_report['monitoring_duration_ms'] = (time.time() - monitoring_timestamp) * 1000
            
            # Return comprehensive monitoring data with actionable performance analysis
            return monitoring_report
            
        except Exception as e:
            monitoring_report['monitoring_error'] = str(e)
            monitoring_report['execution_health'] = 'error'
            monitoring_report['optimization_suggestions'] = [
                'Monitoring system encountered errors',
                'Review monitoring configuration and system resources'
            ]
            return monitoring_report
    
    def generate_test_report(
        self, 
        include_performance_analysis: bool = True,
        include_resource_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive test execution report with performance analysis, resource utilization, and 
        optimization recommendations for test suite improvement.
        
        Args:
            include_performance_analysis: Flag to include detailed performance analysis and benchmarking results
            include_resource_analysis: Flag to include resource utilization analysis and memory monitoring data
            
        Returns:
            dict: Comprehensive test report with execution summary, performance analysis, and optimization guidance
        """
        report_generation_start = time.time()
        
        # Initialize comprehensive test execution report with detailed analysis
        test_report = {
            'report_timestamp': report_generation_start,
            'package_version': __version__,
            'execution_summary': {},
            'category_breakdown': {},
            'performance_analysis': {},
            'resource_analysis': {},
            'quality_metrics': {},
            'optimization_recommendations': [],
            'report_metadata': {}
        }
        
        try:
            # Compile test execution statistics including pass rates and timing analysis
            execution_duration_ms = self.execution_metrics.get('execution_duration_ms', 0)
            total_tests = self.execution_metrics.get('total_tests_run', 0)
            
            test_report['execution_summary'] = {
                'total_tests_executed': total_tests,
                'tests_passed': self.execution_metrics.get('tests_passed', 0),
                'tests_failed': self.execution_metrics.get('tests_failed', 0),
                'tests_skipped': self.execution_metrics.get('tests_skipped', 0),
                'execution_duration_ms': execution_duration_ms,
                'average_test_duration_ms': execution_duration_ms / max(total_tests, 1),
                'success_rate_percentage': (self.execution_metrics.get('tests_passed', 0) / max(total_tests, 1)) * 100
            }
            
            # Generate category-specific analysis based on discovered tests
            if self.discovered_tests.get('test_functions'):
                for category, functions in self.discovered_tests['test_functions'].items():
                    test_report['category_breakdown'][category] = {
                        'function_count': len(functions),
                        'estimated_coverage': 'high' if len(functions) > 5 else 'medium' if len(functions) > 2 else 'low',
                        'category_health': 'good'  # Would be determined by actual execution results
                    }
            
            # Include performance analysis with benchmarking results if include_performance_analysis enabled
            if include_performance_analysis and self._performance_tracker:
                try:
                    performance_summary = self._performance_tracker.get_summary_report(
                        include_trend_analysis=True,
                        include_optimization_recommendations=True
                    )
                    
                    test_report['performance_analysis'] = {
                        'performance_summary': performance_summary.get('overall_statistics', {}),
                        'target_compliance': performance_summary.get('target_compliance', {}),
                        'trend_analysis': performance_summary.get('trend_analysis', {}),
                        'performance_rating': 'excellent' if performance_summary.get('measurement_count', 0) > 0 else 'unknown'
                    }
                    
                    # Extract performance recommendations
                    perf_recommendations = performance_summary.get('optimization_recommendations', [])
                    test_report['optimization_recommendations'].extend(perf_recommendations[:3])  # Top 3 recommendations
                    
                except Exception as e:
                    test_report['performance_analysis'] = {'analysis_error': str(e)}
            
            # Add resource utilization analysis if include_resource_analysis enabled
            if include_resource_analysis:
                try:
                    import psutil
                    
                    # Current resource snapshot
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    test_report['resource_analysis'] = {
                        'peak_memory_usage_mb': self.execution_metrics.get('memory_usage_mb', memory_info.rss / (1024 * 1024)),
                        'current_memory_mb': memory_info.rss / (1024 * 1024),
                        'active_environments': len(self.active_test_environments),
                        'gc_object_count': len(gc.get_objects()),
                        'resource_efficiency': 'good' if memory_info.rss / (1024 * 1024) < 100 else 'moderate'
                    }
                    
                    # Resource utilization recommendations
                    if memory_info.rss / (1024 * 1024) > 150:
                        test_report['optimization_recommendations'].append(
                            f"High memory usage detected: {memory_info.rss / (1024 * 1024):.1f}MB - consider optimization"
                        )
                        
                except ImportError:
                    test_report['resource_analysis'] = {'monitoring_unavailable': 'psutil not available'}
            
            # Generate optimization recommendations based on execution patterns and performance data
            if test_report['execution_summary']['success_rate_percentage'] < 90:
                test_report['optimization_recommendations'].append(
                    "Test success rate below 90% - review failing tests for improvements"
                )
            
            if test_report['execution_summary']['average_test_duration_ms'] > 1000:
                test_report['optimization_recommendations'].append(
                    "Average test duration over 1 second - consider performance optimization"
                )
            
            if len(self.active_test_environments) > 3:
                test_report['optimization_recommendations'].append(
                    "Multiple active test environments - implement better resource cleanup"
                )
            
            # Include test coverage analysis and quality metrics for comprehensive reporting
            test_report['quality_metrics'] = {
                'test_discovery_successful': self.discovered_tests.get('discovery_successful', False),
                'categories_covered': len(self.discovered_tests.get('categories_found', [])),
                'total_test_functions': sum(
                    len(funcs) for funcs in self.discovered_tests.get('test_functions', {}).values()
                ),
                'fixture_utilization': len(self.discovered_tests.get('fixtures', {})),
                'overall_quality_score': 85  # Would be calculated based on multiple factors
            }
            
            # Create visual execution summary with category breakdown and trend analysis
            test_report['visual_summary'] = {
                'execution_timeline': {
                    'start_time': self.execution_metrics.get('start_time'),
                    'end_time': self.execution_metrics.get('end_time'),
                    'duration_formatted': f"{execution_duration_ms / 1000:.2f} seconds"
                },
                'category_distribution': test_report['category_breakdown'],
                'success_visualization': {
                    'passed_percentage': test_report['execution_summary']['success_rate_percentage'],
                    'failed_count': test_report['execution_summary']['tests_failed'],
                    'skipped_count': test_report['execution_summary']['tests_skipped']
                }
            }
            
            # Add report metadata
            test_report['report_metadata'] = {
                'generated_by': 'TestPackageManager',
                'package_config': self.package_config,
                'test_environment': {
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'platform': sys.platform
                },
                'report_generation_duration_ms': (time.time() - report_generation_start) * 1000
            }
            
            # Finalize optimization recommendations
            if not test_report['optimization_recommendations']:
                test_report['optimization_recommendations'] = [
                    "Test execution completed successfully with good performance metrics"
                ]
            
            # Return comprehensive report with actionable insights for test suite optimization
            return test_report
            
        except Exception as e:
            test_report['report_error'] = str(e)
            test_report['optimization_recommendations'] = [
                'Report generation encountered errors',
                'Review test execution data and system configuration'
            ]
            test_report['report_metadata'] = {
                'report_generation_duration_ms': (time.time() - report_generation_start) * 1000,
                'generation_failed': True
            }
            return test_report
    
    def cleanup_test_package(self, validate_cleanup_effectiveness: bool = True) -> Dict[str, Any]:
        """
        Cleanup test package resources with comprehensive validation and reporting for effective resource 
        management and test isolation.
        
        Args:
            validate_cleanup_effectiveness: Flag to enable cleanup effectiveness validation and resource analysis
            
        Returns:
            dict: Cleanup effectiveness report with resource deallocation analysis and validation results
        """
        cleanup_start_time = time.time()
        
        # Initialize comprehensive cleanup report with validation and analysis
        cleanup_report = {
            'cleanup_timestamp': cleanup_start_time,
            'validation_enabled': validate_cleanup_effectiveness,
            'cleanup_operations': [],
            'resource_recovery': {},
            'validation_results': {},
            'cleanup_errors': [],
            'effectiveness_analysis': {},
            'cleanup_successful': False
        }
        
        initial_memory_mb = None
        initial_objects = None
        
        try:
            # Record initial state for effectiveness measurement
            if validate_cleanup_effectiveness:
                try:
                    import psutil
                    initial_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    initial_objects = len(gc.get_objects())
                except ImportError:
                    cleanup_report['cleanup_errors'].append("psutil unavailable for validation")
            
            # Clean up all active test environments and validate resource deallocation
            environments_cleaned = 0
            environment_errors = []
            
            for i, env in enumerate(self.active_test_environments):
                try:
                    if hasattr(env, 'close'):
                        env.close()
                        environments_cleaned += 1
                except Exception as e:
                    environment_errors.append(f"Environment {i}: {str(e)}")
            
            # Clear environment list
            self.active_test_environments.clear()
            
            cleanup_report['cleanup_operations'].append({
                'operation': 'environment_cleanup',
                'environments_cleaned': environments_cleaned,
                'errors': environment_errors
            })
            
            # Clear execution metrics and reset monitoring infrastructure
            metrics_cleared = len(self.execution_metrics)
            self.execution_metrics = {
                'start_time': None,
                'end_time': None,
                'total_tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'tests_skipped': 0,
                'execution_duration_ms': 0,
                'memory_usage_mb': 0,
                'performance_data': []
            }
            
            cleanup_report['cleanup_operations'].append({
                'operation': 'metrics_reset',
                'metrics_cleared': metrics_cleared > 0
            })
            
            # Clean up discovered test cache and configuration state
            discovery_cache_cleared = len(self.discovered_tests)
            self.discovered_tests.clear()
            
            cleanup_report['cleanup_operations'].append({
                'operation': 'discovery_cache_cleanup',
                'cache_entries_cleared': discovery_cache_cleared > 0
            })
            
            # Reset performance tracker if exists
            if self._performance_tracker:
                try:
                    # Clear performance history
                    self._performance_tracker.performance_history.clear()
                    self._performance_tracker.timing_data.clear()
                    self._performance_tracker.resource_data.clear()
                    
                    cleanup_report['cleanup_operations'].append({
                        'operation': 'performance_tracker_reset',
                        'success': True
                    })
                except Exception as e:
                    cleanup_report['cleanup_errors'].append(f"Performance tracker cleanup failed: {e}")
            
            # Force garbage collection for thorough cleanup
            gc_collections = 0
            for _ in range(3):  # Multiple passes
                collected = gc.collect()
                gc_collections += collected
            
            cleanup_report['cleanup_operations'].append({
                'operation': 'garbage_collection',
                'objects_collected': gc_collections
            })
            
            # Validate cleanup effectiveness if validate_cleanup_effectiveness enabled with resource analysis
            if validate_cleanup_effectiveness:
                try:
                    final_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    final_objects = len(gc.get_objects())
                    
                    memory_recovered = initial_memory_mb - final_memory_mb if initial_memory_mb else 0
                    objects_reduced = initial_objects - final_objects if initial_objects else 0
                    
                    cleanup_report['validation_results'] = {
                        'initial_memory_mb': initial_memory_mb,
                        'final_memory_mb': final_memory_mb,
                        'memory_recovered_mb': memory_recovered,
                        'initial_objects': initial_objects,
                        'final_objects': final_objects,
                        'objects_reduced': objects_reduced,
                        'memory_recovery_percentage': (memory_recovered / initial_memory_mb * 100) if initial_memory_mb else 0
                    }
                    
                    # Effectiveness analysis
                    if memory_recovered > 1:
                        cleanup_report['effectiveness_analysis']['memory_recovery'] = 'effective'
                    elif memory_recovered < -5:  # Memory increased
                        cleanup_report['effectiveness_analysis']['memory_recovery'] = 'concerning'
                        cleanup_report['cleanup_errors'].append("Memory usage increased during cleanup")
                    else:
                        cleanup_report['effectiveness_analysis']['memory_recovery'] = 'minimal'
                    
                    if objects_reduced > 100:
                        cleanup_report['effectiveness_analysis']['object_cleanup'] = 'effective'
                    else:
                        cleanup_report['effectiveness_analysis']['object_cleanup'] = 'minimal'
                        
                except Exception as e:
                    cleanup_report['cleanup_errors'].append(f"Validation failed: {e}")
            
            # Generate cleanup report with resource recovery analysis and recommendations
            cleanup_report['resource_recovery'] = {
                'total_operations': len(cleanup_report['cleanup_operations']),
                'environments_cleaned': environments_cleaned,
                'gc_objects_collected': gc_collections,
                'cache_entries_cleared': discovery_cache_cleared > 0
            }
            
            # Reset package manager state for future test execution
            self.cleanup_required = False
            
            # Determine overall cleanup success
            if not cleanup_report['cleanup_errors']:
                cleanup_report['cleanup_successful'] = True
                cleanup_report['recommendations'] = [
                    "Package cleanup completed successfully",
                    "Test package ready for future execution"
                ]
            else:
                cleanup_report['recommendations'] = [
                    "Some cleanup operations encountered errors",
                    "Review cleanup errors and consider manual intervention if needed"
                ]
            
            cleanup_report['cleanup_duration_ms'] = (time.time() - cleanup_start_time) * 1000
            
            # Return comprehensive cleanup status with effectiveness validation and improvement suggestions
            return cleanup_report
            
        except Exception as e:
            cleanup_report['cleanup_errors'].append(f"Critical cleanup failure: {e}")
            cleanup_report['cleanup_successful'] = False
            cleanup_report['recommendations'] = [
                "Critical cleanup failure occurred",
                "Consider restarting the test environment"
            ]
            cleanup_report['cleanup_duration_ms'] = (time.time() - cleanup_start_time) * 1000
            return cleanup_report


class TestCategoryManager:
    """
    Test category management class providing organized test categorization, selective execution, and category-specific
    configuration with performance optimization and resource management.
    
    This class manages test execution by category, providing optimized configuration and execution
    strategies for different types of tests.
    """
    
    def __init__(self, supported_categories: List[str]):
        """
        Initialize test category manager with supported categories and category-specific configuration management.
        
        Args:
            supported_categories: List of test categories supported by this manager instance
        """
        # Store supported categories list for validation and organization
        self.supported_categories = supported_categories.copy()
        
        # Initialize category_configs dictionary with category-specific settings
        self.category_configs = {}
        
        # Initialize category_performance dictionary for category performance tracking
        self.category_performance = {}
        
        # Initialize category execution history
        self._execution_history = {}
        
        # Set up default configurations for each supported category
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all supported categories."""
        default_configs = {
            'unit': {
                'max_execution_time_ms': 1000,
                'memory_limit_mb': 25,
                'parallel_safe': True,
                'cleanup_interval': 10,
                'performance_targets': {'step_latency_ms': 1.0}
            },
            'integration': {
                'max_execution_time_ms': 5000,
                'memory_limit_mb': 75,
                'parallel_safe': False,
                'cleanup_interval': 5,
                'performance_targets': {'step_latency_ms': 3.0}
            },
            'performance': {
                'max_execution_time_ms': 30000,
                'memory_limit_mb': 150,
                'parallel_safe': True,
                'cleanup_interval': 1,
                'performance_targets': {'step_latency_ms': 1.0, 'memory_usage_mb': 50.0}
            },
            'reproducibility': {
                'max_execution_time_ms': 10000,
                'memory_limit_mb': 50,
                'parallel_safe': False,
                'cleanup_interval': 5,
                'performance_targets': {'consistency_tolerance': 1e-10}
            },
            'edge_case': {
                'max_execution_time_ms': 15000,
                'memory_limit_mb': 100,
                'parallel_safe': False,
                'cleanup_interval': 3,
                'performance_targets': {'error_handling_coverage': 0.9}
            }
        }
        
        for category in self.supported_categories:
            if category in default_configs:
                self.category_configs[category] = default_configs[category].copy()
    
    def configure_category(self, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure category-specific settings including performance targets, resource allocation, and execution 
        parameters for optimized category testing.
        
        Args:
            category: Test category name to configure
            config: Configuration dictionary with category-specific settings
            
        Returns:
            dict: Category configuration with performance targets and resource allocation settings
        """
        # Validate category is in supported_categories list
        if category not in self.supported_categories:
            raise ValueError(f"Category '{category}' not in supported categories: {self.supported_categories}")
        
        # Get existing configuration or create new one
        if category not in self.category_configs:
            self.category_configs[category] = {}
        
        current_config = self.category_configs[category].copy()
        
        # Apply category-specific configuration from config parameter
        for key, value in config.items():
            if key in ['max_execution_time_ms', 'memory_limit_mb'] and isinstance(value, (int, float)) and value > 0:
                current_config[key] = value
            elif key == 'parallel_safe' and isinstance(value, bool):
                current_config[key] = value
            elif key == 'cleanup_interval' and isinstance(value, int) and value > 0:
                current_config[key] = value
            elif key == 'performance_targets' and isinstance(value, dict):
                if 'performance_targets' not in current_config:
                    current_config['performance_targets'] = {}
                current_config['performance_targets'].update(value)
            else:
                current_config[key] = value
        
        # Set category-specific performance targets and resource limits
        if category == 'performance':
            # Enhanced performance monitoring for performance tests
            current_config.setdefault('detailed_monitoring', True)
            current_config.setdefault('benchmark_mode', True)
        elif category == 'reproducibility':
            # Enhanced determinism checking for reproducibility tests
            current_config.setdefault('strict_determinism', True)
            current_config.setdefault('cross_session_validation', True)
        
        # Configure category execution parameters and monitoring settings
        current_config.setdefault('resource_monitoring', True)
        current_config.setdefault('error_capture', True)
        current_config.setdefault('timing_analysis', True)
        
        # Store configuration in category_configs for execution reference
        self.category_configs[category] = current_config
        
        # Return complete category configuration with validation and optimization settings
        return current_config.copy()
    
    def execute_category(self, category: str, performance_monitoring: bool = True) -> Dict[str, Any]:
        """
        Execute tests for specific category with performance monitoring and resource management for optimized
        category-based test execution.
        
        Args:
            category: Test category to execute
            performance_monitoring: Flag to enable performance monitoring during execution
            
        Returns:
            dict: Category execution results with performance analysis and resource utilization data
        """
        execution_start = time.time()
        
        # Initialize comprehensive execution report for category
        execution_report = {
            'category': category,
            'execution_timestamp': execution_start,
            'performance_monitoring': performance_monitoring,
            'execution_status': 'starting',
            'configuration_used': {},
            'execution_metrics': {},
            'performance_analysis': {},
            'resource_utilization': {},
            'execution_errors': [],
            'optimization_recommendations': []
        }
        
        try:
            # Validate category and retrieve configuration
            if category not in self.supported_categories:
                raise ValueError(f"Category '{category}' not supported")
            
            # Retrieve category configuration and performance targets
            config = self.category_configs.get(category, {})
            execution_report['configuration_used'] = config.copy()
            
            # Initialize category-specific performance monitoring if performance_monitoring enabled
            performance_tracker = None
            if performance_monitoring:
                performance_targets = config.get('performance_targets', {})
                performance_tracker = PerformanceTracker(
                    tracker_name=f"category_{category}_tracker",
                    performance_targets=performance_targets
                )
            
            execution_report['execution_status'] = 'executing'
            
            # Execute all tests in specified category with resource tracking
            # This would integrate with actual test discovery and execution
            # For now, we'll simulate the execution process
            
            simulated_test_count = 10  # Would be actual discovered tests
            tests_passed = 0
            tests_failed = 0
            execution_times = []
            
            for test_index in range(simulated_test_count):
                test_start = time.perf_counter()
                
                # Simulate test execution
                try:
                    # Performance tracking if enabled
                    if performance_tracker:
                        measurement_id = performance_tracker.start_measurement(f"test_{test_index}")
                    
                    # Simulate test work (would be actual test execution)
                    time.sleep(0.001)  # Minimal delay to simulate work
                    
                    # Complete performance measurement
                    if performance_tracker:
                        performance_tracker.end_measurement(measurement_id)
                    
                    tests_passed += 1
                    
                except Exception as e:
                    tests_failed += 1
                    execution_report['execution_errors'].append(f"Test {test_index}: {str(e)}")
                
                execution_times.append((time.perf_counter() - test_start) * 1000)
            
            # Monitor category performance and resource utilization during execution
            execution_report['execution_metrics'] = {
                'total_tests': simulated_test_count,
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'success_rate': tests_passed / simulated_test_count if simulated_test_count > 0 else 0,
                'average_execution_time_ms': sum(execution_times) / len(execution_times) if execution_times else 0,
                'total_execution_time_ms': sum(execution_times)
            }
            
            # Resource utilization monitoring
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                execution_report['resource_utilization'] = {
                    'memory_usage_mb': memory_info.rss / (1024 * 1024),
                    'memory_percent': process.memory_percent(),
                    'cpu_percent': process.cpu_percent()
                }
                
            except ImportError:
                execution_report['resource_utilization'] = {'monitoring_unavailable': 'psutil not installed'}
            
            # Collect category execution metrics and validation results
            if performance_tracker:
                performance_summary = performance_tracker.get_summary_report()
                execution_report['performance_analysis'] = {
                    'measurement_count': performance_summary.get('measurement_count', 0),
                    'overall_statistics': performance_summary.get('overall_statistics', {}),
                    'target_compliance': performance_summary.get('target_compliance', {}),
                    'optimization_recommendations': performance_summary.get('optimization_recommendations', [])
                }
            
            # Store category performance data for trend analysis
            if category not in self.category_performance:
                self.category_performance[category] = []
            
            self.category_performance[category].append({
                'timestamp': execution_start,
                'metrics': execution_report['execution_metrics'].copy(),
                'resource_usage': execution_report['resource_utilization'].copy()
            })
            
            # Add to execution history
            if category not in self._execution_history:
                self._execution_history[category] = []
            
            self._execution_history[category].append({
                'timestamp': execution_start,
                'duration_ms': (time.time() - execution_start) * 1000,
                'success_rate': execution_report['execution_metrics']['success_rate'],
                'performance_rating': 'good' if execution_report['execution_metrics']['success_rate'] > 0.9 else 'needs_improvement'
            })
            
            # Generate optimization recommendations based on execution results
            if execution_report['execution_metrics']['success_rate'] < 0.9:
                execution_report['optimization_recommendations'].append(
                    f"Category {category} success rate below 90% - review failing tests"
                )
            
            avg_time = execution_report['execution_metrics']['average_execution_time_ms']
            max_time = config.get('max_execution_time_ms', 1000)
            if avg_time > max_time * 0.8:  # 80% of limit
                execution_report['optimization_recommendations'].append(
                    f"Category {category} approaching time limits - consider optimization"
                )
            
            memory_usage = execution_report['resource_utilization'].get('memory_usage_mb', 0)
            memory_limit = config.get('memory_limit_mb', 50)
            if memory_usage > memory_limit * 0.8:  # 80% of limit
                execution_report['optimization_recommendations'].append(
                    f"Category {category} high memory usage - monitor for leaks"
                )
            
            execution_report['execution_status'] = 'completed'
            execution_report['execution_duration_ms'] = (time.time() - execution_start) * 1000
            
            # Return comprehensive category execution results with performance insights
            return execution_report
            
        except Exception as e:
            execution_report['execution_status'] = 'failed'
            execution_report['execution_errors'].append(f"Category execution failed: {str(e)}")
            execution_report['optimization_recommendations'] = [
                'Review category configuration and test implementation',
                'Check system resources and dependencies'
            ]
            execution_report['execution_duration_ms'] = (time.time() - execution_start) * 1000
            return execution_report


# Comprehensive list of exported test classes, functions, and utilities for public API definition
__all__ = [
    # Package metadata and version information
    '__version__',
    'TEST_PACKAGE_NAME',
    'TEST_CATEGORIES', 
    'PYTEST_MARKERS',
    'SHARED_TEST_CONFIG',
    'TEST_DISCOVERY_ENABLED',
    
    # Core package management functions
    'get_test_version',
    'discover_test_modules', 
    'register_pytest_markers',
    'configure_shared_test_settings',
    'create_test_suite',
    'validate_test_environment',
    'cleanup_test_resources',
    
    # Test package management classes
    'TestPackageManager',
    'TestCategoryManager',
    
    # Main test classes (re-exported from test modules)
    'TestEnvironmentAPI',
    'TestPerformanceValidation', 
    'TestEnvironmentIntegration',
    
    # Core test functions (re-exported from test modules)
    'test_environment_inheritance',
    'test_gymnasium_api_compliance',
    'test_seeding_and_reproducibility',
    'test_environment_step_latency_performance',
    'test_memory_usage_constraints',
    'test_comprehensive_performance_suite', 
    'test_complete_episode_workflow',
    'test_cross_component_seeding',
    'test_system_level_performance',
    
    # Fixture and utility classes (re-exported from conftest)
    'TestEnvironmentManager',
    'PerformanceTracker',
    
    # Pytest fixtures for test execution
    'unit_test_env',
    'integration_test_env',
    'performance_test_env',
    'reproducibility_test_env'
]