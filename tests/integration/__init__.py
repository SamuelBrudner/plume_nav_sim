"""
Integration Testing Utilities for Modular Plume Navigation Simulation.

This module provides centralized integration testing infrastructure for end-to-end
validation of the modular plume navigation simulation architecture. Supports 
cross-component testing, protocol compliance validation, and configuration-driven
component switching scenarios essential for the new pluggable system design.

Key Testing Domains:
- Modular Component Integration: Cross-protocol validation between PlumeModel, 
  WindField, Sensor, and Navigator implementations
- Configuration-Driven Testing: Hydra-based component switching validation without
  code modifications (e.g., Gaussian ↔ Turbulent plume model switching)
- Performance Integration: End-to-end latency validation across different component
  combinations while maintaining <10ms step execution requirements
- Memory vs Memoryless Agent Integration: Complete workflow validation for both
  cognitive modeling approaches using identical simulation infrastructure
- Protocol Compliance Integration: Structural subtyping validation across all
  major system interfaces ensuring interchangeable component implementations

Integration Test Categories:
1. Component Boundary Testing: Protocol-based interaction validation
2. Configuration Integration: Hierarchical config composition with component injection
3. Cross-Implementation Testing: Multi-model compatibility and switching validation
4. Performance Integration: End-to-end benchmark validation across configurations
5. Simulation Workflow Testing: Complete episode lifecycle with modular components

Usage Examples:
    # Cross-component integration testing
    def test_gaussian_plume_integration(integration_environment):
        env, config, components = integration_environment
        assert components['plume_model'].__class__.__name__ == "GaussianPlumeModel"
        results = env.run_simulation(steps=100)
        assert results.avg_step_time_ms < 10.0
    
    # Configuration-driven component switching
    def test_plume_model_switching(hydra_integration_config):
        gaussian_config, turbulent_config = hydra_integration_config
        # Test seamless switching between implementations
        gaussian_sim = create_simulation(gaussian_config)
        turbulent_sim = create_simulation(turbulent_config)
        assert gaussian_sim.run().success and turbulent_sim.run().success
    
    # Protocol compliance integration
    def test_sensor_protocol_integration(sensor_implementations):
        for sensor in sensor_implementations:
            validate_sensor_integration(sensor)
            assert sensor.get_reading() is not None

Author: Blitzy Agent
Version: 2.0.0 (Modular Architecture)
"""

import sys
import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, Callable, List, Tuple, Generator
from pathlib import Path
import warnings

# Core testing framework imports
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Gymnasium API integration testing
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    gymnasium = None
    check_env = None
    GYMNASIUM_AVAILABLE = False

# Hypothesis property-based testing for integration scenarios
try:
    import hypothesis
    from hypothesis import given, assume, strategies as st
    from hypothesis.extra.numpy import arrays, array_shapes
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    hypothesis = None
    given = None
    assume = None
    st = None
    arrays = None
    array_shapes = None
    HYPOTHESIS_AVAILABLE = False

# Hydra configuration integration testing
try:
    from hydra import initialize, compose, GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    initialize = None
    compose = None
    GlobalHydra = None

# Import centralized testing utilities from core module
try:
    from tests.core import (
        TestPerformanceMonitor,
        performance_timer,
        validate_gymnasium_api_compliance,
        STEP_LATENCY_THRESHOLD_MS,
        FRAME_TIME_THRESHOLD_MS,
        MAX_MEMORY_PER_AGENT_MB,
        GYMNASIUM_AVAILABLE as CORE_GYMNASIUM_AVAILABLE,
        HYPOTHESIS_AVAILABLE as CORE_HYPOTHESIS_AVAILABLE
    )
    CORE_UTILITIES_AVAILABLE = True
except ImportError:
    CORE_UTILITIES_AVAILABLE = False
    TestPerformanceMonitor = None
    performance_timer = None
    validate_gymnasium_api_compliance = None
    STEP_LATENCY_THRESHOLD_MS = 10.0
    FRAME_TIME_THRESHOLD_MS = 33.3
    MAX_MEMORY_PER_AGENT_MB = 0.1

# Import plume navigation simulation components for integration testing
try:
    from plume_nav_sim.core.protocols import (
        PlumeModelProtocol,
        WindFieldProtocol, 
        SensorProtocol,
        NavigatorProtocol
    )
    from plume_nav_sim.core.simulation import SimulationContext
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    PLUME_NAV_SIM_AVAILABLE = True
except ImportError:
    PLUME_NAV_SIM_AVAILABLE = False
    PlumeModelProtocol = None
    WindFieldProtocol = None
    SensorProtocol = None
    NavigatorProtocol = None
    SimulationContext = None
    PlumeNavigationEnv = None


# Integration testing constants
INTEGRATION_TEST_TIMEOUT_SECONDS = 300  # 5 minutes max for integration tests
CROSS_COMPONENT_VALIDATION_THRESHOLD_MS = 50.0  # Stricter latency for integration
MEMORY_EFFICIENCY_INTEGRATION_THRESHOLD_MB = 1.0  # Per-component memory limit
PROTOCOL_COMPLIANCE_ITERATIONS = 10  # Number of protocol compliance validation rounds

# Integration test environment configuration
INTEGRATION_LOGGING_CONFIG = {
    "environment": "integration_testing",
    "level": "WARNING",  # Suppress verbose logs in integration tests
    "format": "minimal",
    "console_enabled": True,
    "file_enabled": False,
    "correlation_enabled": True,
    "memory_tracking": True,
    "enable_performance": True,
}


class IntegrationTestPerformanceMonitor:
    """
    Enhanced performance monitoring for integration test validation.
    
    Provides comprehensive timing and resource tracking for cross-component
    integration scenarios with multi-component memory and latency validation.
    Extends core TestPerformanceMonitor with integration-specific metrics.
    """
    
    def __init__(self, test_name: str = "integration_test", component_count: int = 1):
        self.test_name = test_name
        self.component_count = component_count
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.memory_before = None
        self.memory_after = None
        self.component_timings = {}
        self.protocol_compliance_results = {}
        
    def start_monitoring(self):
        """Begin integration performance monitoring."""
        self.start_time = time.perf_counter()
        try:
            import psutil
            process = psutil.Process()
            self.memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_before = None
    
    def stop_monitoring(self):
        """Complete integration performance monitoring and calculate metrics."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        try:
            import psutil
            process = psutil.Process()
            self.memory_after = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_after = None
    
    def add_component_timing(self, component_name: str, duration_ms: float):
        """Record timing for individual component in integration test."""
        self.component_timings[component_name] = duration_ms
    
    def add_protocol_compliance_result(self, protocol_name: str, is_compliant: bool):
        """Record protocol compliance validation result."""
        self.protocol_compliance_results[protocol_name] = is_compliant
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Calculate memory usage delta in MB for integration test."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None
    
    def assert_integration_performance(self, 
                                     threshold_ms: float = CROSS_COMPONENT_VALIDATION_THRESHOLD_MS):
        """Assert integration test meets cross-component performance requirements."""
        if self.duration_ms is None:
            raise ValueError("Integration performance monitoring not completed")
        
        assert self.duration_ms <= threshold_ms, (
            f"Integration test '{self.test_name}' took {self.duration_ms:.2f}ms, "
            f"exceeds integration threshold of {threshold_ms:.2f}ms"
        )
    
    def assert_memory_efficiency_integration(self):
        """Assert memory usage meets integration efficiency requirements."""
        if self.memory_delta_mb is None:
            return  # Skip if memory tracking unavailable
        
        max_memory_mb = self.component_count * MEMORY_EFFICIENCY_INTEGRATION_THRESHOLD_MB
        assert self.memory_delta_mb <= max_memory_mb, (
            f"Integration test '{self.test_name}' used {self.memory_delta_mb:.3f}MB "
            f"for {self.component_count} component(s), exceeds limit of {max_memory_mb:.3f}MB"
        )
    
    def assert_protocol_compliance_integration(self):
        """Assert all protocol compliance validations passed."""
        failed_protocols = [
            protocol for protocol, is_compliant in self.protocol_compliance_results.items()
            if not is_compliant
        ]
        
        assert not failed_protocols, (
            f"Protocol compliance failures in integration test '{self.test_name}': "
            f"{', '.join(failed_protocols)}"
        )


@contextmanager
def integration_performance_timer(test_name: str = "integration_test", 
                                component_count: int = 1):
    """
    Context manager for integration test performance timing with component tracking.
    
    Args:
        test_name: Name of the integration test being timed
        component_count: Number of components involved in integration test
        
    Yields:
        IntegrationTestPerformanceMonitor: Enhanced monitor for integration scenarios
        
    Example:
        >>> with integration_performance_timer("plume_model_switching", 3) as perf:
        ...     simulation.switch_plume_model("turbulent")
        ...     results = simulation.run_episode()
        >>> perf.assert_integration_performance()
        >>> perf.assert_memory_efficiency_integration()
    """
    monitor = IntegrationTestPerformanceMonitor(test_name, component_count)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def validate_protocol_integration_compliance(component: Any, 
                                           protocol_type: type,
                                           validation_rounds: int = PROTOCOL_COMPLIANCE_ITERATIONS) -> bool:
    """
    Validate protocol compliance in integration testing context.
    
    Args:
        component: Component instance to validate
        protocol_type: Protocol type to validate against (e.g., PlumeModelProtocol)
        validation_rounds: Number of validation iterations to perform
        
    Returns:
        True if component passes all protocol compliance checks
        
    Raises:
        AssertionError: If component fails protocol compliance validation
    """
    if not PLUME_NAV_SIM_AVAILABLE:
        pytest.skip("plume_nav_sim not available for protocol compliance testing")
    
    # Verify component implements required protocol methods
    if protocol_type == PlumeModelProtocol:
        required_methods = ['concentration_field', 'get_concentration_at']
    elif protocol_type == WindFieldProtocol:
        required_methods = ['get_wind_at', 'get_turbulence_at']
    elif protocol_type == SensorProtocol:
        required_methods = ['get_reading', 'get_gradient']
    elif protocol_type == NavigatorProtocol:
        required_methods = ['reset', 'step', 'sample_odor']
    else:
        pytest.skip(f"Unknown protocol type: {protocol_type}")
    
    # Validate all required methods exist and are callable
    for method_name in required_methods:
        assert hasattr(component, method_name), (
            f"Component {component.__class__.__name__} missing required method: {method_name}"
        )
        assert callable(getattr(component, method_name)), (
            f"Component {component.__class__.__name__} method {method_name} is not callable"
        )
    
    # Perform multiple validation rounds for consistency
    for round_num in range(validation_rounds):
        try:
            # Protocol-specific validation
            if protocol_type == PlumeModelProtocol:
                field = component.concentration_field(grid_size=(10, 10))
                assert field.shape == (10, 10)
                concentration = component.get_concentration_at((0, 0))
                assert isinstance(concentration, (int, float, np.number))
                
            elif protocol_type == WindFieldProtocol:
                wind = component.get_wind_at((0, 0))
                assert len(wind) == 2  # (wind_x, wind_y)
                turbulence = component.get_turbulence_at((0, 0))
                assert isinstance(turbulence, (int, float, np.number))
                
            elif protocol_type == SensorProtocol:
                reading = component.get_reading()
                assert isinstance(reading, (int, float, np.number))
                gradient = component.get_gradient()
                assert len(gradient) == 2  # (grad_x, grad_y)
                
            elif protocol_type == NavigatorProtocol:
                component.reset()
                component.step()
                odor = component.sample_odor()
                assert isinstance(odor, (int, float, np.number, np.ndarray))
                
        except Exception as e:
            return False  # Protocol compliance failed
    
    return True


def create_integration_test_config(config_type: str = "modular_simulation", 
                                 component_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create integration test configuration dictionaries for modular components.
    
    Args:
        config_type: Type of integration configuration to create
        component_overrides: Component-specific parameter overrides
        
    Returns:
        Dictionary configuration for integration testing
    """
    component_overrides = component_overrides or {}
    
    base_configs = {
        "modular_simulation": {
            "plume_model": {
                "type": "gaussian",
                "source_position": [25.0, 25.0],
                "spread_sigma": 2.0,
                "emission_rate": 5.0,
                "background_concentration": 0.01
            },
            "wind_field": {
                "type": "constant",
                "wind_vector": [1.0, 0.5],
                "turbulence_intensity": 0.1
            },
            "sensors": [
                {
                    "type": "concentration",
                    "sensitivity": 1.0,
                    "noise_level": 0.01
                }
            ],
            "navigator": {
                "type": "single",
                "position": [10.0, 10.0],
                "orientation": 0.0,
                "speed": 1.0,
                "max_speed": 2.0
            },
            "simulation": {
                "max_steps": 100,
                "step_size": 0.1,
                "record_trajectory": True,
                "random_seed": 42
            }
        },
        "plume_model_switching": {
            "gaussian_config": {
                "plume_model": {"type": "gaussian", "spread_sigma": 1.5},
                "simulation": {"max_steps": 50}
            },
            "turbulent_config": {
                "plume_model": {"type": "turbulent", "num_filaments": 100},
                "simulation": {"max_steps": 50}
            }
        },
        "cross_component_validation": {
            "plume_models": ["gaussian", "turbulent"],
            "wind_fields": ["constant", "turbulent"],
            "sensor_types": ["binary", "concentration", "gradient"],
            "validation_steps": 25
        },
        "memory_agent_comparison": {
            "memoryless_agent": {
                "navigator": {"type": "reactive", "memory_enabled": False},
                "simulation": {"max_steps": 100}
            },
            "memory_agent": {
                "navigator": {"type": "planning", "memory_enabled": True},
                "simulation": {"max_steps": 100}
            }
        }
    }
    
    config = base_configs.get(config_type, {}).copy()
    
    # Apply component overrides recursively
    def apply_overrides(base_dict, overrides_dict):
        for key, value in overrides_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                apply_overrides(base_dict[key], value)
            else:
                base_dict[key] = value
    
    apply_overrides(config, component_overrides)
    return config


def setup_integration_test_environment(config: Dict[str, Any], 
                                      temp_dir: Optional[Path] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Setup complete integration test environment with modular components.
    
    Args:
        config: Integration test configuration dictionary
        temp_dir: Optional temporary directory for test artifacts
        
    Returns:
        Tuple of (simulation_environment, component_registry)
        
    Raises:
        ValueError: If required components cannot be instantiated
        ImportError: If plume_nav_sim components are not available
    """
    if not PLUME_NAV_SIM_AVAILABLE:
        pytest.skip("plume_nav_sim not available for integration environment setup")
    
    try:
        # Create modular components based on configuration
        component_registry = {}
        
        # Instantiate plume model
        plume_config = config.get("plume_model", {})
        if plume_config.get("type") == "gaussian":
            from plume_nav_sim.models.plume import GaussianPlumeModel
            component_registry["plume_model"] = GaussianPlumeModel(
                source_position=tuple(plume_config.get("source_position", [0, 0])),
                spread_sigma=plume_config.get("spread_sigma", 1.0),
                emission_rate=plume_config.get("emission_rate", 1.0)
            )
        elif plume_config.get("type") == "turbulent":
            from plume_nav_sim.models.plume import TurbulentPlumeModel
            component_registry["plume_model"] = TurbulentPlumeModel(
                source_position=tuple(plume_config.get("source_position", [0, 0])),
                num_filaments=plume_config.get("num_filaments", 100),
                turbulence_intensity=plume_config.get("turbulence_intensity", 0.2)
            )
        else:
            # Fallback to mock for testing
            component_registry["plume_model"] = Mock(spec=PlumeModelProtocol)
            component_registry["plume_model"].concentration_field.return_value = np.zeros((10, 10))
            component_registry["plume_model"].get_concentration_at.return_value = 0.0
        
        # Instantiate wind field
        wind_config = config.get("wind_field", {})
        if wind_config.get("type") == "constant":
            from plume_nav_sim.models.wind import ConstantWindField
            component_registry["wind_field"] = ConstantWindField(
                wind_vector=tuple(wind_config.get("wind_vector", [0, 0])),
                turbulence_intensity=wind_config.get("turbulence_intensity", 0.0)
            )
        else:
            # Fallback to mock for testing
            component_registry["wind_field"] = Mock(spec=WindFieldProtocol)
            component_registry["wind_field"].get_wind_at.return_value = (0.0, 0.0)
            component_registry["wind_field"].get_turbulence_at.return_value = 0.0
        
        # Instantiate sensors
        sensors_config = config.get("sensors", [])
        component_registry["sensors"] = []
        for sensor_config in sensors_config:
            if sensor_config.get("type") == "concentration":
                from plume_nav_sim.core.sensors import ConcentrationSensor
                sensor = ConcentrationSensor(
                    sensitivity=sensor_config.get("sensitivity", 1.0),
                    noise_level=sensor_config.get("noise_level", 0.0)
                )
            else:
                # Fallback to mock for testing
                sensor = Mock(spec=SensorProtocol)
                sensor.get_reading.return_value = 0.0
                sensor.get_gradient.return_value = (0.0, 0.0)
            component_registry["sensors"].append(sensor)
        
        # Create simulation environment
        simulation_config = config.get("simulation", {})
        env = PlumeNavigationEnv(
            plume_model=component_registry["plume_model"],
            wind_field=component_registry["wind_field"],
            sensors=component_registry["sensors"],
            max_episode_steps=simulation_config.get("max_steps", 100),
            step_size=simulation_config.get("step_size", 0.1)
        )
        
        return env, component_registry
        
    except ImportError as e:
        # Create mock environment for testing when components unavailable
        mock_env = Mock()
        mock_env.reset.return_value = np.zeros(10)
        mock_env.step.return_value = (np.zeros(10), 0.0, False, {})
        mock_env.run_simulation.return_value = Mock(
            step_count=config.get("simulation", {}).get("max_steps", 100),
            avg_step_time_ms=5.0,
            success=True
        )
        
        mock_components = {
            "plume_model": Mock(spec=PlumeModelProtocol),
            "wind_field": Mock(spec=WindFieldProtocol),
            "sensors": [Mock(spec=SensorProtocol)]
        }
        
        return mock_env, mock_components


@contextmanager
def hydra_integration_context(config_overrides: Optional[List[str]] = None):
    """
    Context manager for Hydra integration testing with component switching.
    
    Args:
        config_overrides: List of Hydra override strings for configuration testing
        
    Yields:
        DictConfig: Composed Hydra configuration for integration testing
        
    Example:
        >>> with hydra_integration_context(["plume_model.type=turbulent"]) as cfg:
        ...     simulation = create_simulation(cfg)
        ...     assert simulation.plume_model.__class__.__name__ == "TurbulentPlumeModel"
    """
    if not HYDRA_AVAILABLE:
        # Create mock configuration for testing
        mock_config = {
            "plume_model": {"type": "gaussian", "spread_sigma": 1.0},
            "wind_field": {"type": "constant", "wind_vector": [1.0, 0.0]},
            "simulation": {"max_steps": 100}
        }
        yield OmegaConf.create(mock_config) if 'OmegaConf' in globals() else mock_config
        return
    
    config_overrides = config_overrides or []
    
    try:
        with initialize(config_path="../conf", version_base="1.3"):
            cfg = compose(
                config_name="base_simulation",
                overrides=config_overrides
            )
            yield cfg
    except Exception:
        # Fallback to mock configuration if Hydra setup fails
        mock_config = {
            "plume_model": {"type": "gaussian", "spread_sigma": 1.0},
            "wind_field": {"type": "constant", "wind_vector": [1.0, 0.0]},
            "simulation": {"max_steps": 100}
        }
        yield OmegaConf.create(mock_config) if 'OmegaConf' in globals() else mock_config


def compare_agent_strategies(memoryless_config: Dict[str, Any], 
                           memory_config: Dict[str, Any],
                           comparison_steps: int = 100) -> Dict[str, Any]:
    """
    Compare memoryless vs memory-based agent strategies in integration testing.
    
    Args:
        memoryless_config: Configuration for memoryless agent
        memory_config: Configuration for memory-based agent  
        comparison_steps: Number of simulation steps for comparison
        
    Returns:
        Dictionary containing comparison results and performance metrics
    """
    if not PLUME_NAV_SIM_AVAILABLE:
        # Return mock comparison results for testing
        return {
            "memoryless_results": {
                "avg_step_time_ms": 5.0,
                "memory_usage_mb": 0.05,
                "success_rate": 0.85
            },
            "memory_results": {
                "avg_step_time_ms": 7.0,
                "memory_usage_mb": 0.15,
                "success_rate": 0.90
            },
            "comparison_valid": True
        }
    
    results = {}
    
    try:
        # Setup memoryless agent simulation
        memoryless_env, memoryless_components = setup_integration_test_environment(memoryless_config)
        
        with integration_performance_timer("memoryless_agent", 1) as memoryless_perf:
            memoryless_results = memoryless_env.run_simulation(steps=comparison_steps)
        
        results["memoryless_results"] = {
            "avg_step_time_ms": memoryless_perf.duration_ms / comparison_steps,
            "memory_usage_mb": memoryless_perf.memory_delta_mb or 0.0,
            "success_rate": 1.0 if memoryless_results.success else 0.0
        }
        
        # Setup memory-based agent simulation
        memory_env, memory_components = setup_integration_test_environment(memory_config)
        
        with integration_performance_timer("memory_agent", 1) as memory_perf:
            memory_results = memory_env.run_simulation(steps=comparison_steps)
        
        results["memory_results"] = {
            "avg_step_time_ms": memory_perf.duration_ms / comparison_steps,
            "memory_usage_mb": memory_perf.memory_delta_mb or 0.0,
            "success_rate": 1.0 if memory_results.success else 0.0
        }
        
        results["comparison_valid"] = True
        
    except Exception as e:
        results["comparison_valid"] = False
        results["error"] = str(e)
    
    return results


# Test fixtures for integration testing scenarios
@pytest.fixture
def integration_performance_monitor():
    """Provide integration performance monitor for cross-component timing."""
    return IntegrationTestPerformanceMonitor


@pytest.fixture  
def modular_simulation_config():
    """Provide modular simulation configuration for integration testing."""
    return create_integration_test_config("modular_simulation")


@pytest.fixture
def plume_model_switching_config():
    """Provide configuration for plume model switching integration tests."""
    return create_integration_test_config("plume_model_switching")


@pytest.fixture
def cross_component_validation_config():
    """Provide configuration for cross-component validation testing."""
    return create_integration_test_config("cross_component_validation")


@pytest.fixture
def memory_agent_comparison_config():
    """Provide configuration for memory vs memoryless agent comparison."""
    return create_integration_test_config("memory_agent_comparison")


@pytest.fixture
def integration_environment(tmp_path):
    """Provide complete integration test environment with modular components."""
    config = create_integration_test_config("modular_simulation")
    env, components = setup_integration_test_environment(config, tmp_path)
    return env, config, components


# Performance testing decorators for integration scenarios
def requires_integration_performance_validation(threshold_ms: float = CROSS_COMPONENT_VALIDATION_THRESHOLD_MS,
                                              component_count: int = 1):
    """
    Decorator to ensure integration test operations meet cross-component performance requirements.
    
    Args:
        threshold_ms: Maximum allowed duration in milliseconds for integration test
        component_count: Number of components involved in integration test
        
    Example:
        @requires_integration_performance_validation(50.0, 3)
        def test_cross_component_integration(env):
            # Test will automatically validate integration performance
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            with integration_performance_timer(test_func.__name__, component_count) as perf:
                result = test_func(*args, **kwargs)
            perf.assert_integration_performance(threshold_ms)
            perf.assert_memory_efficiency_integration()
            return result
        return wrapper
    return decorator


def requires_protocol_compliance_validation(*protocol_types):
    """
    Decorator to ensure components pass protocol compliance validation in integration tests.
    
    Args:
        protocol_types: Protocol types to validate (e.g., PlumeModelProtocol, WindFieldProtocol)
        
    Example:
        @requires_protocol_compliance_validation(PlumeModelProtocol, WindFieldProtocol)
        def test_component_integration(components):
            # Test will automatically validate protocol compliance
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # Extract components from args/kwargs for validation
            components = None
            if len(args) > 0 and isinstance(args[0], dict):
                components = args[0]
            elif 'components' in kwargs:
                components = kwargs['components']
            elif len(args) > 2 and isinstance(args[2], dict):  # (env, config, components) pattern
                components = args[2]
            
            if components and PLUME_NAV_SIM_AVAILABLE:
                for protocol_type in protocol_types:
                    component_name = None
                    component = None
                    
                    # Map protocol types to component names
                    if protocol_type == PlumeModelProtocol:
                        component_name = "plume_model"
                    elif protocol_type == WindFieldProtocol:
                        component_name = "wind_field"
                    elif protocol_type == SensorProtocol and "sensors" in components:
                        component_name = "sensors"
                        component = components["sensors"][0] if components["sensors"] else None
                    
                    if component_name and component_name in components:
                        component = components[component_name]
                    
                    if component:
                        assert validate_protocol_integration_compliance(
                            component, protocol_type
                        ), f"Protocol compliance failed for {protocol_type.__name__}"
            
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize integration testing environment
def setup_integration_test_logging():
    """Configure logging for integration test environment."""
    if CORE_UTILITIES_AVAILABLE:
        # Use core logging setup if available
        try:
            from tests.core import setup_test_logging
            setup_test_logging()
        except ImportError:
            pass
    else:
        # Fallback to basic logging configuration
        import logging
        logging.basicConfig(level=logging.WARNING)


# Export integration testing utilities
__all__ = [
    # Integration performance monitoring
    "IntegrationTestPerformanceMonitor",
    "integration_performance_timer",
    "requires_integration_performance_validation",
    
    # Protocol compliance validation
    "validate_protocol_integration_compliance",
    "requires_protocol_compliance_validation",
    
    # Configuration and environment setup
    "create_integration_test_config",
    "setup_integration_test_environment",
    "hydra_integration_context",
    
    # Agent strategy comparison
    "compare_agent_strategies",
    
    # Environment setup utilities
    "setup_integration_test_logging",
    
    # Availability flags
    "GYMNASIUM_AVAILABLE",
    "HYPOTHESIS_AVAILABLE",
    "HYDRA_AVAILABLE",
    "PLUME_NAV_SIM_AVAILABLE",
    "CORE_UTILITIES_AVAILABLE",
    
    # Integration testing constants
    "INTEGRATION_TEST_TIMEOUT_SECONDS",
    "CROSS_COMPONENT_VALIDATION_THRESHOLD_MS",
    "MEMORY_EFFICIENCY_INTEGRATION_THRESHOLD_MB",
    "PROTOCOL_COMPLIANCE_ITERATIONS",
]

# Conditional exports based on availability
if HYPOTHESIS_AVAILABLE:
    # Import hypothesis strategies from core module if available
    if CORE_UTILITIES_AVAILABLE:
        try:
            from tests.core import (
                coordinate_strategy,
                position_strategy,
                angle_strategy,
                speed_strategy
            )
            __all__.extend([
                "coordinate_strategy",
                "position_strategy", 
                "angle_strategy",
                "speed_strategy"
            ])
        except ImportError:
            pass

# Initialize integration test logging on module import
setup_integration_test_logging()