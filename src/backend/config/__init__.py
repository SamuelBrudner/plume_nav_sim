"""
Configuration module initialization file providing centralized access to all configuration 
management components including default configurations, environment presets, rendering 
configurations, and test configurations. Serves as the primary interface for the 
plume_nav_sim configuration system with unified imports, factory functions, and registry 
management for all configuration categories.

This module handles the graceful integration of multiple configuration subsystems:
- Default configuration classes and factory functions (with fallback handling)
- Environment configuration presets and registry management
- Rendering configuration presets with dual-mode optimization  
- Test configuration factories with comprehensive validation
- Cross-component configuration validation and compatibility checking
- Centralized error handling and configuration management utilities

The module is designed to work seamlessly whether default_config.py exists or not,
providing robust fallback mechanisms and comprehensive configuration management.
"""

import warnings  # >=3.10 - Configuration warning management, deprecation alerts, and fallback notifications
import logging   # >=3.10 - Structured logging for configuration operations, error reporting, and system monitoring
from typing import Optional, Dict, Any, List, Union, Tuple  # >=3.10 - Type hints for configuration functions and registry management

# Configure logging for configuration module
logger = logging.getLogger(__name__)

# Global configuration for warning management and system initialization
CONFIGURATION_WARNINGS_ENABLED = True
DEFAULT_CONFIG_FALLBACK_ENABLED = True
STRICT_VALIDATION_MODE = False

def _configure_configuration_warnings():
    """
    Configure warning management for configuration system including fallback notifications
    and validation alerts with appropriate filtering and formatting.
    """
    if not CONFIGURATION_WARNINGS_ENABLED:
        warnings.filterwarnings('ignore', category=UserWarning, module='config')
    
    # Configure specific warning categories for configuration management
    warnings.filterwarnings('default', category=ImportWarning, module='config')
    warnings.filterwarnings('default', category=RuntimeWarning, module='config')

# Initialize warning configuration
_configure_configuration_warnings()

# Attempt to import from default_config.py with comprehensive fallback handling
try:
    # Primary imports from default configuration module
    from .default_config import (
        EnvironmentConfig, PlumeConfig, RenderConfig, PerformanceConfig, 
        CompleteConfig, ConfigurationError,
        get_default_environment_config, get_default_plume_config, 
        get_default_render_config, get_default_performance_config, 
        get_complete_default_config, validate_configuration_compatibility,
        merge_configurations, create_config_from_dict
    )
    
    logger.info("Successfully imported default configuration components")
    DEFAULT_CONFIG_AVAILABLE = True
    
except ImportError as e:
    # Handle missing default_config.py with graceful fallback to placeholder implementations
    logger.warning(f"Default configuration module not available: {e}")
    logger.info("Activating fallback configuration implementations from specialized modules")
    
    if DEFAULT_CONFIG_FALLBACK_ENABLED:
        DEFAULT_CONFIG_AVAILABLE = False
        
        # Import fallback implementations from environment_configs.py which provides placeholders
        try:
            from .environment_configs import CompleteConfig as FallbackCompleteConfig
            from .test_configs import CompleteConfig as TestCompleteConfig
            
            # Use the most comprehensive fallback available
            CompleteConfig = FallbackCompleteConfig
            
            # Create minimal fallback implementations for core configuration classes
            class EnvironmentConfig:
                """Fallback environment configuration placeholder with basic validation."""
                def __init__(self, **kwargs):
                    self.config_data = kwargs
                
                def validate(self) -> Tuple[bool, List[str]]:
                    return True, []
                
                def get_registration_kwargs(self) -> Dict[str, Any]:
                    return self.config_data
            
            class PlumeConfig:
                """Fallback plume configuration placeholder with Gaussian parameter management."""
                def __init__(self, **kwargs):
                    self.config_data = kwargs
                
                def validate(self) -> Tuple[bool, List[str]]:
                    return True, []
                
                def get_gaussian_parameters(self) -> Dict[str, Any]:
                    return self.config_data
            
            class RenderConfig:
                """Fallback render configuration placeholder with dual-mode support."""
                def __init__(self, **kwargs):
                    self.config_data = kwargs
                
                def validate(self) -> Tuple[bool, List[str]]:
                    return True, []
                
                def configure_backend(self) -> bool:
                    return True
            
            class PerformanceConfig:
                """Fallback performance configuration placeholder with timing management."""
                def __init__(self, **kwargs):
                    self.config_data = kwargs
                
                def validate(self) -> Tuple[bool, List[str]]:
                    return True, []
                
                def get_target_for_operation(self, operation: str) -> float:
                    return 5.0  # Default 5ms target
            
            class ConfigurationError(Exception):
                """Configuration validation exception with detailed error reporting."""
                def __init__(self, message: str, config_details: Dict[str, Any] = None):
                    super().__init__(message)
                    self.config_details = config_details or {}
                
                def get_formatted_message(self) -> str:
                    return f"{str(self)} - Details: {self.config_details}"
            
            # Create fallback factory functions with minimal default implementations
            def get_default_environment_config(**kwargs) -> EnvironmentConfig:
                """Fallback factory for default environment configuration."""
                return EnvironmentConfig(**kwargs)
            
            def get_default_plume_config(**kwargs) -> PlumeConfig:
                """Fallback factory for default plume configuration."""
                return PlumeConfig(**kwargs)
            
            def get_default_render_config(**kwargs) -> RenderConfig:
                """Fallback factory for default render configuration."""
                return RenderConfig(**kwargs)
            
            def get_default_performance_config(**kwargs) -> PerformanceConfig:
                """Fallback factory for default performance configuration."""
                return PerformanceConfig(**kwargs)
            
            def get_complete_default_config(**kwargs) -> CompleteConfig:
                """Fallback factory for complete default configuration."""
                return CompleteConfig(**kwargs)
            
            def validate_configuration_compatibility(*configs) -> Tuple[bool, List[str]]:
                """Fallback cross-component configuration validation."""
                return True, []
            
            def merge_configurations(*configs, **merge_options) -> Dict[str, Any]:
                """Fallback configuration merging with conflict resolution."""
                merged = {}
                for config in configs:
                    if hasattr(config, 'config_data'):
                        merged.update(config.config_data)
                    elif isinstance(config, dict):
                        merged.update(config)
                return merged
            
            def create_config_from_dict(config_dict: Dict[str, Any], config_type: str = "complete") -> Any:
                """Fallback factory for creating configurations from dictionaries."""
                if config_type == "environment":
                    return EnvironmentConfig(**config_dict)
                elif config_type == "plume":
                    return PlumeConfig(**config_dict)
                elif config_type == "render":
                    return RenderConfig(**config_dict)
                elif config_type == "performance":
                    return PerformanceConfig(**config_dict)
                else:
                    return CompleteConfig(**config_dict)
            
            logger.info("Successfully activated fallback configuration implementations")
            
        except ImportError as fallback_error:
            logger.error(f"Failed to import fallback configurations: {fallback_error}")
            raise ImportError(
                "Configuration system initialization failed - neither default_config.py "
                f"nor fallback implementations are available: {fallback_error}"
            )
    else:
        logger.error("Default configuration fallback disabled, configuration system unavailable")
        raise ImportError(f"Default configuration module required but not available: {e}")

# Import environment configuration components with registry management
try:
    from .environment_configs import (
        create_preset_config, create_research_scenario, create_benchmark_config, 
        create_custom_scenario, ConfigurationRegistry, ENVIRONMENT_REGISTRY,
        PresetMetadata, get_available_presets, validate_preset_name
    )
    
    logger.info("Successfully imported environment configuration components")
    ENVIRONMENT_CONFIG_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Failed to import environment configuration components: {e}")
    ENVIRONMENT_CONFIG_AVAILABLE = False
    
    # Create minimal fallback implementations
    def create_preset_config(preset_name: str, **overrides) -> Any:
        """Fallback preset configuration factory."""
        return get_complete_default_config(**overrides)
    
    def create_research_scenario(scenario_type: str, **parameters) -> Any:
        """Fallback research scenario factory."""
        return get_complete_default_config(**parameters)
    
    def create_benchmark_config(benchmark_type: str, **parameters) -> Any:
        """Fallback benchmark configuration factory."""
        return get_complete_default_config(**parameters)
    
    def create_custom_scenario(custom_parameters: Dict[str, Any]) -> Any:
        """Fallback custom scenario factory."""
        return get_complete_default_config(**custom_parameters)
    
    class ConfigurationRegistry:
        """Fallback configuration registry with minimal functionality."""
        def register_preset(self, name: str, config: Any, metadata: Dict[str, Any] = None):
            pass
        
        def get_preset(self, name: str) -> Any:
            return get_complete_default_config()
        
        def list_presets(self) -> List[str]:
            return []
    
    ENVIRONMENT_REGISTRY = ConfigurationRegistry()
    
    class PresetMetadata:
        """Fallback preset metadata class."""
        def __init__(self, name: str, description: str = "", **kwargs):
            self.name = name
            self.description = description
        
        def to_dict(self) -> Dict[str, Any]:
            return {"name": self.name, "description": self.description}
    
    def get_available_presets(category: str = None) -> List[str]:
        """Fallback function for discovering available presets."""
        return []
    
    def validate_preset_name(preset_name: str) -> Tuple[bool, List[str]]:
        """Fallback preset name validation."""
        return True, []

# Import rendering configuration components with preset management
try:
    from .render_configs import (
        RenderConfigPreset, RenderPresetCategory, RenderPresetRegistry,
        create_rgb_preset, create_matplotlib_preset, create_research_preset,
        create_accessibility_preset, create_performance_preset, RENDER_REGISTRY
    )
    
    logger.info("Successfully imported rendering configuration components")
    RENDER_CONFIG_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Failed to import rendering configuration components: {e}")
    RENDER_CONFIG_AVAILABLE = False
    
    # Create minimal fallback implementations for rendering configurations
    class RenderConfigPreset:
        """Fallback render configuration preset."""
        def __init__(self, **kwargs):
            self.config_data = kwargs
        
        def validate(self) -> Tuple[bool, List[str]]:
            return True, []
        
        def optimize_for_system(self, system_info: Dict[str, Any]) -> 'RenderConfigPreset':
            return self
    
    class RenderPresetCategory:
        """Fallback render preset category enumeration."""
        RGB_ARRAY = "rgb_array"
        MATPLOTLIB = "matplotlib"
        RESEARCH = "research"
        ACCESSIBILITY = "accessibility"
        PERFORMANCE = "performance"
    
    class RenderPresetRegistry:
        """Fallback render preset registry."""
        def get_preset(self, name: str) -> RenderConfigPreset:
            return RenderConfigPreset()
        
        def list_presets_by_category(self, category: str) -> List[str]:
            return []
    
    RENDER_REGISTRY = RenderPresetRegistry()
    
    def create_rgb_preset(**kwargs) -> RenderConfigPreset:
        """Fallback RGB preset factory."""
        return RenderConfigPreset(**kwargs)
    
    def create_matplotlib_preset(**kwargs) -> RenderConfigPreset:
        """Fallback matplotlib preset factory."""
        return RenderConfigPreset(**kwargs)
    
    def create_research_preset(**kwargs) -> RenderConfigPreset:
        """Fallback research preset factory."""
        return RenderConfigPreset(**kwargs)
    
    def create_accessibility_preset(**kwargs) -> RenderConfigPreset:
        """Fallback accessibility preset factory."""
        return RenderConfigPreset(**kwargs)
    
    def create_performance_preset(**kwargs) -> RenderConfigPreset:
        """Fallback performance preset factory."""
        return RenderConfigPreset(**kwargs)

# Import test configuration components with factory management
try:
    from .test_configs import (
        create_unit_test_config, create_integration_test_config, 
        create_performance_test_config, create_reproducibility_test_config,
        TestConfigFactory, validate_test_configuration, REPRODUCIBILITY_SEEDS
    )
    
    logger.info("Successfully imported test configuration components")
    TEST_CONFIG_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Failed to import test configuration components: {e}")
    TEST_CONFIG_AVAILABLE = False
    
    # Create minimal fallback implementations for test configurations
    def create_unit_test_config(**kwargs) -> Any:
        """Fallback unit test configuration factory."""
        return get_complete_default_config(**kwargs)
    
    def create_integration_test_config(**kwargs) -> Any:
        """Fallback integration test configuration factory."""
        return get_complete_default_config(**kwargs)
    
    def create_performance_test_config(**kwargs) -> Any:
        """Fallback performance test configuration factory."""
        return get_complete_default_config(**kwargs)
    
    def create_reproducibility_test_config(seed: int = None, **kwargs) -> Any:
        """Fallback reproducibility test configuration factory."""
        if seed is not None:
            kwargs['seed'] = seed
        return get_complete_default_config(**kwargs)
    
    class TestConfigFactory:
        """Fallback test configuration factory class."""
        @staticmethod
        def create_for_test_type(test_type: str, **parameters) -> Any:
            """Create test configuration for specific test type."""
            return get_complete_default_config(**parameters)
        
        @staticmethod
        def create_suite_configs(test_suite: List[str], **common_parameters) -> List[Any]:
            """Create multiple test configurations for test suite."""
            return [get_complete_default_config(**common_parameters) for _ in test_suite]
    
    def validate_test_configuration(test_config: Any, strict_mode: bool = False) -> Tuple[bool, List[str]]:
        """Fallback test configuration validation."""
        return True, []
    
    REPRODUCIBILITY_SEEDS = [12345, 67890, 54321, 98765, 11111]

# Configuration system status and availability reporting
def get_configuration_system_status() -> Dict[str, Any]:
    """
    Returns comprehensive status of configuration system components including 
    availability, fallback status, and system health for diagnostic purposes.
    """
    return {
        'system_status': {
            'default_config_available': DEFAULT_CONFIG_AVAILABLE,
            'environment_config_available': ENVIRONMENT_CONFIG_AVAILABLE,
            'render_config_available': RENDER_CONFIG_AVAILABLE,
            'test_config_available': TEST_CONFIG_AVAILABLE,
            'fallback_enabled': DEFAULT_CONFIG_FALLBACK_ENABLED,
            'warnings_enabled': CONFIGURATION_WARNINGS_ENABLED
        },
        'component_status': {
            'core_classes': DEFAULT_CONFIG_AVAILABLE,
            'factory_functions': True,
            'environment_presets': ENVIRONMENT_CONFIG_AVAILABLE,
            'render_presets': RENDER_CONFIG_AVAILABLE,
            'test_factories': TEST_CONFIG_AVAILABLE,
            'registries': ENVIRONMENT_CONFIG_AVAILABLE and RENDER_CONFIG_AVAILABLE
        },
        'recommendations': _generate_system_recommendations()
    }

def _generate_system_recommendations() -> List[str]:
    """Generate system recommendations based on component availability."""
    recommendations = []
    
    if not DEFAULT_CONFIG_AVAILABLE:
        recommendations.append(
            "Consider creating default_config.py for enhanced configuration capabilities"
        )
    
    if not ENVIRONMENT_CONFIG_AVAILABLE:
        recommendations.append(
            "Environment configuration presets unavailable - limited preset functionality"
        )
    
    if not RENDER_CONFIG_AVAILABLE:
        recommendations.append(
            "Rendering configuration presets unavailable - using fallback implementations"
        )
    
    if not TEST_CONFIG_AVAILABLE:
        recommendations.append(
            "Test configuration factories unavailable - limited testing configuration support"
        )
    
    if DEFAULT_CONFIG_AVAILABLE and ENVIRONMENT_CONFIG_AVAILABLE and RENDER_CONFIG_AVAILABLE and TEST_CONFIG_AVAILABLE:
        recommendations.append(
            "Full configuration system available - all components operational"
        )
    
    return recommendations

# Advanced configuration management utilities
def create_integrated_config(
    environment_preset: Optional[str] = None,
    render_preset: Optional[str] = None,
    test_type: Optional[str] = None,
    custom_overrides: Dict[str, Any] = None,
    validate_integration: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Creates integrated configuration combining environment presets, rendering configurations, 
    and test settings with cross-component validation and compatibility checking.
    """
    integration_report = {
        'components_used': {},
        'validation_results': {},
        'compatibility_status': {},
        'warnings': []
    }
    
    # Initialize configuration components
    config_components = {}
    
    # Create environment configuration
    if environment_preset and ENVIRONMENT_CONFIG_AVAILABLE:
        try:
            env_config = create_preset_config(environment_preset)
            config_components['environment'] = env_config
            integration_report['components_used']['environment'] = environment_preset
        except Exception as e:
            integration_report['warnings'].append(f"Environment preset '{environment_preset}' failed: {e}")
            config_components['environment'] = get_default_environment_config()
            integration_report['components_used']['environment'] = 'default_fallback'
    else:
        config_components['environment'] = get_default_environment_config()
        integration_report['components_used']['environment'] = 'default'
    
    # Create render configuration
    if render_preset and RENDER_CONFIG_AVAILABLE:
        try:
            # Determine render preset type and create appropriate configuration
            if 'rgb' in render_preset.lower():
                render_config = create_rgb_preset()
            elif 'matplotlib' in render_preset.lower():
                render_config = create_matplotlib_preset()
            elif 'research' in render_preset.lower():
                render_config = create_research_preset()
            elif 'accessibility' in render_preset.lower():
                render_config = create_accessibility_preset()
            elif 'performance' in render_preset.lower():
                render_config = create_performance_preset()
            else:
                render_config = RENDER_REGISTRY.get_preset(render_preset)
            
            config_components['render'] = render_config
            integration_report['components_used']['render'] = render_preset
        except Exception as e:
            integration_report['warnings'].append(f"Render preset '{render_preset}' failed: {e}")
            config_components['render'] = get_default_render_config()
            integration_report['components_used']['render'] = 'default_fallback'
    else:
        config_components['render'] = get_default_render_config()
        integration_report['components_used']['render'] = 'default'
    
    # Create test configuration if specified
    if test_type and TEST_CONFIG_AVAILABLE:
        try:
            if test_type == 'unit':
                test_config = create_unit_test_config()
            elif test_type == 'integration':
                test_config = create_integration_test_config()
            elif test_type == 'performance':
                test_config = create_performance_test_config()
            elif test_type == 'reproducibility':
                test_config = create_reproducibility_test_config()
            else:
                test_config = TestConfigFactory.create_for_test_type(test_type)
            
            config_components['test'] = test_config
            integration_report['components_used']['test'] = test_type
        except Exception as e:
            integration_report['warnings'].append(f"Test type '{test_type}' failed: {e}")
    
    # Apply custom overrides
    if custom_overrides:
        merged_overrides = merge_configurations(*config_components.values(), **custom_overrides)
        integration_report['components_used']['custom_overrides'] = list(custom_overrides.keys())
    else:
        merged_overrides = merge_configurations(*config_components.values())
    
    # Create integrated configuration
    try:
        integrated_config = get_complete_default_config(**merged_overrides)
        integration_report['creation_status'] = 'success'
    except Exception as e:
        integration_report['creation_status'] = 'failed'
        integration_report['warnings'].append(f"Integrated configuration creation failed: {e}")
        # Return basic default configuration
        integrated_config = get_complete_default_config()
    
    # Perform cross-component validation if requested
    if validate_integration and DEFAULT_CONFIG_AVAILABLE:
        try:
            validation_passed, validation_errors = validate_configuration_compatibility(
                *config_components.values()
            )
            integration_report['validation_results'] = {
                'passed': validation_passed,
                'errors': validation_errors
            }
            
            if not validation_passed:
                integration_report['warnings'].extend([
                    f"Configuration validation failed: {error}" for error in validation_errors
                ])
        except Exception as e:
            integration_report['validation_results'] = {
                'passed': False,
                'errors': [f"Validation process failed: {e}"]
            }
    
    return integrated_config, integration_report

def validate_configuration_system(
    check_imports: bool = True,
    check_registries: bool = True,
    check_factories: bool = True,
    detailed_report: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation of configuration system integrity including import status, 
    registry availability, factory functionality, and cross-component compatibility.
    """
    validation_report = {
        'overall_status': True,
        'component_checks': {},
        'functionality_tests': {},
        'recommendations': [],
        'system_health_score': 0.0
    }
    
    health_points = 0
    max_points = 0
    
    # Check import status
    if check_imports:
        max_points += 4
        import_status = {
            'default_config': DEFAULT_CONFIG_AVAILABLE,
            'environment_config': ENVIRONMENT_CONFIG_AVAILABLE,
            'render_config': RENDER_CONFIG_AVAILABLE,
            'test_config': TEST_CONFIG_AVAILABLE
        }
        
        health_points += sum(import_status.values())
        validation_report['component_checks']['imports'] = import_status
        
        if not all(import_status.values()):
            validation_report['overall_status'] = False
    
    # Check registry functionality
    if check_registries:
        max_points += 2
        registry_status = {'environment_registry': False, 'render_registry': False}
        
        if ENVIRONMENT_CONFIG_AVAILABLE:
            try:
                presets = ENVIRONMENT_REGISTRY.list_presets()
                registry_status['environment_registry'] = True
                health_points += 1
            except Exception as e:
                validation_report['recommendations'].append(
                    f"Environment registry check failed: {e}"
                )
        
        if RENDER_CONFIG_AVAILABLE:
            try:
                rgb_presets = RENDER_REGISTRY.list_presets_by_category(RenderPresetCategory.RGB_ARRAY)
                registry_status['render_registry'] = True
                health_points += 1
            except Exception as e:
                validation_report['recommendations'].append(
                    f"Render registry check failed: {e}"
                )
        
        validation_report['component_checks']['registries'] = registry_status
    
    # Check factory functionality
    if check_factories:
        max_points += 6
        factory_tests = {}
        
        # Test environment factories
        try:
            test_env_config = get_default_environment_config()
            factory_tests['environment_factory'] = True
            health_points += 1
        except Exception as e:
            factory_tests['environment_factory'] = False
            validation_report['recommendations'].append(f"Environment factory test failed: {e}")
        
        # Test render factories
        try:
            test_render_config = get_default_render_config()
            factory_tests['render_factory'] = True
            health_points += 1
        except Exception as e:
            factory_tests['render_factory'] = False
            validation_report['recommendations'].append(f"Render factory test failed: {e}")
        
        # Test performance factories
        try:
            test_perf_config = get_default_performance_config()
            factory_tests['performance_factory'] = True
            health_points += 1
        except Exception as e:
            factory_tests['performance_factory'] = False
            validation_report['recommendations'].append(f"Performance factory test failed: {e}")
        
        # Test complete configuration factory
        try:
            test_complete_config = get_complete_default_config()
            factory_tests['complete_factory'] = True
            health_points += 1
        except Exception as e:
            factory_tests['complete_factory'] = False
            validation_report['recommendations'].append(f"Complete factory test failed: {e}")
        
        # Test preset factories
        if ENVIRONMENT_CONFIG_AVAILABLE:
            try:
                test_preset = create_preset_config('default')
                factory_tests['preset_factory'] = True
                health_points += 1
            except Exception as e:
                factory_tests['preset_factory'] = False
                validation_report['recommendations'].append(f"Preset factory test failed: {e}")
        
        # Test test configuration factories
        if TEST_CONFIG_AVAILABLE:
            try:
                test_unit_config = create_unit_test_config()
                factory_tests['test_factory'] = True
                health_points += 1
            except Exception as e:
                factory_tests['test_factory'] = False
                validation_report['recommendations'].append(f"Test factory test failed: {e}")
        
        validation_report['functionality_tests'] = factory_tests
    
    # Calculate system health score
    validation_report['system_health_score'] = (health_points / max(max_points, 1)) * 100
    
    # Add detailed component analysis if requested
    if detailed_report:
        validation_report['detailed_analysis'] = {
            'fallback_usage': not DEFAULT_CONFIG_AVAILABLE,
            'missing_components': [
                comp for comp, available in {
                    'default_config': DEFAULT_CONFIG_AVAILABLE,
                    'environment_config': ENVIRONMENT_CONFIG_AVAILABLE,
                    'render_config': RENDER_CONFIG_AVAILABLE,
                    'test_config': TEST_CONFIG_AVAILABLE
                }.items() if not available
            ],
            'available_factories': [
                name for name, status in validation_report['functionality_tests'].items() 
                if status
            ]
        }
    
    # Generate health-based recommendations
    health_score = validation_report['system_health_score']
    if health_score < 50:
        validation_report['recommendations'].append(
            "Critical: Configuration system health is poor - multiple components unavailable"
        )
        validation_report['overall_status'] = False
    elif health_score < 75:
        validation_report['recommendations'].append(
            "Warning: Configuration system partially functional - some features may be limited"
        )
    elif health_score < 90:
        validation_report['recommendations'].append(
            "Good: Configuration system mostly functional - minor components missing"
        )
    else:
        validation_report['recommendations'].append(
            "Excellent: Configuration system fully operational"
        )
    
    return validation_report['overall_status'], validation_report

# Initialize configuration system on import
def _initialize_configuration_system():
    """
    Initialize configuration system with status checking and warning notifications
    for missing components or degraded functionality.
    """
    try:
        system_status = get_configuration_system_status()
        logger.info(f"Configuration system initialized with status: {system_status['system_status']}")
        
        # Log warnings for missing components
        if not DEFAULT_CONFIG_AVAILABLE:
            logger.warning("Default configuration module not available - using fallback implementations")
        
        if not ENVIRONMENT_CONFIG_AVAILABLE:
            logger.warning("Environment configuration presets not available")
        
        if not RENDER_CONFIG_AVAILABLE:
            logger.warning("Rendering configuration presets not available")
        
        if not TEST_CONFIG_AVAILABLE:
            logger.warning("Test configuration factories not available")
        
        # Validate basic system functionality
        try:
            test_config = get_complete_default_config()
            logger.info("Configuration system basic functionality verified")
        except Exception as e:
            logger.error(f"Configuration system basic functionality test failed: {e}")
        
    except Exception as e:
        logger.error(f"Configuration system initialization failed: {e}")
        if STRICT_VALIDATION_MODE:
            raise

# Perform system initialization on module import
_initialize_configuration_system()

# Export all public interfaces through __all__
__all__ = [
    # Core configuration classes
    "EnvironmentConfig", "PlumeConfig", "RenderConfig", "PerformanceConfig", 
    "CompleteConfig", "ConfigurationError",
    
    # Default configuration factory functions
    "get_default_environment_config", "get_default_plume_config", 
    "get_default_render_config", "get_default_performance_config", 
    "get_complete_default_config",
    
    # Configuration utilities
    "validate_configuration_compatibility", "merge_configurations", 
    "create_config_from_dict",
    
    # Environment configuration presets and management
    "create_preset_config", "create_research_scenario", "create_benchmark_config", 
    "create_custom_scenario", "ConfigurationRegistry", "ENVIRONMENT_REGISTRY", 
    "PresetMetadata", "get_available_presets", "validate_preset_name",
    
    # Rendering configuration presets and management
    "RenderConfigPreset", "RenderPresetCategory", "RenderPresetRegistry",
    "create_rgb_preset", "create_matplotlib_preset", "create_research_preset",
    "create_accessibility_preset", "create_performance_preset", "RENDER_REGISTRY",
    
    # Test configuration factories and utilities
    "create_unit_test_config", "create_integration_test_config", 
    "create_performance_test_config", "create_reproducibility_test_config",
    "TestConfigFactory", "validate_test_configuration", "REPRODUCIBILITY_SEEDS",
    
    # System management and integration functions
    "get_configuration_system_status", "create_integrated_config", 
    "validate_configuration_system"
]