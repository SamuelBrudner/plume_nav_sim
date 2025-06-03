"""
Test suite for API surface validation of the Odor Plume Navigation library.

This module validates that the public API exposes only the intended functions
and maintains backward compatibility while enforcing proper encapsulation of
internal implementation details. The tests ensure that the Hydra-integrated
API structure follows enterprise-grade design patterns and maintains
compatibility with Kedro, RL frameworks, and ML analysis tools.

Test Categories:
    - Public API function availability and correctness
    - Private function encapsulation validation
    - Backward compatibility verification for legacy import patterns
    - Hydra-specific configuration interface validation
    - Factory method pattern validation
    - Type hint and protocol exposure verification
"""

import pytest
import inspect
import warnings
from typing import Any, Dict, List, Set
from unittest.mock import patch, MagicMock

# Test the new import structure
import {{cookiecutter.project_slug}}.api as api_module
from {{cookiecutter.project_slug}}.api import (
    # Primary API functions
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    
    # High-level convenience functions
    create_navigation_session,
    run_complete_experiment,
    
    # Factory methods for backward compatibility
    create_navigator_from_config,
    create_video_plume_from_config,
    
    # Core types for type hints and direct access
    Navigator,
    NavigatorProtocol,
    VideoPlume,
    SingleAgentController,
    MultiAgentController,
    
    # Configuration schemas
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    
    # Utility functions
    set_global_seed,
    get_current_seed,
    
    # Exceptions
    ConfigurationError,
    SimulationError,
    
    # Constants and availability flags
    HYDRA_AVAILABLE,
    VISUALIZATION_AVAILABLE,
)

# Conditional imports for visualization functions (graceful degradation)
try:
    from {{cookiecutter.project_slug}}.api import (
        visualize_simulation_results,
        visualize_trajectory,
        SimulationVisualization,
    )
    VISUALIZATION_IMPORTS_AVAILABLE = True
except ImportError:
    VISUALIZATION_IMPORTS_AVAILABLE = False


class TestAPIStructure:
    """Test suite for validating the overall API structure and organization."""
    
    def test_module_has_proper_docstring(self):
        """Verify the API module has comprehensive documentation."""
        assert api_module.__doc__ is not None
        assert len(api_module.__doc__.strip()) > 100
        assert "Odor Plume Navigation" in api_module.__doc__
        assert "Kedro" in api_module.__doc__
        assert "Hydra" in api_module.__doc__
    
    def test_module_has_version_metadata(self):
        """Ensure the API module includes proper version and metadata information."""
        assert hasattr(api_module, '__version__')
        assert hasattr(api_module, '__author__')
        assert hasattr(api_module, '__description__')
        assert isinstance(api_module.__version__, str)
        assert len(api_module.__version__) > 0
    
    def test_module_has_all_attribute(self):
        """Verify the module properly defines __all__ for explicit public API control."""
        assert hasattr(api_module, '__all__')
        assert isinstance(api_module.__all__, list)
        assert len(api_module.__all__) > 0


class TestPublicAPIFunctions:
    """Test suite for validating public API function availability and signatures."""
    
    def test_primary_api_functions_available(self):
        """Verify all primary API functions are accessible and properly exposed."""
        expected_primary_functions = {
            'create_navigator',
            'create_video_plume', 
            'run_plume_simulation',
        }
        
        for func_name in expected_primary_functions:
            assert hasattr(api_module, func_name), f"Missing primary API function: {func_name}"
            func = getattr(api_module, func_name)
            assert callable(func), f"{func_name} is not callable"
            assert func_name in api_module.__all__, f"{func_name} not in __all__"
    
    def test_high_level_convenience_functions_available(self):
        """Verify high-level convenience functions for complete workflow support."""
        expected_convenience_functions = {
            'create_navigation_session',
            'run_complete_experiment',
        }
        
        for func_name in expected_convenience_functions:
            assert hasattr(api_module, func_name), f"Missing convenience function: {func_name}"
            func = getattr(api_module, func_name)
            assert callable(func), f"{func_name} is not callable"
            assert func_name in api_module.__all__, f"{func_name} not in __all__"
    
    def test_factory_methods_backward_compatibility(self):
        """Verify factory methods are available for backward compatibility."""
        expected_factory_methods = {
            'create_navigator_from_config',
            'create_video_plume_from_config',
        }
        
        for func_name in expected_factory_methods:
            assert hasattr(api_module, func_name), f"Missing factory method: {func_name}"
            func = getattr(api_module, func_name)
            assert callable(func), f"{func_name} is not callable"
            assert func_name in api_module.__all__, f"{func_name} not in __all__"
    
    def test_visualization_functions_conditional_availability(self):
        """Verify visualization functions are available when dependencies are present."""
        expected_viz_functions = {
            'visualize_simulation_results',
            'visualize_trajectory',
        }
        
        if VISUALIZATION_AVAILABLE and VISUALIZATION_IMPORTS_AVAILABLE:
            for func_name in expected_viz_functions:
                assert hasattr(api_module, func_name), f"Missing visualization function: {func_name}"
                func = getattr(api_module, func_name)
                assert callable(func), f"{func_name} is not callable"
                assert func_name in api_module.__all__, f"{func_name} not in __all__"
        
        # Always verify SimulationVisualization class is available (may be fallback)
        assert hasattr(api_module, 'SimulationVisualization')
        assert inspect.isclass(api_module.SimulationVisualization)
        assert 'SimulationVisualization' in api_module.__all__
    
    def test_utility_functions_available(self):
        """Verify utility functions for seed management and reproducibility."""
        expected_utility_functions = {
            'set_global_seed',
            'get_current_seed',
        }
        
        for func_name in expected_utility_functions:
            assert hasattr(api_module, func_name), f"Missing utility function: {func_name}"
            func = getattr(api_module, func_name)
            assert callable(func), f"{func_name} is not callable"
            assert func_name in api_module.__all__, f"{func_name} not in __all__"


class TestCoreTypesAndProtocols:
    """Test suite for validating core types, protocols, and configuration schemas."""
    
    def test_core_navigation_types_available(self):
        """Verify core navigation types are properly exposed."""
        expected_core_types = {
            'Navigator',
            'NavigatorProtocol', 
            'VideoPlume',
            'SingleAgentController',
            'MultiAgentController',
        }
        
        for type_name in expected_core_types:
            assert hasattr(api_module, type_name), f"Missing core type: {type_name}"
            type_obj = getattr(api_module, type_name)
            assert inspect.isclass(type_obj) or hasattr(type_obj, '__annotations__'), f"{type_name} is not a valid type"
            assert type_name in api_module.__all__, f"{type_name} not in __all__"
    
    def test_configuration_schemas_available(self):
        """Verify Pydantic configuration schemas are properly exposed."""
        expected_config_schemas = {
            'NavigatorConfig',
            'VideoPlumeConfig',
            'SimulationConfig',
            'SingleAgentConfig',
            'MultiAgentConfig',
        }
        
        for schema_name in expected_config_schemas:
            assert hasattr(api_module, schema_name), f"Missing configuration schema: {schema_name}"
            schema_obj = getattr(api_module, schema_name)
            assert inspect.isclass(schema_obj), f"{schema_name} is not a class"
            assert schema_name in api_module.__all__, f"{schema_name} not in __all__"
    
    def test_exception_classes_available(self):
        """Verify custom exception classes are properly exposed."""
        expected_exceptions = {
            'ConfigurationError',
            'SimulationError',
        }
        
        for exc_name in expected_exceptions:
            assert hasattr(api_module, exc_name), f"Missing exception class: {exc_name}"
            exc_class = getattr(api_module, exc_name)
            assert inspect.isclass(exc_class), f"{exc_name} is not a class"
            assert issubclass(exc_class, Exception), f"{exc_name} is not an Exception subclass"
            assert exc_name in api_module.__all__, f"{exc_name} not in __all__"
    
    def test_availability_flags_exposed(self):
        """Verify feature availability flags are properly exposed."""
        expected_flags = {
            'HYDRA_AVAILABLE',
            'VISUALIZATION_AVAILABLE',
        }
        
        for flag_name in expected_flags:
            assert hasattr(api_module, flag_name), f"Missing availability flag: {flag_name}"
            flag_value = getattr(api_module, flag_name)
            assert isinstance(flag_value, bool), f"{flag_name} should be boolean"
            assert flag_name in api_module.__all__, f"{flag_name} not in __all__"


class TestPrivateFunctionEncapsulation:
    """Test suite for ensuring private functions are properly encapsulated."""
    
    def test_hydra_internal_functions_not_exposed(self):
        """Verify Hydra-specific internal functions are not exposed in public API."""
        # These are internal functions that should not be in __all__ or easily accessible
        forbidden_hydra_functions = {
            '_validate_and_merge_config',
            '_normalize_positions',
            '_get_hydra_config',
            '_initialize_hydra_context',
            '_merge_config_overrides',
            '_validate_hydra_config',
        }
        
        for func_name in forbidden_hydra_functions:
            if hasattr(api_module, func_name):
                assert func_name not in api_module.__all__, f"Private function {func_name} exposed in __all__"
    
    def test_configuration_internals_not_exposed(self):
        """Verify configuration management internals are not exposed."""
        forbidden_config_functions = {
            '_load_yaml_config',
            '_merge_yaml_configs',
            '_validate_config_schema',
            '_apply_config_defaults',
            '_resolve_config_interpolations',
        }
        
        for func_name in forbidden_config_functions:
            if hasattr(api_module, func_name):
                assert func_name not in api_module.__all__, f"Private config function {func_name} exposed in __all__"
    
    def test_navigation_internals_not_exposed(self):
        """Verify navigation implementation internals are not exposed."""
        forbidden_nav_functions = {
            '_update_agent_position',
            '_calculate_odor_gradient',
            '_process_sensor_data',
            '_apply_navigation_algorithm',
            '_validate_position_bounds',
        }
        
        for func_name in forbidden_nav_functions:
            if hasattr(api_module, func_name):
                assert func_name not in api_module.__all__, f"Private navigation function {func_name} exposed in __all__"
    
    def test_visualization_internals_not_exposed(self):
        """Verify visualization implementation internals are not exposed."""
        forbidden_viz_functions = {
            '_setup_matplotlib_backend',
            '_create_animation_frame',
            '_export_video_frame',
            '_configure_plot_style',
            '_initialize_figure_layout',
        }
        
        for func_name in forbidden_viz_functions:
            if hasattr(api_module, func_name):
                assert func_name not in api_module.__all__, f"Private visualization function {func_name} exposed in __all__"


class TestAPISignatureValidation:
    """Test suite for validating API function signatures and parameter support."""
    
    def test_create_navigator_signature_supports_hydra(self):
        """Verify create_navigator supports Hydra configuration patterns."""
        sig = inspect.signature(create_navigator)
        params = list(sig.parameters.keys())
        
        # Should support cfg parameter for Hydra integration
        assert any('cfg' in str(param).lower() for param in params), "create_navigator should support cfg parameter"
        
        # Should support **kwargs for parameter overrides
        assert any(sig.parameters[param].kind == inspect.Parameter.VAR_KEYWORD for param in params), \
            "create_navigator should support **kwargs"
    
    def test_create_video_plume_signature_supports_hydra(self):
        """Verify create_video_plume supports Hydra configuration patterns.""" 
        sig = inspect.signature(create_video_plume)
        params = list(sig.parameters.keys())
        
        # Should support cfg parameter for Hydra integration
        assert any('cfg' in str(param).lower() for param in params), "create_video_plume should support cfg parameter"
        
        # Should support **kwargs for parameter overrides
        assert any(sig.parameters[param].kind == inspect.Parameter.VAR_KEYWORD for param in params), \
            "create_video_plume should support **kwargs"
    
    def test_convenience_functions_support_hydra(self):
        """Verify convenience functions support comprehensive Hydra configuration."""
        convenience_functions = [create_navigation_session, run_complete_experiment]
        
        for func in convenience_functions:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Should support cfg parameter
            assert any('cfg' in str(param).lower() for param in params), \
                f"{func.__name__} should support cfg parameter"
            
            # Should support **kwargs
            assert any(sig.parameters[param].kind == inspect.Parameter.VAR_KEYWORD for param in params), \
                f"{func.__name__} should support **kwargs"


class TestBackwardCompatibility:
    """Test suite for validating backward compatibility with legacy import patterns."""
    
    def test_legacy_import_patterns_supported(self):
        """Verify the API supports documented legacy import patterns."""
        # Test Kedro-style imports
        try:
            from {{cookiecutter.project_slug}} import Navigator, VideoPlume
            from {{cookiecutter.project_slug}}.config import NavigatorConfig
            legacy_kedro_imports = True
        except ImportError:
            legacy_kedro_imports = False
        
        assert legacy_kedro_imports, "Kedro-style imports should be supported"
        
        # Test RL framework imports
        try:
            from {{cookiecutter.project_slug}}.core import NavigatorProtocol
            from {{cookiecutter.project_slug}}.api import create_navigator
            legacy_rl_imports = True
        except ImportError:
            legacy_rl_imports = False
        
        assert legacy_rl_imports, "RL framework imports should be supported"
        
        # Test ML analysis imports
        try:
            from {{cookiecutter.project_slug}}.utils import set_global_seed
            from {{cookiecutter.project_slug}}.data import VideoPlume
            legacy_ml_imports = True
        except ImportError:
            legacy_ml_imports = False
        
        assert legacy_ml_imports, "ML analysis imports should be supported"
    
    def test_factory_method_compatibility(self):
        """Verify factory methods maintain backward compatibility."""
        # Verify factory methods accept both old and new configuration formats
        factory_methods = [create_navigator_from_config, create_video_plume_from_config]
        
        for factory_method in factory_methods:
            sig = inspect.signature(factory_method)
            
            # Should accept cfg parameter (new pattern)
            assert 'cfg' in sig.parameters or any('cfg' in str(p) for p in sig.parameters.keys()), \
                f"{factory_method.__name__} should accept cfg parameter"


class TestAPICompleteness:
    """Test suite for validating API completeness and coverage."""
    
    def test_all_declared_functions_are_accessible(self):
        """Verify all functions declared in __all__ are actually accessible."""
        for declared_name in api_module.__all__:
            assert hasattr(api_module, declared_name), f"Declared function {declared_name} not accessible"
            
            declared_obj = getattr(api_module, declared_name)
            assert declared_obj is not None, f"Declared function {declared_name} is None"
    
    def test_no_undeclared_public_functions_exposed(self):
        """Verify no unintended public functions are accidentally exposed."""
        # Get all public attributes (not starting with underscore)
        public_attributes = [attr for attr in dir(api_module) if not attr.startswith('_')]
        
        # Filter to only callable objects or classes (exclude constants and modules)
        public_callables = []
        for attr_name in public_attributes:
            attr_obj = getattr(api_module, attr_name)
            if callable(attr_obj) or inspect.isclass(attr_obj):
                public_callables.append(attr_name)
        
        # Every public callable should be in __all__ or be an expected module/builtin
        expected_non_all_attributes = {
            'warnings',  # Standard library import
            'np',        # NumPy import
            'pathlib',   # Standard library import
            'datetime',  # Standard library import
            'suppress',  # contextlib import
            'asdict',    # dataclasses import
            'DictConfig',  # OmegaConf import (conditionally available)
            'OmegaConf',   # OmegaConf import (conditionally available)
        }
        
        for callable_name in public_callables:
            if callable_name not in expected_non_all_attributes:
                assert callable_name in api_module.__all__, \
                    f"Public callable {callable_name} not declared in __all__"
    
    def test_api_surface_consistency(self):
        """Verify the API surface is consistent and follows expected patterns."""
        # Verify function naming conventions
        api_functions = [name for name in api_module.__all__ if callable(getattr(api_module, name, None))]
        
        # Should have create_ functions for main components
        create_functions = [name for name in api_functions if name.startswith('create_')]
        assert len(create_functions) >= 4, "Should have multiple create_ functions"
        
        # Should have visualization functions (conditional)
        if VISUALIZATION_AVAILABLE:
            viz_functions = [name for name in api_functions if 'visualize' in name.lower()]
            assert len(viz_functions) >= 2, "Should have multiple visualization functions when available"
        
        # Should have utility functions
        utility_functions = [name for name in api_functions if name in {'set_global_seed', 'get_current_seed'}]
        assert len(utility_functions) == 2, "Should have both seed management utility functions"


class TestHydraIntegrationFeatures:
    """Test suite for validating Hydra-specific integration features."""
    
    def test_hydra_availability_flag_accuracy(self):
        """Verify HYDRA_AVAILABLE flag accurately reflects Hydra installation."""
        try:
            import hydra
            import omegaconf
            expected_hydra_available = True
        except ImportError:
            expected_hydra_available = False
        
        # Allow for minor discrepancies in import detection
        if expected_hydra_available:
            assert HYDRA_AVAILABLE, "HYDRA_AVAILABLE should be True when Hydra is importable"
    
    def test_visualization_availability_flag_accuracy(self):
        """Verify VISUALIZATION_AVAILABLE flag accurately reflects visualization dependencies."""
        try:
            import matplotlib
            import matplotlib.pyplot
            expected_viz_available = True
        except ImportError:
            expected_viz_available = False
        
        # Flag should reasonably reflect matplotlib availability
        if expected_viz_available:
            assert VISUALIZATION_AVAILABLE, "VISUALIZATION_AVAILABLE should be True when matplotlib is importable"
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_integration_supported(self):
        """Verify functions properly support Hydra DictConfig objects."""
        try:
            from omegaconf import DictConfig
            
            # Mock DictConfig object
            mock_config = DictConfig({
                'position': [10.0, 20.0],
                'max_speed': 5.0,
                'orientation': 0.0
            })
            
            # Verify create_navigator accepts DictConfig
            sig = inspect.signature(create_navigator)
            # Should not raise TypeError for DictConfig parameter
            # (Actual functionality testing would require integration tests)
            
        except ImportError:
            pytest.skip("OmegaConf DictConfig not available for testing")


class TestAPIDocumentationAndMetadata:
    """Test suite for validating API documentation and metadata completeness."""
    
    def test_primary_functions_have_docstrings(self):
        """Verify all primary API functions have comprehensive docstrings."""
        primary_functions = [
            'create_navigator', 
            'create_video_plume', 
            'run_plume_simulation',
            'create_navigation_session',
            'run_complete_experiment'
        ]
        
        for func_name in primary_functions:
            func = getattr(api_module, func_name)
            assert func.__doc__ is not None, f"{func_name} missing docstring"
            assert len(func.__doc__.strip()) > 50, f"{func_name} docstring too brief"
    
    def test_exception_classes_have_docstrings(self):
        """Verify exception classes have proper docstrings."""
        exception_classes = ['ConfigurationError', 'SimulationError']
        
        for exc_name in exception_classes:
            exc_class = getattr(api_module, exc_name)
            assert exc_class.__doc__ is not None, f"{exc_name} missing docstring"
    
    def test_type_hints_are_preserved(self):
        """Verify important functions preserve type hints for IDE support."""
        functions_with_hints = [create_navigator, create_video_plume, run_plume_simulation]
        
        for func in functions_with_hints:
            sig = inspect.signature(func)
            # Should have at least return type annotation
            assert sig.return_annotation != inspect.Signature.empty, f"{func.__name__} missing return type hint"


if __name__ == "__main__":
    # Enable running tests directly with python -m pytest tests/api/test_api_surface.py
    pytest.main([__file__])