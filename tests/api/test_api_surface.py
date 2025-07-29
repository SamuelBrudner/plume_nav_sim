import importlib

def test_api_surface_is_clean():
    """
    Test that the public API surface exposes only intended functions.
    
    Validates that the plume_nav_sim.api module:
    1. Exposes all required public API functions according to current implementation
    2. Does not expose private/internal implementation functions
    3. Includes Gymnasium RL integration and environment registration
    4. Maintains backward compatibility through navigation module access
    
    This test ensures API stability and prevents accidental exposure of 
    internal implementation details.
    """
    api = importlib.import_module("plume_nav_sim.api")
    public_api = set(dir(api))
    
    # Filter out private/internal attributes for the actual set
    actual_public = {name for name in public_api if not name.startswith('_')}
    
    # Core public API functions based on current implementation after setuptools migration
    intended = {
        # Core simulation functions
        "run_plume_simulation",  # Main simulation runner
        "create_navigator",  # Navigator factory function
        "create_video_plume",  # Video-based plume creation
        "visualize_plume_simulation",  # Simulation visualization function
        "create_gymnasium_environment",  # Modern RL environment factory
        
        # Navigation module - For backward compatibility
        "navigation",  # Navigation module access
        
        # Exception classes - Comprehensive error handling
        "SimulationError",  # General simulation errors
        "ConfigurationError",  # Configuration-related errors
    }
    
    # Check all intended functions exist
    missing = intended - actual_public
    extra = actual_public - intended
    
    if missing:
        assert False, f"Missing public API functions: {sorted(missing)}"
    
    if extra:
        assert False, f"Unexpected public API functions: {sorted(extra)}"
    
    # All intended functions should be present and no extras
    assert actual_public == intended


def test_factory_method_patterns():
    """
    Test core factory method patterns exposed through public API.
    
    Validates that factory interfaces support configuration patterns and
    Gymnasium environment creation for RL integration.
    """
    api = importlib.import_module("plume_nav_sim.api")
    
    # Test that primary factory methods are callable
    assert callable(getattr(api, "create_navigator", None)), (
        "create_navigator must be callable factory method"
    )
    assert callable(getattr(api, "create_video_plume", None)), (
        "create_video_plume must be callable factory method"
    )
    assert callable(getattr(api, "run_plume_simulation", None)), (
        "run_plume_simulation must be callable orchestration method"
    )
    
    # Test that Gymnasium factory method is available
    assert callable(getattr(api, "create_gymnasium_environment", None)), (
        "create_gymnasium_environment must be callable for RL integration"
    )
    
    # Test visualization function
    assert callable(getattr(api, "visualize_plume_simulation", None)), (
        "visualize_plume_simulation must be callable visualization method"
    )


def test_exception_classes_available():
    """
    Test that exception classes are properly exposed in the API.
    
    Validates that error handling classes are available for proper
    exception handling in user code.
    """
    api = importlib.import_module("plume_nav_sim.api")
    
    # Test that exception classes are available
    assert hasattr(api, "SimulationError"), (
        "SimulationError must be available for error handling"
    )
    assert hasattr(api, "ConfigurationError"), (
        "ConfigurationError must be available for error handling"
    )
    
    # Test that they are actually exception classes
    assert issubclass(api.SimulationError, Exception), (
        "SimulationError must be an exception class"
    )
    assert issubclass(api.ConfigurationError, Exception), (
        "ConfigurationError must be an exception class"
    )


def test_navigation_module_access():
    """
    Test that navigation module is accessible through the API.
    
    Validates that the navigation module provides backward compatibility
    and access to navigation functionality.
    """
    api = importlib.import_module("plume_nav_sim.api")
    
    # Test that navigation module is available
    assert hasattr(api, "navigation"), (
        "navigation module must be available for backward compatibility"
    )
    
    # The navigation attribute should be a module
    import types
    assert isinstance(api.navigation, types.ModuleType), (
        "navigation should be a module"
    )


def test_gymnasium_integration_available():
    """
    Test that Gymnasium RL integration functions are exposed in the public API.
    
    Validates that RL environment creation functions are properly
    exposed for reinforcement learning integration.
    """
    api = importlib.import_module("plume_nav_sim.api")
    
    # Test that Gymnasium environment function is available
    assert hasattr(api, "create_gymnasium_environment"), (
        "create_gymnasium_environment function must be available for RL integration"
    )
    assert callable(getattr(api, "create_gymnasium_environment")), (
        "create_gymnasium_environment must be callable"
    )


def test_no_private_functions_exposed():
    """
    Test that private functions are not exposed in the public API.
    
    Validates that internal implementation details remain private
    and are not accidentally exposed to users.
    """
    api = importlib.import_module("plume_nav_sim.api")
    public_api = set(dir(api))
    
    # Check that no private functions (starting with _) are exposed
    private_functions = {name for name in public_api if name.startswith('_') and not name.startswith('__')}
    
    assert not private_functions, (
        f"Private functions exposed in public API: {sorted(private_functions)}. "
        f"Private functions should not be accessible to library consumers."
    )