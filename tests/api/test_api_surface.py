import importlib

def test_api_surface_is_clean():
    """
    Test that the public API surface exposes only intended functions.
    
    Validates that the {{cookiecutter.project_slug}}.api module:
    1. Exposes all required public API functions according to Section 7.2.1
    2. Does not expose private/internal implementation functions
    3. Includes new Hydra-integrated factory methods and visualization functions
    4. Maintains backward compatibility through legacy aliases
    
    This test ensures API stability and prevents accidental exposure of 
    internal implementation details as the system migrates to the new
    template structure with Hydra-based configuration management.
    """
    api = importlib.import_module("{{cookiecutter.project_slug}}.api")
    public_api = set(dir(api))
    
    # Core public API functions as defined in Section 7.2.1 Primary API Functions
    intended = {
        # Primary factory methods (Section 7.2.1.1, 7.2.1.2)
        "create_navigator",
        "create_video_plume",
        "run_plume_simulation",
        
        # Enhanced API aliases for improved documentation (api/__init__.py)
        "create_navigator_instance",
        "create_video_plume_instance", 
        "run_navigation_simulation",
        "visualize_results",
        
        # Visualization interface functions (Section 7.2.1.4)
        "visualize_simulation_results",
        "visualize_trajectory",
        "visualize_plume_simulation",
        "SimulationVisualization",
        "batch_visualize_trajectories",
        "setup_headless_mode",
        "get_available_themes",
        "create_simulation_visualization",
        "export_animation",
        
        # Core protocols and classes for advanced usage
        "NavigatorProtocol",
        "VideoPlume",
        
        # Legacy compatibility functions (Section 0.2.6 backward compatibility)
        "create_navigator_from_config",
        "create_video_plume_from_config",
        "run_simulation",
        "create_navigator_legacy",
        "create_video_plume_legacy",
        
        # API introspection for debugging
        "get_api_info",
        
        # Module metadata
        "__version__",
        "__author__",
        "__description__",
    }
    
    # Private/internal functions that should NOT be exposed
    # Includes traditional private helpers plus new Hydra-based internal functions
    forbidden = {
        # Legacy private configuration helpers 
        "_merge_config_with_args",
        "_validate_positions", 
        "_load_config",
        "_load_navigator_from_config",
        
        # Hydra-based internal functions (Section 7.2.2.2 configuration interface)
        "_compose_config",
        "_validate_hydra_config",
        "_merge_hydra_overrides",
        "_setup_config_store",
        "_initialize_hydra_context",
        "_process_dictconfig",
        "_convert_legacy_config",
        
        # Internal API helper functions
        "_get_api_info",  # Private version of get_api_info
        "_validate_api_parameters",
        "_check_dependencies",
        "_setup_logging_context",
        
        # Factory pattern internals
        "_create_navigator_internal",
        "_create_video_plume_internal",
        "_validate_factory_parameters",
        
        # Configuration processing internals
        "_merge_config_with_kwargs",
        "_validate_config_schema",
        "_process_configuration_object",
    }
    
    # Test that all intended public functions are available
    missing_public = intended - public_api
    assert intended.issubset(public_api), (
        f"Missing public API functions: {missing_public}. "
        f"Expected functions from Section 7.2.1 Primary API Functions and "
        f"api/__init__.py __all__ exports are not available."
    )
    
    # Test that forbidden private functions are not exposed
    exposed_private = forbidden & public_api
    assert forbidden.isdisjoint(public_api), (
        f"Private implementation functions exposed in public API: {exposed_private}. "
        f"These internal functions should not be accessible to library consumers "
        f"per Section 7.2.2.2 configuration interface schema."
    )


def test_factory_method_patterns():
    """
    Test new factory method patterns exposed through public API.
    
    Validates that enhanced factory interfaces per Section 7.2.1.1 and 7.2.1.2
    support both Hydra configuration and direct parameter patterns as specified
    in the API design requirements.
    """
    api = importlib.import_module("{{cookiecutter.project_slug}}.api")
    
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
    
    # Test that enhanced factory aliases are available
    assert callable(getattr(api, "create_navigator_instance", None)), (
        "create_navigator_instance alias must be available for enhanced documentation"
    )
    assert callable(getattr(api, "create_video_plume_instance", None)), (
        "create_video_plume_instance alias must be available for enhanced documentation"
    )


def test_hydra_integration_compatibility():
    """
    Test that API maintains Hydra integration while preserving backward compatibility.
    
    Validates that the API supports Section 0.2.6 library import patterns for:
    - Kedro projects with Hydra configuration
    - RL frameworks with protocol-based interfaces  
    - ML analysis tools with standardized data exchange
    """
    api = importlib.import_module("{{cookiecutter.project_slug}}.api")
    
    # Test that legacy compatibility functions are available
    legacy_functions = {
        "create_navigator_from_config",
        "create_video_plume_from_config",
        "run_simulation",
        "create_navigator_legacy",
        "create_video_plume_legacy",
    }
    
    for func_name in legacy_functions:
        assert hasattr(api, func_name), (
            f"Legacy compatibility function {func_name} missing. "
            f"Required for backward compatibility per Section 0.2.6."
        )
        assert callable(getattr(api, func_name)), (
            f"Legacy function {func_name} must be callable"
        )
    
    # Test that core protocols are exposed for advanced usage
    assert hasattr(api, "NavigatorProtocol"), (
        "NavigatorProtocol must be available for RL framework integration"
    )
    assert hasattr(api, "VideoPlume"), (
        "VideoPlume class must be available for ML analysis tools"
    )


def test_visualization_interface_integration():
    """
    Test visualization interface functions from utils.visualization.
    
    Validates that Section 7.2.1.4 visualization interface functions are properly
    re-exported through the public API for seamless integration.
    """
    api = importlib.import_module("{{cookiecutter.project_slug}}.api")
    
    # Core visualization functions that should be re-exported
    visualization_functions = {
        "visualize_simulation_results",
        "visualize_trajectory",
        "visualize_plume_simulation",
        "batch_visualize_trajectories",
        "setup_headless_mode",
        "get_available_themes", 
        "create_simulation_visualization",
        "export_animation",
    }
    
    for func_name in visualization_functions:
        assert hasattr(api, func_name), (
            f"Visualization function {func_name} not available in public API. "
            f"Required per Section 7.2.1.4 visualization interface."
        )
        assert callable(getattr(api, func_name)), (
            f"Visualization function {func_name} must be callable"
        )
    
    # Test that SimulationVisualization class is available
    assert hasattr(api, "SimulationVisualization"), (
        "SimulationVisualization class must be available for advanced visualization usage"
    )
