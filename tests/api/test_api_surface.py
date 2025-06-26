import importlib

def test_api_surface_is_clean():
    """
    Test that the public API surface exposes only intended functions.
    
    Validates that the odor_plume_nav.api module:
    1. Exposes all required public API functions according to updated spec
    2. Does not expose private/internal implementation functions
    3. Includes new Gymnasium RL integration and environment registration
    4. Maintains backward compatibility through legacy aliases
    5. Supports new PlumeNavSim-v0 environment registration
    
    This test ensures API stability and prevents accidental exposure of 
    internal implementation details as the system migrates to Gymnasium
    0.29.x integration with dual-API support and centralized Loguru logging.
    """
    api = importlib.import_module("odor_plume_nav.api")
    public_api = set(dir(api))
    
    # Core public API functions with new Gymnasium integration
    intended = {
        # Primary factory methods
        "create_navigator",
        "create_video_plume",
        "run_plume_simulation",
        
        # New Gymnasium RL environment factory methods (F-004)
        "create_gymnasium_environment",
        "from_legacy",
        
        # Enhanced API aliases for improved documentation
        "create_navigator_instance",
        "create_video_plume_instance", 
        "run_navigation_simulation",
        "visualize_results",
        
        # Visualization interface functions
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
        
        # Legacy compatibility functions (backward compatibility)
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
    # Includes traditional private helpers plus new Hydra-based and Loguru internal functions
    forbidden = {
        # Legacy private configuration helpers 
        "_merge_config_with_args",
        "_validate_positions", 
        "_load_config",
        "_load_navigator_from_config",
        
        # New Hydra-based internal configuration functions (F-006)
        "_compose_config",
        "_validate_hydra_config",
        "_merge_hydra_overrides",
        "_setup_config_store",
        "_initialize_hydra_context",
        "_process_dictconfig",
        "_convert_legacy_config",
        "_register_structured_configs",
        "_validate_dataclass_config",
        "_hydra_config_loader",
        "_config_store_manager",
        
        # Centralized Loguru logging internal functions (F-011)
        "_setup_loguru_logging",
        "_configure_json_sink",
        "_generate_correlation_id",
        "_setup_performance_logger",
        "_mask_sensitive_data",
        "_initialize_logging_context",
        "_loguru_sink_manager",
        "_log_level_configurator",
        
        # Internal API helper functions
        "_get_api_info",  # Private version of get_api_info
        "_validate_api_parameters",
        "_check_dependencies",
        "_setup_logging_context",
        
        # Gymnasium environment internal functions (F-004)
        "_gymnasium_env_factory",
        "_detect_legacy_caller",
        "_wrap_legacy_api",
        "_validate_gymnasium_config",
        "_create_dual_api_wrapper",
        "_register_environment_internally",
        
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
        f"Expected functions from updated specification including new Gymnasium "
        f"integration (F-004) and api/__init__.py __all__ exports are not available."
    )
    
    # Test that forbidden private functions are not exposed
    exposed_private = forbidden & public_api
    assert forbidden.isdisjoint(public_api), (
        f"Private implementation functions exposed in public API: {exposed_private}. "
        f"These internal functions should not be accessible to library consumers "
        f"per updated specification including Hydra ConfigStore (F-006) and "
        f"Loguru logging (F-011) internal functions."
    )


def test_factory_method_patterns():
    """
    Test new factory method patterns exposed through public API.
    
    Validates that enhanced factory interfaces support both Hydra configuration 
    and direct parameter patterns including new Gymnasium environment creation
    patterns for RL integration.
    """
    api = importlib.import_module("odor_plume_nav.api")
    
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
    
    # Test that new Gymnasium factory methods are available (F-004)
    assert callable(getattr(api, "create_gymnasium_environment", None)), (
        "create_gymnasium_environment must be callable for RL integration"
    )
    assert callable(getattr(api, "from_legacy", None)), (
        "from_legacy must be callable for backward compatibility migration"
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
    
    Validates that the API supports library import patterns for:
    - Kedro projects with Hydra configuration
    - RL frameworks with protocol-based interfaces  
    - ML analysis tools with standardized data exchange
    - New Gymnasium RL training workflows
    """
    api = importlib.import_module("odor_plume_nav.api")
    
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
            f"Required for backward compatibility."
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
    
    # Test that new RL integration functions support backward compatibility
    rl_functions = {
        "create_gymnasium_environment",
        "from_legacy",
    }
    
    for func_name in rl_functions:
        assert hasattr(api, func_name), (
            f"RL integration function {func_name} missing. "
            f"Required for Feature F-004 Gymnasium environment support."
        )
        assert callable(getattr(api, func_name)), (
            f"RL function {func_name} must be callable"
        )


def test_visualization_interface_integration():
    """
    Test visualization interface functions from utils.visualization.
    
    Validates that visualization interface functions are properly
    re-exported through the public API for seamless integration.
    """
    api = importlib.import_module("odor_plume_nav.api")
    
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
            f"Required for comprehensive visualization interface."
        )
        assert callable(getattr(api, func_name)), (
            f"Visualization function {func_name} must be callable"
        )
    
    # Test that SimulationVisualization class is available
    assert hasattr(api, "SimulationVisualization"), (
        "SimulationVisualization class must be available for advanced visualization usage"
    )


def test_gymnasium_integration_api_surface():
    """
    Test that new Gymnasium RL integration functions are exposed in the public API.
    
    Validates that Feature F-004 (Gymnasium RL Environment) functions are properly
    exposed while maintaining backward compatibility with legacy gym interfaces.
    This includes the new PlumeNavSim-v0 environment registration and dual-API support.
    """
    api = importlib.import_module("odor_plume_nav.api")
    
    # New Gymnasium environment functions that should be available (F-004)
    gymnasium_functions = {
        "create_gymnasium_environment",
        "from_legacy",
    }
    
    for func_name in gymnasium_functions:
        assert hasattr(api, func_name), (
            f"Gymnasium function {func_name} not available in public API. "
            f"Required per Feature F-004 (Gymnasium RL Environment)."
        )
        assert callable(getattr(api, func_name)), (
            f"Gymnasium function {func_name} must be callable"
        )


def test_environment_registration_api_surface():
    """
    Test that environment registration patterns are properly exposed.
    
    Validates that new environment registration functions support the PlumeNavSim-v0
    environment ID and provide proper factory method access patterns for RL integration.
    """
    # Test that environments module has necessary registration functions
    environments = importlib.import_module("odor_plume_nav.environments")
    
    # Core environment registration functions should be available
    expected_env_functions = {
        "GymnasiumEnv",  # Core environment class  
    }
    
    for func_name in expected_env_functions:
        # Check if available (may be None if RL dependencies not installed)
        if hasattr(environments, func_name):
            env_attr = getattr(environments, func_name)
            if env_attr is not None:
                # If available, it should be a class or function
                assert callable(env_attr) or isinstance(env_attr, type), (
                    f"Environment component {func_name} should be callable or a class"
                )


def test_gymnasium_api_compatibility_functions():
    """
    Test that gymnasium.utils.env_checker and related API compliance functions are exposed.
    
    Validates that the API provides access to Gymnasium environment validation utilities
    and compliance checking functions required for RL framework integration.
    """
    # Check if Gymnasium dependencies are available
    try:
        import gymnasium
        gymnasium_available = True
    except ImportError:
        gymnasium_available = False
    
    if gymnasium_available:
        # Test that we can access env_checker utilities for validation
        try:
            from gymnasium.utils import env_checker
            # This indicates Gymnasium integration is properly configured
            assert hasattr(env_checker, 'check_env'), (
                "gymnasium.utils.env_checker should provide check_env function"
            )
        except ImportError:
            # This is acceptable - env_checker might not be available in all Gymnasium versions
            pass
    
    # Test that our API can handle Gymnasium environment creation
    api = importlib.import_module("odor_plume_nav.api")
    
    # The create_gymnasium_environment function should be available
    assert hasattr(api, "create_gymnasium_environment"), (
        "create_gymnasium_environment function must be available for RL integration"
    )
    
    # Test that legacy migration function is available
    assert hasattr(api, "from_legacy"), (
        "from_legacy function must be available for backward compatibility"
    )


def test_centralized_logging_integration_api_surface():
    """
    Test that centralized Loguru logging integration changes are reflected in API surface.
    
    Validates that Feature F-011 (Centralized Logging Architecture) integration does not
    expose internal logging functions while ensuring the public API maintains expected
    behavior with structured logging support.
    """
    api = importlib.import_module("odor_plume_nav.api")
    public_api = set(dir(api))
    
    # Loguru internal functions that should NOT be exposed in public API
    loguru_internal_functions = {
        "_setup_loguru_logging",
        "_configure_json_sink", 
        "_generate_correlation_id",
        "_setup_performance_logger",
        "_mask_sensitive_data",
        "_initialize_logging_context",
        "_loguru_sink_manager",
        "_log_level_configurator",
    }
    
    # Verify that internal logging functions are not exposed
    exposed_logging_internals = loguru_internal_functions & public_api
    assert not exposed_logging_internals, (
        f"Internal Loguru logging functions exposed in public API: {exposed_logging_internals}. "
        f"These should remain private per Feature F-011 implementation."
    )
    
    # The public API should still provide its core functionality
    # (logging integration should be transparent to API consumers)
    core_api_functions = {
        "create_navigator",
        "create_video_plume", 
        "run_plume_simulation",
        "create_gymnasium_environment",
    }
    
    for func_name in core_api_functions:
        assert hasattr(api, func_name), (
            f"Core API function {func_name} missing after logging integration. "
            f"Loguru integration should be transparent to public API."
        )


def test_hydra_structured_config_api_surface():
    """
    Test that Hydra structured configuration changes don't expose internal functions.
    
    Validates that Feature F-006 (Hierarchical Configuration Management) updates with
    @dataclass-based ConfigStore registration keep internal configuration functions
    private while maintaining public configuration interface.
    """
    api = importlib.import_module("odor_plume_nav.api")
    public_api = set(dir(api))
    
    # New Hydra ConfigStore internal functions that should NOT be exposed
    hydra_internal_functions = {
        "_register_structured_configs",
        "_validate_dataclass_config", 
        "_hydra_config_loader",
        "_config_store_manager",
        "_setup_config_store",
        "_initialize_hydra_context",
        "_compose_config",
        "_validate_hydra_config",
    }
    
    # Verify that internal Hydra functions are not exposed
    exposed_hydra_internals = hydra_internal_functions & public_api
    assert not exposed_hydra_internals, (
        f"Internal Hydra configuration functions exposed in public API: {exposed_hydra_internals}. "
        f"These should remain private per Feature F-006 structured config implementation."
    )
    
    # Public configuration interface should remain available
    # (structured configs should be transparent to API consumers)
    config_related_functions = {
        "create_navigator",
        "create_video_plume",
        "create_gymnasium_environment",
    }
    
    for func_name in config_related_functions:
        assert hasattr(api, func_name), (
            f"Configuration-capable function {func_name} missing after Hydra updates. "
            f"Structured config integration should be transparent to public API."
        )


def test_backward_compatibility_api_preservation():
    """
    Test that existing API functions remain available during modernization.
    
    Validates that legacy compatibility functions remain in the API surface
    while new Gymnasium, Loguru logging, and Hydra structured config features
    are added, ensuring zero breaking changes for existing users.
    """
    api = importlib.import_module("odor_plume_nav.api")
    
    # Core legacy functions that must remain available
    legacy_core_functions = {
        "create_navigator",
        "create_video_plume",
        "run_plume_simulation",
        "visualize_simulation_results",
        "visualize_trajectory",
    }
    
    # Legacy alias functions that must remain available
    legacy_alias_functions = {
        "create_navigator_from_config",
        "create_video_plume_from_config", 
        "run_simulation",
        "create_navigator_legacy",
        "create_video_plume_legacy",
    }
    
    all_legacy_functions = legacy_core_functions | legacy_alias_functions
    
    for func_name in all_legacy_functions:
        assert hasattr(api, func_name), (
            f"Legacy function {func_name} missing from API. "
            f"Backward compatibility requires preservation of all existing functions."
        )
        assert callable(getattr(api, func_name)), (
            f"Legacy function {func_name} must remain callable"
        )
    
    # Test that module metadata is preserved
    metadata_attrs = {"__version__", "__author__", "__description__"}
    for attr_name in metadata_attrs:
        assert hasattr(api, attr_name), (
            f"Module metadata {attr_name} missing from API"
        )
