import logging as std_logging

# Standard library imports
# Type hints imported for potential future use in __all__ exports

# Package version definition for distribution and compatibility tracking
__version__ = "0.1.0"

# Import all exception classes providing structured error handling across components
from .exceptions import (
    ConfigurationError,
    ErrorSeverity,
    PlumeNavSimError,
    RenderingError,
    StateError,
    ValidationError,
    handle_component_error,
)

# Development logging and monitoring interface with component-specific capabilities
# Import logging utilities providing performance tracking and debugging support
from .logging import (
    ComponentLogger,
    LoggingMixin,
    get_component_logger,
)

# Seeding and reproducibility utilities for scientific research compliance
# Import seeding components ensuring deterministic behavior across episodes
from .seeding import (
    SeedManager,
    create_seeded_rng,
    validate_seed,
)

# Gymnasium space management and validation with caching optimization
# Import space utilities providing validated action and observation space creation
from .spaces import (
    create_action_space,
    create_observation_space,
    validate_action,
    validate_observation,
)

# Import validation utilities providing secure parameter validation across components
from .validation import (
    ParameterValidator,
    ValidationResult,
    validate_action_parameter,
    validate_environment_config,
    validate_observation_parameter,
)

# Configuration handling and lifecycle management with validation framework
# Import configuration utilities providing centralized configuration operations
try:
    from .config import (
        ConfigManager,
        create_quick_config,
        validate_config,
    )
except ImportError:  # pragma: no cover - optional configuration utilities may be absent
    create_quick_config = None
    validate_config = None
    ConfigManager = None

_logger = std_logging.getLogger(__name__)

# Initialize module-level logging for package lifecycle tracking and debugging
try:
    # Attempt to get component-specific logger with automatic configuration
    _package_logger = get_component_logger("utils")
    _package_logger.debug("Utility package initialization completed successfully")

    # Log imported component summary for debugging and system verification
    _package_logger.info(
        "Loaded utility components: exceptions, seeding, validation, logging, spaces, config"
    )

except Exception as init_error:
    # Fallback to standard logging if component logger initialization fails
    _logger.warning(f"Could not initialize component logger: {init_error}")
    _logger.info("Utility package initialized with standard logging fallback")

# Performance optimization: Pre-validate critical utility functions during import
# This ensures early detection of configuration issues and improves runtime performance
try:
    # Validate that core seeding functionality is available and working
    test_seed_result = validate_seed(42)

    # Validate that validation framework is functioning properly
    test_validator = ParameterValidator()

    # Log successful utility validation for system verification
    if hasattr(_package_logger, "debug"):
        _package_logger.debug(
            "Core utility functions validated successfully during import"
        )
    else:
        _logger.debug("Core utility functions validated successfully during import")

except Exception as validation_error:
    # Log validation errors but allow package to continue loading for development flexibility
    error_msg = f"Utility function validation failed during import: {validation_error}"

    if hasattr(_package_logger, "warning"):
        _package_logger.warning(error_msg)
    else:
        _logger.warning(error_msg)

# This list defines all publicly available utility components and functions
__all__ = [
    # Exception handling system components providing structured error management
    "PlumeNavSimError",  # Base exception class with consistent error handling interface
    "ValidationError",  # Exception for input validation failures with specific context
    "StateError",  # Exception for invalid state transitions with recovery suggestions
    "RenderingError",  # Exception for visualization failures with fallback options
    "ConfigurationError",  # Exception for configuration issues with valid option suggestions
    "handle_component_error",  # Centralized error handling with component-specific recovery
    "ErrorSeverity",  # Error severity classification for priority and escalation
    # Seeding and reproducibility framework for scientific research compliance
    "validate_seed",
    "create_seeded_rng",  # Gymnasium-compatible seeded RNG creation for deterministic behavior
    "SeedManager",  # Centralized seed management with validation and thread safety
    "validate_environment_config",  # Environment configuration validation with consistency checking
    "validate_action_parameter",  # Action parameter validation with type checking
    "validate_observation_parameter",  # Observation validation with array validation and range checking
    "ValidationResult",  # Validation result data class with error reporting
    "ParameterValidator",  # Parameter validation utility with caching and monitoring
    # Development logging and monitoring interface with performance tracking
    "get_component_logger",  # Factory function for component-specific loggers with configuration
    "ComponentLogger",  # Enhanced logger class with performance tracking and context capture
    "LoggingMixin",  # Mixin class for adding logging capabilities with automatic setup
    # Gymnasium space management and validation with performance optimization
    "create_action_space",  # Factory for validated Discrete action spaces with caching
    "create_observation_space",  # Factory for validated Box observation spaces with bounds
    "validate_action",  # Runtime action validation ensuring space compliance
    "validate_observation",  # Runtime observation validation ensuring space compliance
]

if create_quick_config is not None:
    __all__.extend(
        [
            # Configuration handling and lifecycle management with validation framework
            "create_quick_config",  # Quick configuration creation with overrides and defaults
            "validate_config",
            "ConfigManager",  # Configuration manager for centralized operations and lifecycle
        ]
    )

# Module initialization completion logging for system tracking and debugging
try:
    # Log successful package initialization with version and component information
    init_message = f"plume_nav_sim.utils v{__version__} initialized with {len(__all__)} exported components"

    if hasattr(_package_logger, "info"):
        _package_logger.info(init_message)
    else:
        _logger.info(init_message)

    # Debug-level logging of all exported components for development support
    debug_message = f"Exported utility components: {', '.join(__all__)}"

    if hasattr(_package_logger, "debug"):
        _package_logger.debug(debug_message)
    else:
        _logger.debug(debug_message)

except Exception:
    # Ensure package initialization completes even if logging fails
    pass

# Utility package initialization complete - all components loaded and validated
# performance optimization, and scientific research compliance features.
