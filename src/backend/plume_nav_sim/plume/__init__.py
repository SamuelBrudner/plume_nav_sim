"""
Plume module initialization file providing comprehensive public API for plume navigation simulation
components including static Gaussian plume models, concentration field management, abstract base
classes, and factory functions. Serves as the main entry point for all plume-related functionality
with proper import organization, type safety, and performance optimization for reinforcement learning
environment integration.

This module exposes a comprehensive interface for plume modeling including mathematical implementations,
abstract frameworks, factory functions, specialized exceptions, and utility functions optimized for
gymnasium-compatible reinforcement learning environments with performance targets and extensibility.
"""

# External imports with version comments
import logging  # >=3.10 - Logging integration for module initialization and error reporting
import threading  # >=3.10 - Thread-safe module initialization and registry operations
import time  # >=3.10 - Timestamp generation for performance monitoring and module tracking
from typing import (  # >=3.10 - Type hints for comprehensive type safety
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

from typing_extensions import NotRequired, TypedDict  # 3.10 compatible TypedDict extras


# Typed structures for performance and cache reporting
class PerformanceEntry(TypedDict, total=False):
    last_execution_time: float
    execution_count: int


class PerformanceImpactEntry(TypedDict):
    items_cleared: int
    impact: str


class GarbageCollectionInfo(TypedDict):
    forced: bool
    objects_collected: int
    impact: str


class CacheSummary(TypedDict, total=False):
    total_caches_cleared: int
    total_memory_freed_kb: float
    total_memory_freed_mb: float
    operation_successful: bool
    critical_error: NotRequired[str]


class CacheReport(TypedDict, total=False):
    operation_timestamp: float
    caches_cleared: List[str]
    memory_freed_estimate_kb: float
    performance_impact: Dict[str, PerformanceImpactEntry]
    errors: List[str]
    garbage_collection: GarbageCollectionInfo
    summary: CacheSummary


# Core type system imports from shared types module
from ..core.types import (
    Coordinates,  # 2D coordinate representation for type annotations and parameter validation
)
from ..core.types import (
    CoordinateType,  # Type alias for flexible coordinate parameter validation
)
from ..core.types import (
    GridDimensions,  # Type alias for grid size parameters in factory functions
)
from ..core.types import (
    GridSize,  # Core data structures for coordinate and grid management; Grid dimension representation for type annotations and factory functions
)

# Concentration field data structure imports
from .concentration_field import (
    ConcentrationField,  # Core data structure for efficient 2D field management and sampling
)
from .concentration_field import (
    FieldGenerationError,  # Specialized exception for concentration field generation failures
)
from .concentration_field import (
    FieldSamplingError,  # Specialized exception for field sampling errors with position analysis
)
from .concentration_field import (
    create_concentration_field,  # Factory function for creating validated concentration field instances
)
from .concentration_field import (
    validate_field_parameters,  # Core concentration field class for efficient 2D field management; Factory and utility functions for concentration field operations; Specialized exceptions for concentration field operations; Comprehensive parameter validation for concentration field initialization
)

# Abstract plume model framework imports for extensibility
from .plume_model import (
    BasePlumeModel,  # Abstract base class providing common functionality for plume implementations
)
from .plume_model import (
    ModelRegistrationError,  # Exception for plume model registration failures with detailed context
)
from .plume_model import (
    PlumeModelError,  # General exception for plume model operation failures
)
from .plume_model import (
    PlumeModelInterface,  # Protocol interface for structural typing and duck typing compatibility
)
from .plume_model import (
    PlumeModelRegistry,  # Registry manager for plume model types with factory functionality
)
from .plume_model import (
    create_plume_model,  # Factory function for creating plume model instances with parameter validation
)
from .plume_model import (
    get_supported_plume_types,  # Utility function returning comprehensive information about supported plume types
)
from .plume_model import (
    register_plume_model,  # Abstract base classes and interfaces for plume model implementations; Registry system for plume model management and extensibility; Factory and utility functions for plume model operations; Specialized exceptions for plume model operations; Register custom plume model class with global registry
)

# Static Gaussian plume model implementation imports
from .static_gaussian import (
    GaussianPlumeError,  # Exception with mathematical analysis and recovery guidance
)
from .static_gaussian import (
    StaticGaussianPlume,  # Mathematical implementation providing concentration field calculations
)
from .static_gaussian import (
    calculate_gaussian_concentration,  # Pure mathematical function for Gaussian concentration calculations
)
from .static_gaussian import (
    create_static_gaussian_plume,  # Factory function for creating validated StaticGaussianPlume instances
)
from .static_gaussian import (
    validate_gaussian_parameters,  # Main static Gaussian plume model class with mathematical implementation; Factory and utility functions for Gaussian plume operations; Specialized exception for StaticGaussianPlume-specific errors; Comprehensive parameter validation for static Gaussian plume model
)

# Module version and metadata
__version__ = "0.0.1"
__author__ = "plume_nav_sim development team"

# Module-level constants and configuration
_DEFAULT_PLUME_TYPE = "static_gaussian"
_PLUME_MODULE_INITIALIZED = False
_module_lock = threading.Lock()  # Thread-safe initialization
_initialization_timestamp: Optional[float] = None
_registry_instance: Optional[PlumeModelRegistry] = None

# Performance monitoring and caching
_performance_cache: Dict[str, PerformanceEntry] = {}
_field_cache: Dict[str, object] = {}
_model_cache: Dict[str, object] = {}

# Module initialization logger
_logger = logging.getLogger(__name__)


def initialize_plume_module(
    register_builtin_models: bool = True, validate_dependencies: bool = True
) -> bool:
    """Initialize plume module with registry setup, built-in model registration, and validation
    of component availability for proper module functionality and extensibility support.

    This function performs comprehensive module initialization including dependency validation,
    registry setup, built-in model registration, and performance monitoring configuration.

    Args:
        register_builtin_models (bool): Whether to register built-in models (StaticGaussianPlume)
        validate_dependencies (bool): Whether to validate external dependencies (numpy, typing)

    Returns:
        bool: True if initialization successful, False if validation failures detected

    Raises:
        ImportError: If critical dependencies are missing and validate_dependencies is True
        RuntimeError: If module initialization fails due to configuration issues
    """
    global _PLUME_MODULE_INITIALIZED, _initialization_timestamp, _registry_instance

    # Thread-safe initialization using module lock
    with _module_lock:
        # Check if module already initialized using _PLUME_MODULE_INITIALIZED flag
        if _PLUME_MODULE_INITIALIZED:
            _logger.debug(
                "Plume module already initialized, skipping re-initialization"
            )
            return True

        try:
            _logger.info("Initializing plume module with comprehensive setup")
            initialization_start = time.time()

            # Validate external dependencies (numpy, typing) if validate_dependencies enabled
            if validate_dependencies:
                _logger.debug("Validating external dependencies")
                try:
                    import typing  # >=3.10 - Type system support for interfaces and protocols

                    import numpy  # >=2.1.0 - Mathematical operations and array handling

                    _logger.debug(
                        f"Dependencies validated - NumPy: {numpy.__version__}"
                    )
                except ImportError as e:
                    _logger.error(f"Critical dependency missing: {e}")
                    raise ImportError(
                        f"Required dependency missing for plume module: {e}"
                    )

            # Initialize PlumeModelRegistry with default configuration and thread safety
            _logger.debug("Initializing plume model registry")
            _registry_instance = PlumeModelRegistry()

            # Register built-in StaticGaussianPlume model if register_builtin_models enabled
            if register_builtin_models:
                _logger.debug("Registering built-in StaticGaussianPlume model")
                try:
                    _registry_instance.register_model(
                        model_type=_DEFAULT_PLUME_TYPE,
                        model_class=StaticGaussianPlume,
                        metadata={
                            "description": "Built-in static Gaussian plume model for RL environments"
                        },
                    )
                    _logger.info(
                        f"Successfully registered built-in model: {_DEFAULT_PLUME_TYPE}"
                    )
                except Exception as e:
                    _logger.error(f"Failed to register built-in model: {e}")
                    return False

            # Validate all registered models have proper interface compliance
            _logger.debug("Validating registered models for interface compliance")
            try:
                registered_models = _registry_instance.get_registered_models()
                for model_type, model_info in registered_models.items():
                    model_class = model_info.get("model_class")
                    if model_class and not issubclass(model_class, BasePlumeModel):
                        _logger.warning(
                            f"Model {model_type} does not inherit from BasePlumeModel"
                        )
                _logger.debug(
                    f"Interface validation completed for {len(registered_models)} models"
                )
            except Exception as e:
                _logger.warning(f"Model interface validation warning: {e}")

            # Initialize performance monitoring and caching systems
            _logger.debug("Initializing performance monitoring and caching")
            _performance_cache.clear()
            _field_cache.clear()
            _model_cache.clear()

            # Set initialization timestamp and flag
            _initialization_timestamp = time.time()
            _PLUME_MODULE_INITIALIZED = True

            initialization_duration = _initialization_timestamp - initialization_start

            # Log module initialization with registered model count and configuration
            registered_count = (
                len(_registry_instance.get_registered_models())
                if _registry_instance
                else 0
            )
            _logger.info(
                f"Plume module initialized successfully in {initialization_duration:.3f}s "
                f"with {registered_count} registered models"
            )

            # Return initialization success status with error handling for dependency failures
            return True

        except Exception as e:
            _logger.error(f"Plume module initialization failed: {e}")
            _PLUME_MODULE_INITIALIZED = False
            return False


def get_plume_module_info(
    include_model_details: bool = True, include_performance_data: bool = False
) -> Dict[str, object]:
    """Get comprehensive information about plume module including registered models, capabilities,
    version information, and performance characteristics for debugging and system analysis.

    This function provides detailed module information including model registry status,
    performance metrics, capabilities, and system configuration for debugging and monitoring.

    Args:
        include_model_details (bool): Whether to include detailed model information
        include_performance_data (bool): Whether to include performance benchmarks and optimization data

    Returns:
        dict: Complete plume module information with models, capabilities, and system status
    """
    # Compile basic module information including version, author, and initialization status
    module_info = {
        "module_name": __name__,
        "version": __version__,
        "author": __author__,
        "initialized": _PLUME_MODULE_INITIALIZED,
        "initialization_timestamp": _initialization_timestamp,
        "default_plume_type": _DEFAULT_PLUME_TYPE,
    }

    # Add initialization status and timing information
    if _initialization_timestamp:
        current_time = time.time()
        uptime = current_time - _initialization_timestamp
        module_info["uptime_seconds"] = uptime

    # Add registered model information if include_model_details enabled using get_supported_plume_types
    if include_model_details and _registry_instance:
        try:
            _logger.debug("Gathering detailed model information")
            registered_models = _registry_instance.get_registered_models()
            model_details = {}

            for model_type, model_info in registered_models.items():
                model_details[model_type] = {
                    "model_class": (
                        model_info.get("model_class").__name__
                        if model_info.get("model_class")
                        else "Unknown"
                    ),
                    "description": model_info.get(
                        "description", "No description available"
                    ),
                    "registration_timestamp": model_info.get(
                        "registration_timestamp", "Unknown"
                    ),
                    "interface_compliance": issubclass(
                        model_info.get("model_class", type), BasePlumeModel
                    ),
                }

            module_info["registered_models"] = model_details
            module_info["model_count"] = len(registered_models)

            # Include supported plume types information
            module_info["supported_types"] = get_supported_plume_types()

        except Exception as e:
            _logger.warning(f"Error gathering model details: {e}")
            module_info["model_details_error"] = str(e)

    # Include performance benchmarks and optimization data if include_performance_data enabled
    if include_performance_data:
        try:
            _logger.debug("Gathering performance data")
            performance_data = {
                "cache_sizes": {
                    "performance_cache": len(_performance_cache),
                    "field_cache": len(_field_cache),
                    "model_cache": len(_model_cache),
                },
                "memory_usage_estimates": {
                    "cache_memory_kb": _estimate_cache_memory_kb(),
                    "registry_memory_kb": _estimate_registry_memory_kb(),
                },
            }

            # Include recent performance metrics if available
            if _performance_cache:
                recent_metrics: Dict[str, Dict[str, float | int]] = {}
                for operation, metrics in _performance_cache.items():
                    if "last_execution_time" in metrics:
                        recent_metrics[operation] = {
                            "last_execution_ms": metrics["last_execution_time"]
                            * 1000.0,
                            "execution_count": metrics.get("execution_count", 0),
                        }
                performance_data["recent_metrics"] = cast(
                    Dict[str, object], recent_metrics
                )

            module_info["performance_data"] = performance_data

        except Exception as e:
            _logger.warning(f"Error gathering performance data: {e}")
            module_info["performance_data_error"] = str(e)

    # Add component availability status and dependency validation results
    module_info["component_availability"] = {
        "static_gaussian_available": "StaticGaussianPlume" in globals(),
        "base_model_available": "BasePlumeModel" in globals(),
        "concentration_field_available": "ConcentrationField" in globals(),
        "registry_available": _registry_instance is not None,
        "type_system_available": "Coordinates" in globals() and "GridSize" in globals(),
    }

    # Include module capabilities and supported features for analysis
    capabilities = {
        "model_types_supported": [_DEFAULT_PLUME_TYPE],
        "factory_functions_available": [
            "create_static_gaussian_plume",
            "create_plume_model",
            "create_concentration_field",
        ],
        "validation_functions_available": [
            "validate_gaussian_parameters",
            "validate_field_parameters",
        ],
        "extensibility_features": [
            "custom_model_registration",
            "plugin_architecture",
            "protocol_based_interfaces",
        ],
        "performance_optimizations": [
            "caching_system",
            "memory_management",
            "thread_safe_operations",
        ],
    }

    if _registry_instance:
        registered_models = _registry_instance.get_registered_models()
        capabilities["model_types_supported"] = list(registered_models.keys())

    module_info["capabilities"] = capabilities

    # Return comprehensive module information dictionary for debugging and monitoring
    return module_info


def clear_plume_caches(
    clear_field_cache: bool = True,
    clear_model_cache: bool = True,
    force_gc: bool = False,
) -> CacheReport:
    """Clear all plume-related caches including field generation cache, model instance cache,
    and performance data for memory optimization and testing scenarios.

    This function provides comprehensive cache management for memory optimization,
    testing scenarios, and system maintenance with detailed reporting of cleared resources.

    Args:
        clear_field_cache (bool): Whether to clear concentration field cache
        clear_model_cache (bool): Whether to clear model instance cache
        force_gc (bool): Whether to force garbage collection for complete memory cleanup

    Returns:
        dict: Cache clearing report with memory freed and performance impact analysis
    """
    cache_report: CacheReport = {
        "operation_timestamp": time.time(),
        "caches_cleared": [],
        "memory_freed_estimate_kb": 0,
        "performance_impact": {},
        "errors": [],
    }

    try:
        _logger.debug("Starting cache clearing operation")

        # Clear concentration field cache if clear_field_cache enabled using clear_field_cache function
        if clear_field_cache:
            try:
                field_cache_size = len(_field_cache)
                field_memory_estimate = _estimate_cache_memory_kb(_field_cache)

                _field_cache.clear()
                cache_report["caches_cleared"].append("field_cache")
                cache_report["memory_freed_estimate_kb"] += field_memory_estimate
                cache_report["performance_impact"]["field_cache"] = {
                    "items_cleared": field_cache_size,
                    "impact": "Subsequent field generation may be slower until cache rebuilds",
                }
                _logger.debug(
                    f"Cleared field cache: {field_cache_size} items, ~{field_memory_estimate}KB"
                )
            except Exception as e:
                cache_report["errors"].append(f"Error clearing field cache: {e}")
                _logger.error(f"Error clearing field cache: {e}")

        # Clear model registry cache if clear_model_cache enabled
        if clear_model_cache:
            try:
                model_cache_size = len(_model_cache)
                model_memory_estimate = _estimate_cache_memory_kb(_model_cache)

                _model_cache.clear()
                cache_report["caches_cleared"].append("model_cache")
                cache_report["memory_freed_estimate_kb"] += model_memory_estimate
                cache_report["performance_impact"]["model_cache"] = {
                    "items_cleared": model_cache_size,
                    "impact": "Model instantiation may be slower until cache rebuilds",
                }
                _logger.debug(
                    f"Cleared model cache: {model_cache_size} items, ~{model_memory_estimate}KB"
                )
            except Exception as e:
                cache_report["errors"].append(f"Error clearing model cache: {e}")
                _logger.error(f"Error clearing model cache: {e}")

        # Clear Gaussian calculation cache from static_gaussian module
        try:
            # Clear performance cache for operation timing data
            performance_cache_size = len(_performance_cache)
            performance_memory_estimate = _estimate_cache_memory_kb(_performance_cache)

            _performance_cache.clear()
            cache_report["caches_cleared"].append("performance_cache")
            cache_report["memory_freed_estimate_kb"] += performance_memory_estimate
            cache_report["performance_impact"]["performance_cache"] = {
                "items_cleared": performance_cache_size,
                "impact": "Performance metrics will be reset",
            }
            _logger.debug(f"Cleared performance cache: {performance_cache_size} items")
        except Exception as e:
            cache_report["errors"].append(f"Error clearing performance cache: {e}")
            _logger.error(f"Error clearing performance cache: {e}")

        # Force garbage collection if force_gc enabled for complete memory cleanup
        if force_gc:
            try:
                import gc

                collected_objects = gc.collect()
                cache_report["garbage_collection"] = {
                    "forced": True,
                    "objects_collected": collected_objects,
                    "impact": "Full memory cleanup performed",
                }
                _logger.debug(
                    f"Forced garbage collection: {collected_objects} objects collected"
                )
            except Exception as e:
                cache_report["errors"].append(f"Error during garbage collection: {e}")
                _logger.error(f"Error during garbage collection: {e}")

        # Calculate total memory freed from all cache clearing operations
        total_memory_freed: float = cache_report["memory_freed_estimate_kb"]
        cache_report["summary"] = {
            "total_caches_cleared": len(cache_report["caches_cleared"]),
            "total_memory_freed_kb": total_memory_freed,
            "total_memory_freed_mb": total_memory_freed / 1024,
            "operation_successful": len(cache_report["errors"]) == 0,
        }

        # Log cache clearing operation with memory freed and performance impact
        if cache_report["summary"]["operation_successful"]:
            _logger.info(
                f"Cache clearing completed successfully: "
                f"{cache_report['summary']['total_caches_cleared']} caches cleared, "
                f"~{cache_report['summary']['total_memory_freed_mb']:.2f}MB freed"
            )
        else:
            _logger.warning(
                f"Cache clearing completed with {len(cache_report['errors'])} errors: "
                f"{cache_report['errors']}"
            )

    except Exception as e:
        cache_report["errors"].append(f"Critical error during cache clearing: {e}")
        cache_report["summary"] = {
            "operation_successful": False,
            "critical_error": str(e),
        }
        _logger.error(f"Critical error during cache clearing: {e}")

    # Return comprehensive cache clearing report with statistics and analysis
    return cache_report


# Internal utility functions for module management


def _estimate_cache_memory_kb(
    cache_dict: Optional[Mapping[str, object]] = None,
) -> float:
    """Estimate memory usage of cache dictionary in kilobytes."""
    if cache_dict is None:
        return 0.0

    try:
        import sys

        total_size = 0
        for key, value in cache_dict.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(value)
        return total_size / 1024  # Convert bytes to KB
    except Exception:
        return float(len(cache_dict))  # Fallback estimate


def _estimate_registry_memory_kb() -> float:
    """Estimate memory usage of model registry in kilobytes."""
    if not _registry_instance:
        return 0.0

    try:
        registered_models = _registry_instance.get_registered_models()
        return _estimate_cache_memory_kb(registered_models)
    except Exception:
        return 0.0


# Comprehensive public interface for plume navigation simulation
__all__ = [
    # Core plume model classes for mathematical implementations and abstract frameworks
    "StaticGaussianPlume",  # Main static Gaussian plume model implementation
    "BasePlumeModel",  # Abstract base class providing common functionality
    "PlumeModelInterface",  # Protocol interface defining structural contract
    "ConcentrationField",  # Core concentration field data structure
    # Factory functions for creating validated instances with optimized configuration
    "create_static_gaussian_plume",  # Factory function for StaticGaussianPlume instances
    "create_plume_model",  # Generic factory function for plume model instances
    "create_concentration_field",  # Factory function for concentration field instances
    # Mathematical and utility functions for plume operations and validation
    "calculate_gaussian_concentration",  # Pure mathematical function for Gaussian calculations
    "validate_gaussian_parameters",  # Parameter validation for static Gaussian plume model
    "validate_field_parameters",  # Parameter validation for concentration field initialization
    "get_supported_plume_types",  # Utility function for supported plume model information
    "register_plume_model",  # Function for registering custom plume model classes
    # Registry system for extensible plume model management
    "PlumeModelRegistry",  # Registry manager for plume model types
    # Specialized exceptions for plume-specific error handling with recovery guidance
    "GaussianPlumeError",  # StaticGaussianPlume-specific errors with mathematical analysis
    "PlumeModelError",  # General plume model operation failures with diagnostics
    "ModelRegistrationError",  # Plume model registration failures with context
    "FieldGenerationError",  # Concentration field generation failures with recovery suggestions
    "FieldSamplingError",  # Field sampling errors with position analysis
    # Core type system for type annotations and parameter validation
    "Coordinates",  # 2D coordinate representation for plume operations
    "GridSize",  # Grid dimension representation for factory functions
    "CoordinateType",  # Type alias for flexible coordinate parameter validation
    "GridDimensions",  # Type alias for grid size parameters in factory functions
    # Module management functions for initialization and maintenance
    "initialize_plume_module",  # Initialize plume module with comprehensive setup
    "get_plume_module_info",  # Get comprehensive module information for debugging
    "clear_plume_caches",  # Clear plume-related caches for memory optimization
]


# Auto-initialize module on import if not already initialized
if not _PLUME_MODULE_INITIALIZED:
    try:
        initialize_plume_module(
            register_builtin_models=True, validate_dependencies=False
        )
    except Exception as e:
        _logger.warning(
            f"Auto-initialization failed: {e}. Call initialize_plume_module() manually."
        )
