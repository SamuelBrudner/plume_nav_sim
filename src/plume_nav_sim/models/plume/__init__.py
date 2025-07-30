"""
Plume Models Package: Modular odor plume simulation with pluggable implementations.

This package provides the core infrastructure for the modular plume modeling architecture
enabling seamless switching between different simulation approaches while maintaining
consistent interfaces and performance requirements. The modular design supports both
fast analytical models and realistic physics-based simulations for diverse research needs.

Key Features:
- Protocol-based architecture for seamless model substitution
- Configuration-driven instantiation via factory patterns  
- Performance-optimized implementations with <10ms step execution
- Hydra integration for structured configuration management
- Backward compatibility with existing VideoPlume workflows
- Runtime model discovery and validation through plugin registry

Modular Architecture Components:
- GaussianPlumeModel: Fast analytical dispersion using Gaussian equations
- TurbulentPlumeModel: Realistic filament-based turbulent physics simulation
- VideoPlumeAdapter: Backward-compatible wrapper for video-based plume data
- Plugin registry enabling runtime discovery and validation of implementations
- Factory functions supporting both programmatic and configuration-driven instantiation

Performance Characteristics:
- concentration_at(): <1ms single agent, <10ms for 100 concurrent agents
- step(): <5ms per time step for real-time simulation compatibility
- Memory efficiency: <100MB for typical simulation scenarios
- Zero-copy operations and vectorized computations for scalability

Usage Patterns:
    Direct instantiation:
    >>> from plume_nav_sim.models.plume import GaussianPlumeModel, TurbulentPlumeModel
    >>> gaussian_plume = GaussianPlumeModel(source_position=(50, 50), source_strength=1000)
    >>> turbulent_plume = TurbulentPlumeModel(filament_count=500, turbulence_intensity=0.3)
    
    Factory-based creation:
    >>> from plume_nav_sim.models.plume import create_plume_model, AVAILABLE_PLUME_MODELS
    >>> config = {'type': 'GaussianPlumeModel', 'source_position': (25, 75)}
    >>> plume_model = create_plume_model(config)
    
    Configuration-driven instantiation:
    >>> from hydra import compose, initialize
    >>> with initialize(config_path="../conf"):
    ...     cfg = compose(config_name="config")
    ...     plume_model = create_plume_model(cfg.plume_model)
    
    Environment integration:
    >>> from plume_nav_sim.envs import PlumeNavigationEnv  
    >>> env = PlumeNavigationEnv.from_config({
    ...     "plume_model": {"type": "TurbulentPlumeModel", "turbulence_intensity": 0.2}
    ... })
    
    Model discovery and validation:
    >>> print(f"Available models: {list(AVAILABLE_PLUME_MODELS.keys())}")
    >>> model_info = get_model_info('GaussianPlumeModel')
    >>> print(f"Model features: {model_info['features']}")

Technical Integration:
- Implements PlumeModelProtocol from core.protocols for type safety
- Integrates with WindFieldProtocol for realistic transport dynamics  
- Supports SensorProtocol integration for configurable observation modalities
- Maintains compatibility with NavigatorProtocol and GymnasiumEnv interfaces
- Provides seamless Hydra configuration management with structured configs
"""

from __future__ import annotations
import warnings
import logging
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from pathlib import Path

# Core imports with graceful fallbacks during migration
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Protocol imports for type safety and interface compliance
try:
    from ...core.protocols import PlumeModelProtocol
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Fallback during migration - create minimal protocol interface
    from typing import Protocol
    
    class PlumeModelProtocol(Protocol):
        """Minimal fallback protocol interface during migration."""
        def concentration_at(self, positions) -> Any: ...
        def step(self, dt: float = 1.0) -> None: ...
        def reset(self, **kwargs: Any) -> None: ...
    
    PROTOCOLS_AVAILABLE = False

# Configuration management
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False

# Enhanced logging for debugging and performance monitoring
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Import all plume model implementations with graceful fallbacks
PLUME_MODEL_IMPLEMENTATIONS = {}  # Registry for successful imports

# GaussianPlumeModel implementation
try:
    from .gaussian_plume import GaussianPlumeModel, GaussianPlumeConfig, create_gaussian_plume_model
    PLUME_MODEL_IMPLEMENTATIONS['GaussianPlumeModel'] = {
        'class': GaussianPlumeModel,
        'config_class': GaussianPlumeConfig,
        'factory_function': create_gaussian_plume_model,
        'description': 'Fast analytical plume modeling using Gaussian dispersion equations',
        'features': ['analytical_solution', 'fast_computation', 'wind_integration', 'vectorized_operations'],
        'performance': {'concentration_query_ms': 0.1, 'step_time_ms': 1.0, 'memory_mb': 1},
        'use_cases': ['rapid_prototyping', 'algorithm_development', 'baseline_comparison'],
        'available': True
    }
    GAUSSIAN_PLUME_AVAILABLE = True
    logger.debug("GaussianPlumeModel imported successfully") if LOGURU_AVAILABLE else None
except ImportError as e:
    GAUSSIAN_PLUME_AVAILABLE = False
    logger.warning(f"GaussianPlumeModel not available: {e}") if LOGURU_AVAILABLE else None
    GaussianPlumeModel = None
    GaussianPlumeConfig = None
    create_gaussian_plume_model = None

# TurbulentPlumeModel implementation  
try:
    from .turbulent_plume import TurbulentPlumeModel, TurbulentPlumeConfig
    PLUME_MODEL_IMPLEMENTATIONS['TurbulentPlumeModel'] = {
        'class': TurbulentPlumeModel,
        'config_class': TurbulentPlumeConfig,
        'factory_function': None,  # Uses class constructor
        'description': 'Realistic filament-based turbulent physics simulation',
        'features': ['filament_tracking', 'turbulent_physics', 'intermittent_signals', 'eddy_interactions'],
        'performance': {'concentration_query_ms': 1.0, 'step_time_ms': 5.0, 'memory_mb': 100},
        'use_cases': ['realistic_simulation', 'biological_navigation', 'turbulence_research'],
        'available': True
    }
    TURBULENT_PLUME_AVAILABLE = True
    logger.debug("TurbulentPlumeModel imported successfully") if LOGURU_AVAILABLE else None
except ImportError as e:
    TURBULENT_PLUME_AVAILABLE = False
    logger.warning(f"TurbulentPlumeModel not available: {e}") if LOGURU_AVAILABLE else None
    TurbulentPlumeModel = None
    TurbulentPlumeConfig = None

# VideoPlumeAdapter implementation
try:
    from .video_plume_adapter import VideoPlumeAdapter, VideoPlumeConfig
    PLUME_MODEL_IMPLEMENTATIONS['VideoPlumeAdapter'] = {
        'class': VideoPlumeAdapter,
        'config_class': VideoPlumeConfig,
        'factory_function': None,  # Uses class constructor and from_config
        'description': 'Backward-compatible adapter for video-based plume data',
        'features': ['video_integration', 'frame_caching', 'preprocessing_pipeline', 'spatial_interpolation'],
        'performance': {'concentration_query_ms': 0.5, 'step_time_ms': 1.0, 'memory_mb': 200},
        'use_cases': ['experimental_data', 'validation_studies', 'legacy_compatibility'],
        'available': True
    }
    VIDEO_PLUME_ADAPTER_AVAILABLE = True
    logger.debug("VideoPlumeAdapter imported successfully") if LOGURU_AVAILABLE else None
except ImportError as e:
    VIDEO_PLUME_ADAPTER_AVAILABLE = False
    logger.warning(f"VideoPlumeAdapter not available: {e}") if LOGURU_AVAILABLE else None
    VideoPlumeAdapter = None
    VideoPlumeConfig = None

# Calculate total available implementations
TOTAL_IMPLEMENTATIONS = len(PLUME_MODEL_IMPLEMENTATIONS)
logger.info(f"Loaded {TOTAL_IMPLEMENTATIONS} plume model implementations") if LOGURU_AVAILABLE else None

# Ensure at least one implementation is available
if TOTAL_IMPLEMENTATIONS == 0:
    warnings.warn(
        "No plume model implementations are currently available. "
        "This may indicate missing dependencies or incomplete migration. "
        "At least one model implementation is required for functionality.",
        ImportWarning,
        stacklevel=2
    )

# Create the public registry for model discovery and validation
AVAILABLE_PLUME_MODELS: Dict[str, Dict[str, Any]] = PLUME_MODEL_IMPLEMENTATIONS.copy()
"""
Registry of available plume model implementations with metadata.

This registry enables runtime discovery, validation, and selection of plume models
based on feature requirements, performance characteristics, and use case suitability.

Registry Structure:
    Each model entry contains:
    - class: Model implementation class
    - config_class: Associated configuration schema
    - factory_function: Optional factory function for creation
    - description: Human-readable model description
    - features: List of supported features and capabilities
    - performance: Expected performance characteristics
    - use_cases: Recommended application scenarios
    - available: Boolean indicating successful import status

Examples:
    Model discovery:
    >>> available_models = list(AVAILABLE_PLUME_MODELS.keys())
    >>> print(f"Available models: {available_models}")
    
    Feature-based selection:
    >>> fast_models = [name for name, info in AVAILABLE_PLUME_MODELS.items() 
    ...                if info['performance']['concentration_query_ms'] < 1.0]
    
    Use case matching:
    >>> research_models = [name for name, info in AVAILABLE_PLUME_MODELS.items()
    ...                   if 'realistic_simulation' in info['use_cases']]
"""

# Model type mappings for configuration-driven instantiation
MODEL_TYPE_MAPPING = {
    'gaussian': 'GaussianPlumeModel',
    'turbulent': 'TurbulentPlumeModel', 
    'video': 'VideoPlumeAdapter',
    'analytical': 'GaussianPlumeModel',
    'physics_based': 'TurbulentPlumeModel',
    'file_based': 'VideoPlumeAdapter'
}
"""Mapping of user-friendly model type names to implementation classes."""


def create_plume_model(
    config: Union[Dict[str, Any], DictConfig], 
    validate_protocol: bool = True,
    **kwargs: Any
) -> PlumeModelProtocol:
    """
    Factory function for creating plume models from configuration.
    
    This function provides the primary interface for configuration-driven plume model
    instantiation with comprehensive parameter validation, type checking, and
    graceful error handling. Supports both Hydra-style and dictionary configurations.
    
    Args:
        config: Configuration dictionary or DictConfig specifying plume model
            parameters. Must include 'type' or '_target_' field for model selection.
            Supported configuration formats:
            - Hydra style: {'_target_': 'fully.qualified.ClassName', ...}
            - Factory style: {'type': 'ModelName', ...}
            - Friendly names: {'type': 'gaussian', ...}
        validate_protocol: Whether to validate PlumeModelProtocol compliance
            of the created instance (default: True).
        **kwargs: Additional parameters to override or supplement configuration.
            These take precedence over config values for flexibility.
            
    Returns:
        PlumeModelProtocol: Configured plume model instance implementing the
            complete protocol interface with validated parameters.
            
    Raises:
        ValueError: If configuration is invalid, incomplete, or specifies
            unknown model type.
        TypeError: If configuration format is unsupported.
        ImportError: If required model implementation is not available.
        RuntimeError: If model instantiation fails or protocol validation fails.
        
    Notes:
        Model Selection Priority:
        1. '_target_' field for Hydra instantiation (highest priority)
        2. 'type' field with exact class name matching
        3. 'type' field with friendly name mapping
        4. Default to GaussianPlumeModel if no type specified
        
        Parameter Merging:
        - kwargs override config parameters for runtime flexibility
        - Hydra interpolation is resolved before parameter validation
        - Type conversion is applied for numeric parameters
        
        Performance Considerations:
        - Model instances are created fresh for each call (no caching)
        - Configuration validation adds ~1ms overhead
        - Protocol compliance checking adds ~0.5ms overhead
        
    Examples:
        Hydra-style configuration:
        >>> config = {
        ...     '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
        ...     'source_position': (50, 50),
        ...     'source_strength': 1000.0,
        ...     'sigma_x': 5.0,
        ...     'sigma_y': 3.0
        ... }
        >>> plume_model = create_plume_model(config)
        
        Factory-style configuration:
        >>> config = {
        ...     'type': 'TurbulentPlumeModel',
        ...     'source_position': (25, 75),
        ...     'filament_count': 500,
        ...     'turbulence_intensity': 0.3,
        ...     'mean_wind_velocity': (2.0, 0.5)
        ... }
        >>> plume_model = create_plume_model(config)
        
        Friendly name configuration:
        >>> config = {'type': 'gaussian', 'source_strength': 2000}
        >>> plume_model = create_plume_model(config, sigma_x=10.0)
        
        Video-based plume configuration:
        >>> config = {
        ...     'type': 'video',
        ...     'video_path': 'data/plume_experiment.mp4',
        ...     'preprocessing_config': {'grayscale': True, 'blur_kernel': 3}
        ... }
        >>> plume_model = create_plume_model(config)
        
        Configuration with validation disabled:
        >>> plume_model = create_plume_model(config, validate_protocol=False)
    """
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    # Convert DictConfig to regular dict if needed
    if hasattr(config, 'to_container') and HYDRA_AVAILABLE:
        config_dict = config.to_container(resolve=True)
    else:
        config_dict = dict(config)
    
    # Apply parameter overrides from kwargs
    config_dict.update(kwargs)
    
    # Determine model type and validate availability
    model_type = None
    target_class = None
    
    # Priority 1: Hydra-style _target_ field
    if '_target_' in config_dict:
        if HYDRA_AVAILABLE:
            try:
                from hydra import utils as hydra_utils
                logger.debug(f"Using Hydra instantiation with target: {config_dict['_target_']}")
                plume_model = hydra_utils.instantiate(config_dict)
                
                # Validate protocol compliance if requested
                if validate_protocol and PROTOCOLS_AVAILABLE:
                    if not isinstance(plume_model, PlumeModelProtocol):
                        raise RuntimeError(
                            f"Instantiated model does not implement PlumeModelProtocol: "
                            f"{type(plume_model)}"
                        )
                
                logger.info(f"Successfully created plume model via Hydra: {type(plume_model).__name__}")
                return plume_model
            except Exception as e:
                raise RuntimeError(f"Hydra instantiation failed: {e}") from e
        else:
            warnings.warn(
                "Hydra not available but _target_ specified. "
                "Falling back to factory instantiation.",
                UserWarning
            )
    
    # Priority 2: Factory-style type field
    if 'type' in config_dict:
        specified_type = config_dict['type']
        
        # Check for exact class name match
        if specified_type in AVAILABLE_PLUME_MODELS:
            model_type = specified_type
            target_class = AVAILABLE_PLUME_MODELS[specified_type]['class']
        
        # Check for friendly name mapping
        elif specified_type in MODEL_TYPE_MAPPING:
            model_type = MODEL_TYPE_MAPPING[specified_type] 
            if model_type in AVAILABLE_PLUME_MODELS:
                target_class = AVAILABLE_PLUME_MODELS[model_type]['class']
            else:
                raise ImportError(
                    f"Mapped model type '{model_type}' is not available. "
                    f"Available models: {list(AVAILABLE_PLUME_MODELS.keys())}"
                )
        else:
            raise ValueError(
                f"Unknown plume model type: '{specified_type}'. "
                f"Available types: {list(AVAILABLE_PLUME_MODELS.keys())} "
                f"or friendly names: {list(MODEL_TYPE_MAPPING.keys())}"
            )
    
    # Priority 3: Default to GaussianPlumeModel
    else:
        if 'GaussianPlumeModel' in AVAILABLE_PLUME_MODELS:
            model_type = 'GaussianPlumeModel'
            target_class = AVAILABLE_PLUME_MODELS[model_type]['class']
            logger.debug("No model type specified, defaulting to GaussianPlumeModel")
        else:
            raise RuntimeError(
                "No model type specified and GaussianPlumeModel not available. "
                f"Please specify 'type' field. Available models: {list(AVAILABLE_PLUME_MODELS.keys())}"
            )
    
    # Remove internal configuration keys before instantiation
    model_params = {k: v for k, v in config_dict.items() 
                   if k not in ['type', '_target_']}
    
    # Instantiate the model with error handling
    try:
        logger.debug(f"Instantiating {model_type} with parameters: {list(model_params.keys())}")
        plume_model = target_class(**model_params)
        
        # Validate protocol compliance if requested
        if validate_protocol and PROTOCOLS_AVAILABLE:
            if not isinstance(plume_model, PlumeModelProtocol):
                raise RuntimeError(
                    f"Instantiated model does not implement PlumeModelProtocol: "
                    f"{type(plume_model)}"
                )
        
        logger.info(f"Successfully created {model_type} plume model")
        return plume_model
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {model_type} with parameters {model_params}. "
            f"Error: {e}"
        ) from e


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a specific plume model implementation.
    
    Args:
        model_type: Name of the plume model implementation to query.
            Must be a key in AVAILABLE_PLUME_MODELS registry.
            
    Returns:
        Dict[str, Any]: Comprehensive model information including:
            - description: Human-readable model description
            - features: List of supported features and capabilities
            - performance: Expected performance characteristics
            - use_cases: Recommended application scenarios
            - available: Import and availability status
            - class: Model implementation class (if available)
            - config_class: Associated configuration schema (if available)
            
    Raises:
        KeyError: If model_type is not found in the registry.
        
    Examples:
        Get Gaussian plume model information:
        >>> info = get_model_info('GaussianPlumeModel')
        >>> print(f"Description: {info['description']}")
        >>> print(f"Features: {info['features']}")
        >>> print(f"Expected query time: {info['performance']['concentration_query_ms']}ms")
        
        Check model availability:
        >>> info = get_model_info('TurbulentPlumeModel')
        >>> if info['available']:
        ...     print("TurbulentPlumeModel is available for use")
    """
    if model_type not in AVAILABLE_PLUME_MODELS:
        available_types = list(AVAILABLE_PLUME_MODELS.keys())
        raise KeyError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available_types}"
        )
    
    return AVAILABLE_PLUME_MODELS[model_type].copy()


def list_available_models() -> List[str]:
    """
    Get list of all available plume model implementation names.
    
    Returns:
        List[str]: List of model names that can be used with create_plume_model().
        
    Examples:
        List all available models:
        >>> models = list_available_models()
        >>> print(f"Available plume models: {models}")
        
        Check if specific model is available:
        >>> if 'GaussianPlumeModel' in list_available_models():
        ...     print("Gaussian plume modeling is available")
    """
    return list(AVAILABLE_PLUME_MODELS.keys())


def get_models_by_feature(feature: str) -> List[str]:
    """
    Get list of plume models that support a specific feature.
    
    Args:
        feature: Feature name to search for. Examples: 'fast_computation',
            'turbulent_physics', 'video_integration', 'wind_integration'.
            
    Returns:
        List[str]: List of model names that support the specified feature.
        
    Examples:
        Find models with fast computation:
        >>> fast_models = get_models_by_feature('fast_computation')
        >>> print(f"Fast computation models: {fast_models}")
        
        Find models with realistic physics:
        >>> physics_models = get_models_by_feature('turbulent_physics')
        >>> print(f"Physics-based models: {physics_models}")
    """
    matching_models = []
    for model_name, model_info in AVAILABLE_PLUME_MODELS.items():
        if feature in model_info.get('features', []):
            matching_models.append(model_name)
    return matching_models


def get_models_by_use_case(use_case: str) -> List[str]:
    """
    Get list of plume models recommended for a specific use case.
    
    Args:
        use_case: Use case name to search for. Examples: 'rapid_prototyping',
            'realistic_simulation', 'experimental_data', 'algorithm_development'.
            
    Returns:
        List[str]: List of model names recommended for the specified use case.
        
    Examples:
        Find models for rapid prototyping:
        >>> prototype_models = get_models_by_use_case('rapid_prototyping')
        >>> print(f"Prototyping models: {prototype_models}")
        
        Find models for realistic simulation:
        >>> simulation_models = get_models_by_use_case('realistic_simulation')
        >>> print(f"Simulation models: {simulation_models}")
    """
    matching_models = []
    for model_name, model_info in AVAILABLE_PLUME_MODELS.items():
        if use_case in model_info.get('use_cases', []):
            matching_models.append(model_name)
    return matching_models


def validate_model_config(
    config: Union[Dict[str, Any], DictConfig], 
    model_type: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate plume model configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate.
        model_type: Optional specific model type to validate against.
            If None, attempts to infer from config.
            
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
            - is_valid: True if configuration is valid
            - error_messages: List of validation errors (empty if valid)
            
    Examples:
        Validate complete configuration:
        >>> config = {'type': 'GaussianPlumeModel', 'source_position': (50, 50)}
        >>> is_valid, errors = validate_model_config(config)
        >>> if not is_valid:
        ...     print(f"Configuration errors: {errors}")
        
        Validate specific model configuration:
        >>> config = {'source_strength': 1000, 'sigma_x': 5.0}
        >>> is_valid, errors = validate_model_config(config, 'GaussianPlumeModel')
    """
    errors = []
    
    # Check for None configuration
    if config is None:
        errors.append("Configuration cannot be None")
        return False, errors
    
    # Convert to dict if needed
    if hasattr(config, 'to_container'):
        config_dict = config.to_container(resolve=True)
    else:
        config_dict = dict(config)
    
    # Determine model type for validation
    if model_type is None:
        if '_target_' in config_dict:
            # Extract class name from _target_ path
            target_path = config_dict['_target_']
            model_type = target_path.split('.')[-1]
        elif 'type' in config_dict:
            specified_type = config_dict['type']
            model_type = MODEL_TYPE_MAPPING.get(specified_type, specified_type)
        else:
            model_type = 'GaussianPlumeModel'  # Default
    
    # Check if model type is available
    if model_type not in AVAILABLE_PLUME_MODELS:
        errors.append(f"Unknown or unavailable model type: {model_type}")
        return False, errors
    
    # Perform model-specific validation
    model_info = AVAILABLE_PLUME_MODELS[model_type]
    
    # Check availability
    if not model_info['available']:
        errors.append(f"Model type {model_type} is not currently available")
    
    # Basic parameter validation based on model type
    if model_type == 'GaussianPlumeModel':
        # Validate Gaussian plume parameters
        if 'source_position' in config_dict:
            pos = config_dict['source_position']
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                errors.append("source_position must be a 2-element list or tuple")
        
        if 'sigma_x' in config_dict and config_dict['sigma_x'] <= 0:
            errors.append("sigma_x must be positive")
        
        if 'sigma_y' in config_dict and config_dict['sigma_y'] <= 0:
            errors.append("sigma_y must be positive")
    
    elif model_type == 'TurbulentPlumeModel':
        # Validate turbulent plume parameters
        if 'filament_count' in config_dict:
            count = config_dict['filament_count']
            if not isinstance(count, int) or count <= 0:
                errors.append("filament_count must be a positive integer")
        
        if 'turbulence_intensity' in config_dict:
            intensity = config_dict['turbulence_intensity']
            if not (0 <= intensity <= 1):
                errors.append("turbulence_intensity must be between 0 and 1")
    
    elif model_type == 'VideoPlumeAdapter':
        # Validate video plume parameters
        if 'video_path' in config_dict:
            video_path = config_dict['video_path']
            if not isinstance(video_path, str):
                errors.append("video_path must be a string")
            else:
                # Check if path exists (if it's not a placeholder)
                path_obj = Path(video_path)
                if not str(path_obj).startswith('<') and not path_obj.exists():
                    errors.append(f"Video file not found: {video_path}")
    
    # Return validation results
    is_valid = len(errors) == 0
    return is_valid, errors


def get_default_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific plume model type.
    
    Args:
        model_type: Name of the plume model implementation.
            
    Returns:
        Dict[str, Any]: Default configuration dictionary with sensible defaults
            for the specified model type.
            
    Raises:
        KeyError: If model_type is not found in the registry.
        
    Examples:
        Get default Gaussian plume configuration:
        >>> config = get_default_config('GaussianPlumeModel')
        >>> print(f"Default source position: {config['source_position']}")
        
        Create model with defaults and overrides:
        >>> config = get_default_config('TurbulentPlumeModel')
        >>> config['turbulence_intensity'] = 0.4  # Override default
        >>> plume_model = create_plume_model(config)
    """
    if model_type not in AVAILABLE_PLUME_MODELS:
        available_types = list(AVAILABLE_PLUME_MODELS.keys())
        raise KeyError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available_types}"
        )
    
    # Define default configurations for each model type
    defaults = {
        'GaussianPlumeModel': {
            'source_position': (50.0, 50.0),
            'source_strength': 1000.0,
            'sigma_x': 5.0,
            'sigma_y': 3.0,
            'background_concentration': 0.0,
            'max_concentration': 1.0,
            'wind_speed': 0.0,
            'wind_direction': 0.0
        },
        'TurbulentPlumeModel': {
            'source_position': (50.0, 50.0),
            'source_strength': 1000.0,
            'filament_count': 500,
            'turbulence_intensity': 0.2,
            'mean_wind_velocity': (2.0, 0.0),
            'diffusion_coefficient': 0.1,
            'filament_lifetime': 100.0
        },
        'VideoPlumeAdapter': {
            'video_path': 'path/to/plume_data.mp4',
            'temporal_mode': 'cyclic',
            'preprocessing_config': {
                'grayscale': True,
                'blur_kernel': 0,
                'normalize': True
            },
            'spatial_interpolation_config': {
                'method': 'bilinear',
                'boundary_mode': 'constant',
                'boundary_value': 0.0
            }
        }
    }
    
    return defaults.get(model_type, {}).copy()


# Performance monitoring and diagnostics
def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary for all available plume model implementations.
    
    Returns:
        Dict[str, Any]: Performance summary including model characteristics,
            expected execution times, and memory usage estimates.
            
    Examples:
        Display performance comparison:
        >>> summary = get_performance_summary()
        >>> for model, perf in summary['models'].items():
        ...     print(f"{model}: {perf['concentration_query_ms']}ms query time")
    """
    performance_data = {
        'total_models': len(AVAILABLE_PLUME_MODELS),
        'available_models': [name for name, info in AVAILABLE_PLUME_MODELS.items() 
                           if info['available']],
        'models': {}
    }
    
    for model_name, model_info in AVAILABLE_PLUME_MODELS.items():
        performance_data['models'][model_name] = {
            'available': model_info['available'],
            'performance': model_info['performance'],
            'features': model_info['features'],
            'description': model_info['description']
        }
    
    return performance_data


# Module initialization and integrity validation
def _validate_module_integrity() -> Dict[str, Any]:
    """
    Validate module integrity and component availability.
    
    Returns:
        Dict[str, Any]: Validation results with status and component details.
    """
    validation_results = {
        'status': 'healthy',
        'issues': [],
        'components': {
            'numpy_available': NUMPY_AVAILABLE,
            'protocols_available': PROTOCOLS_AVAILABLE,
            'hydra_available': HYDRA_AVAILABLE,
            'loguru_available': LOGURU_AVAILABLE
        },
        'implementations': {
            'total': len(AVAILABLE_PLUME_MODELS),
            'available': len([m for m in AVAILABLE_PLUME_MODELS.values() if m['available']]),
            'models': list(AVAILABLE_PLUME_MODELS.keys())
        }
    }
    
    # Check critical dependencies
    if not NUMPY_AVAILABLE:
        validation_results['issues'].append("NumPy not available - array operations will fail")
    
    if not PROTOCOLS_AVAILABLE:
        validation_results['issues'].append("Protocol interfaces not available - type safety limited")
    
    # Check implementation availability
    available_count = validation_results['implementations']['available']
    if available_count == 0:
        validation_results['issues'].append("No plume model implementations available")
        validation_results['status'] = 'critical'
    elif available_count < 2:
        validation_results['issues'].append("Limited plume model implementations available")
        validation_results['status'] = 'degraded'
    
    # Check for basic functionality
    if available_count > 0:
        try:
            # Test factory function with minimal config
            test_config = {'type': list(AVAILABLE_PLUME_MODELS.keys())[0]}
            test_model = create_plume_model(test_config, validate_protocol=False)
            validation_results['factory_test'] = 'passed'
        except Exception as e:
            validation_results['issues'].append(f"Factory function test failed: {e}")
            validation_results['factory_test'] = 'failed'
            validation_results['status'] = 'degraded'
    
    return validation_results


# Perform module integrity validation on import
_validation_results = _validate_module_integrity()
if _validation_results['status'] != 'healthy':
    warning_msg = (
        f"Plume models package validation {_validation_results['status']}: "
        f"{len(_validation_results['issues'])} issues detected. "
        f"Available implementations: {_validation_results['implementations']['available']}/"
        f"{_validation_results['implementations']['total']}"
    )
    warnings.warn(warning_msg, ImportWarning, stacklevel=2)

# Log successful initialization
if LOGURU_AVAILABLE:
    logger.info(
        f"Plume models package initialized successfully",
        total_implementations=_validation_results['implementations']['total'],
        available_implementations=_validation_results['implementations']['available'],
        status=_validation_results['status']
    )

# Define comprehensive public API for maximum flexibility
__all__ = [
    # Core model implementations (conditionally exported based on availability)
    *(["GaussianPlumeModel", "GaussianPlumeConfig", "create_gaussian_plume_model"] 
      if GAUSSIAN_PLUME_AVAILABLE else []),
    *(["TurbulentPlumeModel", "TurbulentPlumeConfig"] 
      if TURBULENT_PLUME_AVAILABLE else []),
    *(["VideoPlumeAdapter", "VideoPlumeConfig"] 
      if VIDEO_PLUME_ADAPTER_AVAILABLE else []),
    
    # Factory functions and registry
    "create_plume_model",
    "AVAILABLE_PLUME_MODELS", 
    "MODEL_TYPE_MAPPING",
    
    # Discovery and information functions
    "get_model_info",
    "list_available_models",
    "get_models_by_feature",
    "get_models_by_use_case",
    
    # Configuration utilities
    "validate_model_config",
    "get_default_config",
    
    # Performance and diagnostics
    "get_performance_summary",
    
    # Module metadata
    "TOTAL_IMPLEMENTATIONS",
]

# Module metadata for documentation and debugging
__version__ = "0.1.0"
__author__ = "Plume Navigation Team"
__description__ = "Modular plume model implementations with pluggable architecture"
__module_status__ = _validation_results['status']
__available_models__ = _validation_results['implementations']['available']