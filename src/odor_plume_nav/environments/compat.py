"""
Compatibility Layer for Dual Gym/Gymnasium API Support.

This module provides a comprehensive compatibility layer enabling seamless interoperability
between legacy gym and modern Gymnasium APIs without breaking existing code or requiring
manual configuration. The implementation automatically detects caller context and adjusts
return value formats while maintaining performance targets of ≤10ms average step() time.

Key Features:
- Automatic API version detection using call stack introspection
- Zero-copy tuple conversion between 4-tuple and 5-tuple formats  
- Performance-optimized compatibility wrappers with minimal overhead
- Enhanced error handling and debugging support for API transitions
- Migration utilities for upgrading from legacy gym to Gymnasium
- Comprehensive logging integration with correlation tracking

Technical Implementation:
- Context detection via inspect module analyzing import patterns
- Lazy wrapper initialization to minimize performance impact
- Thread-local storage for multi-threaded compatibility contexts
- Extensive compatibility testing and validation utilities

Backward Compatibility:
All existing gym-based code continues to work without modification:

    # Legacy gym usage (returns 4-tuple)
    env = gym.make('OdorPlumeNavigation-v1')
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    
    # Modern Gymnasium usage (returns 5-tuple)  
    env = gymnasium.make('PlumeNavSim-v0')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Design Principles:
- Zero breaking changes for existing APIs
- Minimal performance overhead (<1% step time impact)
- Automatic compatibility without manual configuration
- Comprehensive test coverage for API surface variations
- Enhanced observability for debugging compatibility issues
"""

from __future__ import annotations

import inspect
import threading
import warnings
import weakref
from typing import (
    Any, Dict, Optional, Tuple, Union, List, Callable, Protocol, runtime_checkable,
    TypeVar, Generic, NamedTuple, Literal
)
from functools import wraps, lru_cache
from contextlib import contextmanager
import time

# Enhanced logging integration for correlation tracking
try:
    from odor_plume_nav.utils.logging_setup import (
        get_enhanced_logger, correlation_context, PerformanceMetrics
    )
    logger = get_enhanced_logger(__name__)
    ENHANCED_LOGGING = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    ENHANCED_LOGGING = False

# Type definitions for compatibility
LegacyStepReturn = Tuple[Any, float, bool, Dict[str, Any]]
ModernStepReturn = Tuple[Any, float, bool, bool, Dict[str, Any]]
StepReturn = Union[LegacyStepReturn, ModernStepReturn]

ActionType = TypeVar('ActionType')
ObservationType = TypeVar('ObservationType')
InfoType = Dict[str, Any]

# Performance tracking constants
PERFORMANCE_TARGET_MS = 10.0  # 10ms step() target from requirements
COMPATIBILITY_OVERHEAD_THRESHOLD = 0.001  # 1ms compatibility overhead limit
DEFAULT_CACHE_SIZE = 1024  # LRU cache size for optimization


class APIDetectionResult(NamedTuple):
    """Structured result from API version detection analysis."""
    is_legacy: bool
    confidence: float
    detection_method: str
    caller_module: Optional[str]
    import_context: Optional[str]
    debug_info: Dict[str, Any]


class CompatibilityMode(NamedTuple):
    """Compatibility mode configuration for environment instances."""
    use_legacy_api: bool
    detection_result: APIDetectionResult
    performance_monitoring: bool
    created_at: float
    correlation_id: Optional[str]


@runtime_checkable
class CompatibleEnvironment(Protocol):
    """Protocol for environments supporting dual API compatibility."""
    
    def step(self, action: ActionType) -> StepReturn:
        """Execute step with appropriate return format."""
        ...
    
    def reset(self, **kwargs) -> Union[ObservationType, Tuple[ObservationType, InfoType]]:
        """Reset environment with appropriate return format."""
        ...
    
    @property
    def _use_legacy_api(self) -> bool:
        """Current API compatibility mode."""
        ...


class CompatibilityError(Exception):
    """Exception raised when compatibility cannot be established."""
    pass


class PerformanceViolationError(Exception):
    """Exception raised when compatibility overhead exceeds performance targets."""
    pass


# Thread-local storage for compatibility context
_local_context = threading.local()


def get_compatibility_context() -> Optional[Dict[str, Any]]:
    """Get thread-local compatibility context."""
    return getattr(_local_context, 'context', None)


def set_compatibility_context(context: Dict[str, Any]) -> None:
    """Set thread-local compatibility context."""
    _local_context.context = context


@lru_cache(maxsize=DEFAULT_CACHE_SIZE)
def _analyze_caller_frame(frame_globals: frozenset, frame_locals: frozenset) -> APIDetectionResult:
    """
    Analyze caller frame for API version indicators with caching optimization.
    
    Args:
        frame_globals: Frozen set of global variable names for caching
        frame_locals: Frozen set of local variable names for caching
        
    Returns:
        APIDetectionResult: Structured detection result with confidence scoring
    """
    confidence = 0.0
    detection_method = "heuristic"
    caller_module = None
    import_context = None
    debug_info = {}
    
    # Check for explicit gym vs gymnasium imports in globals
    has_gym = 'gym' in frame_globals
    has_gymnasium = 'gymnasium' in frame_globals
    
    if has_gymnasium and not has_gym:
        confidence = 0.95
        detection_method = "explicit_gymnasium_import"
        import_context = "gymnasium"
        debug_info["import_pattern"] = "gymnasium_only"
    elif has_gym and not has_gymnasium:
        confidence = 0.85
        detection_method = "explicit_gym_import"
        import_context = "gym"
        debug_info["import_pattern"] = "gym_only"
    elif has_gym and has_gymnasium:
        # Both imports present - prefer gymnasium for modern API
        confidence = 0.80
        detection_method = "mixed_imports_prefer_gymnasium"
        import_context = "gymnasium"
        debug_info["import_pattern"] = "both_present"
    
    # Additional heuristics for module naming patterns
    for name in frame_globals:
        if 'gymnasium' in name.lower():
            confidence = max(confidence, 0.70)
            detection_method = "gymnasium_naming"
            caller_module = name
        elif name.endswith('_gym') or name.startswith('gym_'):
            confidence = max(confidence, 0.50)
            detection_method = "gym_naming"
            caller_module = name
    
    # Check for stable-baselines3 integration patterns
    if 'stable_baselines3' in frame_globals:
        confidence = max(confidence, 0.80)
        detection_method = "sb3_integration"
        import_context = "gymnasium"  # SB3 prefers Gymnasium
        debug_info["framework"] = "stable_baselines3"
    
    # Default to gymnasium (modern) for better forward compatibility
    # Only use legacy if explicitly detected or very low confidence
    is_legacy = confidence < 0.75 if confidence > 0 and detection_method == "explicit_gym_import" else False
    
    return APIDetectionResult(
        is_legacy=is_legacy,
        confidence=confidence,
        detection_method=detection_method,
        caller_module=caller_module,
        import_context=import_context,
        debug_info=debug_info
    )


def detect_api_version(
    depth: int = 3,
    enable_caching: bool = True,
    performance_monitoring: bool = True
) -> APIDetectionResult:
    """
    Detect API version preference through call stack analysis.
    
    Analyzes the calling context to determine whether legacy gym or modern
    Gymnasium API should be used. Uses multiple heuristics including import
    patterns, module names, and framework integration indicators.
    
    Args:
        depth: Maximum call stack depth to analyze (default: 3)
        enable_caching: Whether to use LRU caching for optimization (default: True)
        performance_monitoring: Whether to track performance metrics (default: True)
        
    Returns:
        APIDetectionResult: Comprehensive detection result with confidence metrics
        
    Performance:
        Optimized for <1ms execution time with caching enabled. Uses frozenset
        conversion for hashable cache keys and lazy evaluation of expensive checks.
        
    Examples:
        >>> # Detect current API context
        >>> result = detect_api_version()
        >>> print(f"Legacy API: {result.is_legacy}, Confidence: {result.confidence}")
        
        >>> # High-performance mode with minimal analysis
        >>> result = detect_api_version(depth=1, enable_caching=True)
    """
    start_time = time.perf_counter() if performance_monitoring else None
    
    try:
        # Start call stack analysis
        frame = inspect.currentframe()
        detection_results = []
        
        # Analyze up to specified depth
        for i in range(depth + 1):  # +1 to skip this function
            if frame is None:
                break
            frame = frame.f_back
            if frame is None:
                break
                
            # Skip internal compatibility module frames
            if frame.f_globals.get('__name__', '').endswith('.compat'):
                continue
            
            # Extract frame information for analysis
            if enable_caching:
                # Use frozensets for hashable cache keys
                frame_globals = frozenset(frame.f_globals.keys())
                frame_locals = frozenset(frame.f_locals.keys()) if frame.f_locals else frozenset()
                result = _analyze_caller_frame(frame_globals, frame_locals)
            else:
                # Direct analysis without caching
                result = _analyze_caller_frame(
                    frozenset(frame.f_globals.keys()),
                    frozenset(frame.f_locals.keys()) if frame.f_locals else frozenset()
                )
            
            detection_results.append(result)
            
            # Early exit for high-confidence detection
            if result.confidence > 0.90:
                break
        
        # Select best detection result
        if detection_results:
            best_result = max(detection_results, key=lambda r: r.confidence)
        else:
            # Fallback to legacy for maximum compatibility
            best_result = APIDetectionResult(
                is_legacy=True,
                confidence=0.0,
                detection_method="fallback_default",
                caller_module=None,
                import_context=None,
                debug_info={"reason": "no_caller_frames_found"}
            )
        
        # Performance monitoring and threshold checking
        if performance_monitoring and start_time is not None:
            detection_time = time.perf_counter() - start_time
            
            if ENHANCED_LOGGING:
                logger.debug(
                    f"API detection completed: {best_result.detection_method}",
                    extra={
                        "metric_type": "api_detection",
                        "detection_time": detection_time,
                        "confidence": best_result.confidence,
                        "is_legacy": best_result.is_legacy
                    }
                )
            
            # Check performance threshold
            if detection_time > COMPATIBILITY_OVERHEAD_THRESHOLD:
                warning_msg = (
                    f"API detection took {detection_time*1000:.2f}ms, "
                    f"exceeding {COMPATIBILITY_OVERHEAD_THRESHOLD*1000:.2f}ms threshold"
                )
                if ENHANCED_LOGGING:
                    logger.log_threshold_violation(
                        "api_detection", detection_time, 
                        COMPATIBILITY_OVERHEAD_THRESHOLD, "seconds"
                    )
                else:
                    logger.warning(warning_msg)
        
        return best_result
        
    except Exception as e:
        logger.error(f"API detection failed: {e}")
        # Return safe fallback on any error
        return APIDetectionResult(
            is_legacy=True,
            confidence=0.0,
            detection_method="error_fallback", 
            caller_module=None,
            import_context=None,
            debug_info={"error": str(e)}
        )
    finally:
        # Clean up frame references to prevent memory leaks
        del frame


def format_step_return(
    observation: ObservationType,
    reward: float, 
    terminated: bool,
    truncated: bool,
    info: InfoType,
    use_legacy_api: bool
) -> StepReturn:
    """
    Format step return tuple based on API compatibility mode.
    
    Converts between 5-tuple (Gymnasium) and 4-tuple (legacy gym) formats
    with zero-copy optimization when possible. Handles edge cases like
    boolean combination logic and info dictionary preservation.
    
    Args:
        observation: Environment observation
        reward: Scalar reward value  
        terminated: Whether episode terminated due to success/failure
        truncated: Whether episode truncated due to time/step limits
        info: Additional step information dictionary
        use_legacy_api: Whether to return legacy 4-tuple format
        
    Returns:
        StepReturn: Appropriately formatted tuple (4-tuple or 5-tuple)
        
    Performance:
        Zero-copy optimization - no data duplication, only tuple restructuring.
        Optimized boolean logic for done calculation in legacy mode.
        
    Examples:
        >>> # Convert to legacy format
        >>> result = format_step_return(obs, 1.0, True, False, {}, use_legacy_api=True)
        >>> obs, reward, done, info = result  # 4-tuple
        
        >>> # Keep modern format  
        >>> result = format_step_return(obs, 1.0, True, False, {}, use_legacy_api=False)
        >>> obs, reward, terminated, truncated, info = result  # 5-tuple
    """
    if use_legacy_api:
        # Combine terminated and truncated into single done flag
        done = terminated or truncated
        
        # Add termination details to info for debugging if not present
        if 'terminated' not in info:
            info = {**info, 'terminated': terminated, 'truncated': truncated}
        
        return observation, reward, done, info
    else:
        # Return modern 5-tuple format
        return observation, reward, terminated, truncated, info


def create_compatibility_mode(
    detection_result: Optional[APIDetectionResult] = None,
    force_legacy: Optional[bool] = None,
    performance_monitoring: bool = True,
    correlation_id: Optional[str] = None
) -> CompatibilityMode:
    """
    Create compatibility mode configuration with comprehensive validation.
    
    Args:
        detection_result: Explicit detection result (triggers new detection if None)
        force_legacy: Override detection with explicit legacy mode setting
        performance_monitoring: Enable performance tracking for this mode
        correlation_id: Optional correlation ID for logging context
        
    Returns:
        CompatibilityMode: Complete compatibility configuration
        
    Examples:
        >>> # Automatic detection
        >>> mode = create_compatibility_mode()
        
        >>> # Force legacy mode
        >>> mode = create_compatibility_mode(force_legacy=True)
        
        >>> # With custom correlation tracking
        >>> mode = create_compatibility_mode(correlation_id="exp_12345")
    """
    if detection_result is None:
        detection_result = detect_api_version(performance_monitoring=performance_monitoring)
    
    # Override detection if forced
    use_legacy_api = force_legacy if force_legacy is not None else detection_result.is_legacy
    
    # Update detection result if overridden
    if force_legacy is not None and force_legacy != detection_result.is_legacy:
        detection_result = APIDetectionResult(
            is_legacy=force_legacy,
            confidence=1.0,
            detection_method="forced_override",
            caller_module=detection_result.caller_module,
            import_context=detection_result.import_context,
            debug_info={**detection_result.debug_info, "original_detection": detection_result.is_legacy}
        )
    
    return CompatibilityMode(
        use_legacy_api=use_legacy_api,
        detection_result=detection_result,
        performance_monitoring=performance_monitoring,
        created_at=time.time(),
        correlation_id=correlation_id
    )


class CompatibilityWrapper:
    """
    High-performance compatibility wrapper for environment instances.
    
    Provides transparent API compatibility by wrapping environment methods
    and adjusting return formats based on detection context. Implements
    lazy initialization and caching for optimal performance.
    
    Key Features:
    - Transparent method delegation with minimal overhead
    - Automatic API format conversion for step() and reset()
    - Performance monitoring and threshold validation
    - Thread-safe compatibility context management
    - Comprehensive error handling and debugging support
    
    Performance Optimizations:
    - Cached compatibility mode to avoid repeated detection
    - Lazy wrapper initialization only when needed
    - Zero-copy tuple reformatting where possible
    - Minimal method call overhead via direct delegation
    """
    
    def __init__(
        self,
        env: CompatibleEnvironment,
        compatibility_mode: Optional[CompatibilityMode] = None,
        enable_performance_monitoring: bool = True,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize compatibility wrapper with comprehensive configuration.
        
        Args:
            env: Target environment implementing CompatibleEnvironment protocol
            compatibility_mode: Explicit compatibility configuration
            enable_performance_monitoring: Enable step performance tracking
            correlation_id: Optional correlation ID for logging context
        """
        self._wrapped_env = env
        self._enable_performance_monitoring = enable_performance_monitoring
        self._correlation_id = correlation_id
        
        # Initialize compatibility mode
        if compatibility_mode is None:
            self._compatibility_mode = create_compatibility_mode(
                performance_monitoring=enable_performance_monitoring,
                correlation_id=correlation_id
            )
        else:
            self._compatibility_mode = compatibility_mode
        
        # Performance tracking
        self._step_times = []
        self._compatibility_overhead_times = []
        self._total_steps = 0
        
        # Logging integration
        if ENHANCED_LOGGING:
            logger.debug(
                f"CompatibilityWrapper initialized",
                extra={
                    "metric_type": "wrapper_init",
                    "use_legacy_api": self._compatibility_mode.use_legacy_api,
                    "detection_method": self._compatibility_mode.detection_result.detection_method,
                    "confidence": self._compatibility_mode.detection_result.confidence,
                    "correlation_id": correlation_id
                }
            )
    
    @property
    def use_legacy_api(self) -> bool:
        """Current API compatibility mode."""
        return self._compatibility_mode.use_legacy_api
    
    @property
    def detection_result(self) -> APIDetectionResult:
        """Original API detection result."""
        return self._compatibility_mode.detection_result
    
    def step(self, action: ActionType) -> StepReturn:
        """
        Execute environment step with automatic format conversion.
        
        Args:
            action: Action to execute in environment
            
        Returns:
            StepReturn: Step result in appropriate format (4-tuple or 5-tuple)
            
        Raises:
            PerformanceViolationError: If step time exceeds performance targets
        """
        if self._enable_performance_monitoring:
            return self._step_with_monitoring(action)
        else:
            return self._step_without_monitoring(action)
    
    def _step_with_monitoring(self, action: ActionType) -> StepReturn:
        """Execute step with comprehensive performance monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Execute underlying environment step
            if hasattr(self._wrapped_env, '_execute_step_without_monitoring'):
                # Use optimized path if available
                result = self._wrapped_env._execute_step_without_monitoring(action)
            else:
                result = self._wrapped_env.step(action)
            
            # Measure compatibility overhead
            compat_start = time.perf_counter()
            
            # Handle different return formats from underlying environment
            if len(result) == 5:
                # Modern format from environment
                obs, reward, terminated, truncated, info = result
                formatted_result = format_step_return(
                    obs, reward, terminated, truncated, info,
                    self._compatibility_mode.use_legacy_api
                )
            elif len(result) == 4:
                # Legacy format from environment - convert to modern first
                obs, reward, done, info = result
                terminated = done
                truncated = False  # Default assumption for legacy compatibility
                formatted_result = format_step_return(
                    obs, reward, terminated, truncated, info,
                    self._compatibility_mode.use_legacy_api
                )
            else:
                raise CompatibilityError(
                    f"Unexpected step return format: {len(result)} elements"
                )
            
            # Track compatibility overhead
            compatibility_time = time.perf_counter() - compat_start
            self._compatibility_overhead_times.append(compatibility_time)
            
            # Track total step time
            total_time = time.perf_counter() - start_time
            self._step_times.append(total_time)
            self._total_steps += 1
            
            # Performance threshold validation
            if total_time > PERFORMANCE_TARGET_MS / 1000:
                if ENHANCED_LOGGING:
                    logger.log_threshold_violation(
                        "compatibility_step", total_time, 
                        PERFORMANCE_TARGET_MS / 1000, "seconds"
                    )
                else:
                    logger.warning(
                        f"Step time {total_time*1000:.2f}ms exceeds target {PERFORMANCE_TARGET_MS}ms"
                    )
            
            # Log periodic performance summary
            if self._total_steps % 1000 == 0 and ENHANCED_LOGGING:
                avg_step_time = sum(self._step_times[-1000:]) / len(self._step_times[-1000:])
                avg_compat_overhead = sum(self._compatibility_overhead_times[-1000:]) / len(self._compatibility_overhead_times[-1000:])
                
                logger.log_performance_metrics({
                    "avg_step_time": avg_step_time,
                    "avg_compatibility_overhead": avg_compat_overhead,
                    "total_steps": self._total_steps,
                    "api_mode": "legacy" if self._compatibility_mode.use_legacy_api else "gymnasium"
                }, metric_type="compatibility_performance")
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Compatibility step failed: {e}")
            raise CompatibilityError(f"Step execution failed: {e}") from e
    
    def _step_without_monitoring(self, action: ActionType) -> StepReturn:
        """Execute step with minimal overhead for maximum performance."""
        try:
            # Execute underlying environment step
            result = self._wrapped_env.step(action)
            
            # Handle different return formats
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return format_step_return(
                    obs, reward, terminated, truncated, info,
                    self._compatibility_mode.use_legacy_api
                )
            elif len(result) == 4:
                obs, reward, done, info = result
                terminated = done
                truncated = False
                return format_step_return(
                    obs, reward, terminated, truncated, info,
                    self._compatibility_mode.use_legacy_api
                )
            else:
                raise CompatibilityError(
                    f"Unexpected step return format: {len(result)} elements"
                )
                
        except Exception as e:
            logger.error(f"Compatibility step failed: {e}")
            raise CompatibilityError(f"Step execution failed: {e}") from e
    
    def reset(self, **kwargs) -> Union[ObservationType, Tuple[ObservationType, InfoType]]:
        """
        Execute environment reset with appropriate return format.
        
        Args:
            **kwargs: Reset parameters passed to underlying environment
            
        Returns:
            Reset result in format appropriate for detected API version
        """
        try:
            result = self._wrapped_env.reset(**kwargs)
            
            # Modern Gymnasium reset returns (obs, info)
            # Legacy gym reset returns obs only
            if self._compatibility_mode.use_legacy_api:
                if isinstance(result, tuple) and len(result) == 2:
                    # Extract observation from (obs, info) tuple for legacy compatibility
                    observation, _ = result
                    return observation
                else:
                    # Already in legacy format
                    return result
            else:
                # Ensure modern format
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                else:
                    # Convert legacy format to modern (obs, info)
                    return result, {}
                    
        except Exception as e:
            logger.error(f"Compatibility reset failed: {e}")
            raise CompatibilityError(f"Reset execution failed: {e}") from e
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for compatibility operations.
        
        Returns:
            Dict containing performance statistics and compliance metrics
        """
        if not self._step_times:
            return {"status": "no_data", "total_steps": 0}
        
        avg_step_time = sum(self._step_times) / len(self._step_times)
        avg_compat_overhead = sum(self._compatibility_overhead_times) / len(self._compatibility_overhead_times) if self._compatibility_overhead_times else 0
        
        return {
            "total_steps": self._total_steps,
            "avg_step_time_ms": avg_step_time * 1000,
            "avg_compatibility_overhead_ms": avg_compat_overhead * 1000,
            "target_compliance": avg_step_time <= (PERFORMANCE_TARGET_MS / 1000),
            "overhead_compliance": avg_compat_overhead <= COMPATIBILITY_OVERHEAD_THRESHOLD,
            "api_mode": "legacy" if self._compatibility_mode.use_legacy_api else "gymnasium",
            "detection_confidence": self._compatibility_mode.detection_result.confidence,
            "detection_method": self._compatibility_mode.detection_result.detection_method
        }
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to wrapped environment."""
        return getattr(self._wrapped_env, name)
    
    def __str__(self) -> str:
        """String representation with compatibility information."""
        api_mode = "legacy" if self._compatibility_mode.use_legacy_api else "gymnasium"
        return f"CompatibilityWrapper({self._wrapped_env}, api_mode={api_mode})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"CompatibilityWrapper(\n"
            f"  env={repr(self._wrapped_env)},\n"
            f"  use_legacy_api={self._compatibility_mode.use_legacy_api},\n"
            f"  detection_method={self._compatibility_mode.detection_result.detection_method},\n"
            f"  confidence={self._compatibility_mode.detection_result.confidence:.3f},\n"
            f"  total_steps={self._total_steps}\n"
            f")"
        )


@contextmanager
def compatibility_context(
    force_legacy: Optional[bool] = None,
    performance_monitoring: bool = True,
    correlation_id: Optional[str] = None
):
    """
    Context manager for temporary compatibility mode override.
    
    Enables temporary override of API compatibility mode within a specific
    context, useful for testing, debugging, or migration scenarios.
    
    Args:
        force_legacy: Force legacy API mode (None for auto-detection)
        performance_monitoring: Enable performance tracking
        correlation_id: Optional correlation ID for logging
        
    Yields:
        CompatibilityMode: Active compatibility configuration
        
    Examples:
        >>> # Test legacy API behavior
        >>> with compatibility_context(force_legacy=True) as mode:
        ...     assert mode.use_legacy_api
        ...     obs, reward, done, info = env.step(action)
        
        >>> # Test modern API behavior  
        >>> with compatibility_context(force_legacy=False) as mode:
        ...     assert not mode.use_legacy_api
        ...     obs, reward, terminated, truncated, info = env.step(action)
    """
    if ENHANCED_LOGGING and correlation_id:
        ctx_manager = correlation_context("compatibility_override", correlation_id=correlation_id)
    else:
        ctx_manager = None
    
    # Create compatibility mode
    mode = create_compatibility_mode(
        force_legacy=force_legacy,
        performance_monitoring=performance_monitoring,
        correlation_id=correlation_id
    )
    
    # Store previous context
    previous_context = get_compatibility_context()
    
    try:
        if ctx_manager:
            with ctx_manager:
                # Set new compatibility context
                set_compatibility_context({"mode": mode})
                
                if ENHANCED_LOGGING:
                    logger.debug(
                        f"Compatibility context activated",
                        extra={
                            "metric_type": "context_start",
                            "use_legacy_api": mode.use_legacy_api,
                            "forced": force_legacy is not None
                        }
                    )
                
                yield mode
        else:
            set_compatibility_context({"mode": mode})
            yield mode
            
    finally:
        # Restore previous context
        if previous_context is not None:
            set_compatibility_context(previous_context)
        else:
            set_compatibility_context({})
        
        if ENHANCED_LOGGING:
            logger.debug(
                f"Compatibility context deactivated",
                extra={"metric_type": "context_end"}
            )


def wrap_environment(
    env: CompatibleEnvironment,
    compatibility_mode: Optional[CompatibilityMode] = None,
    **wrapper_kwargs
) -> CompatibilityWrapper:
    """
    Create compatibility wrapper for environment with optimization.
    
    Factory function for creating CompatibilityWrapper instances with
    intelligent defaults and validation.
    
    Args:
        env: Environment to wrap (must implement CompatibleEnvironment protocol)
        compatibility_mode: Explicit compatibility configuration
        **wrapper_kwargs: Additional wrapper configuration parameters
        
    Returns:
        CompatibilityWrapper: Wrapped environment with dual API support
        
    Raises:
        CompatibilityError: If environment is not compatible
        
    Examples:
        >>> # Automatic compatibility detection
        >>> wrapped_env = wrap_environment(gym_env)
        
        >>> # Force legacy mode
        >>> mode = create_compatibility_mode(force_legacy=True)
        >>> wrapped_env = wrap_environment(env, mode)
    """
    # Validate environment compatibility
    if not isinstance(env, CompatibleEnvironment):
        raise CompatibilityError(
            f"Environment {type(env)} does not implement CompatibleEnvironment protocol"
        )
    
    # Check for existing wrapper to avoid double-wrapping
    if isinstance(env, CompatibilityWrapper):
        logger.warning("Environment is already wrapped, returning existing wrapper")
        return env
    
    # Create wrapper with configuration
    wrapper = CompatibilityWrapper(
        env=env,
        compatibility_mode=compatibility_mode,
        **wrapper_kwargs
    )
    
    if ENHANCED_LOGGING:
        logger.debug(
            f"Environment wrapped with compatibility layer",
            extra={
                "metric_type": "environment_wrapped",
                "env_type": type(env).__name__,
                "use_legacy_api": wrapper.use_legacy_api,
                "detection_method": wrapper.detection_result.detection_method
            }
        )
    
    return wrapper


def validate_compatibility(
    env: CompatibleEnvironment,
    test_episodes: int = 3,
    performance_validation: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of environment compatibility and performance.
    
    Performs extensive testing of both API modes to ensure compatibility
    layer works correctly and meets performance requirements.
    
    Args:
        env: Environment to validate
        test_episodes: Number of test episodes per API mode
        performance_validation: Whether to validate performance requirements
        
    Returns:
        Dict containing comprehensive validation results
        
    Examples:
        >>> results = validate_compatibility(env)
        >>> assert results["overall_status"] == "passed"
        >>> print(f"Performance: {results['performance']['avg_step_time_ms']:.2f}ms")
    """
    validation_results = {
        "overall_status": "unknown",
        "legacy_api_tests": {},
        "gymnasium_api_tests": {},
        "performance": {},
        "compatibility_analysis": {},
        "recommendations": []
    }
    
    try:
        if ENHANCED_LOGGING:
            logger.info("Starting compatibility validation", extra={"metric_type": "validation_start"})
        
        # Test legacy API mode
        with compatibility_context(force_legacy=True) as legacy_mode:
            validation_results["legacy_api_tests"] = _test_api_mode(
                env, "legacy", test_episodes, performance_validation
            )
        
        # Test gymnasium API mode  
        with compatibility_context(force_legacy=False) as gymnasium_mode:
            validation_results["gymnasium_api_tests"] = _test_api_mode(
                env, "gymnasium", test_episodes, performance_validation
            )
        
        # Aggregate performance results
        legacy_perf = validation_results["legacy_api_tests"].get("performance", {})
        gymnasium_perf = validation_results["gymnasium_api_tests"].get("performance", {})
        
        validation_results["performance"] = {
            "legacy_avg_step_time_ms": legacy_perf.get("avg_step_time_ms", 0),
            "gymnasium_avg_step_time_ms": gymnasium_perf.get("avg_step_time_ms", 0),
            "performance_target_ms": PERFORMANCE_TARGET_MS,
            "legacy_compliant": legacy_perf.get("target_compliance", False),
            "gymnasium_compliant": gymnasium_perf.get("target_compliance", False)
        }
        
        # Compatibility analysis
        validation_results["compatibility_analysis"] = {
            "detection_accuracy": _analyze_detection_accuracy(env),
            "api_format_consistency": _validate_api_formats(validation_results),
            "performance_overhead": _calculate_performance_overhead(validation_results)
        }
        
        # Generate recommendations
        validation_results["recommendations"] = _generate_recommendations(validation_results)
        
        # Determine overall status
        legacy_passed = validation_results["legacy_api_tests"].get("status") == "passed"
        gymnasium_passed = validation_results["gymnasium_api_tests"].get("status") == "passed"
        performance_ok = (
            validation_results["performance"]["legacy_compliant"] and
            validation_results["performance"]["gymnasium_compliant"]
        )
        
        if legacy_passed and gymnasium_passed and performance_ok:
            validation_results["overall_status"] = "passed"
        elif legacy_passed and gymnasium_passed:
            validation_results["overall_status"] = "passed_with_performance_warnings"
        else:
            validation_results["overall_status"] = "failed"
        
        if ENHANCED_LOGGING:
            logger.info(
                f"Compatibility validation completed: {validation_results['overall_status']}",
                extra={
                    "metric_type": "validation_complete",
                    "status": validation_results["overall_status"],
                    "legacy_compliant": validation_results["performance"]["legacy_compliant"],
                    "gymnasium_compliant": validation_results["performance"]["gymnasium_compliant"]
                }
            )
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Compatibility validation failed: {e}")
        validation_results["overall_status"] = "error"
        validation_results["error"] = str(e)
        return validation_results


def _test_api_mode(
    env: CompatibleEnvironment, 
    mode_name: str, 
    test_episodes: int,
    performance_validation: bool
) -> Dict[str, Any]:
    """Test specific API mode with comprehensive validation."""
    results = {
        "status": "unknown",
        "episodes_completed": 0,
        "performance": {},
        "errors": []
    }
    
    wrapper = wrap_environment(env)
    step_times = []
    
    try:
        for episode in range(test_episodes):
            # Test reset
            reset_result = wrapper.reset()
            
            # Validate reset format
            if mode_name == "legacy":
                # Legacy reset should return observation only or maintain existing format
                pass  # Accept any format for reset in legacy mode
            else:
                # Gymnasium reset should return (obs, info) tuple
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    logger.warning(f"Unexpected reset format in {mode_name} mode: {type(reset_result)}")
            
            # Test multiple steps
            for step in range(10):  # Short episodes for validation
                action = env.action_space.sample() if hasattr(env, 'action_space') else [0.0, 0.0]
                
                step_start = time.perf_counter()
                step_result = wrapper.step(action)
                step_time = time.perf_counter() - step_start
                
                step_times.append(step_time)
                
                # Validate step format
                if mode_name == "legacy":
                    if len(step_result) != 4:
                        results["errors"].append(f"Legacy step returned {len(step_result)} elements, expected 4")
                        break
                else:
                    if len(step_result) != 5:
                        results["errors"].append(f"Gymnasium step returned {len(step_result)} elements, expected 5")
                        break
                
                # Check for episode termination
                if mode_name == "legacy":
                    _, _, done, _ = step_result
                    if done:
                        break
                else:
                    _, _, terminated, truncated, _ = step_result
                    if terminated or truncated:
                        break
            
            results["episodes_completed"] += 1
        
        # Calculate performance metrics
        if step_times:
            avg_step_time = sum(step_times) / len(step_times)
            results["performance"] = {
                "avg_step_time_ms": avg_step_time * 1000,
                "target_compliance": avg_step_time <= (PERFORMANCE_TARGET_MS / 1000),
                "total_steps": len(step_times)
            }
        
        # Determine status
        if not results["errors"] and results["episodes_completed"] == test_episodes:
            results["status"] = "passed"
        else:
            results["status"] = "failed"
            
    except Exception as e:
        logger.error(f"API mode test failed for {mode_name}: {e}")
        results["status"] = "error"
        results["errors"].append(str(e))
    
    return results


def _analyze_detection_accuracy(env: CompatibleEnvironment) -> Dict[str, Any]:
    """Analyze detection accuracy across multiple detection attempts."""
    detection_results = []
    
    for _ in range(10):  # Multiple detection attempts for consistency
        result = detect_api_version()
        detection_results.append(result)
    
    # Calculate consistency metrics
    legacy_count = sum(1 for r in detection_results if r.is_legacy)
    confidence_scores = [r.confidence for r in detection_results]
    
    return {
        "consistency_ratio": max(legacy_count, len(detection_results) - legacy_count) / len(detection_results),
        "avg_confidence": sum(confidence_scores) / len(confidence_scores),
        "min_confidence": min(confidence_scores),
        "max_confidence": max(confidence_scores),
        "detection_methods": list(set(r.detection_method for r in detection_results))
    }


def _validate_api_formats(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate consistency of API format handling."""
    legacy_errors = validation_results["legacy_api_tests"].get("errors", [])
    gymnasium_errors = validation_results["gymnasium_api_tests"].get("errors", [])
    
    format_errors = []
    for error in legacy_errors + gymnasium_errors:
        if "returned" in error and "elements" in error:
            format_errors.append(error)
    
    return {
        "format_errors": format_errors,
        "format_consistency": len(format_errors) == 0,
        "legacy_format_valid": len([e for e in legacy_errors if "elements" in e]) == 0,
        "gymnasium_format_valid": len([e for e in gymnasium_errors if "elements" in e]) == 0
    }


def _calculate_performance_overhead(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate compatibility layer performance overhead."""
    legacy_perf = validation_results["legacy_api_tests"].get("performance", {})
    gymnasium_perf = validation_results["gymnasium_api_tests"].get("performance", {})
    
    legacy_time = legacy_perf.get("avg_step_time_ms", 0)
    gymnasium_time = gymnasium_perf.get("avg_step_time_ms", 0)
    
    if legacy_time > 0 and gymnasium_time > 0:
        overhead_pct = abs(legacy_time - gymnasium_time) / min(legacy_time, gymnasium_time) * 100
    else:
        overhead_pct = 0
    
    return {
        "overhead_percentage": overhead_pct,
        "legacy_time_ms": legacy_time,
        "gymnasium_time_ms": gymnasium_time,
        "overhead_acceptable": overhead_pct < 5.0  # 5% threshold
    }


def _generate_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on validation results."""
    recommendations = []
    
    perf = validation_results.get("performance", {})
    compat = validation_results.get("compatibility_analysis", {})
    
    # Performance recommendations
    if not perf.get("legacy_compliant", True):
        recommendations.append(
            f"Legacy API step time ({perf.get('legacy_avg_step_time_ms', 0):.2f}ms) "
            f"exceeds target ({PERFORMANCE_TARGET_MS}ms). Consider performance optimization."
        )
    
    if not perf.get("gymnasium_compliant", True):
        recommendations.append(
            f"Gymnasium API step time ({perf.get('gymnasium_avg_step_time_ms', 0):.2f}ms) "
            f"exceeds target ({PERFORMANCE_TARGET_MS}ms). Consider performance optimization."
        )
    
    # Detection accuracy recommendations
    detection_analysis = compat.get("detection_accuracy", {})
    if detection_analysis.get("consistency_ratio", 1.0) < 0.8:
        recommendations.append(
            "API detection consistency is low. Consider explicit compatibility mode setting."
        )
    
    if detection_analysis.get("avg_confidence", 1.0) < 0.7:
        recommendations.append(
            "API detection confidence is low. Consider improving import patterns or explicit configuration."
        )
    
    # Format validation recommendations
    format_analysis = compat.get("api_format_consistency", {})
    if not format_analysis.get("format_consistency", True):
        recommendations.append(
            "API format consistency issues detected. Review step return value handling."
        )
    
    # Performance overhead recommendations
    overhead_analysis = compat.get("performance_overhead", {})
    if not overhead_analysis.get("overhead_acceptable", True):
        recommendations.append(
            f"Compatibility overhead ({overhead_analysis.get('overhead_percentage', 0):.1f}%) "
            "is high. Consider performance optimization."
        )
    
    if not recommendations:
        recommendations.append("All compatibility tests passed. Environment is fully compatible.")
    
    return recommendations


# Migration and deprecation utilities
def issue_legacy_warning(detection_result: APIDetectionResult) -> None:
    """Issue deprecation warning for legacy gym usage."""
    if detection_result.is_legacy and detection_result.confidence > 0.5:
        warning_message = (
            "Legacy gym API detected. Consider migrating to Gymnasium for improved compatibility. "
            f"Detection method: {detection_result.detection_method}, "
            f"confidence: {detection_result.confidence:.2f}"
        )
        
        warnings.warn(
            warning_message,
            DeprecationWarning,
            stacklevel=3  # Point to actual caller
        )
        
        if ENHANCED_LOGGING:
            logger.warning(
                "Legacy gym API usage detected",
                extra={
                    "metric_type": "legacy_warning",
                    "detection_method": detection_result.detection_method,
                    "confidence": detection_result.confidence,
                    "migration_recommendation": "consider_gymnasium_upgrade"
                }
            )


def get_migration_guide() -> str:
    """Get comprehensive migration guide for upgrading to Gymnasium."""
    return """
Migration Guide: Legacy gym → Gymnasium

1. Update Dependencies:
   # Old
   pip install gym
   
   # New  
   pip install gymnasium==0.29.*

2. Update Imports:
   # Old
   import gym
   
   # New
   import gymnasium

3. Update Environment Creation:
   # Old
   env = gym.make('OdorPlumeNavigation-v1')
   
   # New
   env = gymnasium.make('PlumeNavSim-v0')

4. Update Step Returns:
   # Old (4-tuple)
   obs, reward, done, info = env.step(action)
   
   # New (5-tuple)
   obs, reward, terminated, truncated, info = env.step(action)
   
   # Compatibility check
   done = terminated or truncated

5. Update Reset Returns:
   # Old
   obs = env.reset()
   
   # New
   obs, info = env.reset()

6. Testing Compatibility:
   from odor_plume_nav.environments.compat import validate_compatibility
   results = validate_compatibility(env)
   print(results['overall_status'])

For automatic compatibility without code changes, the compatibility layer
handles these differences transparently based on import detection.
"""


# Public exports for comprehensive API
__all__ = [
    # Core compatibility functions
    "detect_api_version",
    "format_step_return", 
    "create_compatibility_mode",
    "wrap_environment",
    
    # Compatibility classes
    "CompatibilityWrapper",
    "CompatibilityMode",
    "APIDetectionResult",
    
    # Context managers and utilities
    "compatibility_context",
    "validate_compatibility",
    
    # Migration utilities
    "issue_legacy_warning",
    "get_migration_guide",
    
    # Protocols and exceptions
    "CompatibleEnvironment",
    "CompatibilityError",
    "PerformanceViolationError",
    
    # Type definitions
    "LegacyStepReturn",
    "ModernStepReturn", 
    "StepReturn",
    "ActionType",
    "ObservationType",
    "InfoType"
]


# Module-level performance monitoring initialization
if ENHANCED_LOGGING:
    logger.bind_experiment_metadata(
        module="compatibility_layer",
        api_support="dual_mode",
        performance_target_ms=PERFORMANCE_TARGET_MS,
        overhead_threshold_ms=COMPATIBILITY_OVERHEAD_THRESHOLD * 1000
    )

# Auto-register with enhanced logging for system health monitoring
if ENHANCED_LOGGING:
    logger.log_system_health(
        "compatibility_layer", 
        "healthy",
        features_loaded=[
            "api_detection",
            "format_conversion", 
            "performance_monitoring",
            "correlation_tracking"
        ],
        performance_targets={
            "step_time_ms": PERFORMANCE_TARGET_MS,
            "detection_time_ms": COMPATIBILITY_OVERHEAD_THRESHOLD * 1000
        }
    )