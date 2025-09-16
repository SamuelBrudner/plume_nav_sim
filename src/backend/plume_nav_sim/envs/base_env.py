"""
Abstract base environment class providing Gymnasium-compatible interface template, shared functionality, 
common initialization patterns, and abstract method definitions for all plume navigation environments 
with comprehensive error handling, performance monitoring integration, and validation framework for 
systematic environment development and maintenance.

This module establishes the foundational architecture for all plume navigation environments, implementing
the Template Method pattern with Gymnasium API compliance, performance monitoring, and comprehensive
validation. All concrete environment implementations must inherit from BaseEnvironment and implement
the abstract methods while leveraging the shared infrastructure.

Key Components:
- BaseEnvironment: Abstract base class defining Gymnasium-compatible interface template
- AbstractEnvironmentError: Specialized exception for abstract method enforcement
- Factory functions: Validated configuration creation and setup validation
- Performance monitoring: Integrated timing and resource tracking
- Validation framework: Comprehensive parameter and state validation

Architecture Integration:
- Interface layer component providing Gymnasium API compliance
- Template Method pattern foundation for systematic environment development
- Performance targets: <1ms step execution, <10ms reset operations
- Resource management with automatic cleanup and graceful degradation
- Extensible design supporting future environment implementations
"""

# Standard library imports - Python >=3.10
import abc  # >=3.10 - Abstract base class decorators for environment template definition and interface contracts
import copy  # >=3.10 - Deep copying for safe configuration management and state manipulation
import logging  # >=3.10 - Environment operation logging, error reporting, and performance monitoring integration
import time  # >=3.10 - High-precision timing measurements for performance monitoring and benchmarking
import warnings  # >=3.10 - Development warnings for deprecated patterns, performance issues, and compatibility concerns
from typing import Any, Dict, Optional, Tuple, Union  # >=3.10 - Type hints for abstract methods and generic type support

# Third-party imports
import gymnasium  # >=0.29.0 - Core reinforcement learning environment framework with standard API methods
import numpy as np  # >=2.1.0 - Array operations, mathematical calculations, and performance-optimized numerical computing

# Internal imports - Core types and constants
from ..core import (
    Action, ActionType, Coordinates, GridSize, EnvironmentConfig, RenderMode  # Core data types for environment state management
)

from ..core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, DEFAULT_MAX_STEPS, DEFAULT_GOAL_RADIUS,  # Environment configuration defaults
    ACTION_SPACE_SIZE, SUPPORTED_RENDER_MODES, OBSERVATION_DTYPE,  # Gymnasium space definitions and validation
    PERFORMANCE_TARGET_STEP_LATENCY_MS, PERFORMANCE_TARGET_RGB_RENDER_MS  # Performance targets and benchmarking
)

# Internal imports - Rendering system
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Avoid importing heavy rendering stack at import time
    from ..render.base_renderer import BaseRenderer, RenderContext

# Internal imports - Utility framework
from ..utils.logging import get_component_logger, monitor_performance  # Component logging and performance tracking
from ..utils.validation import (
    validate_seed_value, validate_action_parameter, validate_render_mode, create_validation_context  # Parameter validation
)
from ..utils.exceptions import (
    ValidationError, StateError, RenderingError, ComponentError, handle_component_error  # Exception handling framework
)

# Global configuration constants
ABSTRACT_METHOD_ERROR_MSG = "Subclasses must implement this abstract method for environment-specific functionality"
PERFORMANCE_WARNING_THRESHOLD = 2.0  # Multiple of target performance before warning
DEFAULT_RENDER_MODE = "rgb_array"  # Default render mode for environment operations

# Module exports
__all__ = [
    'BaseEnvironment',  # Abstract base environment class providing Gymnasium-compatible interface template
    'AbstractEnvironmentError',  # Exception class for abstract method enforcement with implementation guidance
    'create_base_environment_config',  # Factory function for validated environment configuration creation
    'validate_base_environment_setup'  # Comprehensive validation function for environment setup feasibility
]


class AbstractEnvironmentError(Exception):
    """
    Exception class for abstract method enforcement and base environment error handling with 
    implementation guidance and debugging support for environment development.
    
    This specialized exception provides detailed context for abstract method implementation requirements,
    includes implementation guidance for developers, and integrates with the logging framework for
    comprehensive error reporting and debugging support during environment development.
    
    Attributes:
        method_name: Name of the abstract method requiring implementation
        class_name: Name of the class where implementation is needed
        implementation_hint: Optional guidance for method implementation
    """
    
    def __init__(self, method_name: str, class_name: str, implementation_hint: Optional[str] = None):
        """
        Initialize abstract environment error with method identification and implementation guidance.
        
        Args:
            method_name: Name of abstract method requiring implementation
            class_name: Name of class requiring method implementation
            implementation_hint: Optional guidance for implementation approach
        """
        # Store method identification for error context and debugging
        self.method_name = method_name
        self.class_name = class_name
        self.implementation_hint = implementation_hint
        
        # Format comprehensive error message with implementation requirements
        error_msg = (
            f"Abstract method '{method_name}' must be implemented in class '{class_name}'. "
            f"{ABSTRACT_METHOD_ERROR_MSG}"
        )
        
        # Include implementation hint if provided for developer guidance
        if implementation_hint:
            error_msg += f"\n\nImplementation hint: {implementation_hint}"
        
        # Add general guidance for abstract method implementation
        error_msg += (
            f"\n\nRequired implementation pattern:"
            f"\n  def {method_name}(self, ...):"
            f"\n    # Implement {method_name} logic here"
            f"\n    # Return appropriate value based on method signature"
        )
        
        # Call parent exception constructor with formatted message
        super().__init__(error_msg)
    
    def get_implementation_guidance(self) -> str:
        """
        Generate comprehensive implementation guidance for abstract method with examples and best practices.
        
        Returns:
            str: Detailed implementation guidance with examples, patterns, and best practices
        """
        # Generate method-specific implementation guidance based on method name
        guidance = f"Implementation guidance for {self.method_name} in {self.class_name}:\n\n"
        
        if self.method_name == "_reset_environment_state":
            guidance += (
                "This method should:\n"
                "- Initialize all environment-specific state variables\n"
                "- Reset component systems (plume model, agent state, etc.)\n"
                "- Prepare environment for new episode execution\n"
                "- Handle initialization failures gracefully\n\n"
                "Example implementation:\n"
                "def _reset_environment_state(self) -> None:\n"
                "    self.agent_position = self.config.start_position\n"
                "    self.step_count = 0\n"
                "    # Initialize other environment-specific state\n"
            )
        
        elif self.method_name == "_process_action":
            guidance += (
                "This method should:\n"
                "- Validate action parameter against action space\n"
                "- Calculate agent movement based on action\n"
                "- Apply boundary enforcement and collision detection\n"
                "- Update agent position and related state variables\n\n"
                "Example implementation:\n"
                "def _process_action(self, action: ActionType) -> None:\n"
                "    # Convert action to movement vector\n"
                "    # Apply movement with boundary checking\n"
                "    # Update agent position\n"
            )
        
        elif self.method_name == "_get_observation":
            guidance += (
                "This method should:\n"
                "- Sample concentration field at agent position\n"
                "- Format observation according to observation space\n"
                "- Ensure proper data type and value range\n"
                "- Return numpy array ready for RL agent\n\n"
                "Example implementation:\n"
                "def _get_observation(self) -> np.ndarray:\n"
                "    concentration = self.plume_model.sample_at_position(self.agent_position)\n"
                "    return np.array([concentration], dtype=OBSERVATION_DTYPE)\n"
            )
        
        else:
            # Generic guidance for other abstract methods
            guidance += (
                "General implementation requirements:\n"
                "- Follow method signature exactly as defined in base class\n"
                "- Handle errors gracefully with appropriate exceptions\n"
                "- Include logging for debugging and monitoring\n"
                "- Validate inputs and outputs as appropriate\n"
                "- Maintain performance targets for the operation\n"
            )
        
        # Add performance considerations and optimization suggestions
        guidance += (
            "\nPerformance considerations:\n"
            "- Optimize for target performance requirements\n"
            "- Use vectorized operations where possible\n"
            "- Minimize memory allocations in hot paths\n"
            "- Include performance monitoring if appropriate\n"
        )
        
        # Include error handling patterns and exception management
        guidance += (
            "\nError handling patterns:\n"
            "- Use appropriate exception types from utils.exceptions\n"
            "- Provide detailed error context for debugging\n"
            "- Log errors at appropriate levels\n"
            "- Ensure graceful degradation where possible\n"
        )
        
        return guidance


class BaseEnvironment(gymnasium.Env, abc.ABC):
    """
    Abstract base environment class providing Gymnasium-compatible interface template, shared functionality, 
    common initialization patterns, performance monitoring integration, and abstract method contracts for 
    systematic plume navigation environment development with comprehensive error handling and validation framework.
    
    This abstract base class establishes the foundation for all plume navigation environments, implementing
    the Template Method pattern with Gymnasium API compliance, comprehensive shared functionality, and
    performance monitoring. All concrete environment implementations must inherit from this class and
    implement the abstract methods while leveraging the shared infrastructure.
    
    Key Design Patterns:
        - Template Method: Shared workflow with customizable implementation points
        - Gymnasium Interface: Complete API compliance with standard RL environment methods
        - Performance Monitoring: Integrated timing and resource tracking
        - Validation Framework: Comprehensive input validation and error handling
        - Resource Management: Automatic cleanup and lifecycle management
    
    Shared Functionality:
        - Gymnasium space initialization and validation
        - Performance tracking and analysis
        - Rendering system integration and management
        - Error handling with graceful degradation
        - Logging integration with structured output
        - Configuration management and validation
        - Seeding and reproducibility support
    
    Abstract Interface Requirements:
        - _reset_environment_state(): Environment-specific state initialization
        - _process_action(): Action processing and movement calculation
        - _update_environment_state(): State synchronization after action processing
        - _calculate_reward(): Reward calculation based on environment state
        - _check_terminated(): Episode termination condition evaluation
        - _check_truncated(): Episode truncation condition evaluation
        - _get_observation(): Environment observation generation
        - _create_render_context(): Rendering context creation
        - _create_renderer(): Renderer instantiation
        - _seed_components(): Component-specific seeding
        - _cleanup_components(): Component-specific resource cleanup
        - _validate_component_states(): Component state validation
    """
    
    def __init__(self, config: EnvironmentConfig, render_mode: Optional[str] = None):
        """
        Initialize base environment with configuration validation, Gymnasium space setup, rendering 
        initialization, and performance monitoring framework for systematic environment development.
        
        Args:
            config: Validated environment configuration with parameter consistency checking
            render_mode: Optional rendering mode for visualization ('rgb_array' or 'human')
        """
        # Validate configuration using comprehensive validation with strict checking
        if not validate_base_environment_setup(config, strict_validation=True, check_performance_feasibility=True):
            raise ValidationError(
                "Base environment setup validation failed",
                context={'config_type': type(config).__name__}
            )
        
        # Store validated configuration and initialize component logger
        self.config = config
        self.logger = get_component_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize Gymnasium action space as Discrete with ACTION_SPACE_SIZE
        self.action_space = gymnasium.spaces.Discrete(ACTION_SPACE_SIZE)
        
        # Initialize Gymnasium observation space as Box with OBSERVATION_DTYPE and shape (1,)
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=OBSERVATION_DTYPE
        )
        
        # Validate and store render_mode with fallback to DEFAULT_RENDER_MODE
        if render_mode is not None:
            if not validate_render_mode(render_mode):
                self.logger.warning(f"Invalid render mode '{render_mode}', falling back to '{DEFAULT_RENDER_MODE}'")
                render_mode = DEFAULT_RENDER_MODE
        self.render_mode = render_mode or DEFAULT_RENDER_MODE
        
        # Initialize metadata dictionary with environment information and performance targets
        self.metadata = {
            'render_modes': SUPPORTED_RENDER_MODES,
            'environment_type': 'plume_navigation',
            'performance_targets': {
                'step_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS,
                'rgb_render_ms': PERFORMANCE_TARGET_RGB_RENDER_MS
            },
            'config_summary': {
                'grid_size': f"{config.grid_size.width}x{config.grid_size.height}",
                'max_steps': config.max_steps,
                'goal_radius': config.goal_radius
            }
        }
        
        # Initialize performance monitoring with timing metrics and resource tracking
        self.performance_metrics = {
            'step_times': [],
            'reset_times': [],
            'render_times': [],
            'total_steps': 0,
            'total_episodes': 0,
            'average_step_time_ms': 0.0,
            'average_reset_time_ms': 0.0
        }
        
        # Set environment initialization flags and counters
        self._environment_initialized = False
        self._step_count = 0
        self._episode_count = 0
        
        # Initialize renderer reference to None for lazy initialization
        self._renderer: Optional[BaseRenderer] = None
        
        # Initialize random number generator reference for seeding
        self.np_random: Optional[np.random.Generator] = None
        self._seed: Optional[int] = None
        
        # Log environment initialization with configuration summary
        self.logger.info(
            f"BaseEnvironment initialized: grid={config.grid_size.width}x{config.grid_size.height}, "
            f"max_steps={config.max_steps}, render_mode={self.render_mode}"
        )
    
    @monitor_performance('base_reset', 10.0, True)
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state following Gymnasium API with optional seeding, abstract method 
        coordination, performance monitoring, and comprehensive error handling returning observation and info tuple.
        
        Args:
            seed: Optional seed value for reproducible episode initialization
            options: Optional dictionary of episode options and configuration overrides
            
        Returns:
            Tuple of (observation, info) with initial environment observation and episode metadata
        """
        reset_start_time = time.perf_counter()
        
        try:
            # Validate seed parameter using validate_seed_value with comprehensive error handling
            if seed is not None:
                if not validate_seed_value(seed):
                    raise ValidationError(
                        f"Invalid seed value: {seed}",
                        context={'seed_type': type(seed).__name__, 'seed_value': seed}
                    )
                # Apply seeding using self.seed() method with validation
                self.seed(seed)
            
            # Reset environment state using abstract _reset_environment_state() method
            try:
                self._reset_environment_state()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_reset_environment_state",
                    self.__class__.__name__,
                    "Initialize environment state variables and prepare for new episode"
                )
            
            # Initialize step counter and increment episode counter
            self._step_count = 0
            self._episode_count += 1
            
            # Generate initial observation using abstract _get_observation() method
            try:
                observation = self._get_observation()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_get_observation",
                    self.__class__.__name__,
                    "Sample environment state and return properly formatted observation array"
                )
            
            # Create comprehensive info dictionary with episode metadata
            info = {
                'episode_count': self._episode_count,
                'step_count': self._step_count,
                'seed_used': seed,
                'config_summary': {
                    'grid_size': f"{self.config.grid_size.width}x{self.config.grid_size.height}",
                    'max_steps': self.config.max_steps
                },
                'performance_info': {
                    'reset_time_ms': 0.0  # Will be updated below
                }
            }
            
            # Set environment initialized flag
            self._environment_initialized = True
            
            # Record reset timing for performance analysis
            reset_time_ms = (time.perf_counter() - reset_start_time) * 1000
            self.performance_metrics['reset_times'].append(reset_time_ms)
            self.performance_metrics['average_reset_time_ms'] = np.mean(self.performance_metrics['reset_times'])
            info['performance_info']['reset_time_ms'] = reset_time_ms
            
            # Log reset operation with timing and configuration
            self.logger.info(
                f"Environment reset completed: episode={self._episode_count}, seed={seed}, "
                f"reset_time={reset_time_ms:.2f}ms"
            )
            
            return observation, info
            
        except Exception as e:
            # Handle reset errors with comprehensive context and cleanup
            reset_time_ms = (time.perf_counter() - reset_start_time) * 1000
            self.logger.error(f"Environment reset failed: {e}, reset_time={reset_time_ms:.2f}ms")
            
            # Attempt to set safe state
            self._environment_initialized = False
            
            # Re-raise with appropriate error type
            if isinstance(e, (ValidationError, StateError, AbstractEnvironmentError)):
                raise
            else:
                raise StateError(f"Environment reset failed: {e}")
    
    @monitor_performance('base_step', 1.0, False)
    def step(self, action: ActionType) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step with action processing following Gymnasium API returning 5-tuple with 
        observation, reward, termination status, truncation status, and info dictionary with comprehensive 
        validation and performance monitoring.
        
        Args:
            action: Action parameter for agent movement (int 0-3 or Action enum)
            
        Returns:
            5-tuple (observation, reward, terminated, truncated, info) following Gymnasium step specification
        """
        step_start_time = time.perf_counter()
        
        try:
            # Validate environment is initialized and ready for steps
            if not self._environment_initialized:
                raise StateError(
                    "Environment not initialized - call reset() before step()",
                    context={'environment_state': 'uninitialized'}
                )
            
            # Validate action parameter; returns canonical integer in [0, ACTION_SPACE_SIZE-1]
            action = validate_action_parameter(action, allow_enum_types=True)
            
            # Increment step counter and update performance timing
            self._step_count += 1
            self.performance_metrics['total_steps'] += 1
            
            # Process action using abstract _process_action() method
            try:
                self._process_action(action)
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_process_action",
                    self.__class__.__name__,
                    "Process action parameter and update agent position with boundary enforcement"
                )
            
            # Update environment state using abstract _update_environment_state() method
            try:
                self._update_environment_state()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_update_environment_state",
                    self.__class__.__name__,
                    "Synchronize component states after action processing"
                )
            
            # Calculate reward using abstract _calculate_reward() method
            try:
                reward = self._calculate_reward()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_calculate_reward",
                    self.__class__.__name__,
                    "Calculate reward based on agent position and goal achievement"
                )
            
            # Check episode termination using abstract _check_terminated() method
            try:
                terminated = self._check_terminated()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_check_terminated",
                    self.__class__.__name__,
                    "Evaluate goal achievement and success conditions"
                )
            
            # Check episode truncation using abstract _check_truncated() method
            try:
                truncated = self._check_truncated()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_check_truncated",
                    self.__class__.__name__,
                    "Check step limits and truncation conditions"
                )
            
            # Generate observation using abstract _get_observation() method
            try:
                observation = self._get_observation()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_get_observation",
                    self.__class__.__name__,
                    "Generate environment observation array for agent consumption"
                )
            
            # Create comprehensive info dictionary
            step_time_ms = (time.perf_counter() - step_start_time) * 1000
            info = {
                'step_count': self._step_count,
                'episode_count': self._episode_count,
                'action_taken': int(action) if hasattr(action, '__int__') else action,
                'performance_info': {
                    'step_time_ms': step_time_ms,
                    'average_step_time_ms': self.performance_metrics['average_step_time_ms']
                },
                'episode_info': {
                    'terminated': terminated,
                    'truncated': truncated,
                    'reward': reward
                }
            }
            
            # Record step timing for performance analysis
            self.performance_metrics['step_times'].append(step_time_ms)
            self.performance_metrics['average_step_time_ms'] = np.mean(self.performance_metrics['step_times'][-100:])
            
            # Check performance warning threshold
            if step_time_ms > PERFORMANCE_TARGET_STEP_LATENCY_MS * PERFORMANCE_WARNING_THRESHOLD:
                self.logger.warning(
                    f"Step exceeded performance target: {step_time_ms:.2f}ms > "
                    f"{PERFORMANCE_TARGET_STEP_LATENCY_MS * PERFORMANCE_WARNING_THRESHOLD:.2f}ms"
                )
            
            # Log step completion
            self.logger.debug(
                f"Step completed: action={action}, reward={reward}, terminated={terminated}, "
                f"truncated={truncated}, step_time={step_time_ms:.2f}ms"
            )
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            # Handle step errors with performance recording
            step_time_ms = (time.perf_counter() - step_start_time) * 1000
            self.logger.error(f"Environment step failed: {e}, step_time={step_time_ms:.2f}ms")
            
            # Re-raise with appropriate error type
            if isinstance(e, (ValidationError, StateError, AbstractEnvironmentError)):
                raise
            else:
                raise StateError(f"Environment step failed: {e}")
    
    @monitor_performance('base_render', 50.0, False)
    def render(self) -> Union[np.ndarray, None]:
        """
        Render environment visualization in specified mode with lazy renderer initialization, performance 
        monitoring, error handling, and fallback strategies following Gymnasium render specification.
        
        Returns:
            RGB array for rgb_array mode, None for human mode following Gymnasium render API
        """
        render_start_time = time.perf_counter()
        
        try:
            # Validate environment is initialized
            if not self._environment_initialized:
                self.logger.warning("Render called on uninitialized environment")
                return None
            
            # Validate render_mode using validate_render_mode
            if not validate_render_mode(self.render_mode):
                raise RenderingError(
                    f"Invalid render mode: {self.render_mode}",
                    context={'supported_modes': SUPPORTED_RENDER_MODES}
                )
            
            # Get or create appropriate renderer using _get_or_create_renderer()
            try:
                renderer = self._get_or_create_renderer()
            except Exception as e:
                raise RenderingError(f"Failed to initialize renderer: {e}")
            
            # Create render context using abstract _create_render_context() method
            try:
                render_context = self._create_render_context()
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_create_render_context",
                    self.__class__.__name__,
                    "Create RenderContext with current environment state and visualization data"
                )
            
            # Execute rendering operation using renderer.render()
            try:
                result = renderer.render(render_context)
            except Exception as e:
                # Handle rendering errors with fallback
                self.logger.error(f"Rendering failed: {e}")
                if hasattr(e, 'get_fallback_suggestions'):
                    suggestions = e.get_fallback_suggestions()
                    self.logger.info(f"Fallback suggestions: {suggestions}")
                return None
            
            # Record rendering timing
            render_time_ms = (time.perf_counter() - render_start_time) * 1000
            self.performance_metrics['render_times'].append(render_time_ms)
            
            # Check performance against target
            target_ms = PERFORMANCE_TARGET_RGB_RENDER_MS
            if render_time_ms > target_ms * PERFORMANCE_WARNING_THRESHOLD:
                self.logger.warning(
                    f"Rendering exceeded performance target: {render_time_ms:.2f}ms > {target_ms:.2f}ms"
                )
            
            # Log rendering completion
            self.logger.debug(f"Rendering completed: mode={self.render_mode}, time={render_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            # Handle rendering errors gracefully
            render_time_ms = (time.perf_counter() - render_start_time) * 1000
            self.logger.error(f"Rendering failed: {e}, render_time={render_time_ms:.2f}ms")
            
            if isinstance(e, (RenderingError, AbstractEnvironmentError)):
                raise
            else:
                raise RenderingError(f"Rendering operation failed: {e}")
    
    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set random seed for environment reproducibility with validation, gymnasium seeding integration, 
        and comprehensive logging following Gymnasium seeding specification.
        
        Args:
            seed: Optional seed value for random number generator initialization
            
        Returns:
            List containing the seed value used following Gymnasium specification
        """
        try:
            # Validate seed parameter using validate_seed_value
            if seed is not None and not validate_seed_value(seed):
                raise ValidationError(
                    f"Invalid seed value: {seed}",
                    context={'seed_type': type(seed).__name__, 'seed_value': seed}
                )
            
            # Create seeded random number generator using gymnasium seeding
            self.np_random, actual_seed = gymnasium.utils.seeding.np_random(seed)
            self._seed = actual_seed
            
            # Call abstract _seed_components() method for component-specific seeding
            try:
                self._seed_components(actual_seed)
            except NotImplementedError:
                raise AbstractEnvironmentError(
                    "_seed_components",
                    self.__class__.__name__,
                    "Propagate seed to environment subsystems for reproducible behavior"
                )
            
            # Log seeding operation
            self.logger.info(f"Environment seeded with value: {actual_seed}")
            
            return [actual_seed]
            
        except Exception as e:
            self.logger.error(f"Seeding failed: {e}")
            if isinstance(e, (ValidationError, AbstractEnvironmentError)):
                raise
            else:
                raise ValidationError(f"Environment seeding failed: {e}")
    
    def close(self) -> None:
        """
        Close environment and clean up resources with renderer cleanup, component cleanup, performance 
        reporting, and memory management following Gymnasium specification.
        """
        try:
            # Set environment initialization flag to False
            self._environment_initialized = False
            
            # Clean up renderer resources using _cleanup_renderer()
            self._cleanup_renderer()
            
            # Call abstract _cleanup_components() method for component-specific cleanup
            try:
                self._cleanup_components()
            except NotImplementedError:
                self.logger.warning("_cleanup_components not implemented - skipping component cleanup")
            except Exception as e:
                self.logger.error(f"Component cleanup failed: {e}")
            
            # Generate final performance report
            performance_summary = {
                'total_episodes': self._episode_count,
                'total_steps': self.performance_metrics['total_steps'],
                'average_step_time_ms': self.performance_metrics['average_step_time_ms'],
                'average_reset_time_ms': self.performance_metrics['average_reset_time_ms']
            }
            
            # Clear performance metrics
            self.performance_metrics = {
                'step_times': [],
                'reset_times': [],
                'render_times': [],
                'total_steps': 0,
                'total_episodes': 0,
                'average_step_time_ms': 0.0,
                'average_reset_time_ms': 0.0
            }
            
            # Reset internal state variables
            self._step_count = 0
            self._episode_count = 0
            self._renderer = None
            self.np_random = None
            self._seed = None
            
            # Log environment closure with performance summary
            self.logger.info(f"Environment closed - Performance summary: {performance_summary}")
            
        except Exception as e:
            self.logger.error(f"Environment close failed: {e}")
            # Force cleanup even if errors occurred
            self._environment_initialized = False
            self._renderer = None
    
    def _get_or_create_renderer(self) -> 'BaseRenderer':
        """
        Get existing renderer or create new renderer based on render mode with lazy initialization, 
        backend selection, and error handling for dual-mode visualization system.
        
        Returns:
            BaseRenderer: Initialized renderer instance optimized for current render mode
        """
        # Check if renderer already exists and is compatible
        if self._renderer is not None:
            try:
                # Validate renderer supports current render mode
                render_mode_enum = RenderMode.RGB_ARRAY if self.render_mode == 'rgb_array' else RenderMode.HUMAN
                if self._renderer.supports_render_mode(render_mode_enum):
                    return self._renderer
                else:
                    # Cleanup incompatible renderer
                    self._cleanup_renderer()
            except Exception as e:
                self.logger.warning(f"Renderer compatibility check failed: {e}")
                self._cleanup_renderer()
        
        # Create new renderer using abstract _create_renderer() method
        try:
            self._renderer = self._create_renderer()
            if self._renderer is None:
                raise RenderingError("Renderer creation returned None")
            
            # Initialize renderer if not already initialized
            if not getattr(self._renderer, '_initialized', False):
                self._renderer.initialize()
            
            self.logger.debug(f"Renderer created and initialized: {type(self._renderer).__name__}")
            return self._renderer
            
        except NotImplementedError:
            raise AbstractEnvironmentError(
                "_create_renderer",
                self.__class__.__name__,
                "Create appropriate renderer instance based on render_mode configuration"
            )
        except Exception as e:
            raise RenderingError(f"Renderer creation failed: {e}")
    
    def _cleanup_renderer(self) -> None:
        """
        Clean up renderer resources with proper cleanup methods, error handling, and memory management.
        """
        if self._renderer is not None:
            try:
                # Call renderer cleanup method
                if hasattr(self._renderer, 'cleanup_resources'):
                    self._renderer.cleanup_resources()
                elif hasattr(self._renderer, 'close'):
                    self._renderer.close()
                
                self.logger.debug("Renderer resources cleaned up successfully")
            except Exception as e:
                self.logger.warning(f"Renderer cleanup failed: {e}")
            finally:
                # Clear renderer reference
                self._renderer = None
    
    def get_environment_status(self, include_performance_data: bool = True, 
                             include_component_details: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive environment status including initialization state, component status, 
        performance metrics, and debugging information for monitoring and analysis.
        
        Args:
            include_performance_data: Whether to include performance metrics and timing data
            include_component_details: Whether to include detailed component information
            
        Returns:
            Dictionary containing comprehensive environment status for debugging and monitoring
        """
        # Compile basic environment status
        status = {
            'initialized': self._environment_initialized,
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'render_mode': self.render_mode,
            'seeded': self._seed is not None,
            'seed_value': self._seed,
            'config_summary': {
                'grid_size': f"{self.config.grid_size.width}x{self.config.grid_size.height}",
                'max_steps': self.config.max_steps,
                'goal_radius': self.config.goal_radius
            }
        }
        
        # Include performance data if requested
        if include_performance_data:
            status['performance_metrics'] = {
                'total_steps': self.performance_metrics['total_steps'],
                'average_step_time_ms': self.performance_metrics['average_step_time_ms'],
                'average_reset_time_ms': self.performance_metrics['average_reset_time_ms'],
                'step_times_count': len(self.performance_metrics['step_times']),
                'reset_times_count': len(self.performance_metrics['reset_times']),
                'render_times_count': len(self.performance_metrics['render_times'])
            }
        
        # Include component details if requested
        if include_component_details:
            status['component_details'] = {
                'renderer_initialized': self._renderer is not None,
                'renderer_type': type(self._renderer).__name__ if self._renderer else None,
                'action_space_type': type(self.action_space).__name__,
                'observation_space_type': type(self.observation_space).__name__,
                'np_random_initialized': self.np_random is not None
            }
        
        return status
    
    def validate_environment_state(self, strict_validation: bool = True, 
                                 check_performance_targets: bool = True) -> Dict[str, Any]:
        """
        Validate complete environment state consistency with component validation, performance checking, 
        and comprehensive error reporting for debugging and quality assurance.
        
        Args:
            strict_validation: Apply comprehensive validation including edge cases
            check_performance_targets: Whether to check performance against targets
            
        Returns:
            Dictionary containing validation report with analysis and recommendations
        """
        validation_context = create_validation_context("validate_environment_state", "BaseEnvironment")
        
        validation_report = {
            'overall_valid': True,
            'validation_timestamp': time.time(),
            'issues_found': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Validate environment initialization status
            if not self._environment_initialized:
                validation_report['issues_found'].append("Environment not initialized")
                validation_report['overall_valid'] = False
            
            # Check internal state consistency
            if self._step_count < 0:
                validation_report['issues_found'].append("Invalid step count")
                validation_report['overall_valid'] = False
            
            if self._episode_count < 0:
                validation_report['issues_found'].append("Invalid episode count")
                validation_report['overall_valid'] = False
            
            # Validate configuration parameters
            try:
                self.config.validate()
            except Exception as e:
                validation_report['issues_found'].append(f"Configuration validation failed: {e}")
                validation_report['overall_valid'] = False
            
            # Check performance metrics against targets if enabled
            if check_performance_targets:
                avg_step_time = self.performance_metrics['average_step_time_ms']
                if avg_step_time > PERFORMANCE_TARGET_STEP_LATENCY_MS * 2:
                    validation_report['warnings'].append(f"Step time ({avg_step_time:.2f}ms) exceeds target")
            
            # Apply strict validation if enabled
            if strict_validation:
                # Check renderer compatibility
                if self._renderer is not None:
                    try:
                        render_mode_enum = RenderMode.RGB_ARRAY if self.render_mode == 'rgb_array' else RenderMode.HUMAN
                        if not self._renderer.supports_render_mode(render_mode_enum):
                            validation_report['issues_found'].append("Renderer incompatible with current render mode")
                            validation_report['overall_valid'] = False
                    except Exception as e:
                        validation_report['warnings'].append(f"Renderer validation failed: {e}")
            
            # Call abstract _validate_component_states() method for component-specific validation
            try:
                component_validation = self._validate_component_states(strict_validation)
                if isinstance(component_validation, dict):
                    if not component_validation.get('valid', True):
                        validation_report['issues_found'].extend(component_validation.get('issues', []))
                        validation_report['overall_valid'] = False
                    validation_report['component_validation'] = component_validation
            except NotImplementedError:
                validation_report['warnings'].append("Component state validation not implemented")
            except Exception as e:
                validation_report['warnings'].append(f"Component validation failed: {e}")
            
            # Generate recommendations based on findings
            if not validation_report['overall_valid']:
                validation_report['recommendations'].append("Address validation issues before continuing operation")
            
            if validation_report['warnings']:
                validation_report['recommendations'].append("Review warnings for potential improvements")
            
            if not validation_report['issues_found'] and not validation_report['warnings']:
                validation_report['recommendations'].append("Environment state is valid and ready for operation")
            
        except Exception as e:
            validation_report['issues_found'].append(f"Validation failed with exception: {e}")
            validation_report['overall_valid'] = False
        
        return validation_report
    
    # Abstract methods that must be implemented by concrete environment classes
    
    @abc.abstractmethod
    def _reset_environment_state(self) -> None:
        """
        Abstract method for environment-specific state reset including component initialization, 
        state variables, and system preparation for new episode start.
        """
        raise AbstractEnvironmentError(
            "_reset_environment_state",
            self.__class__.__name__,
            "Initialize environment state variables, reset component systems, and prepare for new episode"
        )
    
    @abc.abstractmethod
    def _process_action(self, action: ActionType) -> None:
        """
        Abstract method for environment-specific action processing including validation, movement 
        calculation, boundary enforcement, and state updates with comprehensive error handling.
        
        Args:
            action: Validated action parameter for agent movement
        """
        raise AbstractEnvironmentError(
            "_process_action",
            self.__class__.__name__,
            "Process action parameter, calculate agent movement, and apply boundary enforcement"
        )
    
    @abc.abstractmethod
    def _update_environment_state(self) -> None:
        """
        Abstract method for environment state update after action processing including component 
        synchronization, state validation, and consistency checking.
        """
        raise AbstractEnvironmentError(
            "_update_environment_state",
            self.__class__.__name__,
            "Synchronize component states after action processing and validate consistency"
        )
    
    @abc.abstractmethod
    def _calculate_reward(self) -> float:
        """
        Abstract method for environment-specific reward calculation including goal detection, 
        distance measurements, and reward structure implementation.
        
        Returns:
            float: Calculated reward value based on environment-specific reward structure
        """
        raise AbstractEnvironmentError(
            "_calculate_reward",
            self.__class__.__name__,
            "Calculate reward based on agent position, goal achievement, and reward structure"
        )
    
    @abc.abstractmethod
    def _check_terminated(self) -> bool:
        """
        Abstract method for episode termination checking based on goal achievement, failure 
        conditions, and environment-specific termination criteria.
        
        Returns:
            bool: True if episode should terminate due to goal achievement or failure
        """
        raise AbstractEnvironmentError(
            "_check_terminated",
            self.__class__.__name__,
            "Check goal achievement and evaluate success/failure termination conditions"
        )
    
    @abc.abstractmethod
    def _check_truncated(self) -> bool:
        """
        Abstract method for episode truncation checking based on step limits, time constraints, 
        and environment-specific truncation criteria.
        
        Returns:
            bool: True if episode should truncate due to step limit or other constraints
        """
        raise AbstractEnvironmentError(
            "_check_truncated",
            self.__class__.__name__,
            "Check step count against limits and evaluate truncation conditions"
        )
    
    @abc.abstractmethod
    def _get_observation(self) -> np.ndarray:
        """
        Abstract method for environment-specific observation generation including data sampling, 
        format conversion, and observation space compliance.
        
        Returns:
            numpy.ndarray: Environment observation as numpy array with proper shape and dtype
        """
        raise AbstractEnvironmentError(
            "_get_observation",
            self.__class__.__name__,
            "Generate observation array from environment state with proper format and dtype"
        )
    
    @abc.abstractmethod
    def _create_render_context(self) -> 'RenderContext':
        """
        Abstract method for render context creation containing environment state, visualization 
        data, and rendering metadata for visualization pipeline.
        
        Returns:
            RenderContext: Complete render context with environment state and visualization data
        """
        raise AbstractEnvironmentError(
            "_create_render_context",
            self.__class__.__name__,
            "Create RenderContext with current environment state for visualization pipeline"
        )
    
    @abc.abstractmethod
    def _create_renderer(self) -> 'BaseRenderer':
        """
        Abstract method for renderer creation based on render mode with backend selection, 
        optimization configuration, and error handling.
        
        Returns:
            BaseRenderer: Initialized renderer instance optimized for current render mode
        """
        raise AbstractEnvironmentError(
            "_create_renderer",
            self.__class__.__name__,
            "Create appropriate renderer instance based on render_mode configuration"
        )
    
    @abc.abstractmethod
    def _seed_components(self, seed: int) -> None:
        """
        Abstract method for component-specific seeding including subsystem seed propagation 
        and reproducibility setup.
        
        Args:
            seed: Seed value to propagate to environment subsystems
        """
        raise AbstractEnvironmentError(
            "_seed_components",
            self.__class__.__name__,
            "Propagate seed to environment subsystems for reproducible behavior"
        )
    
    @abc.abstractmethod
    def _cleanup_components(self) -> None:
        """
        Abstract method for component-specific cleanup including resource deallocation, 
        memory management, and proper shutdown procedures.
        """
        raise AbstractEnvironmentError(
            "_cleanup_components",
            self.__class__.__name__,
            "Clean up environment subsystems and release allocated resources"
        )
    
    @abc.abstractmethod
    def _validate_component_states(self, strict_validation: bool) -> Dict[str, Any]:
        """
        Abstract method for component state validation including consistency checking, 
        integrity verification, and debugging support.
        
        Args:
            strict_validation: Whether to apply strict validation rules
            
        Returns:
            Dictionary containing component validation results with status and recommendations
        """
        raise AbstractEnvironmentError(
            "_validate_component_states",
            self.__class__.__name__,
            "Validate component states and return comprehensive validation report"
        )


# Factory and utility functions

def create_base_environment_config(grid_size: tuple = DEFAULT_GRID_SIZE,
                                 source_location: tuple = DEFAULT_SOURCE_LOCATION,
                                 max_steps: int = DEFAULT_MAX_STEPS,
                                 goal_radius: float = DEFAULT_GOAL_RADIUS,
                                 render_mode: Optional[str] = None,
                                 additional_config: Optional[dict] = None) -> EnvironmentConfig:
    """
    Factory function to create validated base environment configuration with parameter consistency 
    checking, resource validation, and performance target verification for environment initialization.
    
    Args:
        grid_size: Grid dimensions tuple with positive integer width and height
        source_location: Source position tuple with coordinates within grid bounds
        max_steps: Maximum episode steps with positive integer validation
        goal_radius: Goal detection radius with non-negative float validation
        render_mode: Optional rendering mode with supported mode validation
        additional_config: Optional configuration overrides and extensions
        
    Returns:
        EnvironmentConfig: Validated configuration ready for environment initialization
    """
    try:
        # Validate grid_size tuple contains positive integer dimensions
        if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
            raise ValidationError(
                "Grid size must be tuple or list of length 2",
                context={'grid_size_type': type(grid_size).__name__, 'grid_size_value': grid_size}
            )
        
        width, height = grid_size
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValidationError(
                "Grid dimensions must be positive integers",
                context={'width': width, 'height': height}
            )
        
        # Check memory feasibility for large grids
        total_cells = width * height
        if total_cells > 1000000:  # 1M cells threshold
            warnings.warn(f"Large grid ({total_cells} cells) may impact performance", UserWarning)
        
        # Validate source_location coordinates are within grid bounds
        if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
            raise ValidationError(
                "Source location must be tuple or list of length 2",
                context={'source_location_type': type(source_location).__name__}
            )
        
        src_x, src_y = source_location
        if not isinstance(src_x, int) or not isinstance(src_y, int):
            raise ValidationError(
                "Source coordinates must be integers",
                context={'src_x': src_x, 'src_y': src_y}
            )
        
        if src_x < 0 or src_x >= width or src_y < 0 or src_y >= height:
            raise ValidationError(
                "Source location outside grid boundaries",
                context={
                    'source_location': (src_x, src_y),
                    'grid_bounds': f"0 <= x < {width}, 0 <= y < {height}"
                }
            )
        
        # Validate max_steps is positive integer within reasonable limits
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValidationError(
                "Max steps must be positive integer",
                context={'max_steps': max_steps, 'max_steps_type': type(max_steps).__name__}
            )
        
        if max_steps > 100000:  # Reasonable upper limit
            warnings.warn(f"Very high max_steps ({max_steps}) may impact performance", UserWarning)
        
        # Validate goal_radius is non-negative float
        if not isinstance(goal_radius, (int, float)) or goal_radius < 0:
            raise ValidationError(
                "Goal radius must be non-negative number",
                context={'goal_radius': goal_radius, 'goal_radius_type': type(goal_radius).__name__}
            )
        
        # Validate render_mode against SUPPORTED_RENDER_MODES
        if render_mode is not None:
            if not validate_render_mode(render_mode):
                raise ValidationError(
                    f"Unsupported render mode: {render_mode}",
                    context={'supported_modes': SUPPORTED_RENDER_MODES}
                )
        
        # Apply additional_config overrides with parameter validation
        config_params = {
            'grid_size': GridSize(width=width, height=height),
            'source_location': Coordinates(x=src_x, y=src_y),
            'max_steps': max_steps,
            'goal_radius': float(goal_radius)
        }
        
        if additional_config and isinstance(additional_config, dict):
            # Validate additional config parameters
            for key, value in additional_config.items():
                if key in config_params:
                    # Override with validation
                    if key == 'grid_size' and isinstance(value, (tuple, list)):
                        config_params[key] = GridSize(width=value[0], height=value[1])
                    elif key == 'source_location' and isinstance(value, (tuple, list)):
                        config_params[key] = Coordinates(x=value[0], y=value[1])
                    else:
                        config_params[key] = value
        
        # Create EnvironmentConfig with validated parameters
        config = EnvironmentConfig(
            grid_size=config_params['grid_size'],
            source_location=config_params['source_location'],
            max_steps=config_params['max_steps'],
            goal_radius=config_params['goal_radius']
        )
        
        # Validate complete configuration using config.validate()
        config.validate()
        
        # Estimate resources and check feasibility
        resource_estimate = config.estimate_resources()
        if resource_estimate.get('memory_mb', 0) > 500:  # 500MB threshold
            warnings.warn(f"High memory estimate: {resource_estimate['memory_mb']}MB", UserWarning)
        
        return config
        
    except ValidationError:
        # Re-raise validation errors with existing context
        raise
    except Exception as e:
        # Handle unexpected errors in configuration creation
        raise ValidationError(f"Configuration creation failed: {e}")


def validate_base_environment_setup(config: EnvironmentConfig,
                                   strict_validation: bool = True,
                                   check_performance_feasibility: bool = True,
                                   validation_context: Optional[dict] = None) -> bool:
    """
    Comprehensive validation function for base environment setup ensuring Gymnasium API compliance, 
    component initialization readiness, and performance target feasibility with detailed error reporting.
    
    Args:
        config: EnvironmentConfig instance to validate for setup feasibility
        strict_validation: Apply rigorous validation including edge case testing
        check_performance_feasibility: Validate performance against system targets
        validation_context: Optional context information for validation tracking
        
    Returns:
        bool: True if environment setup is valid and feasible
        
    Raises:
        ValidationError: If validation fails with detailed context and recommendations
    """
    try:
        # Create validation context with operation details
        context = validation_context or create_validation_context(
            "validate_base_environment_setup",
            "BaseEnvironment"
        )
        
        # Validate configuration completeness and parameter consistency
        if not isinstance(config, EnvironmentConfig):
            raise ValidationError(
                "Config must be EnvironmentConfig instance",
                context={'config_type': type(config).__name__}
            )
        
        # Use EnvironmentConfig.validate() for comprehensive validation
        config.validate()
        
        # Verify Gymnasium API compliance requirements
        grid_size = config.grid_size
        if grid_size.width <= 0 or grid_size.height <= 0:
            raise ValidationError(
                "Grid dimensions must be positive for Gymnasium compliance",
                context={'grid_size': f"{grid_size.width}x{grid_size.height}"}
            )
        
        # Check component initialization feasibility
        source_pos = config.source_location
        if not (0 <= source_pos.x < grid_size.width and 0 <= source_pos.y < grid_size.height):
            raise ValidationError(
                "Source position must be within grid boundaries",
                context={
                    'source_position': f"({source_pos.x}, {source_pos.y})",
                    'grid_bounds': f"0 <= x < {grid_size.width}, 0 <= y < {grid_size.height}"
                }
            )
        
        # Validate performance feasibility if check_performance_feasibility enabled
        if check_performance_feasibility:
            resource_estimate = config.estimate_resources()
            memory_mb = resource_estimate.get('memory_mb', 0)
            
            if memory_mb > 1000:  # 1GB threshold
                raise ValidationError(
                    "Memory requirements exceed feasibility threshold",
                    context={'estimated_memory_mb': memory_mb, 'threshold_mb': 1000}
                )
            
            # Check computational complexity
            total_cells = grid_size.width * grid_size.height
            if total_cells > 2048 * 2048 and strict_validation:
                raise ValidationError(
                    "Grid size may not meet performance targets",
                    context={'total_cells': total_cells, 'warning_threshold': 2048 * 2048}
                )
        
        # Apply strict validation rules if strict_validation enabled
        if strict_validation:
            # Check goal radius consistency
            if config.goal_radius > min(grid_size.width, grid_size.height) / 2:
                raise ValidationError(
                    "Goal radius too large relative to grid size",
                    context={
                        'goal_radius': config.goal_radius,
                        'max_reasonable': min(grid_size.width, grid_size.height) / 2
                    }
                )
            
            # Validate max_steps is reasonable
            if config.max_steps > grid_size.width * grid_size.height * 10:
                warnings.warn(
                    f"Max steps ({config.max_steps}) may be excessive for grid size",
                    UserWarning
                )
        
        # Validate rendering system compatibility if render mode specified
        # This would be validated by the concrete implementation
        
        # Check memory requirements and resource constraints
        if check_performance_feasibility:
            estimated_step_time = total_cells / 1000000  # Rough estimate in ms
            if estimated_step_time > PERFORMANCE_TARGET_STEP_LATENCY_MS * 5:
                warnings.warn(
                    f"Estimated step time ({estimated_step_time:.2f}ms) may exceed targets",
                    UserWarning
                )
        
        return True
        
    except ValidationError:
        # Re-raise validation errors with existing context
        raise
    except Exception as e:
        # Handle unexpected validation errors
        raise ValidationError(
            f"Base environment setup validation failed: {e}",
            context={
                'error_type': type(e).__name__,
                'validation_function': 'validate_base_environment_setup'
            }
        )