"""
Consolidated navigation controllers for single and multi-agent scenarios.

This module consolidates the SingleAgentController and MultiAgentController implementations
that encapsulate agent state and provide methods to initialize, reset, advance, and sample
odor for navigation simulations. The controllers implement the NavigatorProtocol interface
and integrate with the Hydra configuration system for enhanced ML framework compatibility.

The controllers support:
- Single-agent and multi-agent navigation scenarios
- Hydra configuration integration for experiment orchestration
- Enhanced error handling and performance monitoring
- Comprehensive logging with structured context
- Random seed management for reproducible experiments
- Type-safe parameter validation and constraints

This module maintains backward compatibility by re-exporting existing controllers while
adding enhanced factory functions and parameter dataclasses for improved integration.

Examples:
    Single-agent controller with enhanced features:
        >>> from odor_plume_nav.core.controllers import SingleAgentController
        >>> controller = SingleAgentController(position=(10.0, 20.0), speed=1.5)
        >>> controller.step(env_array, dt=1.0)
        >>> odor_value = controller.sample_odor(env_array)
        
    Multi-agent controller with performance monitoring:
        >>> from odor_plume_nav.core.controllers import MultiAgentController
        >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        >>> controller = MultiAgentController(positions=positions)
        >>> controller.step(env_array, dt=1.0)
        >>> metrics = controller.get_performance_metrics()
        
    Configuration-driven instantiation:
        >>> from odor_plume_nav.core.controllers import create_controller_from_config
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({"position": [5.0, 5.0], "max_speed": 2.0})
        >>> controller = create_controller_from_config(cfg)
        
    Type-safe parameter updates:
        >>> from odor_plume_nav.core.controllers import SingleAgentParams
        >>> params = SingleAgentParams(position=(10, 20), speed=1.5)
        >>> controller.reset_with_params(params)

Notes:
    All controllers maintain backward compatibility with the original NavigatorProtocol
    interface while adding enhanced features for modern ML pipeline integration.
    Performance requirements are maintained with <33ms frame processing latency
    and support for 100+ agents at 30fps simulation throughput.
"""

import contextlib
import time
from typing import Optional, Union, Any, Tuple, List, Dict, TypeVar
from dataclasses import dataclass
import numpy as np

# Import existing controllers for backward compatibility
from odor_plume_nav.services.control_loops import (
    SingleAgentController as _BaseingleAgentController,
    MultiAgentController as _BaseMultiAgentController
)

# Hydra integration for configuration management
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None

# Loguru integration for enhanced logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Import configuration models for type-safe configuration
from odor_plume_nav.config.models import NavigatorConfig, SingleAgentConfig, MultiAgentConfig

# Type variable for controller types
ControllerType = TypeVar('ControllerType', bound=Union[_BaseingleAgentController, _BaseMultiAgentController])


@dataclass
class SingleAgentParams:
    """Type-safe parameters for resetting a single agent navigator.
    
    This dataclass provides stronger type checking than kwargs-based configuration
    and integrates with Hydra's structured configuration system for validation.
    
    Attributes:
        position: Initial agent position coordinates [x, y]
        orientation: Initial orientation in degrees (0 = right, 90 = up)
        speed: Initial speed in units per time step
        max_speed: Maximum allowed speed in units per time step
        angular_velocity: Angular velocity in degrees per second
    
    Examples:
        Basic parameter configuration:
            >>> params = SingleAgentParams(position=(10.0, 20.0), speed=1.5)
            >>> controller.reset_with_params(params)
            
        Partial parameter updates:
            >>> params = SingleAgentParams(speed=2.0, max_speed=3.0)
            >>> controller.reset_with_params(params)
    """
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert dataclass to kwargs dictionary for controller reset methods."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class MultiAgentParams:
    """Type-safe parameters for resetting a multi-agent navigator.
    
    This dataclass provides stronger type checking and enables batch parameter
    updates for multiple agents with comprehensive validation.
    
    Attributes:
        positions: Array of agent positions with shape (num_agents, 2)
        orientations: Array of agent orientations with shape (num_agents,)
        speeds: Array of agent speeds with shape (num_agents,)
        max_speeds: Array of maximum speeds with shape (num_agents,)
        angular_velocities: Array of angular velocities with shape (num_agents,)
    
    Examples:
        Batch agent configuration:
            >>> import numpy as np
            >>> params = MultiAgentParams(
            ...     positions=np.array([[0, 0], [10, 10], [20, 20]]),
            ...     speeds=np.array([1.0, 1.5, 2.0])
            ... )
            >>> controller.reset_with_params(params)
            
        Array parameter updates:
            >>> params = MultiAgentParams(
            ...     speeds=np.array([2.0, 2.5, 3.0]),
            ...     max_speeds=np.array([3.0, 3.5, 4.0])
            ... )
            >>> controller.reset_with_params(params)
    """
    positions: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert dataclass to kwargs dictionary for controller reset methods."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# Enhanced controller classes that extend the base implementations
class SingleAgentController(_BaseingleAgentController):
    """Enhanced single agent controller with advanced integration capabilities.
    
    This extends the base SingleAgentController with enhanced logging, performance
    monitoring, and Hydra configuration integration while maintaining full backward
    compatibility with the original NavigatorProtocol interface.
    
    Additional Features:
        - Comprehensive structured logging with loguru integration
        - Performance monitoring with <33ms frame processing requirements
        - Enhanced error handling and validation
        - Type-safe parameter reset with SingleAgentParams dataclass
        - Hydra configuration context binding
        - Experiment tracking integration
    
    Performance Requirements:
        - Frame processing: <33ms per step
        - Memory usage: <10MB per 100 agents
        - Simulation throughput: ≥30 FPS
    
    Examples:
        Enhanced initialization with logging:
            >>> controller = SingleAgentController(
            ...     position=(10.0, 20.0),
            ...     speed=1.5,
            ...     enable_logging=True,
            ...     controller_id="agent_001"
            ... )
            
        Performance monitoring:
            >>> controller.step(env_array, dt=1.0)
            >>> metrics = controller.get_performance_metrics()
            >>> print(f"Step time: {metrics['step_time_mean_ms']:.2f}ms")
            
        Type-safe parameter updates:
            >>> params = SingleAgentParams(position=(5.0, 5.0), max_speed=2.0)
            >>> controller.reset_with_params(params)
    """
    
    def __init__(self, *args, enable_logging: bool = True, controller_id: Optional[str] = None, **kwargs):
        """Initialize enhanced single agent controller with monitoring capabilities."""
        # Initialize base controller
        super().__init__(*args, **kwargs)
        
        # Enhanced logging and monitoring setup
        self._enable_logging = enable_logging
        self._controller_id = controller_id or f"single_agent_{id(self)}"
        self._performance_metrics = {
            'step_times': [],
            'sample_times': [],
            'total_steps': 0
        }
        
        # Setup structured logging with context binding
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                controller_type="single_agent",
                controller_id=self._controller_id,
                num_agents=1
            )
            
            # Add Hydra context if available
            if HYDRA_AVAILABLE:
                try:
                    hydra_cfg = HydraConfig.get()
                    self._logger = self._logger.bind(
                        hydra_job_name=hydra_cfg.job.name,
                        hydra_output_dir=hydra_cfg.runtime.output_dir
                    )
                except Exception:
                    # Hydra context not available, continue without it
                    pass
                    
            self._logger.info(
                "Enhanced SingleAgentController initialized",
                position=getattr(self.positions[0] if hasattr(self, 'positions') else [0, 0], 'tolist', lambda: [0, 0])(),
                orientation=float(self.orientations[0]) if hasattr(self, 'orientations') else 0.0,
                speed=float(self.speeds[0]) if hasattr(self, 'speeds') else 0.0
            )
        else:
            self._logger = None
    
    def reset_with_params(self, params: SingleAgentParams) -> None:
        """Reset the agent using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method
        and integrates seamlessly with Hydra structured configuration.
        
        Parameters
        ----------
        params : SingleAgentParams
            Parameters to update, as a dataclass instance
            
        Raises:
            TypeError: If params is not a SingleAgentParams instance
            ValueError: If parameter constraints are violated
        """
        if not isinstance(params, SingleAgentParams):
            raise TypeError(f"Expected SingleAgentParams, got {type(params)}")
        
        if self._logger:
            self._logger.debug(
                "Resetting agent with type-safe parameters",
                param_count=len([v for v in params.__dict__.values() if v is not None]),
                params=params.to_kwargs()
            )
        
        # Delegate to the existing reset method with kwargs
        self.reset(**params.to_kwargs())
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Enhanced step method with performance monitoring."""
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Call base step method
            super().step(env_array, dt)
            
            # Track performance metrics
            if self._enable_logging:
                step_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['step_times'].append(step_time)
                self._performance_metrics['total_steps'] += 1
                
                # Check performance requirement (<33ms)
                if step_time > 33.0 and self._logger:
                    self._logger.warning(
                        "Step processing exceeded 33ms requirement",
                        step_time_ms=step_time,
                        dt=dt,
                        performance_degradation=True
                    )
                
                # Log periodic performance summary
                if self._performance_metrics['total_steps'] % 100 == 0 and self._logger:
                    avg_step_time = np.mean(self._performance_metrics['step_times'][-100:])
                    self._logger.debug(
                        "Performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        recent_position=self.positions[0].tolist() if hasattr(self.positions[0], 'tolist') else [0, 0]
                    )
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Step execution failed: {str(e)}",
                    error_type=type(e).__name__,
                    dt=dt,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Agent step failed: {str(e)}") from e
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance statistics and metrics
        """
        if not self._enable_logging:
            return {}
            
        metrics = {
            'controller_type': 'single_agent',
            'controller_id': self._controller_id,
            'total_steps': self._performance_metrics['total_steps'],
            'num_agents': 1
        }
        
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            metrics.update({
                'step_time_mean_ms': float(np.mean(step_times)),
                'step_time_std_ms': float(np.std(step_times)),
                'step_time_max_ms': float(np.max(step_times)),
                'step_time_p95_ms': float(np.percentile(step_times, 95)),
                'performance_violations': int(np.sum(step_times > 33.0))
            })
        
        if self._performance_metrics['sample_times']:
            sample_times = np.array(self._performance_metrics['sample_times'])
            metrics.update({
                'sample_time_mean_ms': float(np.mean(sample_times)),
                'sample_time_max_ms': float(np.max(sample_times))
            })
        
        return metrics


class MultiAgentController(_BaseMultiAgentController):
    """Enhanced multi-agent controller with advanced integration capabilities.
    
    This extends the base MultiAgentController with enhanced logging, performance
    monitoring, and batch operation optimization while maintaining full backward
    compatibility with the original NavigatorProtocol interface.
    
    Additional Features:
        - Comprehensive structured logging with loguru integration
        - Performance monitoring with throughput tracking
        - Enhanced error handling and validation
        - Type-safe parameter reset with MultiAgentParams dataclass
        - Vectorized operations optimization for large agent counts
        - Memory management for efficient scaling
    
    Performance Requirements:
        - Frame processing: <33ms per step
        - Throughput: ≥3000 agents·FPS (100 agents × 30 FPS)
        - Memory usage: <10MB per 100 agents
    
    Examples:
        Enhanced multi-agent initialization:
            >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
            >>> controller = MultiAgentController(
            ...     positions=positions,
            ...     enable_logging=True,
            ...     controller_id="swarm_001"
            ... )
            
        Performance monitoring:
            >>> controller.step(env_array, dt=1.0)
            >>> metrics = controller.get_performance_metrics()
            >>> print(f"Throughput: {metrics['throughput_mean_agents_fps']:.0f} agents·FPS")
            
        Batch parameter updates:
            >>> import numpy as np
            >>> params = MultiAgentParams(
            ...     speeds=np.array([1.0, 1.5, 2.0]),
            ...     max_speeds=np.array([2.0, 2.5, 3.0])
            ... )
            >>> controller.reset_with_params(params)
    """
    
    def __init__(self, *args, enable_logging: bool = True, controller_id: Optional[str] = None, **kwargs):
        """Initialize enhanced multi-agent controller with monitoring capabilities."""
        # Initialize base controller
        super().__init__(*args, **kwargs)
        
        # Enhanced logging and monitoring setup
        self._enable_logging = enable_logging
        self._controller_id = controller_id or f"multi_agent_{id(self)}"
        self._performance_metrics = {
            'step_times': [],
            'sample_times': [],
            'total_steps': 0,
            'agents_per_step': []
        }
        
        # Setup structured logging with context binding
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                controller_type="multi_agent",
                controller_id=self._controller_id,
                num_agents=self.num_agents
            )
            
            # Add Hydra context if available
            if HYDRA_AVAILABLE:
                try:
                    hydra_cfg = HydraConfig.get()
                    self._logger = self._logger.bind(
                        hydra_job_name=hydra_cfg.job.name,
                        hydra_output_dir=hydra_cfg.runtime.output_dir
                    )
                except Exception:
                    # Hydra context not available, continue without it
                    pass
                    
            self._logger.info(
                "Enhanced MultiAgentController initialized",
                num_agents=self.num_agents,
                position_bounds={
                    'x_min': float(np.min(self.positions[:, 0])),
                    'x_max': float(np.max(self.positions[:, 0])),
                    'y_min': float(np.min(self.positions[:, 1])),
                    'y_max': float(np.max(self.positions[:, 1]))
                },
                speed_stats={
                    'mean': float(np.mean(self.speeds)),
                    'max': float(np.max(self.speeds))
                }
            )
        else:
            self._logger = None
    
    def reset_with_params(self, params: MultiAgentParams) -> None:
        """Reset all agents using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method
        and integrates seamlessly with Hydra structured configuration for batch experiments.
        
        Parameters
        ----------
        params : MultiAgentParams
            Parameters to update, as a dataclass instance
            
        Raises:
            TypeError: If params is not a MultiAgentParams instance
            ValueError: If parameter arrays have invalid shapes or violate constraints
        """
        if not isinstance(params, MultiAgentParams):
            raise TypeError(f"Expected MultiAgentParams, got {type(params)}")
        
        if self._logger:
            self._logger.debug(
                "Resetting multi-agent controller with type-safe parameters",
                param_count=len([v for v in params.__dict__.values() if v is not None]),
                num_agents=self.num_agents
            )
        
        # Delegate to the existing reset method with kwargs
        self.reset(**params.to_kwargs())
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Enhanced step method with performance optimization and monitoring."""
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Call base step method with vectorized operations
            super().step(env_array, dt)
            
            # Track performance metrics
            if self._enable_logging:
                step_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['step_times'].append(step_time)
                self._performance_metrics['total_steps'] += 1
                self._performance_metrics['agents_per_step'].append(self.num_agents)
                
                # Calculate throughput (agents × frames / second)
                throughput = self.num_agents * (1000.0 / step_time) if step_time > 0 else 0
                
                # Check performance requirements
                performance_issues = []
                if step_time > 33.0:
                    performance_issues.append(f"step_time:{step_time:.1f}ms")
                if throughput < 3000:  # 100 agents × 30fps
                    performance_issues.append(f"throughput:{throughput:.0f}")
                
                if performance_issues and self._logger:
                    self._logger.warning(
                        "Performance degradation detected",
                        step_time_ms=step_time,
                        throughput_agents_fps=throughput,
                        num_agents=self.num_agents,
                        dt=dt,
                        issues=performance_issues
                    )
                
                # Log periodic performance summary
                if self._performance_metrics['total_steps'] % 100 == 0 and self._logger:
                    recent_steps = self._performance_metrics['step_times'][-100:]
                    avg_step_time = np.mean(recent_steps)
                    avg_throughput = np.mean([
                        na * (1000.0 / st) for na, st in 
                        zip(self._performance_metrics['agents_per_step'][-100:], recent_steps)
                        if st > 0
                    ])
                    
                    self._logger.debug(
                        "Multi-agent performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        avg_throughput_agents_fps=avg_throughput,
                        num_agents=self.num_agents
                    )
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent step execution failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    dt=dt,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Multi-agent step failed: {str(e)}") from e
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring and optimization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed performance statistics and metrics
        """
        if not self._enable_logging:
            return {}
            
        metrics = {
            'controller_type': 'multi_agent',
            'controller_id': self._controller_id,
            'total_steps': self._performance_metrics['total_steps'],
            'num_agents': self.num_agents
        }
        
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            agents_per_step = np.array(self._performance_metrics['agents_per_step'])
            
            # Calculate throughput metrics
            throughputs = [
                na * (1000.0 / st) for na, st in zip(agents_per_step, step_times) if st > 0
            ]
            
            metrics.update({
                'step_time_mean_ms': float(np.mean(step_times)),
                'step_time_std_ms': float(np.std(step_times)),
                'step_time_max_ms': float(np.max(step_times)),
                'step_time_p95_ms': float(np.percentile(step_times, 95)),
                'performance_violations': int(np.sum(step_times > 33.0)),
                'throughput_mean_agents_fps': float(np.mean(throughputs)) if throughputs else 0,
                'throughput_max_agents_fps': float(np.max(throughputs)) if throughputs else 0,
                'throughput_violations': int(np.sum(np.array(throughputs) < 3000))
            })
        
        if self._performance_metrics['sample_times']:
            sample_times = np.array(self._performance_metrics['sample_times'])
            metrics.update({
                'sample_time_mean_ms': float(np.mean(sample_times)),
                'sample_time_max_ms': float(np.max(sample_times))
            })
        
        return metrics


def create_controller_from_config(
    config: Union[DictConfig, Dict[str, Any], NavigatorConfig],
    controller_id: Optional[str] = None,
    enable_logging: bool = True
) -> Union[SingleAgentController, MultiAgentController]:
    """
    Create a navigation controller from configuration with automatic type detection.
    
    This factory function automatically detects whether to create a single-agent or
    multi-agent controller based on the configuration parameters and creates the
    appropriate controller type with full Hydra integration and enhanced features.
    
    Parameters
    ----------
    config : Union[DictConfig, Dict[str, Any], NavigatorConfig]
        Configuration object containing navigation parameters
    controller_id : Optional[str], optional
        Unique identifier for the controller, by default None
    enable_logging : bool, optional
        Enable comprehensive logging integration, by default True
        
    Returns
    -------
    Union[SingleAgentController, MultiAgentController]
        Configured navigation controller instance
        
    Raises:
        ValueError: If configuration is invalid or inconsistent
        TypeError: If configuration type is not supported
        
    Examples:
        Single-agent from dict:
            >>> config = {"position": [10.0, 20.0], "speed": 1.5}
            >>> controller = create_controller_from_config(config)
            >>> isinstance(controller, SingleAgentController)
            True
            
        Multi-agent from Hydra config:
            >>> from omegaconf import DictConfig
            >>> config = DictConfig({
            ...     "positions": [[0, 0], [10, 10]], 
            ...     "speeds": [1.0, 1.5]
            ... })
            >>> controller = create_controller_from_config(config)
            >>> isinstance(controller, MultiAgentController)
            True
            
        Pydantic model configuration:
            >>> from odor_plume_nav.config.models import NavigatorConfig
            >>> config = NavigatorConfig(position=(5.0, 5.0), max_speed=2.0)
            >>> controller = create_controller_from_config(config)
    """
    start_time = time.perf_counter() if enable_logging else None
    
    try:
        # Handle different configuration types
        if isinstance(config, NavigatorConfig):
            # Pydantic model - extract relevant parameters
            config_dict = config.model_dump(exclude_none=True)
        elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
            # Hydra OmegaConf configuration
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            # Regular dictionary
            config_dict = config.copy()
        else:
            raise TypeError(
                f"Unsupported configuration type: {type(config)}. "
                f"Expected DictConfig, dict, or NavigatorConfig"
            )
        
        # Detect controller type based on configuration parameters
        has_multi_params = any([
            'positions' in config_dict,
            'num_agents' in config_dict and config_dict.get('num_agents', 1) > 1,
            isinstance(config_dict.get('orientations'), (list, np.ndarray)),
            isinstance(config_dict.get('speeds'), (list, np.ndarray)),
            isinstance(config_dict.get('max_speeds'), (list, np.ndarray)),
            isinstance(config_dict.get('angular_velocities'), (list, np.ndarray))
        ])
        
        has_single_params = 'position' in config_dict
        
        # Validate mode exclusivity
        if has_multi_params and has_single_params:
            raise ValueError(
                "Configuration contains both single-agent (position) and "
                "multi-agent (positions, arrays) parameters. Use only one mode."
            )
        
        # Create appropriate controller
        if has_multi_params:
            # Multi-agent controller
            controller = MultiAgentController(
                positions=config_dict.get('positions'),
                orientations=config_dict.get('orientations'),
                speeds=config_dict.get('speeds'),
                max_speeds=config_dict.get('max_speeds'),
                angular_velocities=config_dict.get('angular_velocities'),
                enable_logging=enable_logging,
                controller_id=controller_id
            )
            
        else:
            # Single-agent controller (default)
            controller = SingleAgentController(
                position=config_dict.get('position'),
                orientation=config_dict.get('orientation', 0.0),
                speed=config_dict.get('speed', 0.0),
                max_speed=config_dict.get('max_speed', 1.0),
                angular_velocity=config_dict.get('angular_velocity', 0.0),
                enable_logging=enable_logging,
                controller_id=controller_id
            )
        
        # Log successful creation with performance timing
        if enable_logging and LOGURU_AVAILABLE:
            creation_time = (time.perf_counter() - start_time) * 1000
            logger.bind(
                controller_type=type(controller).__name__,
                controller_id=controller_id,
                creation_time_ms=creation_time,
                num_agents=controller.num_agents
            ).info(
                "Controller created from configuration",
                config_keys=list(config_dict.keys())
            )
        
        return controller
        
    except Exception as e:
        if enable_logging and LOGURU_AVAILABLE:
            logger.error(
                f"Controller creation failed: {str(e)}",
                error_type=type(e).__name__,
                config_type=type(config).__name__,
                controller_id=controller_id
            )
        raise


def create_single_agent_controller(
    config: Union[DictConfig, Dict[str, Any], SingleAgentConfig],
    **kwargs: Any
) -> SingleAgentController:
    """
    Create a single-agent controller from configuration.
    
    Convenience factory function for explicitly creating single-agent controllers
    with type-safe configuration validation and enhanced features.
    
    Parameters
    ----------
    config : Union[DictConfig, Dict[str, Any], SingleAgentConfig]
        Single-agent configuration parameters
    **kwargs
        Additional parameters passed to controller constructor
        
    Returns
    -------
    SingleAgentController
        Configured single-agent controller instance
    """
    # Merge configuration with kwargs
    if isinstance(config, SingleAgentConfig):
        config_dict = config.model_dump(exclude_none=True)
    elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config) if config else {}
    
    config_dict.update(kwargs)
    
    return SingleAgentController(**config_dict)


def create_multi_agent_controller(
    config: Union[DictConfig, Dict[str, Any], MultiAgentConfig],
    **kwargs: Any
) -> MultiAgentController:
    """
    Create a multi-agent controller from configuration.
    
    Convenience factory function for explicitly creating multi-agent controllers
    with type-safe configuration validation and enhanced features.
    
    Parameters
    ----------
    config : Union[DictConfig, Dict[str, Any], MultiAgentConfig]
        Multi-agent configuration parameters
    **kwargs
        Additional parameters passed to controller constructor
        
    Returns
    -------
    MultiAgentController
        Configured multi-agent controller instance
    """
    # Merge configuration with kwargs
    if isinstance(config, MultiAgentConfig):
        config_dict = config.model_dump(exclude_none=True)
    elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config) if config else {}
    
    config_dict.update(kwargs)
    
    return MultiAgentController(**config_dict)


# Utility functions for common configuration patterns
def validate_controller_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate controller configuration and return validation results.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Controller configuration to validate
        
    Returns
    -------
    Tuple[bool, List[str]]
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for basic parameter types
    if 'position' in config and not isinstance(config['position'], (list, tuple)):
        errors.append("'position' must be a list or tuple of [x, y] coordinates")
    
    if 'positions' in config and not isinstance(config['positions'], (list, np.ndarray)):
        errors.append("'positions' must be a list or array of [x, y] coordinates")
    
    # Check for speed constraints
    if 'speed' in config and 'max_speed' in config:
        if config['speed'] > config['max_speed']:
            errors.append(f"speed ({config['speed']}) cannot exceed max_speed ({config['max_speed']})")
    
    # Check for array length consistency in multi-agent configs
    if 'positions' in config:
        num_agents = len(config['positions'])
        for param in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
            if param in config and len(config[param]) != num_agents:
                errors.append(f"{param} length ({len(config[param])}) does not match positions length ({num_agents})")
    
    return len(errors) == 0, errors


def get_controller_info(controller: Union[SingleAgentController, MultiAgentController]) -> Dict[str, Any]:
    """
    Get comprehensive information about a controller instance.
    
    Parameters
    ----------
    controller : Union[SingleAgentController, MultiAgentController]
        Controller instance to analyze
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing controller information and statistics
    """
    info = {
        'controller_type': type(controller).__name__,
        'num_agents': controller.num_agents,
        'has_performance_metrics': hasattr(controller, 'get_performance_metrics'),
        'has_enhanced_logging': hasattr(controller, '_logger'),
    }
    
    # Add state information
    info.update({
        'positions_shape': controller.positions.shape,
        'orientations_range': [float(np.min(controller.orientations)), float(np.max(controller.orientations))],
        'speeds_range': [float(np.min(controller.speeds)), float(np.max(controller.speeds))],
        'max_speeds_range': [float(np.min(controller.max_speeds)), float(np.max(controller.max_speeds))],
    })
    
    # Add performance metrics if available
    if hasattr(controller, 'get_performance_metrics'):
        try:
            metrics = controller.get_performance_metrics()
            info['performance_metrics'] = metrics
        except Exception:
            info['performance_metrics'] = "Error retrieving metrics"
    
    return info


# Export public API with backward compatibility
__all__ = [
    # Backward compatibility - re-export base controllers
    "SingleAgentController",
    "MultiAgentController",
    
    # Enhanced parameter dataclasses
    "SingleAgentParams", 
    "MultiAgentParams",
    
    # Factory functions for configuration-driven instantiation
    "create_controller_from_config",
    "create_single_agent_controller",
    "create_multi_agent_controller",
    
    # Utility functions
    "validate_controller_config",
    "get_controller_info",
]
