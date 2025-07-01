"""
Examples package for demonstrating modular plume navigation architecture capabilities.

This module provides comprehensive entry points and utilities for showcasing how the same 
configurable simulation foundation supports diverse research scenarios through configuration 
rather than code changes. Serves as the central hub for demonstrating memory-based vs 
memory-less navigation strategies, CLI execution capabilities, and the extensible protocol-based 
agent abstractions that enable seamless algorithm switching.

The examples package fulfills the critical requirement to show how researchers can:
- Toggle between memory-based and non-memory-based navigation scenarios via configuration
- Use identical simulator infrastructure for different cognitive modeling approaches
- Run example agents via CLI for reproducible experimentation
- Extend the system with custom navigation strategies while maintaining compatibility

Key Architecture Features Demonstrated:
- Configurable plume models (Gaussian, Turbulent, Video-based) via PlumeModelProtocol
- Pluggable wind dynamics (Constant, Turbulent, Time-varying) via WindFieldProtocol  
- Flexible sensor modalities (Binary, Concentration, Gradient) via SensorProtocol
- Agent-agnostic design ensuring no assumptions about internal logic or memory usage
- Identical observation and action spaces across all agent implementations
- Performance-preserving modularity maintaining sub-10ms step latency requirements

Agent Categories:
1. **Memory-less (Reactive) Agents**: Make decisions based solely on current observations
   - ReactiveAgent: Pure gradient-following with no internal state
   - CastingAgent: Biologically-inspired zigzag search pattern
   
2. **Memory-based (Planning) Agents**: Maintain internal state for strategic decisions
   - InfotaxisAgent: Information-theoretic source localization with spatial memory
   - Custom example agents with trajectory history and belief state management

CLI Execution Examples:
    Run reactive (memory-less) agent with Gaussian plume:
    >>> python -m plume_nav_sim.examples run_agent \
    ...     --agent_type=ReactiveAgent \
    ...     --plume_model=GaussianPlumeModel \
    ...     --num_steps=1000 \
    ...     --config_file=examples/configs/reactive_gaussian.yaml

    Run memory-based agent with turbulent plume:
    >>> python -m plume_nav_sim.examples run_agent \
    ...     --agent_type=InfotaxisAgent \
    ...     --plume_model=TurbulentPlumeModel \
    ...     --enable_memory=true \
    ...     --num_steps=1000 \
    ...     --config_file=examples/configs/infotaxis_turbulent.yaml

    Compare agents with identical environment configuration:
    >>> python -m plume_nav_sim.examples compare_agents \
    ...     --agents=ReactiveAgent,InfotaxisAgent \
    ...     --environment_config=examples/configs/standard_environment.yaml \
    ...     --num_episodes=50 \
    ...     --output_dir=results/comparison

Configuration-Driven Approach Examples:
    Memory-less reactive configuration:
    >>> agent_config = {
    ...     'type': 'ReactiveAgent',
    ...     'memory_enabled': False,
    ...     'sensor_config': {'type': 'ConcentrationSensor'},
    ...     'navigation_strategy': 'gradient_following'
    ... }
    >>> agent = create_example_agent(agent_config)

    Memory-based planning configuration:
    >>> agent_config = {
    ...     'type': 'InfotaxisAgent', 
    ...     'memory_enabled': True,
    ...     'belief_state_size': (100, 100),
    ...     'information_gain_threshold': 0.1,
    ...     'trajectory_history_length': 500
    ... }
    >>> agent = create_example_agent(agent_config)

Performance Requirements:
- Agent instantiation: <50ms for complex memory-based agents  
- Step execution: <1ms for reactive agents, <5ms for planning agents
- Memory usage: <10MB per agent for trajectory storage and belief states
- Configuration switching: <100ms for runtime agent type changes
- CLI execution: Support for 1000+ time steps with real-time visualization
"""

from __future__ import annotations
import sys
import time
import warnings
from typing import Optional, Dict, List, Any, Union, Type, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import importlib
import argparse

# Core system imports for agent integration
from ..core.protocols import NavigatorProtocol, PlumeModelProtocol, WindFieldProtocol, SensorProtocol
from ..core.simulation import SimulationContext, run_simulation
from ..models import create_plume_model, create_wind_field, create_sensors, get_model_registry

# Enhanced logging and configuration support
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize_config_dir
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False

# Optional dependencies for enhanced functionality
try:
    import numpy as np
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    import numpy as np
    VISUALIZATION_AVAILABLE = False

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False


# Global registry for example agents
_EXAMPLE_AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Import status tracking for graceful degradation
_AGENT_IMPORT_STATUS: Dict[str, Dict[str, Any]] = {}


@dataclass
class AgentExecutionConfig:
    """
    Configuration for agent execution scenarios supporting both memory-based and memory-less approaches.
    
    This dataclass provides type-safe configuration for demonstrating different navigation strategies
    while maintaining identical environmental conditions and evaluation metrics. Enables fair
    comparison between cognitive modeling approaches through standardized parameter management.
    
    Attributes:
        agent_type: Name of agent implementation (ReactiveAgent, InfotaxisAgent, CastingAgent)
        memory_enabled: Whether agent maintains internal state and trajectory history
        plume_model_config: Configuration for plume physics model
        wind_field_config: Optional wind dynamics configuration
        sensor_configs: List of sensor configurations for multi-modal sensing
        environment_config: Environment-specific parameters (bounds, obstacles, etc.)
        execution_config: Simulation execution parameters (steps, dt, visualization)
        output_config: Result collection and analysis configuration
    """
    agent_type: str = 'ReactiveAgent'
    memory_enabled: bool = False
    plume_model_config: Dict[str, Any] = field(default_factory=lambda: {'type': 'GaussianPlumeModel'})
    wind_field_config: Optional[Dict[str, Any]] = None
    sensor_configs: List[Dict[str, Any]] = field(default_factory=lambda: [{'type': 'ConcentrationSensor'}])
    environment_config: Dict[str, Any] = field(default_factory=dict)
    execution_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_steps': 1000,
        'dt': 1.0,
        'target_fps': 30.0,
        'enable_visualization': False,
        'save_trajectory': True
    })
    output_config: Dict[str, Any] = field(default_factory=lambda: {
        'save_results': True,
        'output_dir': 'results',
        'metrics_collection': True,
        'performance_monitoring': True
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization and logging."""
        return {
            'agent_type': self.agent_type,
            'memory_enabled': self.memory_enabled,
            'plume_model_config': self.plume_model_config,
            'wind_field_config': self.wind_field_config,
            'sensor_configs': self.sensor_configs,
            'environment_config': self.environment_config,
            'execution_config': self.execution_config,
            'output_config': self.output_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentExecutionConfig':
        """Create configuration from dictionary with validation."""
        return cls(**config_dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters for consistency and requirements.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate agent type is available
        if self.agent_type not in list_available_example_agents():
            errors.append(f"Unknown agent type: {self.agent_type}")
        
        # Validate memory configuration consistency
        if self.agent_type == 'ReactiveAgent' and self.memory_enabled:
            errors.append("ReactiveAgent is memory-less by design, cannot enable memory")
        elif self.agent_type == 'InfotaxisAgent' and not self.memory_enabled:
            errors.append("InfotaxisAgent requires memory for belief state management")
        
        # Validate plume model configuration
        if 'type' not in self.plume_model_config:
            errors.append("plume_model_config must include 'type' field")
        
        # Validate execution parameters
        exec_config = self.execution_config
        if exec_config.get('num_steps', 0) <= 0:
            errors.append("num_steps must be positive")
        if exec_config.get('dt', 0) <= 0:
            errors.append("dt must be positive")
        if exec_config.get('target_fps', 0) <= 0:
            errors.append("target_fps must be positive")
        
        return len(errors) == 0, errors


def register_example_agent(
    agent_name: str,
    agent_class: Type[NavigatorProtocol],
    description: str,
    memory_required: bool = False,
    config_schema: Optional[Dict[str, Any]] = None,
    performance_characteristics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register an example agent implementation in the global registry.
    
    Args:
        agent_name: Unique identifier for the agent (e.g., 'ReactiveAgent')
        agent_class: Class implementing NavigatorProtocol
        description: Human-readable description of the agent's capabilities and approach
        memory_required: Whether agent requires memory to function properly
        config_schema: Optional schema defining expected configuration parameters
        performance_characteristics: Optional performance metrics and requirements
    
    Examples:
        Register a custom reactive agent:
        >>> register_example_agent(
        ...     'CustomReactiveAgent',
        ...     CustomReactiveAgent,
        ...     'Custom gradient-following agent with enhanced obstacle avoidance',
        ...     memory_required=False,
        ...     config_schema={'obstacle_avoidance_distance': float},
        ...     performance_characteristics={'step_time_ms': 0.5, 'memory_usage_mb': 1}
        ... )
    """
    _EXAMPLE_AGENT_REGISTRY[agent_name] = {
        'class': agent_class,
        'description': description,
        'memory_required': memory_required,
        'config_schema': config_schema or {},
        'performance_characteristics': performance_characteristics or {},
        'module_path': f"{agent_class.__module__}.{agent_class.__name__}"
    }
    
    if LOGURU_AVAILABLE:
        logger.debug(
            f"Registered example agent: {agent_name}",
            agent_class=agent_class.__name__,
            memory_required=memory_required,
            description=description
        )


def create_example_agent(config: Union[Dict[str, Any], DictConfig, AgentExecutionConfig]) -> NavigatorProtocol:
    """
    Factory function to create example agent instances from configuration.
    
    This function demonstrates the configurable architecture by enabling seamless switching
    between different agent implementations while maintaining identical environment interfaces.
    Supports both memory-based and memory-less agents through unified configuration approach.
    
    Args:
        config: Agent configuration containing type and parameters.
            Supports AgentExecutionConfig, dict, or DictConfig formats.
            
    Returns:
        NavigatorProtocol: Configured agent instance implementing the protocol
        
    Raises:
        ValueError: If agent type is unknown or configuration is invalid
        ImportError: If required agent implementation is not available
        TypeError: If created instance doesn't implement NavigatorProtocol
        
    Examples:
        Create memory-less reactive agent:
        >>> config = {
        ...     'agent_type': 'ReactiveAgent',
        ...     'memory_enabled': False,
        ...     'gradient_threshold': 0.01,
        ...     'step_size': 1.0
        ... }
        >>> agent = create_example_agent(config)
        
        Create memory-based planning agent:
        >>> config = AgentExecutionConfig(
        ...     agent_type='InfotaxisAgent',
        ...     memory_enabled=True,
        ...     execution_config={'num_steps': 2000}
        ... )
        >>> agent = create_example_agent(config)
        
        Configuration switching demonstration:
        >>> # Same environment, different agents
        >>> base_config = {'environment_size': (100, 100), 'source_position': (50, 50)}
        >>> reactive_config = {**base_config, 'agent_type': 'ReactiveAgent', 'memory_enabled': False}
        >>> planning_config = {**base_config, 'agent_type': 'InfotaxisAgent', 'memory_enabled': True}
        >>> 
        >>> reactive_agent = create_example_agent(reactive_config)
        >>> planning_agent = create_example_agent(planning_config)
    """
    start_time = time.perf_counter() if LOGURU_AVAILABLE else None
    
    try:
        # Normalize configuration format
        if isinstance(config, AgentExecutionConfig):
            agent_type = config.agent_type
            agent_config = config.to_dict()
        elif isinstance(config, (dict, DictConfig)):
            agent_type = config.get('agent_type', 'ReactiveAgent')
            agent_config = dict(config) if isinstance(config, DictConfig) else config
        else:
            raise TypeError(f"Unsupported configuration type: {type(config)}")
        
        # Validate configuration
        if isinstance(config, AgentExecutionConfig):
            is_valid, errors = config.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
        
        # Try registry lookup first
        if agent_type in _EXAMPLE_AGENT_REGISTRY:
            try:
                agent_info = _EXAMPLE_AGENT_REGISTRY[agent_type]
                agent_class = agent_info['class']
                
                # Validate memory requirements
                memory_enabled = agent_config.get('memory_enabled', False)
                memory_required = agent_info['memory_required']
                
                if memory_required and not memory_enabled:
                    raise ValueError(
                        f"Agent {agent_type} requires memory but memory_enabled=False"
                    )
                elif not memory_required and memory_enabled:
                    logger.warning(
                        f"Agent {agent_type} is memory-less but memory_enabled=True. "
                        f"Memory features will be ignored."
                    )
                
                # Extract agent-specific configuration
                agent_params = {k: v for k, v in agent_config.items() 
                              if k not in ['agent_type', 'plume_model_config', 'wind_field_config', 
                                         'sensor_configs', 'environment_config', 'execution_config', 'output_config']}
                
                agent = agent_class(**agent_params)
                
                if LOGURU_AVAILABLE:
                    creation_time = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"Created example agent via registry",
                        agent_type=agent_type,
                        memory_enabled=memory_enabled,
                        creation_time_ms=creation_time,
                        config_keys=list(agent_params.keys())
                    )
                
                return agent
                
            except Exception as e:
                raise ValueError(
                    f"Failed to create example agent '{agent_type}' from registry. Error: {e}"
                )
        
        # Fallback to direct import attempts for common agents
        try:
            if agent_type == 'ReactiveAgent':
                from .agents.reactive_agent import ReactiveAgent
                agent_class = ReactiveAgent
            elif agent_type == 'InfotaxisAgent':
                from .agents.infotaxis_agent import InfotaxisAgent
                agent_class = InfotaxisAgent
            elif agent_type == 'CastingAgent':
                from .agents.casting_agent import CastingAgent
                agent_class = CastingAgent
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Create agent instance with configuration
            agent_params = {k: v for k, v in agent_config.items()
                          if k not in ['agent_type', 'plume_model_config', 'wind_field_config',
                                     'sensor_configs', 'environment_config', 'execution_config', 'output_config']}
            
            agent = agent_class(**agent_params)
            
            if LOGURU_AVAILABLE:
                creation_time = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Created example agent via direct import",
                    agent_type=agent_type,
                    module=agent_class.__module__,
                    creation_time_ms=creation_time
                )
            
            return agent
            
        except ImportError as e:
            # Update import status for diagnostics
            _AGENT_IMPORT_STATUS[agent_type] = {
                'available': False,
                'error': str(e),
                'attempted_import': True
            }
            
            raise ImportError(
                f"Example agent implementation not available: {agent_type}. "
                f"Ensure the corresponding module has been created. "
                f"Available agents: {list_available_example_agents()}. "
                f"Error: {e}"
            )
            
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.error(
                f"Failed to create example agent: {str(e)}",
                agent_type=agent_type if 'agent_type' in locals() else 'unknown',
                error_type=type(e).__name__
            )
        raise


def create_modular_simulation_environment(config: Union[Dict[str, Any], AgentExecutionConfig]) -> Dict[str, Any]:
    """
    Create complete simulation environment with modular components for agent demonstrations.
    
    This function showcases the modular architecture by assembling plume models, wind fields,
    sensors, and agents from configuration while maintaining component interoperability. 
    Demonstrates how the same simulation foundation supports diverse research scenarios.
    
    Args:
        config: Complete configuration for environment and agent setup
        
    Returns:
        Dict[str, Any]: Dictionary containing all simulation components:
            - 'agent': Configured agent instance
            - 'plume_model': Plume physics model
            - 'wind_field': Optional wind dynamics model  
            - 'sensors': List of sensor instances
            - 'environment': Environment wrapper/context
            - 'metadata': Configuration and component information
            
    Examples:
        Memory-less reactive simulation:
        >>> config = AgentExecutionConfig(
        ...     agent_type='ReactiveAgent',
        ...     memory_enabled=False,
        ...     plume_model_config={'type': 'GaussianPlumeModel', 'source_position': (50, 50)},
        ...     sensor_configs=[{'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}]
        ... )
        >>> sim_env = create_modular_simulation_environment(config)
        >>> agent = sim_env['agent']
        >>> plume_model = sim_env['plume_model']
        
        Memory-based planning simulation with complex environment:
        >>> config = AgentExecutionConfig(
        ...     agent_type='InfotaxisAgent',
        ...     memory_enabled=True,
        ...     plume_model_config={'type': 'TurbulentPlumeModel', 'filament_count': 500},
        ...     wind_field_config={'type': 'TurbulentWindField', 'turbulence_intensity': 0.3},
        ...     sensor_configs=[
        ...         {'type': 'ConcentrationSensor'},
        ...         {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
        ...     ]
        ... )
        >>> sim_env = create_modular_simulation_environment(config)
        >>> # All components use identical interfaces despite complexity differences
    """
    start_time = time.perf_counter() if LOGURU_AVAILABLE else None
    
    try:
        # Normalize configuration
        if isinstance(config, dict):
            exec_config = AgentExecutionConfig.from_dict(config)
        elif isinstance(config, AgentExecutionConfig):
            exec_config = config
        else:
            raise TypeError(f"Expected dict or AgentExecutionConfig, got {type(config)}")
        
        # Validate configuration
        is_valid, errors = exec_config.validate()
        if not is_valid:
            raise ValueError(f"Invalid simulation configuration: {errors}")
        
        components = {}
        
        # Create agent
        components['agent'] = create_example_agent(exec_config)
        
        # Create plume model
        components['plume_model'] = create_plume_model(exec_config.plume_model_config)
        
        # Create wind field (optional)
        if exec_config.wind_field_config:
            components['wind_field'] = create_wind_field(exec_config.wind_field_config)
        else:
            components['wind_field'] = None
        
        # Create sensors
        components['sensors'] = create_sensors(exec_config.sensor_configs)
        
        # Create simulation context/environment
        if GYMNASIUM_AVAILABLE:
            # TODO: Create Gymnasium environment when available
            components['environment'] = None
        else:
            components['environment'] = None
        
        # Store configuration and metadata
        components['configuration'] = exec_config
        components['metadata'] = {
            'agent_type': exec_config.agent_type,
            'memory_enabled': exec_config.memory_enabled,
            'plume_model_type': type(components['plume_model']).__name__,
            'wind_field_type': type(components['wind_field']).__name__ if components['wind_field'] else None,
            'sensor_count': len(components['sensors']),
            'sensor_types': [type(s).__name__ for s in components['sensors']],
            'modular_architecture_demo': True,
            'created_components': list(components.keys())
        }
        
        if LOGURU_AVAILABLE:
            setup_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Created modular simulation environment",
                setup_time_ms=setup_time,
                **components['metadata']
            )
        
        return components
        
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.error(
                f"Failed to create modular simulation environment: {str(e)}",
                error_type=type(e).__name__
            )
        raise


def run_agent_demonstration(
    config: Union[Dict[str, Any], AgentExecutionConfig],
    save_results: bool = True,
    enable_visualization: bool = False
) -> Dict[str, Any]:
    """
    Execute complete agent demonstration showcasing modular architecture capabilities.
    
    This function provides end-to-end demonstration of agent execution with the modular
    simulation framework, collecting performance metrics and results that highlight the
    benefits of the configurable architecture. Supports both memory-based and memory-less
    agents with identical evaluation protocols.
    
    Args:
        config: Agent and environment configuration
        save_results: Whether to save execution results and metrics
        enable_visualization: Whether to enable real-time visualization
        
    Returns:
        Dict[str, Any]: Complete execution results including:
            - 'trajectory': Agent movement history
            - 'performance_metrics': Timing and efficiency statistics  
            - 'environment_data': Plume and wind field information
            - 'agent_state': Final agent state and memory contents (if applicable)
            - 'configuration': Complete configuration used for execution
            
    Examples:
        Run reactive agent demonstration:
        >>> config = AgentExecutionConfig(
        ...     agent_type='ReactiveAgent',
        ...     execution_config={'num_steps': 1000, 'enable_visualization': False}
        ... )
        >>> results = run_agent_demonstration(config, save_results=True)
        >>> print(f"Success rate: {results['performance_metrics']['success_rate']:.2%}")
        
        Compare memory vs non-memory performance:
        >>> reactive_config = AgentExecutionConfig(agent_type='ReactiveAgent', memory_enabled=False)
        >>> planning_config = AgentExecutionConfig(agent_type='InfotaxisAgent', memory_enabled=True)
        >>> 
        >>> reactive_results = run_agent_demonstration(reactive_config)
        >>> planning_results = run_agent_demonstration(planning_config)
        >>> 
        >>> print("Reactive agent steps to source:", reactive_results['performance_metrics']['steps_to_source'])
        >>> print("Planning agent steps to source:", planning_results['performance_metrics']['steps_to_source'])
    """
    if LOGURU_AVAILABLE:
        logger.info("Starting agent demonstration", config=config.to_dict() if hasattr(config, 'to_dict') else config)
    
    try:
        # Create simulation environment
        sim_env = create_modular_simulation_environment(config)
        
        # Extract execution parameters
        if isinstance(config, AgentExecutionConfig):
            exec_params = config.execution_config
        else:
            exec_params = config.get('execution_config', {})
        
        num_steps = exec_params.get('num_steps', 1000)
        dt = exec_params.get('dt', 1.0)
        target_fps = exec_params.get('target_fps', 30.0)
        
        # Initialize tracking variables
        trajectory = []
        performance_metrics = {
            'start_time': time.time(),
            'step_times': [],
            'total_steps': 0,
            'success': False,
            'steps_to_source': None,
            'average_step_time_ms': 0,
            'total_execution_time_s': 0,
            'effective_fps': 0,
            'memory_usage_mb': 0
        }
        
        agent = sim_env['agent']
        plume_model = sim_env['plume_model']
        wind_field = sim_env['wind_field']
        sensors = sim_env['sensors']
        
        # Reset agent and environment
        agent.reset()
        if hasattr(plume_model, 'reset'):
            plume_model.reset()
        
        if LOGURU_AVAILABLE:
            logger.info(
                f"Starting execution of {type(agent).__name__} for {num_steps} steps",
                memory_enabled=sim_env['metadata']['memory_enabled'],
                plume_model=sim_env['metadata']['plume_model_type']
            )
        
        # Main execution loop
        for step in range(num_steps):
            step_start = time.perf_counter()
            
            try:
                # Get current environment state
                # Note: This would interface with actual plume model when fully implemented
                current_state = np.random.random((100, 100))  # Placeholder for demonstration
                
                # Agent step
                agent.step(current_state, dt=dt)
                
                # Record trajectory
                if hasattr(agent, 'positions'):
                    current_position = agent.positions[0] if agent.positions.ndim > 1 else agent.positions
                    trajectory.append({
                        'step': step,
                        'position': current_position.tolist() if hasattr(current_position, 'tolist') else current_position,
                        'time': time.time()
                    })
                
                # Check success condition (placeholder logic)
                if len(trajectory) > 0:
                    pos = trajectory[-1]['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        # Success if within 5 units of center (50, 50)
                        distance_to_center = ((pos[0] - 50)**2 + (pos[1] - 50)**2)**0.5
                        if distance_to_center < 5.0:
                            performance_metrics['success'] = True
                            performance_metrics['steps_to_source'] = step + 1
                            if LOGURU_AVAILABLE:
                                logger.info(f"Success! Agent reached source in {step + 1} steps")
                            break
                
                # Track step performance
                step_time = (time.perf_counter() - step_start) * 1000
                performance_metrics['step_times'].append(step_time)
                performance_metrics['total_steps'] = step + 1
                
                # Frame rate limiting
                if target_fps > 0:
                    target_step_time = 1.0 / target_fps
                    actual_step_time = (time.perf_counter() - step_start)
                    if actual_step_time < target_step_time:
                        time.sleep(target_step_time - actual_step_time)
                
                # Periodic progress logging
                if LOGURU_AVAILABLE and (step + 1) % 100 == 0:
                    avg_step_time = np.mean(performance_metrics['step_times'][-100:])
                    logger.debug(
                        f"Progress: {step + 1}/{num_steps} steps",
                        avg_step_time_ms=avg_step_time,
                        current_position=trajectory[-1]['position'] if trajectory else None
                    )
                
            except Exception as e:
                if LOGURU_AVAILABLE:
                    logger.error(f"Error during step {step}: {e}")
                break
        
        # Calculate final performance metrics
        performance_metrics['total_execution_time_s'] = time.time() - performance_metrics['start_time']
        if performance_metrics['step_times']:
            performance_metrics['average_step_time_ms'] = np.mean(performance_metrics['step_times'])
            performance_metrics['effective_fps'] = 1000.0 / performance_metrics['average_step_time_ms']
        
        # Collect agent state information
        agent_state = {
            'final_position': trajectory[-1]['position'] if trajectory else None,
            'total_steps_executed': performance_metrics['total_steps'],
            'agent_type': type(agent).__name__,
            'memory_enabled': sim_env['metadata']['memory_enabled']
        }
        
        # Add memory-specific information if applicable
        if hasattr(agent, 'get_memory_state') and sim_env['metadata']['memory_enabled']:
            try:
                agent_state['memory_state'] = agent.get_memory_state()
            except:
                agent_state['memory_state'] = 'Not available'
        
        # Compile complete results
        results = {
            'trajectory': trajectory,
            'performance_metrics': performance_metrics,
            'environment_data': {
                'plume_model_type': sim_env['metadata']['plume_model_type'],
                'wind_field_type': sim_env['metadata']['wind_field_type'],
                'sensor_types': sim_env['metadata']['sensor_types']
            },
            'agent_state': agent_state,
            'configuration': config.to_dict() if hasattr(config, 'to_dict') else config,
            'execution_metadata': {
                'demonstration_type': 'modular_architecture_showcase',
                'framework_version': '0.1.0',
                'total_trajectory_points': len(trajectory)
            }
        }
        
        # Save results if requested
        if save_results:
            try:
                output_dir = Path('results/demonstrations')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                agent_type = agent_state['agent_type']
                filename = f"{agent_type}_demonstration_{timestamp}.json"
                
                # Save as JSON (simplified version)
                import json
                with open(output_dir / filename, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_results = {}
                    for key, value in results.items():
                        if isinstance(value, dict):
                            json_results[key] = {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
                        else:
                            json_results[key] = value
                    json.dump(json_results, f, indent=2)
                
                if LOGURU_AVAILABLE:
                    logger.info(f"Results saved to {output_dir / filename}")
                    
            except Exception as e:
                if LOGURU_AVAILABLE:
                    logger.warning(f"Failed to save results: {e}")
        
        if LOGURU_AVAILABLE:
            logger.info(
                "Agent demonstration completed",
                success=performance_metrics['success'],
                total_steps=performance_metrics['total_steps'],
                avg_step_time_ms=performance_metrics['average_step_time_ms'],
                agent_type=agent_state['agent_type']
            )
        
        return results
        
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.error(f"Agent demonstration failed: {e}")
        raise


def list_available_example_agents() -> List[str]:
    """
    Get list of available example agent types for demonstration.
    
    Returns:
        List[str]: List of agent type names that can be instantiated
        
    Examples:
        Discover available agents:
        >>> agents = list_available_example_agents()
        >>> print(f"Available example agents: {agents}")
        >>> for agent in agents:
        ...     info = get_agent_info(agent)
        ...     print(f"  {agent}: {info['description']}")
    """
    available_agents = list(_EXAMPLE_AGENT_REGISTRY.keys())
    
    # Add fallback agents that have direct import support
    fallback_agents = ['ReactiveAgent', 'InfotaxisAgent', 'CastingAgent']
    for agent in fallback_agents:
        if agent not in available_agents:
            # Check if import status indicates availability
            status = _AGENT_IMPORT_STATUS.get(agent, {})
            if not status.get('attempted_import', False) or status.get('available', False):
                available_agents.append(agent)
    
    return sorted(available_agents)


def get_agent_info(agent_type: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a specific agent type.
    
    Args:
        agent_type: Name of the agent type to inspect
        
    Returns:
        Dict[str, Any]: Dictionary containing agent information:
            - description: Human-readable description
            - memory_required: Whether agent requires memory
            - config_schema: Expected configuration parameters
            - performance_characteristics: Performance metrics and requirements
            
    Examples:
        Inspect agent capabilities:
        >>> info = get_agent_info('ReactiveAgent')
        >>> print(f"Memory required: {info['memory_required']}")
        >>> print(f"Description: {info['description']}")
    """
    if agent_type in _EXAMPLE_AGENT_REGISTRY:
        return _EXAMPLE_AGENT_REGISTRY[agent_type].copy()
    
    # Provide basic information for known agents
    fallback_info = {
        'ReactiveAgent': {
            'description': 'Memory-less gradient-following agent with pure reactive behavior',
            'memory_required': False,
            'config_schema': {'gradient_threshold': float, 'step_size': float},
            'performance_characteristics': {'step_time_ms': 0.5, 'memory_usage_mb': 1}
        },
        'InfotaxisAgent': {
            'description': 'Memory-based information-theoretic agent with spatial belief state',
            'memory_required': True,
            'config_schema': {'belief_state_size': tuple, 'information_gain_threshold': float},
            'performance_characteristics': {'step_time_ms': 2, 'memory_usage_mb': 10}
        },
        'CastingAgent': {
            'description': 'Biologically-inspired zigzag search agent with casting behavior',
            'memory_required': False,
            'config_schema': {'casting_angle': float, 'search_radius': float},
            'performance_characteristics': {'step_time_ms': 0.8, 'memory_usage_mb': 2}
        }
    }
    
    return fallback_info.get(agent_type, {
        'description': f'Unknown agent type: {agent_type}',
        'memory_required': False,
        'config_schema': {},
        'performance_characteristics': {}
    })


def compare_agent_approaches(
    agent_configs: List[Union[Dict[str, Any], AgentExecutionConfig]],
    environment_config: Optional[Dict[str, Any]] = None,
    num_episodes: int = 10,
    save_comparison: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple agent approaches using identical environmental conditions.
    
    This function demonstrates the key benefit of the modular architecture: the ability to
    evaluate different navigation strategies (memory-based vs memory-less) using identical
    simulation infrastructure and environmental conditions for fair comparison.
    
    Args:
        agent_configs: List of agent configurations to compare
        environment_config: Shared environment configuration for all agents
        num_episodes: Number of episodes to run for statistical significance
        save_comparison: Whether to save detailed comparison results
        
    Returns:
        Dict[str, Any]: Comprehensive comparison results including:
            - 'individual_results': Results for each agent type
            - 'comparative_metrics': Cross-agent performance comparison
            - 'statistical_analysis': Statistical significance tests
            - 'recommendations': Performance-based recommendations
            
    Examples:
        Compare reactive vs planning approaches:
        >>> agent_configs = [
        ...     AgentExecutionConfig(agent_type='ReactiveAgent', memory_enabled=False),
        ...     AgentExecutionConfig(agent_type='InfotaxisAgent', memory_enabled=True)
        ... ]
        >>> comparison = compare_agent_approaches(agent_configs, num_episodes=20)
        >>> 
        >>> print("Performance Comparison:")
        >>> for agent_type, metrics in comparison['comparative_metrics'].items():
        ...     print(f"  {agent_type}: {metrics['avg_steps_to_source']:.1f} steps")
    """
    if LOGURU_AVAILABLE:
        logger.info(
            f"Starting agent comparison study",
            num_agents=len(agent_configs),
            num_episodes=num_episodes,
            agent_types=[cfg.agent_type if hasattr(cfg, 'agent_type') else cfg.get('agent_type', 'unknown') 
                        for cfg in agent_configs]
        )
    
    comparison_results = {
        'individual_results': {},
        'comparative_metrics': {},
        'statistical_analysis': {},
        'recommendations': {},
        'execution_metadata': {
            'comparison_type': 'memory_vs_memoryless_demonstration',
            'num_episodes': num_episodes,
            'environment_config': environment_config,
            'start_time': time.time()
        }
    }
    
    try:
        # Run episodes for each agent configuration
        for i, agent_config in enumerate(agent_configs):
            # Normalize configuration
            if isinstance(agent_config, dict):
                config = AgentExecutionConfig.from_dict(agent_config)
            else:
                config = agent_config
            
            # Apply shared environment configuration if provided
            if environment_config:
                config.environment_config.update(environment_config)
                
            agent_type = config.agent_type
            
            if LOGURU_AVAILABLE:
                logger.info(f"Running episodes for {agent_type} ({i+1}/{len(agent_configs)})")
            
            agent_results = {
                'episodes': [],
                'performance_summary': {},
                'configuration': config.to_dict()
            }
            
            # Run multiple episodes for statistical significance
            episode_metrics = []
            
            for episode in range(num_episodes):
                try:
                    # Run single episode
                    episode_result = run_agent_demonstration(
                        config, 
                        save_results=False,  # Don't save individual episodes
                        enable_visualization=False
                    )
                    
                    # Extract key metrics
                    episode_metric = {
                        'episode': episode,
                        'success': episode_result['performance_metrics']['success'],
                        'steps_to_source': episode_result['performance_metrics'].get('steps_to_source', config.execution_config['num_steps']),
                        'total_steps': episode_result['performance_metrics']['total_steps'],
                        'avg_step_time_ms': episode_result['performance_metrics']['average_step_time_ms'],
                        'trajectory_length': len(episode_result['trajectory'])
                    }
                    
                    episode_metrics.append(episode_metric)
                    agent_results['episodes'].append(episode_result)
                    
                    if LOGURU_AVAILABLE and (episode + 1) % 5 == 0:
                        logger.debug(f"  Completed episode {episode + 1}/{num_episodes} for {agent_type}")
                    
                except Exception as e:
                    if LOGURU_AVAILABLE:
                        logger.warning(f"Episode {episode} failed for {agent_type}: {e}")
                    continue
            
            # Calculate summary statistics
            if episode_metrics:
                success_rate = np.mean([m['success'] for m in episode_metrics])
                avg_steps = np.mean([m['steps_to_source'] for m in episode_metrics])
                std_steps = np.std([m['steps_to_source'] for m in episode_metrics])
                avg_step_time = np.mean([m['avg_step_time_ms'] for m in episode_metrics])
                
                agent_results['performance_summary'] = {
                    'success_rate': success_rate,
                    'avg_steps_to_source': avg_steps,
                    'std_steps_to_source': std_steps,
                    'avg_step_time_ms': avg_step_time,
                    'total_episodes': len(episode_metrics),
                    'memory_enabled': config.memory_enabled
                }
                
                comparison_results['individual_results'][agent_type] = agent_results
                comparison_results['comparative_metrics'][agent_type] = agent_results['performance_summary']
                
                if LOGURU_AVAILABLE:
                    logger.info(
                        f"Completed {agent_type} evaluation",
                        success_rate=f"{success_rate:.2%}",
                        avg_steps=f"{avg_steps:.1f}",
                        avg_step_time_ms=f"{avg_step_time:.2f}"
                    )
        
        # Perform comparative analysis
        if len(comparison_results['comparative_metrics']) >= 2:
            metrics = comparison_results['comparative_metrics']
            
            # Find best performing agents
            best_success_rate = max(metrics.values(), key=lambda x: x['success_rate'])
            best_efficiency = min(metrics.values(), key=lambda x: x['avg_steps_to_source'])
            fastest_execution = min(metrics.values(), key=lambda x: x['avg_step_time_ms'])
            
            # Analyze memory vs non-memory performance
            memory_agents = {k: v for k, v in metrics.items() if v['memory_enabled']}
            memoryless_agents = {k: v for k, v in metrics.items() if not v['memory_enabled']}
            
            comparison_results['statistical_analysis'] = {
                'best_success_rate': {
                    'agent_type': next(k for k, v in metrics.items() if v == best_success_rate),
                    'success_rate': best_success_rate['success_rate']
                },
                'most_efficient': {
                    'agent_type': next(k for k, v in metrics.items() if v == best_efficiency),
                    'avg_steps': best_efficiency['avg_steps_to_source']
                },
                'fastest_execution': {
                    'agent_type': next(k for k, v in metrics.items() if v == fastest_execution),
                    'avg_step_time_ms': fastest_execution['avg_step_time_ms']
                },
                'memory_vs_memoryless': {
                    'memory_agents_count': len(memory_agents),
                    'memoryless_agents_count': len(memoryless_agents),
                    'memory_avg_success_rate': np.mean([v['success_rate'] for v in memory_agents.values()]) if memory_agents else 0,
                    'memoryless_avg_success_rate': np.mean([v['success_rate'] for v in memoryless_agents.values()]) if memoryless_agents else 0
                }
            }
        
        # Generate recommendations
        recommendations = []
        if comparison_results['statistical_analysis']:
            analysis = comparison_results['statistical_analysis']
            
            # Performance recommendations
            recommendations.append(
                f"Best overall success rate: {analysis['best_success_rate']['agent_type']} "
                f"({analysis['best_success_rate']['success_rate']:.2%})"
            )
            
            recommendations.append(
                f"Most efficient pathfinding: {analysis['most_efficient']['agent_type']} "
                f"({analysis['most_efficient']['avg_steps']:.1f} steps average)"
            )
            
            recommendations.append(
                f"Fastest execution: {analysis['fastest_execution']['agent_type']} "
                f"({analysis['fastest_execution']['avg_step_time_ms']:.2f}ms per step)"
            )
            
            # Memory vs non-memory analysis
            memory_vs_memoryless = analysis['memory_vs_memoryless']
            if memory_vs_memoryless['memory_agents_count'] > 0 and memory_vs_memoryless['memoryless_agents_count'] > 0:
                memory_advantage = (memory_vs_memoryless['memory_avg_success_rate'] - 
                                   memory_vs_memoryless['memoryless_avg_success_rate'])
                
                if memory_advantage > 0.1:  # 10% advantage
                    recommendations.append(
                        "Memory-based agents show significant performance advantage. "
                        "Consider memory-enabled approaches for complex navigation tasks."
                    )
                elif memory_advantage < -0.1:
                    recommendations.append(
                        "Memory-less agents perform better. Consider reactive approaches "
                        "for computational efficiency and real-time requirements."
                    )
                else:
                    recommendations.append(
                        "Memory-based and memory-less agents show comparable performance. "
                        "Choose based on computational constraints and task complexity."
                    )
        
        comparison_results['recommendations'] = recommendations
        comparison_results['execution_metadata']['total_execution_time_s'] = time.time() - comparison_results['execution_metadata']['start_time']
        
        # Save comparison results if requested
        if save_comparison:
            try:
                output_dir = Path('results/comparisons')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"agent_comparison_{timestamp}.json"
                
                # Save detailed comparison results
                import json
                with open(output_dir / filename, 'w') as f:
                    # Simplified version for JSON serialization
                    json_results = {
                        'comparative_metrics': comparison_results['comparative_metrics'],
                        'statistical_analysis': comparison_results['statistical_analysis'],
                        'recommendations': comparison_results['recommendations'],
                        'execution_metadata': comparison_results['execution_metadata']
                    }
                    json.dump(json_results, f, indent=2)
                
                if LOGURU_AVAILABLE:
                    logger.info(f"Comparison results saved to {output_dir / filename}")
                    
            except Exception as e:
                if LOGURU_AVAILABLE:
                    logger.warning(f"Failed to save comparison results: {e}")
        
        if LOGURU_AVAILABLE:
            logger.info(
                "Agent comparison completed",
                total_agents=len(comparison_results['comparative_metrics']),
                total_episodes=num_episodes * len(agent_configs),
                execution_time_s=comparison_results['execution_metadata']['total_execution_time_s']
            )
            
            # Log key findings
            for recommendation in comparison_results['recommendations']:
                logger.info(f"Finding: {recommendation}")
        
        return comparison_results
        
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.error(f"Agent comparison failed: {e}")
        raise


def main_cli():
    """
    Command-line interface for running example agent demonstrations.
    
    This function provides CLI access to all example functionality, enabling reproducible
    execution of agent demonstrations and comparisons as required by Section 0.6.2 output
    constraints: "Example agents must be runnable via CLI".
    
    Examples:
        Run single agent demonstration:
        $ python -m plume_nav_sim.examples run_agent --agent_type=ReactiveAgent --num_steps=1000
        
        Compare multiple agents:
        $ python -m plume_nav_sim.examples compare_agents --agents=ReactiveAgent,InfotaxisAgent --num_episodes=20
        
        List available agents:
        $ python -m plume_nav_sim.examples list_agents
    """
    parser = argparse.ArgumentParser(
        description='Plume Navigation Examples - Demonstrating Modular Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run reactive (memory-less) agent demonstration
  python -m plume_nav_sim.examples run_agent --agent_type=ReactiveAgent --memory_enabled=false
  
  # Run memory-based agent demonstration  
  python -m plume_nav_sim.examples run_agent --agent_type=InfotaxisAgent --memory_enabled=true
  
  # Compare different agent approaches
  python -m plume_nav_sim.examples compare_agents --agents=ReactiveAgent,InfotaxisAgent
  
  # List all available example agents
  python -m plume_nav_sim.examples list_agents
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run single agent command
    run_parser = subparsers.add_parser('run_agent', help='Run single agent demonstration')
    run_parser.add_argument('--agent_type', default='ReactiveAgent', 
                           help='Agent type to run (default: ReactiveAgent)')
    run_parser.add_argument('--memory_enabled', type=bool, default=None,
                           help='Enable memory for agent (auto-detected if not specified)')
    run_parser.add_argument('--num_steps', type=int, default=1000,
                           help='Number of simulation steps (default: 1000)')
    run_parser.add_argument('--plume_model', default='GaussianPlumeModel',
                           help='Plume model type (default: GaussianPlumeModel)')
    run_parser.add_argument('--enable_visualization', action='store_true',
                           help='Enable real-time visualization')
    run_parser.add_argument('--save_results', action='store_true', default=True,
                           help='Save execution results (default: True)')
    run_parser.add_argument('--config_file', type=str,
                           help='YAML configuration file for detailed setup')
    
    # Compare agents command
    compare_parser = subparsers.add_parser('compare_agents', help='Compare multiple agent approaches')
    compare_parser.add_argument('--agents', required=True,
                               help='Comma-separated list of agent types to compare')
    compare_parser.add_argument('--num_episodes', type=int, default=10,
                               help='Number of episodes per agent (default: 10)')
    compare_parser.add_argument('--save_comparison', action='store_true', default=True,
                               help='Save comparison results (default: True)')
    compare_parser.add_argument('--environment_config', type=str,
                               help='Shared environment configuration file')
    
    # List agents command  
    list_parser = subparsers.add_parser('list_agents', help='List available example agents')
    list_parser.add_argument('--detailed', action='store_true',
                            help='Show detailed information for each agent')
    
    # Diagnostics command
    diag_parser = subparsers.add_parser('diagnostics', help='Show system diagnostics')
    diag_parser.add_argument('--check_imports', action='store_true',
                            help='Check availability of all example agents')
    
    args = parser.parse_args()
    
    if args.command == 'run_agent':
        try:
            # Create configuration
            config = AgentExecutionConfig(
                agent_type=args.agent_type,
                memory_enabled=args.memory_enabled if args.memory_enabled is not None else None,
                plume_model_config={'type': args.plume_model},
                execution_config={
                    'num_steps': args.num_steps,
                    'enable_visualization': args.enable_visualization
                }
            )
            
            # Auto-detect memory requirement if not specified
            if config.memory_enabled is None:
                agent_info = get_agent_info(args.agent_type)
                config.memory_enabled = agent_info.get('memory_required', False)
            
            print(f"Running {args.agent_type} demonstration...")
            print(f"Memory enabled: {config.memory_enabled}")
            print(f"Plume model: {args.plume_model}")
            print(f"Number of steps: {args.num_steps}")
            
            # Run demonstration
            results = run_agent_demonstration(
                config,
                save_results=args.save_results,
                enable_visualization=args.enable_visualization
            )
            
            # Print results summary
            metrics = results['performance_metrics']
            print("\nExecution Results:")
            print(f"  Success: {metrics['success']}")
            print(f"  Total steps: {metrics['total_steps']}")
            if metrics['steps_to_source']:
                print(f"  Steps to source: {metrics['steps_to_source']}")
            print(f"  Average step time: {metrics['average_step_time_ms']:.2f} ms")
            print(f"  Effective FPS: {metrics['effective_fps']:.1f}")
            
        except Exception as e:
            print(f"Error running agent demonstration: {e}")
            return 1
    
    elif args.command == 'compare_agents':
        try:
            agent_types = [a.strip() for a in args.agents.split(',')]
            print(f"Comparing agents: {agent_types}")
            print(f"Episodes per agent: {args.num_episodes}")
            
            # Create configurations for each agent
            agent_configs = []
            for agent_type in agent_types:
                agent_info = get_agent_info(agent_type)
                config = AgentExecutionConfig(
                    agent_type=agent_type,
                    memory_enabled=agent_info.get('memory_required', False),
                    execution_config={'num_steps': 1000}
                )
                agent_configs.append(config)
            
            # Run comparison
            comparison = compare_agent_approaches(
                agent_configs,
                num_episodes=args.num_episodes,
                save_comparison=args.save_comparison
            )
            
            # Print comparison results
            print("\nComparison Results:")
            for agent_type, metrics in comparison['comparative_metrics'].items():
                print(f"\n{agent_type}:")
                print(f"  Success rate: {metrics['success_rate']:.2%}")
                print(f"  Avg steps to source: {metrics['avg_steps_to_source']:.1f}")
                print(f"  Avg step time: {metrics['avg_step_time_ms']:.2f} ms")
                print(f"  Memory enabled: {metrics['memory_enabled']}")
            
            print("\nRecommendations:")
            for rec in comparison['recommendations']:
                print(f"  • {rec}")
                
        except Exception as e:
            print(f"Error running agent comparison: {e}")
            return 1
    
    elif args.command == 'list_agents':
        try:
            available_agents = list_available_example_agents()
            print(f"Available example agents ({len(available_agents)}):")
            
            for agent_type in available_agents:
                if args.detailed:
                    info = get_agent_info(agent_type)
                    print(f"\n{agent_type}:")
                    print(f"  Description: {info.get('description', 'No description available')}")
                    print(f"  Memory required: {info.get('memory_required', False)}")
                    if info.get('performance_characteristics'):
                        perf = info['performance_characteristics']
                        print(f"  Step time: {perf.get('step_time_ms', 'N/A')} ms")
                        print(f"  Memory usage: {perf.get('memory_usage_mb', 'N/A')} MB")
                else:
                    memory_status = "(memory)" if get_agent_info(agent_type).get('memory_required', False) else "(memory-less)"
                    print(f"  {agent_type} {memory_status}")
                    
        except Exception as e:
            print(f"Error listing agents: {e}")
            return 1
    
    elif args.command == 'diagnostics':
        try:
            print("System Diagnostics:")
            
            # Check model availability
            from ..models import get_import_diagnostics
            diagnostics = get_import_diagnostics()
            
            print("\nAvailable Components:")
            for component_type, components in diagnostics['available_components'].items():
                print(f"  {component_type}: {len(components)} available")
                if components:
                    print(f"    {', '.join(components)}")
            
            print(f"\nExample Agents: {len(list_available_example_agents())} available")
            
            if diagnostics['errors']:
                print(f"\nImport Errors ({len(diagnostics['errors'])}):")
                for error in diagnostics['errors']:
                    print(f"  {error['component_type']}.{error['component_name']}: {error['error']}")
            
            if diagnostics['warnings']:
                print(f"\nWarnings ({len(diagnostics['warnings'])}):")
                for warning in diagnostics['warnings']:
                    print(f"  {warning}")
            
            if args.check_imports:
                print("\nTesting agent imports:")
                for agent_type in list_available_example_agents():
                    try:
                        create_example_agent({'agent_type': agent_type})
                        print(f"  ✓ {agent_type}")
                    except Exception as e:
                        print(f"  ✗ {agent_type}: {e}")
                        
        except Exception as e:
            print(f"Error running diagnostics: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


# Package initialization and auto-discovery
def _initialize_examples_package():
    """Initialize the examples package with auto-discovery of agent implementations."""
    try:
        # Attempt to auto-register common example agents
        agent_implementations = [
            ('ReactiveAgent', 'agents.reactive_agent', 'ReactiveAgent'),
            ('InfotaxisAgent', 'agents.infotaxis_agent', 'InfotaxisAgent'),  
            ('CastingAgent', 'agents.casting_agent', 'CastingAgent')
        ]
        
        for agent_name, module_path, class_name in agent_implementations:
            try:
                module = importlib.import_module(f'.{module_path}', package=__name__)
                agent_class = getattr(module, class_name)
                
                # Determine if agent requires memory based on name/class characteristics
                memory_required = 'infotaxis' in agent_name.lower() or 'planning' in agent_name.lower()
                
                register_example_agent(
                    agent_name,
                    agent_class,
                    f"Example {agent_name} demonstrating {'memory-based' if memory_required else 'memory-less'} navigation",
                    memory_required=memory_required,
                    performance_characteristics={'auto_discovered': True}
                )
                
                _AGENT_IMPORT_STATUS[agent_name] = {
                    'available': True,
                    'attempted_import': True,
                    'module_path': module_path
                }
                
            except ImportError:
                _AGENT_IMPORT_STATUS[agent_name] = {
                    'available': False,
                    'attempted_import': True,
                    'error': f"Module {module_path} not available"
                }
                continue
        
        if LOGURU_AVAILABLE:
            available_count = len([s for s in _AGENT_IMPORT_STATUS.values() if s.get('available', False)])
            logger.debug(
                f"Examples package initialized",
                registered_agents=len(_EXAMPLE_AGENT_REGISTRY),
                available_agents=available_count,
                import_errors=len([s for s in _AGENT_IMPORT_STATUS.values() if not s.get('available', True)])
            )
            
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.warning(f"Examples package initialization had issues: {e}")
        else:
            warnings.warn(f"Examples package initialization warning: {e}", UserWarning, stacklevel=2)


# Initialize package on import
_initialize_examples_package()


# Public API exports
__all__ = [
    # Core configuration and execution classes
    'AgentExecutionConfig',
    
    # Factory functions for agent creation
    'create_example_agent',
    'create_modular_simulation_environment',
    
    # Demonstration and comparison functions
    'run_agent_demonstration', 
    'compare_agent_approaches',
    
    # Registry management functions
    'register_example_agent',
    'list_available_example_agents',
    'get_agent_info',
    
    # CLI entry point
    'main_cli'
]

# Package metadata
__version__ = "0.1.0"
__description__ = "Example agents and demonstrations for modular plume navigation architecture"
__author__ = "Plume Navigation Team"

# Performance characteristics for monitoring and validation
__performance_characteristics__ = {
    "agent_instantiation_ms": 50,      # Target <50ms for complex memory-based agents
    "reactive_step_time_ms": 1,        # Target <1ms for reactive agents
    "planning_step_time_ms": 5,        # Target <5ms for planning agents  
    "cli_startup_time_s": 2,           # Target <2s for CLI initialization
    "demonstration_memory_mb": 50,     # Target <50MB for demonstration execution
    "comparison_overhead_percent": 10   # Target <10% overhead for comparison framework
}

# Compatibility and feature flags
__compatibility_features__ = {
    "hydra_integration": HYDRA_AVAILABLE,
    "enhanced_logging": LOGURU_AVAILABLE,
    "visualization_support": VISUALIZATION_AVAILABLE,
    "gymnasium_integration": GYMNASIUM_AVAILABLE,
    "cli_execution": True,
    "agent_comparison": True,
    "memory_vs_memoryless_demo": True,
    "configuration_driven_switching": True
}


# CLI execution support
if __name__ == '__main__':
    sys.exit(main_cli())