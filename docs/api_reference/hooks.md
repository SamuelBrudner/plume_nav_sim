# Hook System API Reference

## Overview

The plume_nav_sim v1.0 hook system provides a comprehensive framework for zero-code extensibility through configuration-driven dependency injection. This system enables researchers to customize simulation behavior, add data collection capabilities, and integrate debugging tools without modifying core simulation code.

## Core Architecture

### Hook System Protocol

The hook system is built around lifecycle callbacks and extension points that provide non-invasive access to simulation state at key execution points.

```python
from typing import Protocol, Callable, Optional, Dict, Any
import hydra
from hydra.utils import instantiate

class HookSystemProtocol(Protocol):
    """Protocol interface defining the core hook system architecture."""
    
    def register_pre_step_hook(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback executed before each simulation step."""
        ...
    
    def register_post_step_hook(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback executed after each simulation step."""
        ...
    
    def register_episode_end_hook(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback executed when episodes complete."""
        ...
    
    def execute_lifecycle_hooks(self, hook_type: str, context: Dict[str, Any]) -> None:
        """Execute all registered hooks of the specified type."""
        ...
```

## Lifecycle Hooks

### Pre-Step Hooks

Executed before each simulation step, providing access to current state for validation and preparation.

```python
def state_validation_pre_step(simulation_state: Dict[str, Any]) -> None:
    """Validate simulation state before step execution."""
    agent_positions = simulation_state.get('agent_positions')
    if agent_positions is not None:
        # Validate agent positions are within bounds
        domain_bounds = simulation_state.get('domain_bounds', (100, 100))
        for pos in agent_positions:
            assert 0 <= pos[0] <= domain_bounds[0], f"Agent x-position {pos[0]} out of bounds"
            assert 0 <= pos[1] <= domain_bounds[1], f"Agent y-position {pos[1]} out of bounds"

def performance_monitor_pre_step(simulation_state: Dict[str, Any]) -> None:
    """Monitor performance metrics before step execution."""
    start_time = time.perf_counter()
    simulation_state['step_start_time'] = start_time
    simulation_state['performance_tracking'] = True
```

### Post-Step Hooks

Executed after each simulation step, enabling data collection and analysis.

```python
def trajectory_recorder_post_step(simulation_state: Dict[str, Any]) -> None:
    """Record agent trajectory data after step completion."""
    recorder = simulation_state.get('recorder')
    if recorder and recorder.enabled:
        step_data = {
            'step_number': simulation_state.get('step_count', 0),
            'agent_positions': simulation_state.get('agent_positions'),
            'agent_orientations': simulation_state.get('agent_orientations'),
            'odor_concentrations': simulation_state.get('odor_readings'),
            'applied_actions': simulation_state.get('actions'),
            'step_rewards': simulation_state.get('rewards'),
            'timestamp': time.time()
        }
        recorder.record_step(step_data, simulation_state['step_count'])

def performance_monitor_post_step(simulation_state: Dict[str, Any]) -> None:
    """Monitor performance metrics after step completion."""
    if simulation_state.get('performance_tracking'):
        start_time = simulation_state.get('step_start_time')
        if start_time:
            step_duration = time.perf_counter() - start_time
            if step_duration > 0.033:  # 33ms threshold
                logger.warning(f"Step duration {step_duration*1000:.1f}ms exceeds 33ms target")
```

### Episode-End Hooks

Executed when episodes complete, providing opportunities for summary generation and cleanup.

```python
def episode_summarizer_episode_end(final_info: Dict[str, Any]) -> None:
    """Generate comprehensive episode summary on completion."""
    stats_aggregator = final_info.get('stats_aggregator')
    if stats_aggregator:
        trajectory_data = {
            'positions': final_info.get('trajectory_positions', []),
            'total_steps': final_info.get('total_steps', 0),
            'total_reward': final_info.get('total_reward', 0.0),
            'success': final_info.get('terminated', False),
            'final_position': final_info.get('final_position'),
            'exploration_coverage': final_info.get('exploration_coverage', 0.0)
        }
        
        episode_stats = stats_aggregator.calculate_episode_stats(
            trajectory_data, 
            episode_id=final_info.get('episode_id')
        )
        final_info['episode_stats'] = episode_stats

def cleanup_manager_episode_end(final_info: Dict[str, Any]) -> None:
    """Perform cleanup operations on episode completion."""
    recorder = final_info.get('recorder')
    if recorder:
        recorder.flush()  # Ensure all data is written
    
    # Clear temporary buffers
    if 'trajectory_buffer' in final_info:
        final_info['trajectory_buffer'].clear()
```

## Extension Points

### Extra Observation Functions

Extend environment observations with custom data through `extra_obs_fn`.

```python
def wind_direction_observation(base_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Add wind direction information to observations."""
    wind_velocity = base_obs.get('wind_velocity', np.array([0.0, 0.0]))
    wind_magnitude = np.linalg.norm(wind_velocity)
    
    if wind_magnitude > 0.1:
        wind_direction = np.degrees(np.arctan2(wind_velocity[1], wind_velocity[0])) % 360
    else:
        wind_direction = 0.0
    
    return {
        'wind_direction': np.array([wind_direction], dtype=np.float32),
        'wind_magnitude': np.array([wind_magnitude], dtype=np.float32),
        'wind_cardinal': get_cardinal_direction(wind_direction)
    }

def plume_gradient_analysis_observation(base_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate spatial gradients for enhanced navigation."""
    agent_position = base_obs.get('agent_position')
    concentration = base_obs.get('odor_concentration', 0.0)
    
    # Calculate spatial gradient using finite differences
    gradient = calculate_spatial_gradient(agent_position, concentration)
    
    return {
        'plume_gradient': gradient.astype(np.float32),
        'gradient_magnitude': np.array([np.linalg.norm(gradient)], dtype=np.float32),
        'gradient_direction': np.array([np.degrees(np.arctan2(gradient[1], gradient[0])) % 360], dtype=np.float32)
    }
```

### Extra Reward Functions

Customize reward calculation through `extra_reward_fn`.

```python
def exploration_bonus_reward(base_reward: float, info: Dict[str, Any]) -> float:
    """Provide exploration bonus for visiting novel areas."""
    exploration_coverage = info.get('exploration_coverage', 0.0)
    current_position = info.get('agent_position')
    
    # Check if position is novel (not recently visited)
    visited_positions = info.get('visited_positions', [])
    is_novel = True
    for prev_pos in visited_positions[-50:]:  # Check last 50 positions
        if np.linalg.norm(np.array(current_position) - np.array(prev_pos)) < 5.0:
            is_novel = False
            break
    
    if is_novel:
        exploration_bonus = 0.1 * (1.0 - exploration_coverage)  # Diminishing returns
        return exploration_bonus
    
    return 0.0

def efficiency_penalty_reward(base_reward: float, info: Dict[str, Any]) -> float:
    """Penalize inefficient movement patterns."""
    action = info.get('action', np.array([0.0, 0.0]))
    speed = action[0] if len(action) > 0 else 0.0
    angular_velocity = action[1] if len(action) > 1 else 0.0
    
    # Energy consumption penalty
    energy_cost = 0.01 * (speed**2 + angular_velocity**2)
    
    # Path efficiency check
    path_efficiency = info.get('path_efficiency', 1.0)
    efficiency_penalty = -0.005 * (1.0 - path_efficiency)
    
    return -energy_cost + efficiency_penalty
```

### Episode End Functions

Custom episode completion handling through `episode_end_fn`.

```python
def adaptive_curriculum_episode_end(final_info: Dict[str, Any]) -> None:
    """Adjust environment difficulty based on performance."""
    success_rate = final_info.get('success_rate', 0.0)
    episode_count = final_info.get('episode_count', 0)
    
    # Adjust difficulty every 100 episodes
    if episode_count % 100 == 0:
        if success_rate > 0.8:
            # Increase difficulty
            final_info['curriculum_adjustment'] = 'increase_difficulty'
            logger.info(f"Increasing difficulty after episode {episode_count} (success rate: {success_rate:.2%})")
        elif success_rate < 0.3:
            # Decrease difficulty
            final_info['curriculum_adjustment'] = 'decrease_difficulty'
            logger.info(f"Decreasing difficulty after episode {episode_count} (success rate: {success_rate:.2%})")

def experiment_tracking_episode_end(final_info: Dict[str, Any]) -> None:
    """Log experiment metrics to external tracking systems."""
    episode_id = final_info.get('episode_id')
    total_reward = final_info.get('total_reward', 0.0)
    success = final_info.get('terminated', False)
    
    # Log to wandb if available
    try:
        import wandb
        wandb.log({
            'episode_reward': total_reward,
            'episode_success': success,
            'episode_length': final_info.get('total_steps', 0),
            'exploration_coverage': final_info.get('exploration_coverage', 0.0)
        }, step=episode_id)
    except ImportError:
        logger.debug("wandb not available for experiment tracking")
    
    # Export episode artifacts
    if success:
        export_successful_episode(final_info)
```

## Configuration Examples

### Default Configuration (none.yaml)

Zero-overhead configuration with all hooks disabled:

```yaml
hooks:
  enabled: false
  
  pre_step:
    enabled: false
    hooks: []
    
  post_step:
    enabled: false
    hooks: []
    
  episode_end:
    enabled: false
    hooks: []
  
  extensions:
    extra_obs_fn: null
    extra_reward_fn: null
    extra_done_fn: null
  
  performance:
    monitor_overhead: false
    max_hook_overhead_ms: 0.0
```

### Research Configuration (research.yaml)

Comprehensive data collection for scientific experiments:

```yaml
recorder:
  _target_: plume_nav_sim.recording.backends.ParquetRecorder
  backend: parquet
  output_dir: ./data/research_recordings
  buffer_size: 1000
  compression: snappy
  enable_metrics: true

stats_aggregator:
  _target_: plume_nav_sim.analysis.stats.StatsAggregator
  config:
    metrics_definitions:
      trajectory:
        - mean
        - std
        - total_distance
        - displacement_efficiency
        - tortuosity
      concentration:
        - mean
        - detection_rate
        - max_detection_streak
      timing:
        - episode_length
        - mean_step_duration
        - performance_compliance_rate

observation_hooks:
  enable_extra_observations: true
  extra_observations:
    wind_direction: true
    exploration_progress: true
    concentration_history: true

reward_hooks:
  enable_extra_rewards: true
  extra_rewards:
    exploration_bonus:
      enabled: true
      weight: 0.1
    efficiency_bonus:
      enabled: true
      weight: 0.05
```

### Debug Configuration (debug.yaml)

Interactive debugging with GUI integration:

```yaml
hooks:
  enabled: true
  debug:
    enabled: true
    backend: "auto"  # qt, streamlit, auto
    
    visualization:
      enabled: true
      refresh_rate: 30
      real_time_plots: true
      
    interactive:
      step_through_enabled: true
      breakpoints_enabled: true
      state_inspection_enabled: true
      
    callbacks:
      pre_step:
        - _target_: plume_nav_sim.debug.hooks.performance_monitor_pre_step
          enabled: true
          
      post_step:
        - _target_: plume_nav_sim.debug.hooks.visualization_update_post_step
          enabled: true
          refresh_rate: 30
          
      episode_end:
        - _target_: plume_nav_sim.debug.hooks.debug_summary_episode_end
          enabled: true
          export_trajectory: true

visualization:
  hook_registration:
    - _target_: plume_nav_sim.utils.visualization.register_visualization_hooks
      hooks:
        - name: "debug_state_visualization"
          callback: plume_nav_sim.debug.visualization.update_state_plot
          frequency: 30
          enabled: true
```

### Full Configuration (full.yaml)

Complete hook system with all extension points enabled:

```yaml
hooks:
  enabled: true
  
  lifecycle:
    pre_step:
      enabled: true
      hooks:
        - name: "state_validator"
          _target_: plume_nav_sim.hooks.lifecycle.StateValidationHook
          validate_positions: true
          validate_boundaries: true
          
        - name: "performance_profiler"
          _target_: plume_nav_sim.hooks.lifecycle.PerformanceProfilerHook
          enable_memory_tracking: true
          enable_timing_breakdown: true
    
    post_step:
      enabled: true
      hooks:
        - name: "trajectory_recorder"
          _target_: plume_nav_sim.hooks.lifecycle.TrajectoryRecorderHook
          record_full_state: true
          buffer_size: 1000
          
        - name: "statistics_collector"
          _target_: plume_nav_sim.hooks.lifecycle.StatisticsCollectorHook
          collect_step_metrics: true
          real_time_analysis: true
    
    episode_end:
      enabled: true
      hooks:
        - name: "episode_summarizer"
          _target_: plume_nav_sim.hooks.lifecycle.EpisodeSummarizerHook
          generate_comprehensive_summary: true
          export_summary_json: true
  
  extensions:
    extra_obs_fn:
      enabled: true
      functions:
        - name: "plume_gradient_analysis"
          _target_: plume_nav_sim.hooks.extensions.PlumeGradientAnalysisObs
          calculate_spatial_gradients: true
          vectorized_computation: true
          
    extra_reward_fn:
      enabled: true
      functions:
        - name: "exploration_bonus"
          _target_: plume_nav_sim.hooks.extensions.ExplorationBonusReward
          novelty_weight: 0.1
          coverage_weight: 0.05
          
    episode_end_fn:
      enabled: true
      functions:
        - name: "adaptive_curriculum"
          _target_: plume_nav_sim.hooks.extensions.AdaptiveCurriculumEpisodeEnd
          adjust_difficulty: true
          track_learning_progress: true
```

## Integration Patterns

### Recorder Integration

Hook integration with the recording system for automated data collection:

```python
class RecorderManager:
    """Hook manager for recorder integration."""
    
    def register_recording_hooks(self, hook_system: HookSystemProtocol) -> None:
        """Register recorder hooks with the hook system."""
        if self.recorder and self.recorder.enabled:
            hook_system.register_post_step_hook(self._record_step_data)
            hook_system.register_episode_end_hook(self._finalize_recording)
    
    def _record_step_data(self, simulation_state: Dict[str, Any]) -> None:
        """Hook callback for step data recording."""
        step_data = {
            'step_number': simulation_state.get('step_count', 0),
            'agent_position': simulation_state.get('agent_position'),
            'odor_concentration': simulation_state.get('odor_concentration'),
            'action': simulation_state.get('action'),
            'reward': simulation_state.get('reward'),
            'timestamp': time.time()
        }
        
        try:
            self.recorder.record_step(step_data, simulation_state['step_count'])
        except Exception as e:
            logger.warning(f"Failed to record step data: {e}")
    
    def _finalize_recording(self, final_info: Dict[str, Any]) -> None:
        """Hook callback for recording finalization."""
        episode_data = {
            'episode_id': final_info.get('episode_id'),
            'total_steps': final_info.get('total_steps'),
            'total_reward': final_info.get('total_reward'),
            'success': final_info.get('terminated', False),
            'final_position': final_info.get('final_position')
        }
        
        try:
            self.recorder.record_episode(episode_data, final_info['episode_id'])
            self.recorder.flush()
        except Exception as e:
            logger.warning(f"Failed to finalize episode recording: {e}")
```

### Statistics Integration

Hook integration with the statistics aggregation system:

```python
class StatsHookManager:
    """Hook manager for statistics integration."""
    
    def register_stats_hooks(self, hook_system: HookSystemProtocol) -> None:
        """Register statistics hooks with the hook system."""
        if self.stats_aggregator:
            hook_system.register_post_step_hook(self._collect_step_metrics)
            hook_system.register_episode_end_hook(self._calculate_episode_metrics)
    
    def _collect_step_metrics(self, simulation_state: Dict[str, Any]) -> None:
        """Collect real-time metrics during simulation steps."""
        metrics = {
            'step_number': simulation_state.get('step_count', 0),
            'computation_time': simulation_state.get('step_duration', 0.0),
            'memory_usage': get_memory_usage(),
            'agent_count': len(simulation_state.get('agent_positions', []))
        }
        
        # Update running statistics
        self.stats_aggregator.update_running_stats(metrics)
    
    def _calculate_episode_metrics(self, final_info: Dict[str, Any]) -> None:
        """Calculate comprehensive episode metrics."""
        trajectory_data = {
            'positions': final_info.get('trajectory_positions', []),
            'total_steps': final_info.get('total_steps', 0),
            'total_reward': final_info.get('total_reward', 0.0),
            'success': final_info.get('terminated', False)
        }
        
        episode_stats = self.stats_aggregator.calculate_episode_stats(
            trajectory_data, 
            episode_id=final_info.get('episode_id')
        )
        
        final_info['episode_stats'] = episode_stats
        
        # Export summary if configured
        if self.auto_export:
            self.stats_aggregator.export_summary(
                f"./results/episode_{final_info['episode_id']}_summary.json"
            )
```

### Debug Integration

Hook integration with debugging and visualization systems:

```python
class DebugHookManager:
    """Hook manager for debug and visualization integration."""
    
    def register_debug_hooks(self, hook_system: HookSystemProtocol) -> None:
        """Register debug hooks with the hook system."""
        if self.debug_enabled:
            hook_system.register_pre_step_hook(self._pre_step_debug)
            hook_system.register_post_step_hook(self._post_step_debug)
            hook_system.register_episode_end_hook(self._episode_debug_summary)
    
    def _pre_step_debug(self, simulation_state: Dict[str, Any]) -> None:
        """Pre-step debugging and validation."""
        if self.step_through_enabled and self.paused:
            self._wait_for_step_signal()
        
        if self.breakpoints_enabled:
            self._check_breakpoints(simulation_state)
        
        if self.state_validation_enabled:
            self._validate_simulation_state(simulation_state)
    
    def _post_step_debug(self, simulation_state: Dict[str, Any]) -> None:
        """Post-step visualization and data collection."""
        if self.visualization_enabled:
            self._update_visualization(simulation_state)
        
        if self.performance_monitoring_enabled:
            self._monitor_performance(simulation_state)
        
        if self.trajectory_recording_enabled:
            self._record_debug_trajectory(simulation_state)
    
    def _episode_debug_summary(self, final_info: Dict[str, Any]) -> None:
        """Generate debug summary on episode completion."""
        if self.export_debug_data:
            debug_summary = {
                'episode_id': final_info.get('episode_id'),
                'performance_metrics': self._get_performance_summary(),
                'breakpoint_events': self._get_breakpoint_events(),
                'trajectory_data': self._get_trajectory_summary(),
                'visualization_frames': self._get_saved_frames()
            }
            
            output_path = f"./debug_exports/episode_{final_info['episode_id']}_debug.json"
            with open(output_path, 'w') as f:
                json.dump(debug_summary, f, indent=2)
```

### Visualization Integration

Hook registration for extensible visualization callbacks:

```python
def register_visualization_hooks(hooks: List[Dict[str, Any]]) -> None:
    """Register visualization hooks for custom debugging extensions."""
    
    visualization_registry = {}
    
    for hook_config in hooks:
        hook_name = hook_config.get('name')
        callback = hook_config.get('callback')
        frequency = hook_config.get('frequency', 1)
        enabled = hook_config.get('enabled', True)
        
        if enabled and callback:
            try:
                # Import callback function dynamically
                if isinstance(callback, str):
                    module_path, function_name = callback.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    callback_func = getattr(module, function_name)
                else:
                    callback_func = callback
                
                visualization_registry[hook_name] = {
                    'callback': callback_func,
                    'frequency': frequency,
                    'last_called': 0
                }
                
                logger.debug(f"Registered visualization hook: {hook_name}")
                
            except Exception as e:
                logger.warning(f"Failed to register visualization hook {hook_name}: {e}")

def update_visualization_hooks(step_number: int, simulation_state: Dict[str, Any]) -> None:
    """Update all registered visualization hooks based on frequency."""
    
    for hook_name, hook_info in visualization_registry.items():
        frequency = hook_info['frequency']
        last_called = hook_info['last_called']
        
        if step_number - last_called >= frequency:
            try:
                hook_info['callback'](simulation_state)
                hook_info['last_called'] = step_number
            except Exception as e:
                logger.warning(f"Visualization hook {hook_name} failed: {e}")
```

## Performance Considerations

The hook system is designed to maintain the ≤33 ms/step performance target while providing comprehensive extensibility:

### Zero-Overhead When Disabled

```python
def record_step_with_hooks(self, step_data: Dict[str, Any]) -> None:
    """Performance-optimized step recording with hook integration."""
    
    # Ultra-fast early exit when hooks disabled
    if not self.hooks_enabled:
        return  # <0.001ms overhead
    
    # Pre-step hooks
    if self.pre_step_hooks:
        for hook in self.pre_step_hooks:
            hook(step_data)
    
    # Core step processing
    self._execute_core_step(step_data)
    
    # Post-step hooks
    if self.post_step_hooks:
        for hook in self.post_step_hooks:
            hook(step_data)
```

### Buffered I/O for Recording

```python
class BufferedRecordingHook:
    """Performance-optimized recording hook with buffering."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def __call__(self, simulation_state: Dict[str, Any]) -> None:
        """Record step data with minimal overhead."""
        
        # Fast buffer append (O(1) operation)
        self.buffer.append({
            'step': simulation_state['step_count'],
            'position': simulation_state['agent_position'].copy(),
            'concentration': simulation_state['odor_concentration'],
            'timestamp': time.time()
        })
        
        # Async flush when buffer full
        if len(self.buffer) >= self.buffer_size:
            self._flush_async()
    
    def _flush_async(self) -> None:
        """Flush buffer asynchronously to avoid blocking simulation."""
        buffer_copy = self.buffer.copy()
        self.buffer.clear()
        
        # Submit to background thread
        self.flush_executor.submit(self._write_buffer, buffer_copy)
```

### Memory-Efficient State Access

```python
def memory_efficient_hook(simulation_state: Dict[str, Any]) -> None:
    """Hook implementation that minimizes memory allocation."""
    
    # Access data without copying when possible
    agent_positions = simulation_state['agent_positions']  # Reference, not copy
    step_count = simulation_state['step_count']  # Primitive, no allocation
    
    # Perform calculations in-place when safe
    if 'metrics_buffer' not in simulation_state:
        simulation_state['metrics_buffer'] = np.zeros(10)  # Reusable buffer
    
    metrics_buffer = simulation_state['metrics_buffer']
    metrics_buffer[0] = np.mean(agent_positions[:, 0])  # In-place calculation
    metrics_buffer[1] = np.mean(agent_positions[:, 1])
    
    # Avoid string concatenation and formatting in tight loops
    if step_count % 100 == 0:  # Log only occasionally
        logger.debug(f"Step {step_count}: avg position ({metrics_buffer[0]:.2f}, {metrics_buffer[1]:.2f})")
```

## Error Handling and Debugging

### Graceful Hook Failure

```python
def safe_hook_execution(hook_func: Callable, simulation_state: Dict[str, Any]) -> bool:
    """Execute hook with error handling and performance monitoring."""
    
    start_time = time.perf_counter()
    success = True
    
    try:
        hook_func(simulation_state)
    except Exception as e:
        logger.warning(f"Hook {hook_func.__name__} failed: {e}")
        success = False
        
        # Disable hook if it fails repeatedly
        failure_count = getattr(hook_func, '_failure_count', 0) + 1
        setattr(hook_func, '_failure_count', failure_count)
        
        if failure_count >= 3:
            logger.error(f"Disabling hook {hook_func.__name__} after {failure_count} failures")
            return False
    
    # Performance monitoring
    execution_time = time.perf_counter() - start_time
    if execution_time > 0.005:  # 5ms warning threshold
        logger.warning(f"Hook {hook_func.__name__} took {execution_time*1000:.1f}ms (>5ms threshold)")
    
    return success
```

### Hook Validation

```python
def validate_hook_configuration(hook_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate hook configuration before registration."""
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check required fields
    if 'callback' not in hook_config:
        validation_results['errors'].append("Missing required field: callback")
        validation_results['valid'] = False
    
    # Validate callback function
    callback = hook_config.get('callback')
    if callback and isinstance(callback, str):
        try:
            module_path, function_name = callback.rsplit('.', 1)
            module = importlib.import_module(module_path)
            callback_func = getattr(module, function_name)
            
            # Check function signature
            import inspect
            sig = inspect.signature(callback_func)
            if len(sig.parameters) != 1:
                validation_results['warnings'].append(
                    f"Hook callback should accept exactly 1 parameter, got {len(sig.parameters)}"
                )
        except Exception as e:
            validation_results['errors'].append(f"Cannot import callback function: {e}")
            validation_results['valid'] = False
    
    # Performance warnings
    frequency = hook_config.get('frequency', 1)
    if frequency < 1:
        validation_results['warnings'].append("Hook frequency < 1 may impact performance")
    
    return validation_results
```

This comprehensive hook system API enables researchers to extend plume_nav_sim capabilities through configuration-driven dependency injection while maintaining optimal performance and providing extensive debugging and monitoring capabilities.