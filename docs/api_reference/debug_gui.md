# Debug GUI API Reference

## Overview

The Debug GUI system provides comprehensive interactive debugging and visualization capabilities for plume navigation simulations. This toolkit implements a dual-backend architecture supporting both desktop Qt-based interfaces and web-based Streamlit dashboards, enabling real-time simulation monitoring, step-through debugging, performance profiling, and collaborative debugging workflows.

### Key Features

- **Dual-Backend Architecture**: Seamless switching between PySide6 desktop interface and Streamlit web interface
- **Interactive Step-Through Controls**: Frame-by-frame navigation with pause/resume functionality  
- **Real-Time State Visualization**: ≥30 FPS refresh rate with configurable update intervals
- **Performance Monitoring**: Comprehensive profiling with ≤33ms step-time target enforcement
- **Collaborative Debugging**: Shared debugging sessions for team research environments
- **Zero-Overhead Design**: No performance impact when debugging is disabled
- **Extensible Hook System**: Custom debugging extensions without modifying core functionality

### Architecture Components

```python
from plume_nav_sim.debug.gui import DebugGUI, DebugSession, DebugConfig, launch_viewer
from plume_nav_sim.utils.visualization import register_visualization_hooks
```

## Core Classes

### DebugGUI

Main debug interface providing unified access to debugging capabilities with automatic backend selection.

```python
class DebugGUI:
    def __init__(
        self, 
        backend: str = 'auto', 
        config: Optional[DebugConfig] = None, 
        session: Optional[DebugSession] = None
    )
```

**Parameters:**
- `backend`: Backend selection ('qt', 'streamlit', 'auto')
- `config`: Debug configuration object
- `session`: Debug session object

**Backend Selection Strategy:**
1. **Primary Backend - PySide6**: Native desktop application with full debugging capabilities
2. **Fallback Backend - Streamlit**: Web-based interface for remote debugging
3. **Graceful Degradation**: Console-based debugging when GUI backends unavailable

#### Methods

##### show()
Display debug GUI interface with backend-appropriate rendering.

```python
def show(self) -> None
```

**Qt Backend**: Launches desktop application with event loop
**Streamlit Backend**: Creates web dashboard accessible via browser
**Console Backend**: Provides text-based debugging interface

##### set_simulation_state(state)
Update simulation state across all debug components with performance tracking.

```python
def set_simulation_state(self, state: Dict[str, Any]) -> None
```

**Parameters:**
- `state`: Complete simulation state dictionary containing:
  - `observation`: Current environment observation
  - `reward`: Step reward value
  - `terminated`/`done`: Episode completion status
  - `info`: Additional environment information
  - `step_count`: Current simulation step
  - `step_time_ms`: Step execution time in milliseconds

**Features:**
- Automatic performance metrics recording
- Breakpoint condition checking
- State history management with configurable retention
- Thread-safe updates for real-time visualization

##### start_session()
Initialize debug session with correlation tracking and backend activation.

```python
def start_session(self) -> None
```

**Capabilities:**
- Session correlation context creation
- Performance monitoring initialization
- Collaborative debugging setup if configured
- Real-time visualization timer activation

##### step_through()
Execute single simulation step with comprehensive debugging support.

```python
def step_through(self) -> bool
```

**Returns:** `True` if step execution successful

**Features:**
- Step performance measurement and violation detection
- Correlation context for debugging traceability
- Automatic logging of performance violations
- Error handling with session context preservation

##### export_screenshots(output_dir)
Export current visualization as timestamped screenshot with metadata.

```python
def export_screenshots(self, output_dir: str = './debug_exports') -> Optional[str]
```

**Parameters:**
- `output_dir`: Output directory for screenshot files

**Returns:** Path to exported screenshot or `None` if failed

**Export Formats:** PNG, PDF, SVG (configurable via `DebugConfig.export_format`)

##### add_breakpoint(condition, **kwargs)
Add conditional breakpoint with session tracking and GUI integration.

```python
def add_breakpoint(self, condition: str, **kwargs) -> int
```

**Parameters:**
- `condition`: Breakpoint condition expression (e.g., `"odor_reading > 0.8"`)
- `**kwargs`: Additional breakpoint parameters

**Returns:** Breakpoint ID for management operations

**Condition Examples:**
```python
# Basic sensor threshold
gui.add_breakpoint("odor_reading > 0.8")

# Agent position-based
gui.add_breakpoint("agent_x > 50 and agent_y < 25")

# Performance-based
gui.add_breakpoint("step_time_ms > 35.0")
```

##### get_performance_metrics()
Retrieve comprehensive performance statistics with historical analysis.

```python
def get_performance_metrics(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing:
- `avg_step_time_ms`: Average step execution time
- `max_step_time_ms`: Maximum recorded step time
- `performance_violations`: Count of threshold violations
- `memory_usage_mb`: Current memory consumption
- `total_steps`: Total steps executed in session

### DebugSession

Session management with correlation tracking and collaborative debugging support.

```python
@dataclass
class DebugSession:
    def __init__(self, session_id: Optional[str] = None)
```

**Attributes:**
- `session_id`: Unique session identifier (auto-generated if not provided)
- `start_time`: Session start timestamp
- `current_step`: Current simulation step count
- `is_paused`: Pause state for step-through debugging
- `breakpoints`: List of active breakpoints
- `collaborative_config`: Settings for shared debugging sessions

#### Methods

##### configure(**kwargs)
Configure session parameters including collaborative debugging.

```python
def configure(self, **config_kwargs) -> None
```

**Collaborative Configuration:**
```python
session.configure(
    shared=True,
    host='localhost',
    port=8502,
    mode='host'  # 'host' or 'participant'
)
```

##### start()
Activate session with correlation context and collaborative debugging.

```python
def start(self) -> DebugSession
```

**Features:**
- Correlation context creation for traceability
- Collaborative debugging initialization
- Performance monitoring activation
- Session event logging

##### export_session_data(output_path)
Export complete session data including breakpoints and performance metrics.

```python
def export_session_data(self, output_path: Union[str, Path]) -> bool
```

**Export Contents:**
- Session information and configuration
- Breakpoint definitions and hit counts
- Performance metrics and violations
- Collaborative debugging participants

### DebugConfig

Configuration management for debug GUI settings and performance tuning.

```python
@dataclass
class DebugConfig:
    backend: str = 'auto'
    window_size: tuple = (1200, 800)
    refresh_rate: int = 30  # ≥30 FPS requirement
    theme: str = 'auto'
    show_inspector: bool = True
    enable_profiling: bool = True
    export_format: str = 'png'
    performance_target_ms: float = 33.0  # ≤33ms requirement
```

**Performance Settings:**
- `refresh_rate`: Visualization update frequency in Hz
- `performance_target_ms`: Step time threshold for violation detection
- `max_history_length`: Maximum retained states for analysis

**UI Customization:**
- `theme`: Visual theme ('dark', 'light', 'auto')
- `window_size`: Default window dimensions for Qt backend
- `export_format`: Screenshot export format ('png', 'pdf', 'svg')

## Interactive Debugging Features

### Step-Through Controls

The debug GUI provides comprehensive step-through debugging capabilities:

#### Pause/Resume Functionality
```python
# Toggle play/pause state
debug_gui.session.is_paused = not debug_gui.session.is_paused

# Programmatic control
if debug_gui.session.is_paused:
    debug_gui.session.is_paused = False  # Resume
```

#### Frame-by-Frame Navigation
```python
# Execute single step in paused mode
success = debug_gui.step_through()

# Step with custom action
if debug_gui.env:
    action = debug_gui.env.action_space.sample()
    result = debug_gui.env.step(action)
```

#### Keyboard Shortcuts (Qt Backend)
- **Space**: Toggle play/pause
- **Right Arrow**: Single step
- **R**: Reset simulation
- **Ctrl+S**: Export screenshot

### Breakpoint Management

Advanced breakpoint system with condition evaluation and session persistence:

#### Conditional Breakpoints
```python
# Sensor-based breakpoints
bp_id = debug_gui.add_breakpoint("odor_reading > 0.8")

# Multi-condition breakpoints
debug_gui.add_breakpoint("agent_x > 50 and odor_reading > 0.5")

# Performance breakpoints
debug_gui.add_breakpoint("step_time_ms > 35.0")
```

#### Breakpoint Hit Handling
```python
# Automatic pause on breakpoint hit
# Logging of breakpoint events with correlation tracking
# GUI notification and state inspection activation
```

### State Inspection

Comprehensive state inspection with real-time updates:

#### Current State Display
- **Agent Information**: Position, orientation, sensor readings
- **Environment Parameters**: Source locations, boundary conditions
- **Performance Metrics**: Step timing, memory usage, frame rate

#### Custom Inspectors
```python
# Add custom inspection function
def custom_inspector(state):
    return {
        'custom_metric': compute_metric(state),
        'analysis_result': analyze_state(state)
    }

debug_gui.session.add_inspector('custom_analysis', custom_inspector)
```

### Real-Time Visualization

High-performance visualization with configurable refresh rates:

#### Visualization Components
- **Agent Trajectories**: Real-time path tracking with history
- **Plume Concentration**: Background overlay with colormap
- **Source Locations**: Dynamic source positioning and emission rates
- **Performance Overlay**: Real-time metrics display

#### Performance Optimization
- **Adaptive Rendering**: Automatic quality adjustment based on performance
- **Vectorized Operations**: Efficient multi-agent visualization
- **Frame Rate Control**: Configurable FPS with automatic degradation

## Setup and Configuration

### Installation Instructions

#### Required Dependencies
```bash
# Core dependencies (always required)
pip install numpy>=1.26.0 matplotlib>=3.7.0 hydra-core>=1.3.0

# Qt Desktop Backend (optional)
pip install PySide6>=6.9.0

# Streamlit Web Backend (optional)  
pip install streamlit>=1.46.0
```

#### Development Installation
```bash
# Install with all debug dependencies
pip install plume-nav-sim[debug]

# Or install specific backends
pip install plume-nav-sim[qt]      # Qt backend only
pip install plume-nav-sim[web]     # Streamlit backend only
```

### Backend Selection

#### Automatic Backend Selection
```python
# Automatic fallback: Qt → Streamlit → Console
debug_gui = DebugGUI(backend='auto')
```

#### Explicit Backend Configuration
```python
# Qt Desktop Interface
from plume_nav_sim.debug.gui import DebugGUI, DebugConfig

config = DebugConfig(
    backend='qt',
    window_size=(1400, 900),
    refresh_rate=60,
    theme='dark'
)
debug_gui = DebugGUI(backend='qt', config=config)
```

```python
# Streamlit Web Interface  
config = DebugConfig(
    backend='streamlit',
    refresh_rate=30,
    enable_profiling=True
)
debug_gui = DebugGUI(backend='streamlit', config=config)
debug_gui.configure_backend(port=8501, host='0.0.0.0')
```

### Configuration Examples

#### Basic Debug Configuration
```yaml
# conf/base/hooks/debug.yaml
hooks:
  debug:
    enabled: true
    backend: "auto"
    enable_profiling: true
    visualization:
      enabled: true
      refresh_rate: 30
      show_inspector: true
```

#### Advanced Performance Configuration
```yaml
hooks:
  debug:
    enabled: true
    backend: "qt"
    performance_target_ms: 25.0  # Stricter performance target
    advanced:
      profiling:
        enabled: true
        profile_memory: true
        generate_reports: true
```

### Troubleshooting Guide

#### Common Issues

**Qt Backend Not Available:**
```python
try:
    debug_gui = DebugGUI(backend='qt')
except ImportError:
    print("PySide6 not installed. Installing...")
    # pip install PySide6>=6.9.0
```

**Performance Issues:**
```python
# Reduce refresh rate for better performance
config = DebugConfig(refresh_rate=15, enable_profiling=False)

# Use console backend for minimal overhead
debug_gui = DebugGUI(backend='console')
```

**Memory Issues:**
```python
# Limit state history retention
config = DebugConfig(max_history_length=100)

# Disable detailed profiling
config = DebugConfig(enable_profiling=False)
```

## Hook System Integration

### Debug Hooks Configuration

Integration with the extensible hook system for custom debugging workflows:

#### Hook Registration
```python
from plume_nav_sim.utils.visualization import register_visualization_hooks

# Register custom debug hooks
hooks = {}
debug_callbacks = {
    'state_inspector': custom_state_inspector,
    'performance_monitor': custom_performance_monitor
}

register_visualization_hooks(
    hook_registry=hooks,
    debug_callbacks=debug_callbacks
)
```

#### Pre-Step Debug Hooks
```yaml
# conf/base/hooks/debug.yaml
hooks:
  debug:
    callbacks:
      pre_step:
        - _target_: plume_nav_sim.debug.hooks.performance_monitor_pre_step
          enabled: true
          threshold_ms: 33.0
```

#### Post-Step Debug Hooks
```yaml
hooks:
  debug:
    callbacks:
      post_step:
        - _target_: plume_nav_sim.debug.hooks.visualization_update_post_step
          enabled: true
          refresh_rate: 30
```

### Custom Debugging Extensions

#### Visualization Callbacks
```python
def custom_visualization_hook(state, gui_context):
    """Custom visualization overlay for specialized debugging."""
    ax = gui_context['axes']
    
    # Add custom plot elements
    if 'custom_data' in state:
        custom_data = state['custom_data']
        ax.scatter(custom_data['x'], custom_data['y'], 
                  c='red', marker='X', s=100, label='Custom Points')
    
    return True  # Indicate successful rendering
```

#### Performance Monitoring Hooks
```python
def custom_performance_hook(step_metrics):
    """Custom performance analysis and alerting."""
    if step_metrics['step_time_ms'] > 40.0:
        # Custom alert logic
        send_alert(f"Performance degradation: {step_metrics['step_time_ms']:.1f}ms")
    
    return {
        'custom_metric': calculate_efficiency(step_metrics),
        'alert_level': determine_alert_level(step_metrics)
    }
```

### Runtime Hook Management

#### Dynamic Hook Registration
```python
# Add hooks during session
debug_gui.session.add_inspector('performance_analyzer', custom_performance_hook)

# Enable/disable hooks based on conditions
if debug_gui.get_performance_metrics()['avg_step_time_ms'] > 30.0:
    debug_gui.session.enable_detailed_profiling()
```

## Session Management

### Debug Session Lifecycle

Comprehensive session management with persistence and collaboration:

#### Session Creation and Configuration
```python
from plume_nav_sim.debug.gui import DebugSession

# Basic session
session = DebugSession()

# Configured session with collaboration
session = DebugSession(session_id="research_experiment_001")
session.configure(
    shared=True,
    host='localhost',
    port=8502,
    mode='host',
    max_participants=3
)
```

#### Session Persistence
```python
# Export session data
session_file = "debug_session_001.json"
success = session.export_session_data(session_file)

# Session data includes:
# - Complete session configuration
# - All breakpoint definitions and hit counts  
# - Performance metrics and violation history
# - Collaborative debugging participant list
```

### Screenshot Export

Professional-quality export capabilities for documentation:

#### Basic Screenshot Export
```python
# Export current state
screenshot_path = debug_gui.export_screenshots('./exports')

# Custom format and quality
debug_gui.config.export_format = 'pdf'
debug_gui.export_screenshots('./publication_figures')
```

#### Batch Export Automation
```python
# Automated screenshot capture
import time

def auto_screenshot_callback(state):
    """Automatic screenshot on specific conditions."""
    if state.get('episode_complete', False):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"./exports/episode_end_{timestamp}"
        debug_gui.export_screenshots(output_path)

debug_gui.session.add_inspector('auto_screenshot', auto_screenshot_callback)
```

### State Dump Capabilities

Comprehensive state capture for analysis and reproducibility:

#### Full State Export
```python
# Export complete simulation state
state_data = debug_gui.current_state
with open('state_dump.json', 'w') as f:
    json.dump(state_data, f, indent=2, default=str)
```

#### Selective State Capture
```python
# Custom state filtering
def export_filtered_state(state, components=['agents', 'environment']):
    """Export only specified state components."""
    filtered = {k: v for k, v in state.items() if k in components}
    return filtered
```

### Collaborative Debugging

Multi-user debugging sessions for team research:

#### Host Configuration
```python
# Setup collaborative host
session = DebugSession()
session.configure(
    shared=True,
    mode='host',
    host='0.0.0.0',  # Accept connections from any IP
    port=8502
)

debug_gui = DebugGUI(session=session, backend='streamlit')
debug_gui.start_session()
```

#### Participant Connection
```python
# Connect as participant
session = DebugSession()
session.configure(
    shared=True,
    mode='participant',
    host='research.server.edu',
    port=8502
)

debug_gui = DebugGUI(session=session, backend='streamlit')
debug_gui.start_session()
```

#### Shared State Synchronization
- **Real-time State Updates**: All participants see synchronized simulation state
- **Breakpoint Sharing**: Breakpoints set by any participant affect all sessions
- **Performance Metrics**: Aggregated performance data across all participants
- **Chat Integration**: Built-in communication for collaborative analysis

## Performance and Workflows

### Performance Optimization

The debug GUI system is designed to maintain simulation performance targets:

#### Zero-Overhead Design
```python
# Debug GUI with no performance impact when disabled
config = DebugConfig(enabled=False)
debug_gui = DebugGUI(config=config)

# All debug operations become no-ops
debug_gui.set_simulation_state(state)  # <0.1ms overhead
debug_gui.step_through()  # No additional latency
```

#### Adaptive Performance Management
```python
# Automatic performance adjustment
if debug_gui.get_performance_metrics()['avg_step_time_ms'] > 33.0:
    # Reduce visualization refresh rate
    debug_gui.config.refresh_rate = 15
    
    # Disable detailed profiling
    debug_gui.config.enable_profiling = False
    
    # Switch to console backend
    debug_gui.switch_backend('console')
```

#### Performance Monitoring Hooks
```python
def performance_monitor_hook(metrics):
    """Monitor and respond to performance issues."""
    if metrics['step_time_ms'] > 33.0:
        # Log performance violation
        logger.warning(f"Step time violation: {metrics['step_time_ms']:.1f}ms")
        
        # Trigger performance alert
        if metrics['step_time_ms'] > 50.0:
            debug_gui.session.add_breakpoint("step_time_ms > 50.0")
    
    return metrics
```

### Debugging Workflows

#### Development Workflow
```python
# 1. Setup debug environment
config = DebugConfig(
    backend='qt',
    enable_profiling=True,
    show_inspector=True
)

debug_gui = launch_viewer(env=my_env, config=config)

# 2. Set breakpoints for investigation
debug_gui.add_breakpoint("odor_reading > 0.8")
debug_gui.add_breakpoint("step_time_ms > 35.0")

# 3. Run simulation with debugging
for episode in range(num_episodes):
    obs = env.reset()
    debug_gui.set_simulation_state({'observation': obs, 'step_count': 0})
    
    while True:
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        
        state = {
            'observation': obs,
            'reward': reward, 
            'done': done,
            'info': info,
            'step_count': env._elapsed_steps
        }
        debug_gui.set_simulation_state(state)
        
        if done:
            break

# 4. Export session data
debug_gui.session.export_session_data('./debug_results')
```

#### Research Iteration Patterns

**Hypothesis Testing Workflow:**
```python
# Setup experimental conditions with debugging
conditions = ['baseline', 'modified_reward', 'new_algorithm']

for condition in conditions:
    # Configure condition-specific debugging
    session = DebugSession(session_id=f"experiment_{condition}")
    debug_gui = launch_viewer(env=env, session=session)
    
    # Add condition-specific breakpoints
    debug_gui.add_breakpoint(f"reward > baseline_threshold")
    
    # Run experiment with debugging
    run_experiment(condition, debug_gui)
    
    # Export condition-specific results
    debug_gui.session.export_session_data(f'./results/{condition}')
```

**Performance Optimization Workflow:**
```python
# Profile different components
components = ['navigation', 'plume_sampling', 'boundary_checking']

for component in components:
    # Enable component-specific profiling
    config = DebugConfig(
        enable_profiling=True,
        profile_components=[component]
    )
    
    debug_gui = launch_viewer(env=env, config=config)
    
    # Add performance breakpoints
    debug_gui.add_breakpoint(f"{component}_time_ms > threshold")
    
    # Run profiling session
    profile_component(component, debug_gui)
    
    # Analyze performance metrics
    metrics = debug_gui.get_performance_metrics()
    optimize_component(component, metrics)
```

### Common Use Cases

#### Algorithm Development
```python
# Debug new navigation algorithm
def debug_new_algorithm():
    debug_gui = launch_viewer(env=env, backend='qt')
    
    # Monitor algorithm decisions
    debug_gui.add_breakpoint("action_uncertainty > 0.5")
    debug_gui.add_breakpoint("exploration_bonus < 0.1")
    
    # Real-time algorithm state inspection
    def algorithm_inspector(state):
        return {
            'policy_entropy': calculate_entropy(state['action_probs']),
            'value_estimate': state['value_function_output'],
            'gradient_norm': state['gradient_magnitude']
        }
    
    debug_gui.session.add_inspector('algorithm', algorithm_inspector)
```

#### Multi-Agent Coordination
```python
# Debug multi-agent interactions
def debug_coordination():
    config = DebugConfig(
        enable_profiling=True,
        max_history_length=2000  # Extended history for coordination analysis
    )
    
    debug_gui = launch_viewer(env=multi_agent_env, config=config)
    
    # Monitor coordination metrics
    debug_gui.add_breakpoint("inter_agent_distance < collision_threshold")
    debug_gui.add_breakpoint("coordination_score < target_score")
    
    # Coordination-specific visualization
    def coordination_visualizer(state):
        agents = state['agent_positions']
        # Visualize agent communication networks
        # Display coordination effectiveness metrics
        pass
    
    debug_gui.session.add_inspector('coordination', coordination_visualizer)
```

#### Performance Benchmarking
```python
# Benchmark system performance
def benchmark_performance():
    config = DebugConfig(
        enable_profiling=True,
        performance_target_ms=25.0  # Strict performance requirement
    )
    
    debug_gui = launch_viewer(env=env, config=config)
    
    # Performance regression detection
    debug_gui.add_breakpoint("step_time_ms > 25.0")
    debug_gui.add_breakpoint("memory_usage_mb > 512")
    
    # Run benchmark suite
    for test_case in benchmark_cases:
        run_test_case(test_case, debug_gui)
        
        # Collect performance data
        perf_data = debug_gui.get_performance_metrics()
        save_benchmark_results(test_case, perf_data)
```

## Utility Functions

### launch_viewer()

Factory function for convenient debug GUI creation and initialization:

```python
def launch_viewer(
    env: Optional[PlumeNavigationEnv] = None,
    backend: str = 'auto',
    config: Optional[DebugConfig] = None,
    session: Optional[DebugSession] = None,
    **kwargs
) -> DebugGUI
```

**Parameters:**
- `env`: Environment instance to debug
- `backend`: Backend selection ('qt', 'streamlit', 'auto')
- `config`: Debug configuration object  
- `session`: Debug session object
- `**kwargs`: Additional configuration parameters

**Returns:** Configured and initialized `DebugGUI` instance

**Usage Examples:**
```python
# Quick launch with defaults
debug_gui = launch_viewer(env=my_env)

# Custom configuration
debug_gui = launch_viewer(
    env=my_env,
    backend='qt',
    refresh_rate=60,
    enable_profiling=True
)

# Collaborative session
session = DebugSession()
session.configure(shared=True, host='localhost', port=8502)
debug_gui = launch_viewer(env=my_env, session=session)
```

### plot_initial_state()

Visualization function for initial simulation setup:

```python
def plot_initial_state(
    env: PlumeNavigationEnv,
    source: Optional[Any] = None,
    agent_positions: Optional[np.ndarray] = None,
    domain_bounds: Optional[tuple] = None,
    **kwargs
) -> Optional[Figure]
```

**Parameters:**
- `env`: Environment instance providing domain information
- `source`: Source instance for position visualization
- `agent_positions`: Agent starting positions as array
- `domain_bounds`: Domain boundaries as (left, right, bottom, top)
- `**kwargs`: Additional plotting parameters

**Returns:** Matplotlib Figure object for further customization

**Features:**
- **Source Visualization**: Odor source locations with emission indicators
- **Domain Boundaries**: Navigation domain limits and constraints
- **Agent Start Positions**: Initial agent placement with orientation arrows
- **Publication Quality**: High-resolution output suitable for research papers

**Usage Examples:**
```python
# Basic initial state plot
fig = plot_initial_state(env, source=my_source, 
                        agent_positions=start_positions)

# Customized visualization
fig = plot_initial_state(
    env, 
    source=my_source,
    agent_positions=positions,
    domain_bounds=(0, 100, 0, 100),
    title="Experimental Setup",
    dpi=300,
    format='pdf'
)

# Save for publication
fig.savefig('initial_state.pdf', dpi=300, bbox_inches='tight')
```

## Best Practices

### Performance Guidelines

1. **Enable Profiling Selectively**: Only enable detailed profiling when investigating performance issues
2. **Limit History Retention**: Configure `max_history_length` based on available memory
3. **Use Appropriate Refresh Rates**: Match refresh rate to display capabilities and performance requirements
4. **Backend Selection**: Use Qt for development, Streamlit for remote debugging, Console for automated testing

### Memory Management

```python
# Efficient memory usage
config = DebugConfig(
    max_history_length=500,  # Limit state retention
    enable_profiling=False,  # Disable if not needed
    refresh_rate=15          # Reduce if performance is critical
)

# Periodic cleanup
if len(debug_gui.session.state_history) > 1000:
    debug_gui.session.state_history = debug_gui.session.state_history[-500:]
```

### Debugging Strategy

1. **Start Simple**: Begin with basic breakpoints and state inspection
2. **Iterative Refinement**: Add more sophisticated debugging as needed  
3. **Performance Awareness**: Monitor debug overhead and adjust accordingly
4. **Collaboration**: Use shared sessions for team debugging and knowledge transfer
5. **Documentation**: Export session data and screenshots for reproducible research

### Error Handling

```python
try:
    debug_gui = launch_viewer(env=env, backend='qt')
    debug_gui.start_session()
except ImportError as e:
    print(f"GUI backend unavailable: {e}")
    # Fallback to console debugging
    debug_gui = launch_viewer(env=env, backend='console')
except Exception as e:
    print(f"Debug setup failed: {e}")
    # Continue without debugging
    debug_gui = None
```

This comprehensive API reference provides complete documentation for the debug GUI system, covering all aspects from basic usage to advanced customization and performance optimization. The system's dual-backend architecture ensures broad compatibility while maintaining high performance standards for research applications.