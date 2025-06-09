"""
Utility module for odor plume navigation.

This module provides comprehensive utility functions and classes supporting the complete
odor plume navigation system, including:

- **IO Operations**: File handling for YAML, JSON, and NumPy data formats
- **Navigator Utilities**: Agent creation, sensor management, and parameter handling  
- **Logging Infrastructure**: Structured logging setup with loguru integration
- **Reproducibility Management**: Global seed control and deterministic experiment execution
- **Advanced Visualization**: Real-time simulation visualization and publication-quality plotting
- **Configuration Support**: Hydra integration for parameter management

The module consolidates all support functionality required for CLI operations, database
integration, advanced configuration management, and the unified package architecture
resulting from the template rendering refactoring.

Examples:
    Basic utility imports:
        >>> from odor_plume_nav.utils import load_yaml, setup_logger
        >>> config = load_yaml("config.yaml")
        >>> setup_logger(level="INFO")
    
    Reproducibility management:
        >>> from odor_plume_nav.utils import set_global_seed
        >>> set_global_seed(42)  # Ensures deterministic results
    
    Visualization utilities:
        >>> from odor_plume_nav.utils import SimulationVisualization, visualize_trajectory
        >>> viz = SimulationVisualization(fps=30, theme='scientific')
        >>> visualize_trajectory(positions, output_path="trajectory.png")
"""

# Export IO utilities from this module
from odor_plume_nav.utils.io import (
    load_yaml,
    save_yaml,
    load_json,
    save_json,
    load_numpy,
    save_numpy,
)

# Export navigator utilities from this module
from odor_plume_nav.utils.navigator_utils import (
    normalize_array_parameter,
    create_navigator_from_params,
    calculate_sensor_positions,
    sample_odor_at_sensors,
)

# Export logging utilities from this module
from odor_plume_nav.utils.logging_setup import (
    setup_logger,
    get_module_logger,
    DEFAULT_FORMAT,
    MODULE_FORMAT,
    LOG_LEVELS,
)

# Export seed management utilities for reproducibility (F-009 integration)
from odor_plume_nav.utils.seed_manager import (
    set_global_seed,
    get_global_seed_manager,
    configure_from_hydra,
    seed_context,
    get_reproducibility_report,
    SeedManager,
    RandomState,
)

# Export visualization utilities for advanced plotting (F-006 and F-007 integration)  
from odor_plume_nav.utils.visualization import (
    SimulationVisualization,
    visualize_trajectory,
    VisualizationConfig,
)


# Convenience functions for seed management (F-009 integration requirements)
def get_random_state():
    """
    Capture current global random state for checkpointing.
    
    Convenience wrapper around the global seed manager's capture_state method.
    Returns the current random state that can be restored later for reproducibility.
    
    Returns:
        RandomState: Current random state snapshot, or None if no global manager exists
        
    Examples:
        >>> set_global_seed(42)
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> restore_random_state(state)  # Restore to captured state
    """
    manager = get_global_seed_manager()
    return manager.capture_state() if manager else None


def restore_random_state(state):
    """
    Restore global random state from a previous snapshot.
    
    Convenience wrapper around the global seed manager's restore_state method.
    Restores random number generators to the exact state captured in the snapshot.
    
    Args:
        state (RandomState): Random state snapshot to restore
        
    Returns:
        bool: True if restoration was successful, False if no global manager exists
        
    Examples:
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> success = restore_random_state(state)
    """
    manager = get_global_seed_manager()
    return manager.restore_state(state) if manager and state else False


# Convenience functions for visualization (F-006 and F-007 integration requirements)
def create_realtime_visualizer(**config_kwargs):
    """
    Create a real-time visualization instance with simplified configuration.
    
    Convenience wrapper around SimulationVisualization for easy real-time animation setup.
    Provides sensible defaults for interactive visualization workflows.
    
    Args:
        **config_kwargs: Configuration parameters passed to SimulationVisualization
            Common options: fps (int), theme (str), figsize (tuple), headless (bool)
            
    Returns:
        SimulationVisualization: Configured visualization instance for real-time use
        
    Examples:
        >>> viz = create_realtime_visualizer(fps=30, theme='scientific')
        >>> viz.setup_environment(plume_data)
        >>> animation = viz.create_animation(frame_callback, frames=1000)
        >>> viz.show()
    """
    # Set defaults optimized for real-time visualization
    defaults = {
        'fps': 30,
        'theme': 'scientific', 
        'headless': False,
        'max_agents': 100,
        'dpi': 150  # Balanced quality/performance for real-time
    }
    defaults.update(config_kwargs)
    
    return SimulationVisualization(**defaults)


def create_static_plotter(**config_kwargs):
    """
    Create a high-quality static trajectory plotter with publication defaults.
    
    Convenience wrapper around visualize_trajectory for publication-quality static plots.
    Provides optimized defaults for generating publication-ready trajectory visualizations.
    
    Args:
        **config_kwargs: Configuration parameters for static plotting
            Common options: dpi (int), figsize (tuple), theme (str), format (str)
            
    Returns:
        callable: Configured plotting function that can be called with trajectory data
        
    Examples:
        >>> plotter = create_static_plotter(dpi=300, theme='presentation')
        >>> plotter(positions, orientations, output_path="figure.png")
    """
    # Set defaults optimized for publication-quality static plots
    defaults = {
        'dpi': 300,
        'theme': 'scientific',
        'format': 'png',
        'figsize': (12, 8),
        'headless': True,  # Default to headless for batch processing
        'batch_mode': False,
        'legend': True,
        'grid': True,
        'colorbar': True
    }
    defaults.update(config_kwargs)
    
    def plot_trajectory(*args, **kwargs):
        """
        Plot trajectory with pre-configured publication settings.
        
        Args:
            *args: Positional arguments passed to visualize_trajectory
            **kwargs: Keyword arguments passed to visualize_trajectory (override defaults)
            
        Returns:
            Result of visualize_trajectory function
        """
        plot_config = defaults.copy()
        plot_config.update(kwargs)
        return visualize_trajectory(*args, **plot_config)
    
    return plot_trajectory

__all__ = [
    # IO utilities
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "load_numpy",
    "save_numpy",
    
    # Navigator utilities
    "normalize_array_parameter",
    "create_navigator_from_params",
    "calculate_sensor_positions",
    "sample_odor_at_sensors",
    
    # Logging utilities
    "setup_logger",
    "get_module_logger",
    "DEFAULT_FORMAT",
    "MODULE_FORMAT",
    "LOG_LEVELS",
    
    # Seed management utilities (reproducibility support)
    "set_global_seed",
    "get_global_seed_manager", 
    "configure_from_hydra",
    "seed_context",
    "get_reproducibility_report",
    "SeedManager",
    "RandomState",
    "get_random_state",          # Convenience function
    "restore_random_state",      # Convenience function
    
    # Visualization utilities (real-time and static plotting)
    "SimulationVisualization",
    "visualize_trajectory", 
    "VisualizationConfig",
    "create_realtime_visualizer", # Convenience function
    "create_static_plotter",      # Convenience function
]
