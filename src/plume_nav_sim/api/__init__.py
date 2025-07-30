"""
API module for plume_nav_sim.

This module provides high-level functions for creating navigators, running simulations,
and integrating with the plume navigation simulation framework.
"""

from .navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_plume_simulation,
    visualize_trajectory,
    create_gymnasium_environment,
    ConfigurationError,
    SimulationError,
    # Legacy compatibility aliases
    run_simulation,
    visualize_simulation_results,
    create_video_plume_from_config,
    from_legacy,
    create_navigator_from_config
)

__version__ = "1.0.0"
__all__ = [
    "create_navigator",
    "create_video_plume", 
    "run_plume_simulation",
    "visualize_plume_simulation",
    "visualize_trajectory",
    "create_gymnasium_environment",
    "ConfigurationError",
    "SimulationError",
    # Legacy compatibility aliases
    "run_simulation",
    "visualize_simulation_results",
    "create_video_plume_from_config",
    "from_legacy",
    "create_navigator_from_config"
]