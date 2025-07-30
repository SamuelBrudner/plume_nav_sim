"""
API module for plume_nav_sim.

This module provides high-level functions for creating navigators, running simulations,
and integrating with the plume navigation simulation framework.
"""

from . import navigation
from .navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_plume_simulation,
    create_gymnasium_environment,
    ConfigurationError,
    SimulationError,
    # Legacy compatibility aliases
    run_simulation,
    visualize_simulation_results,
)

__version__ = "1.0.0"
__all__ = [
    "create_navigator",
    "create_video_plume", 
    "run_plume_simulation",
    "visualize_plume_simulation",
    "create_gymnasium_environment",
    "navigation",
    "ConfigurationError",
    "SimulationError",
    # Legacy compatibility aliases
    "run_simulation",
    "visualize_simulation_results",
]