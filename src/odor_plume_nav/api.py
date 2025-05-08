"""
Top-level public API alias module.

This module re-exports the core API functions from interfaces/api.py and visualization.
"""
from odor_plume_nav.interfaces.api import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_simulation_results,
)
from odor_plume_nav.interfaces.visualization.visualization import visualize_trajectory

__all__ = [
    "create_navigator",
    "create_video_plume",
    "run_plume_simulation",
    "visualize_simulation_results",
    "visualize_trajectory",
]
