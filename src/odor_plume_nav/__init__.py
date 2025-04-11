"""
Odor Plume Navigation Package

A package for simulating navigation of odor plumes.

This package provides tools for simulating how agents navigate through odor plumes,
with support for both single and multi-agent simulations.
"""

__version__ = "0.1.0"

# API exports for easy access
from odor_plume_nav.api import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_simulation_results,
)

# Submodule exports for better organization
from odor_plume_nav.core import Navigator, SimpleNavigator, VectorizedNavigator, run_simulation
from odor_plume_nav.environments import VideoPlume
from odor_plume_nav.visualization import visualize_trajectory
from odor_plume_nav.config import (
    NavigatorConfig, 
    SingleAgentConfig,
    MultiAgentConfig, 
    VideoPlumeConfig,
    load_config,
    save_config,
)
from odor_plume_nav.utils import (
    # IO utilities
    load_yaml,
    save_yaml,
    load_json, 
    save_json,
    load_numpy,
    save_numpy,
    
    # Logging utilities
    setup_logger,
    get_module_logger,
    DEFAULT_FORMAT,
    MODULE_FORMAT,
    LOG_LEVELS,
)
