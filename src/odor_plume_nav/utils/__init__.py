"""
Utility module for odor plume navigation.

This module contains utility functions and classes for configuration,
file handling, and other support functionality.
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
]
