"""
Configuration module for odor plume navigation.

This module contains configuration models, validators, and utilities
for managing simulation and navigator configurations.
"""

# Export configuration models and utilities from this module
from odor_plume_nav.config.models import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)
from odor_plume_nav.config.utils import (
    load_config,
    save_config,
    validate_config,
    update_config,
)

__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
    "load_config",
    "save_config",
    "validate_config",
    "update_config",
]
