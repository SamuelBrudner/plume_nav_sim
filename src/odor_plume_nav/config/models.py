"""
Configuration models for odor plume navigation.

This module provides Pydantic models for configuration validation.
"""

# Import from the existing config_models module for now
# This will be refactored in a future update
from odor_plume_nav.config_models import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)

# Re-export the models
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
]
