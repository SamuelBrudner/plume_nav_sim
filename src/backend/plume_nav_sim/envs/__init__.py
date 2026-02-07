"""Environment package public exports.

This module intentionally exposes only the current stable environment surface.
Legacy convenience wrappers live in :mod:`plume_nav_sim.envs.compat`.
"""

from __future__ import annotations

from .component_env import ComponentBasedEnvironment
from .factory import create_component_environment
from .plume_env import PlumeEnv, create_plume_env

__all__ = [
    "PlumeEnv",
    "ComponentBasedEnvironment",
    "create_plume_env",
    "create_component_environment",
]
