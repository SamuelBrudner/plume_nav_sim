"""
Unified registration API surface.

This module exposes environment registration helpers under
``plume_nav_sim.config`` to provide a single import path for composition
and configuration workflows.

It forwards to the underlying implementation in ``plume_nav_sim.registration``.
"""

from __future__ import annotations

# Re-export canonical registration API
from plume_nav_sim.registration import ensure_registered
from plume_nav_sim.registration.register import (
    COMPONENT_ENV_ID,
    ENV_ID,
    is_registered,
    register_env,
    unregister_env,
)

__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "ensure_registered",
    "ENV_ID",
    "COMPONENT_ENV_ID",
]
