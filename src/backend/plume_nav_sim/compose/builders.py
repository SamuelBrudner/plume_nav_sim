"""Shim module: unified under plume_nav_sim.config.composition.

This module re-exports high level builders from the new unified API at
``plume_nav_sim.config.composition``. Prefer importing from that location.
"""

from plume_nav_sim.config.composition import (  # noqa: F401
    PolicySpec,
    SimulationSpec,
    build_env,
    build_policy,
    prepare,
)

__all__ = [
    "PolicySpec",
    "SimulationSpec",
    "build_env",
    "build_policy",
    "prepare",
]
