"""Shim module: unified under plume_nav_sim.config.composition.

This module re-exports composition spec types from the new unified API at
``plume_nav_sim.config.composition``. Prefer importing from that location.
"""

from plume_nav_sim.config.composition import (  # noqa: F401
    BuiltinPolicyName,
    PolicySpec,
    SimulationSpec,
)

__all__ = ["BuiltinPolicyName", "PolicySpec", "SimulationSpec"]
