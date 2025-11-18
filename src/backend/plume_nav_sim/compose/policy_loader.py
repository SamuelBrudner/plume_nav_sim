"""Shim module: unified under plume_nav_sim.config.composition.

This module re-exports policy loading helpers from the new unified API at
``plume_nav_sim.config.composition``. Prefer importing from that location.
"""

from plume_nav_sim.config.composition import (  # noqa: F401
    LoadedPolicy,
    load_policy,
    reset_policy_if_possible,
)

__all__ = ["LoadedPolicy", "load_policy", "reset_policy_if_possible"]
