from __future__ import annotations

# Re-export canonical registration API
from plume_nav_sim.registration import ensure_registered
from plume_nav_sim.registration.register import ENV_ID, is_registered, register_env, unregister_env

__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "ensure_registered",
    "ENV_ID",
]
