"""
Internal utilities for the public API (not exposed to users).
"""
from typing import Any

def merge_config_with_args(config: dict, **kwargs) -> dict:
    """Merge config dict with direct arguments, giving precedence to non-None kwargs."""
    merged = dict(config)
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    return merged

# (Optionally move _load_navigator_from_config here if it's not public)
