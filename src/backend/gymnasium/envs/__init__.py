"""Minimal envs package exposing a registry compatible with tests."""

from __future__ import annotations

import gymnasium as _gym

# Re-export the registry instance created in the top-level module
registry = _gym.envs.registry

__all__ = ["registry"]
