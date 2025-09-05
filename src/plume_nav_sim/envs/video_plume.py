"""Compatibility wrapper for video plume environment.

This module exposes :class:`VideoPlume` from the ``odor_plume_nav`` package
under the ``plume_nav_sim.envs`` namespace. It enables existing code that
expects ``plume_nav_sim.envs.video_plume`` to function without requiring a
separate implementation.
"""

from odor_plume_nav.environments.video_plume import VideoPlume

__all__ = ["VideoPlume"]
