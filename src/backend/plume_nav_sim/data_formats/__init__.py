"""Internal data format helpers for plume_nav_sim.

This subpackage currently exposes a lightweight validator for movie-backed
plume datasets used in media experiments and utilities.
"""

from __future__ import annotations

from .movie_schema import MovieSchemaInfo, validate_movie_dataset

__all__ = ["MovieSchemaInfo", "validate_movie_dataset"]
