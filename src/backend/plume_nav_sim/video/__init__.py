"""Video plume dataset schema and validation utilities.

This package defines the contract for movie/video-based plume datasets used by
downstream tooling (ingest, loader, tests, and Hydra integration).

Exports a small set of constants and a Pydantic model describing required
dataset attributes, along with helper validation functions.
"""

from .schema import (  # noqa: F401
    ALLOWED_SOURCE_DTYPES,
    DIMS_TYX,
    SCHEMA_VERSION,
    VARIABLE_NAME,
    VideoPlumeAttrs,
    validate_attrs,
)

__all__ = [
    "DIMS_TYX",
    "VARIABLE_NAME",
    "SCHEMA_VERSION",
    "ALLOWED_SOURCE_DTYPES",
    "VideoPlumeAttrs",
    "validate_attrs",
]
