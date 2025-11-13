"""Schema constants and validation for video-derived plume datasets.

This module defines the public contract for Zarr/xarray datasets that store
video-backed concentration fields for the plume simulator. It exposes:

- Constants for variable name and dimension order
- A Pydantic model describing required dataset attributes
- Lightweight validators for attrs-only mappings and xarray-like datasets

The intent is for other components (ingest CLI, loader, tests, Hydra
integration) to import these constants and the validator to ensure consistent
metadata and layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

# Reuse canonical definitions from the video schema module
from plume_nav_sim.video.schema import DIMS_TYX, SCHEMA_VERSION
from plume_nav_sim.video.schema import VARIABLE_NAME as _VIDEO_VARIABLE_NAME
from plume_nav_sim.video.schema import VideoPlumeAttrs
from plume_nav_sim.video.schema import validate_attrs as _validate_video_attrs

# Backwards-compatible export using the canonical variable name
VARIABLE_CONCENTRATION: str = _VIDEO_VARIABLE_NAME

# Optional Timebase remains defined here for loaders that use it; attrs model
# unification is handled via plume_nav_sim.video.schema.VideoPlumeAttrs.


class Timebase(BaseModel):
    """Rational timebase for frame timestamps (numerator/denominator in Hz)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: PositiveInt
    denominator: PositiveInt = Field(default=1)


def validate_attrs(attrs: Mapping[str, Any]) -> VideoPlumeAttrs:
    """Validate a mapping of dataset attributes and return the parsed model.

    This is the primary entry point for ingest and loaders to ensure metadata
    consistency without importing heavy array libraries.
    """

    # Delegate to the canonical validator from the video schema module so callers
    # receive the canonical VideoPlumeAttrs type.
    return _validate_video_attrs(attrs)


def validate_xarray_like(ds: Any) -> VideoPlumeAttrs:
    """Validate an xarray-like dataset for required variable, dims, and attrs.

    The check is duck-typed to avoid a hard dependency on xarray. It expects:
    - A mapping-like `data_vars` with a variable named `VARIABLE_CONCENTRATION`
    - That variable declares dimension order via array attr `_ARRAY_DIMENSIONS`
      equal to `DIMS_TYX`; if absent, fall back to the variable's `.dims`.
    - An `attrs` mapping that satisfies the canonical `VideoPlumeAttrs`
    """

    # Resolve data variable by name from either mapping or attribute access
    data_vars = getattr(ds, "data_vars", None)
    var = None
    if isinstance(data_vars, Mapping):
        var = data_vars.get(VARIABLE_CONCENTRATION)
    if var is None and hasattr(ds, "__getitem__"):
        try:
            var = ds[VARIABLE_CONCENTRATION]
        except Exception:
            var = None
    if var is None:
        raise ValueError(f"Missing required data variable '{VARIABLE_CONCENTRATION}'")

    # Validate dims. Prefer explicit array attribute if present; otherwise use
    # the variable's dims. Enforce exact match with DIMS_TYX.
    declared_dims = None
    var_attrs = getattr(var, "attrs", None)
    if isinstance(var_attrs, Mapping) and "_ARRAY_DIMENSIONS" in var_attrs:
        declared_dims = var_attrs.get("_ARRAY_DIMENSIONS")
    if declared_dims is None:
        declared_dims = getattr(var, "dims", None)

    if not isinstance(declared_dims, (tuple, list)) or tuple(declared_dims) != DIMS_TYX:
        raise ValueError(
            f"Variable '{VARIABLE_CONCENTRATION}' must have dims {DIMS_TYX}, got {declared_dims!r}"
        )

    attrs = getattr(ds, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise ValueError("Dataset attrs must be a mapping of metadata")

    return validate_attrs(attrs)


__all__ = [
    "VARIABLE_CONCENTRATION",
    "DIMS_TYX",
    "SCHEMA_VERSION",
    "Timebase",
    "VideoPlumeAttrs",
    "validate_attrs",
    "validate_xarray_like",
]
