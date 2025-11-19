"""
Schema constants and lightweight validator for video plume Zarr datasets.

This module defines the expected variable/dimension names and a simple
runtime validator returning parsed metadata needed by MoviePlumeField.

Notes
- This is a minimal slice to unblock the MoviePlumeField loader. It does not
  attempt to be a full bead-87 implementation but mirrors its intent: clear
  constants and a validator callable producing structured info or raising a
  ValidationError with actionable context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from ..utils.exceptions import ValidationError

# Variable and dimension contract
VAR_CONCENTRATION = "concentration"
DIMS_TYX: Tuple[str, str, str] = ("t", "y", "x")

# Required attribute keys on the dataset or variable
ATTR_FPS = "fps"
ATTR_ORIGIN = "origin"  # e.g., "lower" or "upper"
ATTR_EXTENT = "extent"  # [x0, x1, y0, y1]
ATTR_PIXEL_TO_GRID = "pixel_to_grid"  # scalar (float) or (x_scale, y_scale)
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_SOURCE_DTYPE = "source_dtype"


@dataclass(frozen=True)
class MovieSchemaInfo:
    """Parsed metadata for a validated movie plume dataset."""

    width: int
    height: int
    frames: int
    fps: float
    origin: str
    extent: Tuple[float, float, float, float]
    pixel_to_grid: Tuple[float, float]
    dtype_str: str


def _coerce_pixel_to_grid(value: Any) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        f = float(value)
        if f <= 0:
            raise ValidationError("pixel_to_grid must be positive")
        return (f, f)
    if isinstance(value, Sequence) and len(value) == 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError("pixel_to_grid entries must be numeric") from exc
        if x <= 0 or y <= 0:
            raise ValidationError("pixel_to_grid entries must be positive")
        return (x, y)
    raise ValidationError(
        "pixel_to_grid must be a number or 2-length sequence of numbers"
    )


def _require_attr(attrs: Mapping[str, Any], key: str, expected: str) -> Any:
    if key not in attrs:
        raise ValidationError(f"Missing required attribute: {key}", parameter_name=key)

    val = attrs[key]

    def _validate_fps(v: Any) -> float:
        try:
            fps_val = float(v)
        except Exception as exc:
            raise ValidationError("fps must be numeric") from exc
        if fps_val <= 0:
            raise ValidationError("fps must be positive")
        return fps_val

    def _validate_origin(v: Any) -> str:
        if not isinstance(v, str) or v.lower() not in {"lower", "upper"}:
            raise ValidationError('origin must be "lower" or "upper"')
        return v.lower()

    def _validate_extent(v: Any) -> Tuple[float, float, float, float]:
        if not isinstance(v, Sequence) or len(v) != 4:
            raise ValidationError("extent must be a 4-length sequence [x0,x1,y0,y1]")
        try:
            x0, x1, y0, y1 = (
                float(v[0]),
                float(v[1]),
                float(v[2]),
                float(v[3]),
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError("extent entries must be numeric") from exc
        return (x0, x1, y0, y1)

    validators: Mapping[str, Callable[[Any], Any]] = {
        ATTR_FPS: _validate_fps,
        ATTR_ORIGIN: _validate_origin,
        ATTR_EXTENT: _validate_extent,
        ATTR_PIXEL_TO_GRID: _coerce_pixel_to_grid,
    }

    if key in validators:
        return validators[key](val)
    # Default: return as-is
    return val


def validate_movie_dataset(ds: Any) -> MovieSchemaInfo:
    """Validate an xarray Dataset for the movie plume contract.

    Expected:
    - DataArray named VAR_CONCENTRATION with dims DIMS_TYX
    - dtype float32
    - Required attributes present and well-typed
    """
    # Import lazily to avoid hard dependency at import time
    try:  # type: ignore
        import xarray as xr  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise ValidationError(
            "xarray is required to validate movie plume datasets; install extras 'media'",
            underlying_error=exc,  # type: ignore[arg-type]
        )

    if VAR_CONCENTRATION not in ds:
        raise ValidationError(
            f"Dataset missing variable '{VAR_CONCENTRATION}'",
            parameter_name=VAR_CONCENTRATION,
        )

    da = ds[VAR_CONCENTRATION]
    dims = tuple(getattr(da, "dims", ()))
    if dims != DIMS_TYX:
        raise ValidationError(
            f"concentration dims {dims} != expected {DIMS_TYX}",
            parameter_name="dims",
            expected_format=str(DIMS_TYX),
        )

    # Shape: (t, y, x)
    try:
        t, y, x = int(da.sizes["t"]), int(da.sizes["y"]), int(da.sizes["x"])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValidationError("Invalid sizes on concentration variable") from exc

    # dtype: expect float32
    dtype_str = str(getattr(da.data, "dtype", getattr(da, "dtype", "")))
    if "float32" not in dtype_str:
        raise ValidationError(
            f"concentration dtype {dtype_str} != expected float32",
            parameter_name="dtype",
            expected_format="float32",
        )

    # Attributes can live on dataset or on the variable; prefer variable
    attrs: Mapping[str, Any] = {}
    attrs.update(getattr(ds, "attrs", {}) or {})
    attrs.update(getattr(da, "attrs", {}) or {})

    fps = _require_attr(attrs, ATTR_FPS, "float")
    origin = _require_attr(attrs, ATTR_ORIGIN, "str")
    extent = _require_attr(attrs, ATTR_EXTENT, "sequence[4]")
    px_grid = _require_attr(attrs, ATTR_PIXEL_TO_GRID, "float|pair")

    return MovieSchemaInfo(
        width=x,
        height=y,
        frames=t,
        fps=float(fps),
        origin=str(origin),
        extent=(float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])),
        pixel_to_grid=(float(px_grid[0]), float(px_grid[1])),
        dtype_str=dtype_str,
    )


__all__ = [
    "VAR_CONCENTRATION",
    "DIMS_TYX",
    "ATTR_FPS",
    "ATTR_ORIGIN",
    "ATTR_EXTENT",
    "ATTR_PIXEL_TO_GRID",
    "ATTR_SCHEMA_VERSION",
    "ATTR_SOURCE_DTYPE",
    "MovieSchemaInfo",
    "validate_movie_dataset",
]
