from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple, Union

from pydantic import BaseModel, Field, field_validator

# Core dataset contract
VARIABLE_NAME: str = "concentration"
"""Name of the primary variable in the dataset."""

DIMS_TYX: Tuple[str, str, str] = ("t", "y", "x")
"""Canonical dimension order for the concentration variable."""

SCHEMA_VERSION: str = "1.0.0"
"""Schema version for the video plume dataset contract."""

# Conservatively allow a small, explicit set of source dtypes for v0.
ALLOWED_SOURCE_DTYPES = {"uint8", "uint16", "float32", "float64"}


def _tuple2(value: Union[float, Iterable[float]], *, name: str) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v)
    try:
        seq = list(value)  # type: ignore[arg-type]
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a float or a length-2 sequence") from e
    if len(seq) != 2:
        raise ValueError(f"{name} must have length 2, got {len(seq)}")
    return (float(seq[0]), float(seq[1]))


class VideoPlumeAttrs(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    fps: float = Field(gt=0, description="Frames per second")
    source_dtype: str = Field(description="Original frame dtype before normalization")

    # Mapping from pixel units to grid units for (y, x). A scalar applies to both axes.
    pixel_to_grid: Tuple[float, float] = Field(
        description="Scale to convert pixels to grid units (y, x)"
    )

    # Spatial reference frame of the dataset in grid units
    origin: Tuple[float, float] = Field(description="Origin of the grid in (y, x)")
    extent: Tuple[float, float] = Field(description="Extent/size of the grid in (y, x)")

    # The canonical dims for the primary variable
    dims: Tuple[str, str, str] = Field(default=DIMS_TYX)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_matches(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version: {v} (expected {SCHEMA_VERSION})"
            )
        return v

    @field_validator("source_dtype")
    @classmethod
    def _source_dtype_allowed(cls, v: str) -> str:
        if v not in ALLOWED_SOURCE_DTYPES:
            allowed = ", ".join(sorted(ALLOWED_SOURCE_DTYPES))
            raise ValueError(f"source_dtype must be one of: {allowed}")
        return v

    @field_validator("pixel_to_grid", mode="before")
    @classmethod
    def _coerce_pixel_to_grid(cls, v: Any) -> Tuple[float, float]:
        yx = _tuple2(v, name="pixel_to_grid")
        if yx[0] <= 0 or yx[1] <= 0:
            raise ValueError("pixel_to_grid values must be positive")
        return yx

    @field_validator("origin", mode="before")
    @classmethod
    def _coerce_origin(cls, v: Any) -> Tuple[float, float]:
        return _tuple2(v, name="origin")

    @field_validator("extent", mode="before")
    @classmethod
    def _coerce_extent(cls, v: Any) -> Tuple[float, float]:
        yx = _tuple2(v, name="extent")
        if yx[0] <= 0 or yx[1] <= 0:
            raise ValueError("extent values must be positive")
        return yx

    @field_validator("dims")
    @classmethod
    def _dims_are_canonical(cls, v: Tuple[str, str, str]) -> Tuple[str, str, str]:
        if tuple(v) != DIMS_TYX:
            raise ValueError(f"dims must be {DIMS_TYX}")
        return v


def validate_attrs(attrs: Mapping[str, Any]) -> VideoPlumeAttrs:
    try:
        return VideoPlumeAttrs.model_validate(attrs)
    except Exception as e:  # pydantic.ValidationError already informative
        # Re-raise as ValueError to present a consistent surface area to callers
        raise ValueError(str(e)) from e


__all__ = [
    "VARIABLE_NAME",
    "DIMS_TYX",
    "SCHEMA_VERSION",
    "ALLOWED_SOURCE_DTYPES",
    "VideoPlumeAttrs",
    "validate_attrs",
]
