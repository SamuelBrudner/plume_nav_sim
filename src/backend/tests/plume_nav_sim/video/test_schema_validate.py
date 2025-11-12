from __future__ import annotations

import numpy as np
import pytest

# Prefer the dedicated video schema for attrs validation
from plume_nav_sim.video.schema import (
    ALLOWED_SOURCE_DTYPES,
    DIMS_TYX,
    SCHEMA_VERSION,
    VideoPlumeAttrs,
    validate_attrs,
)


def test_validate_attrs_accepts_minimal_required_fields():
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "fps": 12.0,
        "source_dtype": "uint8",
        "pixel_to_grid": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "extent": (32.0, 32.0),
    }
    model = validate_attrs(attrs)
    assert isinstance(model, VideoPlumeAttrs)
    assert model.fps == 12.0
    assert model.schema_version == SCHEMA_VERSION
    assert model.source_dtype in ALLOWED_SOURCE_DTYPES
    # Dims default to canonical (t, y, x)
    assert tuple(model.dims) == DIMS_TYX


def test_validate_attrs_rejects_bad_dtype_and_dims():
    # Bad dtype
    bad = {
        "schema_version": SCHEMA_VERSION,
        "fps": 10.0,
        "source_dtype": "int64",  # not allowed in v0
        "pixel_to_grid": 1.0,
        "origin": (0.0, 0.0),
        "extent": (10.0, 10.0),
    }
    with pytest.raises(ValueError):
        validate_attrs(bad)

    # Bad dims when explicitly provided
    bad_dims = {
        "schema_version": SCHEMA_VERSION,
        "fps": 10.0,
        "source_dtype": "uint8",
        "pixel_to_grid": 1.0,
        "origin": (0.0, 0.0),
        "extent": (10.0, 10.0),
        "dims": ("x", "y", "t"),
    }
    with pytest.raises(ValueError):
        validate_attrs(bad_dims)


def test_validate_xarray_like_happy_path():
    xr = pytest.importorskip("xarray")

    T, Y, X = 4, 8, 8
    data = np.zeros((T, Y, X), dtype=np.float32)
    ds = xr.Dataset({"concentration": (DIMS_TYX, data)})
    ds.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "fps": 15.0,
            "source_dtype": "uint8",
            "pixel_to_grid": (1.0, 1.0),
            "origin": (0.0, 0.0),
            "extent": (float(Y), float(X)),
        }
    )

    # Media schema provides a duck-typed xarray validator
    from plume_nav_sim.media.schema import validate_xarray_like

    model = validate_xarray_like(ds)
    assert isinstance(model, VideoPlumeAttrs)
    assert model.fps == 15.0


def test_validate_xarray_like_rejects_missing_var():
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset({})
    ds.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "fps": 15.0,
            "source_dtype": "uint8",
            "pixel_to_grid": (1.0, 1.0),
            "origin": (0.0, 0.0),
            "extent": (8.0, 8.0),
        }
    )

    from plume_nav_sim.media.schema import validate_xarray_like

    with pytest.raises(ValueError):
        validate_xarray_like(ds)
