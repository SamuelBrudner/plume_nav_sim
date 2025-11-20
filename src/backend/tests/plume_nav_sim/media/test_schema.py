from __future__ import annotations

import pytest

from plume_nav_sim.media.schema import (
    DIMS_TYX,
    SCHEMA_VERSION,
    VARIABLE_CONCENTRATION,
    VideoPlumeAttrs,
    validate_attrs,
    validate_xarray_like,
)


def _valid_attrs(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        fps=12.0,
        pixel_to_grid=(0.5, 0.5),
        origin=(0.0, 0.0),
        extent=(64.0, 64.0),
        source_dtype="uint8",
    )
    base.update(overrides)
    return base


def test_public_constants_present_and_values():
    assert VARIABLE_CONCENTRATION == "concentration"
    assert DIMS_TYX == ("t", "y", "x")
    assert isinstance(SCHEMA_VERSION, str) and len(SCHEMA_VERSION) > 0


def test_validate_attrs_accepts_mapping_and_coerces_pixel_to_grid_scalar():
    attrs = _valid_attrs(pixel_to_grid=1.25)
    model = validate_attrs(attrs)
    assert isinstance(model, VideoPlumeAttrs)
    assert model.pixel_to_grid == (1.25, 1.25)
    assert model.fps == 12.0


def test_validate_attrs_rejects_bad_values():
    with pytest.raises(Exception):
        validate_attrs(_valid_attrs(fps=0.0))
    with pytest.raises(Exception):
        validate_attrs(_valid_attrs(pixel_to_grid=(0.0, 1.0)))
    with pytest.raises(Exception):
        # wrong schema version
        validate_attrs(_valid_attrs(schema_version="999"))


def test_validate_xarray_like_happy_path():
    class _Var:
        dims = DIMS_TYX

    class _DS:
        data_vars = {VARIABLE_CONCENTRATION: _Var()}
        attrs = _valid_attrs()

    out = validate_xarray_like(_DS())
    assert isinstance(out, VideoPlumeAttrs)
    assert out.schema_version == SCHEMA_VERSION


def test_validate_xarray_like_catches_missing_var_and_dims():
    class _VarBad:
        dims = ("t", "x", "y")

    class _DSMissing:
        data_vars = {}
        attrs = _valid_attrs()

    class _DSWrongDims:
        data_vars = {VARIABLE_CONCENTRATION: _VarBad()}
        attrs = _valid_attrs()

    with pytest.raises(Exception):
        validate_xarray_like(_DSMissing())
    with pytest.raises(Exception):
        validate_xarray_like(_DSWrongDims())
