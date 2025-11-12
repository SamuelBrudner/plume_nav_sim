from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure backend package is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src" / "backend"
sys.path.insert(0, str(BACKEND))


def test_constants_and_contract_exports():
    from plume_nav_sim.video import schema as vs

    assert vs.VARIABLE_NAME == "concentration"
    assert tuple(vs.DIMS_TYX) == ("t", "y", "x")
    assert isinstance(vs.SCHEMA_VERSION, str) and vs.SCHEMA_VERSION
    # Allowed dtype set must include common media and working dtype
    for dt in ("uint8", "uint16", "float32"):
        assert dt in vs.ALLOWED_SOURCE_DTYPES


def test_validate_attrs_happy_path_scalar_and_tuple_coercion():
    from plume_nav_sim.video.schema import (
        SCHEMA_VERSION,
        VideoPlumeAttrs,
        validate_attrs,
    )

    attrs = {
        "schema_version": SCHEMA_VERSION,
        "fps": 30.0,
        "source_dtype": "uint8",
        # Accept scalars and tuples; both coerced to (y, x)
        "pixel_to_grid": 0.5,
        "origin": (0.0, 0.0),
        "extent": [64, 64],
        "dims": ("t", "y", "x"),
    }

    parsed = validate_attrs(attrs)
    assert isinstance(parsed, VideoPlumeAttrs)
    assert parsed.pixel_to_grid == (0.5, 0.5)
    assert parsed.origin == (0.0, 0.0)
    assert parsed.extent == (64.0, 64.0)


@pytest.mark.parametrize(
    "missing_key",
    [
        "fps",
        "source_dtype",
        "pixel_to_grid",
        "origin",
        "extent",
    ],
)
def test_validate_attrs_missing_required_keys_raises(missing_key: str):
    from plume_nav_sim.video.schema import SCHEMA_VERSION, validate_attrs

    base = {
        "schema_version": SCHEMA_VERSION,
        "fps": 24.0,
        "source_dtype": "float32",
        "pixel_to_grid": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "extent": (32.0, 32.0),
        "dims": ("t", "y", "x"),
    }
    base.pop(missing_key)

    with pytest.raises(Exception):
        validate_attrs(base)


def test_validate_attrs_rejects_bad_dtype_and_dims():
    from plume_nav_sim.video.schema import SCHEMA_VERSION, validate_attrs

    bad = {
        "schema_version": SCHEMA_VERSION,
        "fps": 60.0,
        "source_dtype": "int32",  # not allowed in v0 contract
        "pixel_to_grid": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "extent": (16.0, 16.0),
        "dims": ("time", "y", "x"),  # must be (t,y,x)
    }

    with pytest.raises(Exception) as excinfo:
        validate_attrs(bad)
    # Helpful error message includes fields that failed
    assert "source_dtype" in str(excinfo.value) or "dims" in str(excinfo.value)
