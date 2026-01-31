from __future__ import annotations

import pytest

from plume_nav_sim._compat import ValidationError
from plume_nav_sim.data_formats.movie_schema import (
    ATTR_EXTENT,
    ATTR_FPS,
    ATTR_ORIGIN,
    ATTR_PIXEL_TO_GRID,
    _require_attr,
)


def _make_attrs(**overrides):
    base = {
        ATTR_FPS: 12.0,
        ATTR_ORIGIN: "lower",
        ATTR_EXTENT: [0.0, 10.0, 0.0, 5.0],
        ATTR_PIXEL_TO_GRID: 2.0,
    }
    base.update(overrides)
    return base


def test_require_attr_validates_known_keys_and_normalizes_values():
    attrs = _make_attrs(origin="UPPER", pixel_to_grid=1.5)

    fps = _require_attr(attrs, ATTR_FPS, "float")
    origin = _require_attr(attrs, ATTR_ORIGIN, "str")
    extent = _require_attr(attrs, ATTR_EXTENT, "sequence[4]")
    pixel_to_grid = _require_attr(attrs, ATTR_PIXEL_TO_GRID, "float|pair")

    assert isinstance(fps, float) and fps == pytest.approx(12.0)
    assert origin == "upper"  # normalized to lowercase
    assert extent == (0.0, 10.0, 0.0, 5.0)
    assert pixel_to_grid == (1.5, 1.5)


def test_require_attr_missing_key_raises_validation_error_with_context():
    expected = "float"
    with pytest.raises(ValidationError) as excinfo:
        _require_attr({}, ATTR_FPS, expected)

    err = excinfo.value
    assert err.parameter_name == ATTR_FPS
    assert err.expected_format == expected
    assert "Missing required attribute" in str(err)


def test_require_attr_invalid_fps_boundaries_raise_validation_error():
    # Non-positive fps
    with pytest.raises(ValidationError):
        _require_attr({ATTR_FPS: 0.0}, ATTR_FPS, "float")
    with pytest.raises(ValidationError):
        _require_attr({ATTR_FPS: -1.0}, ATTR_FPS, "float")

    # Non-numeric fps
    with pytest.raises(ValidationError):
        _require_attr({ATTR_FPS: "not-a-number"}, ATTR_FPS, "float")


def test_require_attr_invalid_origin_raises_validation_error():
    with pytest.raises(ValidationError):
        _require_attr({ATTR_ORIGIN: "left"}, ATTR_ORIGIN, "str")
    with pytest.raises(ValidationError):
        _require_attr({ATTR_ORIGIN: 123}, ATTR_ORIGIN, "str")


def test_require_attr_invalid_extent_variants_raise_validation_error():
    # Wrong length
    with pytest.raises(ValidationError):
        _require_attr({ATTR_EXTENT: [0.0, 1.0, 2.0]}, ATTR_EXTENT, "sequence[4]")

    # Non-numeric entries
    with pytest.raises(ValidationError):
        _require_attr({ATTR_EXTENT: [0.0, "bad", 2.0, 3.0]}, ATTR_EXTENT, "sequence[4]")


def test_require_attr_invalid_pixel_to_grid_variants_raise_validation_error():
    # Non-positive scalar
    with pytest.raises(ValidationError):
        _require_attr({ATTR_PIXEL_TO_GRID: 0.0}, ATTR_PIXEL_TO_GRID, "float|pair")

    # Wrong sequence length
    with pytest.raises(ValidationError):
        _require_attr(
            {ATTR_PIXEL_TO_GRID: (1.0, 2.0, 3.0)},
            ATTR_PIXEL_TO_GRID,
            "float|pair",
        )

    # Non-numeric entries in pair
    with pytest.raises(ValidationError):
        _require_attr(
            {ATTR_PIXEL_TO_GRID: ("bad", 2.0)},
            ATTR_PIXEL_TO_GRID,
            "float|pair",
        )


def test_require_attr_returns_value_as_is_for_unknown_key():
    attrs = {"custom_key": "custom_value"}
    result = _require_attr(attrs, "custom_key", "str")
    assert result == "custom_value"
