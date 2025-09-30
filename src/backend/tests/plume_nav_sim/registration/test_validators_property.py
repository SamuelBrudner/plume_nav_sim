import pytest

hypothesis = pytest.importorskip("hypothesis")  # Skip entire module if not available
from hypothesis import given, settings
from hypothesis import strategies as st

from plume_nav_sim.registration import register as reg


def _new_report(strict: bool = True):
    return {
        "timestamp": 0.0,
        "strict_validation": strict,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "performance_analysis": {},
        "compatibility_check": {},
    }


# Grid size within allowed bounds should not add errors
@given(
    width=st.integers(min_value=1, max_value=1024),
    height=st.integers(min_value=1, max_value=1024),
)
@settings(max_examples=50)
def test_grid_size_within_bounds(width: int, height: int):
    report = _new_report(strict=True)
    ok = reg._validate_grid_size_value((width, height), report, strict_validation=True)
    assert ok is True
    assert report["errors"] == []


# Grid size above bound should warn under strict mode
@given(
    width=st.integers(min_value=1025, max_value=2048),
    height=st.integers(min_value=1025, max_value=2048),
)
@settings(max_examples=25)
def test_grid_size_large_warns_strict(width: int, height: int):
    report = _new_report(strict=True)
    ok = reg._validate_grid_size_value((width, height), report, strict_validation=True)
    assert ok is True
    assert any("Large grid_size" in w for w in report["warnings"])


# Cross-validate source location property: in-bounds vs out-of-bounds
@given(
    width=st.integers(min_value=5, max_value=100),
    height=st.integers(min_value=5, max_value=100),
    x=st.integers(min_value=-10, max_value=110),
    y=st.integers(min_value=-10, max_value=110),
)
@settings(max_examples=100)
def test_cross_validate_source_bounds(width: int, height: int, x: int, y: int):
    report = _new_report()
    reg._cross_validate_params(
        {"grid_size": (width, height), "source_location": (x, y)}, report
    )
    in_bounds = (0 <= x < width) and (0 <= y < height)
    if in_bounds:
        assert all("within grid_size bounds" not in e for e in report["errors"])
    else:
        assert any("within grid_size bounds" in e for e in report["errors"])


# Validate max_episode_steps properties
@given(value=st.integers(min_value=1, max_value=100000))
@settings(max_examples=50)
def test_max_episode_steps_valid_range(value: int):
    report = _new_report()
    ok = reg._validate_max_episode_steps(value, report)
    assert ok is True
    assert report["errors"] == []


@given(value=st.integers(min_value=-1000, max_value=0))
@settings(max_examples=25)
def test_max_episode_steps_non_positive_invalid(value: int):
    report = _new_report()
    ok = reg._validate_max_episode_steps(value, report)
    assert ok is False
    assert any("positive" in e for e in report["errors"])


@given(value=st.integers(min_value=100001, max_value=200000))
@settings(max_examples=25)
def test_max_episode_steps_too_large_invalid(value: int):
    report = _new_report()
    ok = reg._validate_max_episode_steps(value, report)
    assert ok is False
    assert any("exceeds recommended maximum" in e for e in report["errors"])


# Strict checks: goal radius beyond edge should warn when all required params present
@given(
    width=st.integers(min_value=5, max_value=50),
    height=st.integers(min_value=5, max_value=50),
    sx=st.integers(min_value=0, max_value=49),
    sy=st.integers(min_value=0, max_value=49),
)
@settings(max_examples=50)
def test_strict_checks_goal_radius_edge_warning(
    width: int, height: int, sx: int, sy: int
):
    # Clamp source within provided grid
    sx = min(sx, width - 1)
    sy = min(sy, height - 1)

    # Compute a radius that guarantees exceeding the nearest edge
    mde = min(sx, sy, width - sx - 1, height - sy - 1)
    radius = float(mde + 1)

    report = _new_report()
    kwargs = {
        "grid_size": (width, height),
        "source_location": (sx, sy),
        "goal_radius": radius,
    }
    reg._strict_checks("AnyEnv-v0", kwargs, report)

    assert any(
        "Goal radius extends beyond grid boundaries" in w for w in report["warnings"]
    )
