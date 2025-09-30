import numpy as np
import pytest

import plume_nav_sim.render as render
from plume_nav_sim.render import GridSize, RenderMode


class _RgbStub:
    def __init__(self, *args, **kwargs):
        pass

    def render(self, context, mode):
        assert mode == RenderMode.RGB_ARRAY
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def cleanup_resources(self):
        pass


class _MplStub:
    def __init__(self, *args, **kwargs):
        pass

    def render(self, context, mode):
        # Return None for HUMAN mode, array for RGB if asked
        if mode == RenderMode.RGB_ARRAY:
            return np.zeros((2, 2, 3), dtype=np.uint8)
        return None

    def cleanup_resources(self):
        pass


@pytest.fixture(autouse=True)
def _clear_registry():
    # Ensure clean registry per test
    render._RENDERER_REGISTRY.clear()
    yield
    render._RENDERER_REGISTRY.clear()


@pytest.fixture
def _fake_color_scheme(monkeypatch):
    class _Scheme:
        name = "test"

    monkeypatch.setattr(render, "_get_color_scheme", lambda name: _Scheme())
    return _Scheme()


def _fake_tests_success(monkeypatch):
    monkeypatch.setattr(
        render,
        "_test_renderer_rgb",
        lambda grid, r: {"success": True, "duration_ms": 0.1},
    )
    monkeypatch.setattr(
        render,
        "_test_renderer_matplotlib",
        lambda grid, r: {"success": True, "duration_ms": 0.2},
    )


def test_create_dual_mode_renderer_success_basic(monkeypatch, _fake_color_scheme):
    # Arrange
    monkeypatch.setattr(render, "create_rgb_renderer", lambda *a, **k: _RgbStub())
    monkeypatch.setattr(
        render, "create_matplotlib_renderer", lambda *a, **k: _MplStub()
    )
    _fake_tests_success(monkeypatch)

    # Act
    result = render.create_dual_mode_renderer(GridSize(4, 4))

    # Assert
    assert result["renderer_id"].startswith("dual_mode_")
    rid = result["renderer_id"]
    assert rid in render._RENDERER_REGISTRY

    caps = result["capabilities"]
    assert caps["rgb_available"] is True
    assert caps["matplotlib_available"] is True
    # primary auto-selected to RGB since rgb renderer exists
    assert result["configuration"]["primary_mode"] == RenderMode.RGB_ARRAY


def test_create_dual_mode_renderer_fallback_to_mpl_when_rgb_missing(
    monkeypatch, _fake_color_scheme
):
    # Arrange: RGB creation fails, Matplotlib succeeds
    def _fail_rgb(*a, **k):
        raise RuntimeError("rgb unavailable")

    monkeypatch.setattr(render, "create_rgb_renderer", _fail_rgb)
    monkeypatch.setattr(
        render, "create_matplotlib_renderer", lambda *a, **k: _MplStub()
    )
    _fake_tests_success(monkeypatch)

    # Act
    result = render.create_dual_mode_renderer(GridSize(4, 4))

    # Assert
    assert result["rgb_renderer"] is None
    assert "rgb_error" in result["errors"] and result["errors"]["rgb_error"]
    # With no RGB renderer, auto-select HUMAN
    assert result["configuration"]["primary_mode"] == RenderMode.HUMAN
    # Fallback enabled only if the counterpart renderer exists
    assert result["capabilities"]["fallback_enabled"] is True


def test_create_dual_mode_renderer_fallback_to_rgb_when_mpl_missing(
    monkeypatch, _fake_color_scheme
):
    # Arrange: Matplotlib creation fails, RGB succeeds
    def _fail_mpl(*a, **k):
        raise RuntimeError("mpl unavailable")

    monkeypatch.setattr(render, "create_rgb_renderer", lambda *a, **k: _RgbStub())
    monkeypatch.setattr(render, "create_matplotlib_renderer", _fail_mpl)
    _fake_tests_success(monkeypatch)

    # Act
    result = render.create_dual_mode_renderer(GridSize(4, 4))

    # Assert
    assert result["matplotlib_renderer"] is None
    assert (
        "matplotlib_error" in result["errors"] and result["errors"]["matplotlib_error"]
    )
    # Auto-select RGB since present
    assert result["configuration"]["primary_mode"] == RenderMode.RGB_ARRAY
    assert result["capabilities"]["fallback_enabled"] is True
