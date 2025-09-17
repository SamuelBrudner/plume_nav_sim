"""Contract tests ensuring rendering package reuses canonical exception types."""

import importlib

import pytest


def test_render_validation_error_alias():
    """Render module should expose the shared ValidationError type."""

    render_module = pytest.importorskip(
        "plume_nav_sim.render",
        reason="Rendering package not importable in this minimal test environment.",
    )
    utils_exceptions = importlib.import_module("plume_nav_sim.utils.exceptions")

    assert render_module.ValidationError is utils_exceptions.ValidationError
    assert issubclass(utils_exceptions.ValidationError, ValueError)
