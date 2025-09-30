"""Regression test for plume model registry idempotent registration."""

import pytest

from plume_nav_sim.core.constants import STATIC_GAUSSIAN_MODEL_TYPE
from plume_nav_sim.core.types import Coordinates, GridSize
from plume_nav_sim.plume.plume_model import PlumeModelRegistry
from plume_nav_sim.plume.static_gaussian import StaticGaussianPlume


def test_registry_allows_idempotent_builtin_registration():
    """Registry should allow re-registering built-in models without raising errors."""
    registry = PlumeModelRegistry()

    # First registration should succeed
    result1 = registry.register_model(
        STATIC_GAUSSIAN_MODEL_TYPE,
        StaticGaussianPlume,
        metadata={"description": "Built-in static Gaussian plume model"},
    )
    assert result1 is True

    # Second registration of the same model should either:
    # 1. Return True silently (idempotent)
    # 2. Or raise ValueError if override_existing=False (current behavior)
    # For TDD, we want idempotent behavior for built-in models
    result2 = registry.register_model(
        STATIC_GAUSSIAN_MODEL_TYPE,
        StaticGaussianPlume,
        metadata={"description": "Built-in static Gaussian plume model"},
    )
    assert result2 is True  # Should not raise, should return True

    # Verify model is still registered and functional
    assert STATIC_GAUSSIAN_MODEL_TYPE in registry.get_registered_models()

    # Can still create instances
    model = registry.create_model(
        STATIC_GAUSSIAN_MODEL_TYPE,
        grid_size=GridSize(64, 64),
        source_location=Coordinates(32, 32),
        sigma=12.0,
    )
    assert model is not None
