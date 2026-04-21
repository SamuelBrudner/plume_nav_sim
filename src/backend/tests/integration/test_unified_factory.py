"""Smoke tests for the thin create_environment() shim."""

from plume_nav_sim.envs import create_environment


def test_create_plume_env_via_unified_factory():
    env = create_environment(
        env_type="plume_env", grid_size=(16, 16), source_location=(8, 8), max_steps=5
    )
    obs, info = env.reset(seed=0)
    # Accept both dict and Box observations
    import numpy as np

    assert isinstance(obs, (dict, np.ndarray))
    env.close()


def test_create_environment_rejects_removed_component_env_type():
    import pytest

    with pytest.raises(ValueError, match="Unsupported environment type: component"):
        create_environment(
            env_type="component",
            grid_size=(16, 16),
            source_location=(8, 8),
            max_steps=5,
        )
