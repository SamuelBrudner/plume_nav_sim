"""
Smoke tests for the unified create_environment() helper to ensure it can
instantiate both the legacy and component-based environments.
"""

from plume_nav_sim.envs import create_environment


def test_create_legacy_via_unified_factory():
    env = create_environment(
        env_type="plume_search", grid_size=(16, 16), source_location=(8, 8), max_steps=5
    )
    obs, info = env.reset(seed=0)
    # Accept both dict and Box observations
    import numpy as np

    assert isinstance(obs, (dict, np.ndarray))
    env.close()


def test_create_component_via_unified_factory():
    env = create_environment(
        env_type="component", grid_size=(16, 16), source_location=(8, 8), max_steps=5
    )
    obs, info = env.reset()
    # Accept both dict and Box observations
    import numpy as np

    assert isinstance(obs, (dict, np.ndarray))
    env.close()
