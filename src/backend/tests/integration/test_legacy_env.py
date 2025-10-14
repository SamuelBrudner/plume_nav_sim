"""
Integration checks for the legacy PlumeSearchEnv to ensure backward compatibility
while the component-based path is introduced.

This keeps the blast radius small for Phase B by exercising a few critical
behaviours without depending on Gymnasium's registry.
"""

import numpy as np

from plume_nav_sim.envs import create_plume_search_env


def test_legacy_env_reset_and_step_basic():
    env = create_plume_search_env(
        grid_size=(32, 32), source_location=(16, 16), max_steps=10
    )

    obs, info = env.reset(seed=42)

    # Observation may be dict (legacy contract) or a 2D Box field; accept both
    if isinstance(obs, dict):
        assert "agent_position" in obs
        assert "concentration_field" in obs
        assert "source_location" in obs
    else:
        # Expect a 2D field in [0,1]
        assert hasattr(obs, "shape") and len(obs.shape) == 2
        import numpy as _np

        assert not _np.any(_np.isnan(obs)) and not _np.any(_np.isinf(obs))

    # Info contains legacy convenience keys plus counters
    assert isinstance(info, dict)
    for key in ("agent_xy", "plume_peak_xy", "distance_to_source", "step_count"):
        assert key in info

    # Step through one action
    step = env.step(0)
    assert isinstance(step, tuple) and len(step) == 5
    next_obs, reward, terminated, truncated, step_info = step

    # Accept dict or 2D field
    assert isinstance(next_obs, (dict, np.ndarray))
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)

    env.close()
