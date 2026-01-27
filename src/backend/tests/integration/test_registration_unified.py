"""Registration smoke tests for the default entry point."""

import gymnasium as gym

from plume_nav_sim.registration import ENV_ID, register_env, unregister_env


def _get_entry_point_string(env_id: str) -> str:
    """Return the entry point as a string for a given env id across Gymnasium variants."""
    spec = None
    reg = getattr(gym.envs, "registry", None)
    if hasattr(reg, "env_specs"):
        spec = reg.env_specs.get(env_id)
    if spec is None:
        raise AssertionError(f"Environment id {env_id} not found in registry")
    # entry_point may be a string or callable; normalize to string for assertions
    ep = getattr(spec, "entry_point", None)
    return str(ep)


def test_register_legacy_entry_point_default():
    # Ensure clean state
    unregister_env(ENV_ID, suppress_warnings=True)
    # Default registration uses PlumeEnv factory entry point
    register_env(force_reregister=True)
    entry_point = _get_entry_point_string(ENV_ID)
    assert "plume_nav_sim.envs.plume_env:create_plume_env" in entry_point

