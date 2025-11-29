"""
Registration smoke tests to verify the optional component-based entry point flag
does not break the default legacy registration and can be toggled on demand.

These tests do not instantiate environments (no gym.make), they only inspect the
Gymnasium registry state to avoid Phase C runtime dependency issues.
"""

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
    # Default registration uses legacy PlumeSearchEnv
    register_env(force_reregister=True)
    entry_point = _get_entry_point_string(ENV_ID)
    assert "plume_nav_sim.envs.plume_search_env:PlumeSearchEnv" in entry_point


def test_register_component_env_id():
    # Ensure clean state
    unregister_env(ENV_ID, suppress_warnings=True)
    # Register DI env id and verify factory entry point
    di_env_id = "PlumeNav-Components-v0"
    unregister_env(di_env_id, suppress_warnings=True)
    register_env(env_id=di_env_id, force_reregister=True)
    entry_point = _get_entry_point_string(di_env_id)
    assert "plume_nav_sim.envs.factory:create_component_environment" in entry_point
