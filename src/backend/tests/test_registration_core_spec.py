import gymnasium
import pytest

from plume_nav_sim.registration.register import (
    register_env, unregister_env, is_registered, ENV_ID
)
from plume_nav_sim.registration import ensure_registered
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv


def test_register_make_isinstance_and_cleanup():
    # Ensure not registered
    if is_registered(ENV_ID):
        unregister_env(ENV_ID)
    # Register
    env_id = register_env()
    assert is_registered(env_id)

    # make() should yield an instance recognized as PlumeSearchEnv (even if wrapped)
    env = gymnasium.make(env_id)
    try:
        assert isinstance(env, PlumeSearchEnv)
    finally:
        env.close()

    # Cleanup
    assert unregister_env(env_id)
    assert not is_registered(env_id)


def test_ensure_registered_idempotent():
    try:
        ensure_registered()
        assert is_registered(ENV_ID)
        ensure_registered()  # second call should be safe
        assert is_registered(ENV_ID)
    finally:
        unregister_env(ENV_ID)
