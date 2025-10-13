"""
Component-based (DI) environment usage example.

This example demonstrates how to register the DI environment id and instantiate
the factory-backed environment via Gymnasium. It also shows how to opt into DI
behind the default env id using an environment variable.
"""

from __future__ import annotations

import os
import warnings

import gymnasium as gym
from plume_nav_sim.registration import (
    COMPONENT_ENV_ID,
    ENV_ID,
    ensure_component_env_registered,
    register_env,
)


def main() -> None:
    # 1) Preferred: register and use the first‑class DI env id
    di_env_id = register_env(env_id=COMPONENT_ENV_ID, force_reregister=True)
    print(f"Registered DI env id: {di_env_id}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = gym.make(di_env_id)
    try:
        obs, info = env.reset(seed=123)
        # Observation is a Dict in the public wrapper; show keys and sensor value
        if isinstance(obs, dict):
            sensor = float(obs.get("sensor_reading", [0.0])[0])
            print(
                f"Reset OK, obs keys={list(obs.keys())}, sensor={sensor:.3f}, info={info}"
            )
        else:
            print(f"Reset OK, obs shape={getattr(obs, 'shape', None)} info={info}")
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"Step OK: reward={reward} terminated={terminated} truncated={truncated}")
    finally:
        env.close()

    # 2) Alternate: opt‑in globally using an env var (keeps the default env id)
    #    This routes register_env() for ENV_ID to use the DI entry point.
    os.environ["PLUMENAV_DEFAULT"] = "components"
    legacy_id = register_env(force_reregister=True)
    assert legacy_id == ENV_ID
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env2 = gym.make(legacy_id)
    try:
        env2.reset()
        print("Default env id routed to DI via PLUMENAV_DEFAULT=components")
    finally:
        env2.close()

    # 3) Helper: ensure DI env id is registered
    ensured = ensure_component_env_registered(validate_creation=True)
    print(f"ensure_component_env_registered() -> {ensured}")


if __name__ == "__main__":
    main()
