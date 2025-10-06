"""
Integration test to confirm end-behavior parity between the legacy environment
and the component-based (DI) environment under comparable configurations.

Parity criteria validated:
- Identical agent positions after a fixed action sequence
- Matching termination/truncation flags at each step
- Binary sparse rewards match at each step
- Observation value at agent position matches (legacy field[y,x] vs DI concentration)

Notes:
- Legacy env has random start on reset; we capture the start then inject it as
  `start_location` for the DI env to align trajectories.
- DI configuration mirrors legacy defaults: discrete actions, concentration
  observation, sparse rewards, plume sigma set to default (12.0).
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import gymnasium as gym
from plume_nav_sim.core.constants import DEFAULT_PLUME_SIGMA
from plume_nav_sim.registration import (
    COMPONENT_ENV_ID,
    ENV_ID,
    register_env,
    unregister_env,
)


def _make_legacy_env(config) -> gym.Env:
    unregister_env(ENV_ID, suppress_warnings=True)
    register_env(env_id=ENV_ID, kwargs=config, force_reregister=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(ENV_ID)


def _make_di_env(config) -> gym.Env:
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)
    register_env(env_id=COMPONENT_ENV_ID, kwargs=config, force_reregister=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make(COMPONENT_ENV_ID)


def test_di_vs_legacy_parity_simple_path():
    grid_size = (32, 24)
    source_location = (16, 12)
    goal_radius = 1.0
    max_steps = 15
    plume_sigma = DEFAULT_PLUME_SIGMA  # 12.0, aligns DI and legacy fields

    legacy_cfg = {
        "grid_size": grid_size,
        "source_location": source_location,
        "goal_radius": goal_radius,
        "max_steps": max_steps,
        "plume_params": {"sigma": plume_sigma},
    }

    legacy_env = _make_legacy_env(legacy_cfg)
    try:
        # Reset legacy and capture its start position
        legacy_obs, legacy_info = legacy_env.reset(seed=123)
        start_xy = tuple(legacy_info.get("agent_xy", (0, 0)))

        # Build DI config to mirror legacy and align start
        di_cfg = {
            "grid_size": grid_size,
            "source_location": source_location,  # mapped to goal_location
            "start_location": start_xy,
            "goal_radius": goal_radius,
            "max_steps": max_steps,
            "action_type": "discrete",
            "observation_type": "concentration",
            "reward_type": "sparse",
            "plume_sigma": plume_sigma,
        }

        di_env = _make_di_env(di_cfg)
        try:
            di_obs, di_info = di_env.reset()

            # Fixed action sequence exercising movement and boundaries
            actions: List[int] = [1, 1, 0, 0, 3, 2, 2, 1, 0, 3]

            for a in actions:
                # Step legacy
                legacy_obs, legacy_r, legacy_term, legacy_trunc, legacy_info = (
                    legacy_env.step(a)
                )
                lpos = tuple(legacy_info.get("agent_xy"))
                # Step DI
                di_obs, di_r, di_term, di_trunc, di_info = di_env.step(a)
                dpos = tuple(di_info.get("agent_position"))

                # Positions must match
                assert dpos == lpos

                # Termination/truncation flags must match
                assert di_term == legacy_term
                assert di_trunc == legacy_trunc

                # Sparse rewards must match
                assert float(di_r) == float(legacy_r)

                # Observation parity at agent position: DI provides a scalar concentration,
                # legacy provides the full field; compare value at current position.
                ly, lx = lpos[1], lpos[0]
                legacy_val = float(legacy_obs[ly, lx])
                di_val = float(di_obs[0])
                assert abs(di_val - legacy_val) < 1e-6

                if di_term or di_trunc:
                    break
        finally:
            di_env.close()
    finally:
        legacy_env.close()

    # Cleanup registry
    unregister_env(ENV_ID, suppress_warnings=True)
    unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)
