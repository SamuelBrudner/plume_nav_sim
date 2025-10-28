from __future__ import annotations

import pytest

from plume_nav_sim.compose.builders import prepare
from plume_nav_sim.compose.specs import PolicySpec, SimulationSpec


def test_prepare_rejects_policy_superset_of_env_space():
    # Oriented policy (n=3) with run_tumble env (n=2) must raise at composition time
    sim = SimulationSpec(
        grid_size=(8, 8),
        max_steps=10,
        render=False,
        action_type="run_tumble",
        policy=PolicySpec(builtin="deterministic_td"),
        seed=1,
    )
    with pytest.raises(ValueError):
        prepare(sim)


def test_prepare_allows_equal_or_subset_spaces():
    # Equal: oriented policy with oriented env
    sim_equal = SimulationSpec(
        grid_size=(8, 8),
        max_steps=10,
        render=False,
        action_type="oriented",
        policy=PolicySpec(builtin="deterministic_td"),
        seed=1,
    )
    env, pol = prepare(sim_equal)
    env.close()

    # Subset: run_tumble policy (n=2) with oriented env (n=3) is allowed at compose-time
    # Note: runner will impose stricter identity and can error later during execution.
    sim_subset = SimulationSpec(
        grid_size=(8, 8),
        max_steps=10,
        render=False,
        action_type="oriented",
        policy=PolicySpec(builtin="run_tumble_td"),
        seed=1,
    )
    env2, pol2 = prepare(sim_subset)
    env2.close()
