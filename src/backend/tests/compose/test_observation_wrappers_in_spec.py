from __future__ import annotations

import pytest

from plume_nav_sim.compose import PolicySpec, SimulationSpec, WrapperSpec, prepare
from tests.test_env_utils_helper import step_once_safely


def test_prepare_applies_observation_wrappers():
    sim = SimulationSpec(
        grid_size=(16, 16),
        max_steps=5,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
        seed=123,
        policy=PolicySpec(builtin="random"),  # sampler uses env.action_space
        observation_wrappers=[
            WrapperSpec(
                spec="plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
                kwargs={"n": 2},
            )
        ],
    )
    env, pol = prepare(sim)

    # Observation space should reflect 2-back wrapper
    space = getattr(env, "observation_space", None)
    assert space is not None
    assert getattr(space, "shape", None) == (2,)

    # Reset and step should yield shape (2,)
    obs, info = env.reset(seed=sim.seed)
    assert getattr(obs, "shape", None) == (2,)


@pytest.mark.parametrize(
    "spec_str",
    [
        "plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
        "plume_nav_sim.observations.history_wrappers.ConcentrationNBackWrapper",
    ],
)
def test_wrapper_spec_import_forms(spec_str: str):
    sim = SimulationSpec(
        grid_size=(8, 8),
        max_steps=3,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
        seed=1,
        policy=PolicySpec(builtin="random"),
        observation_wrappers=[WrapperSpec(spec=spec_str, kwargs={"n": 2})],
    )
    env, pol = prepare(sim)
    # Reset and a single step should yield shape (2,)
    # Note: episodes may terminate on the first step depending on configuration,
    # so avoid stepping multiple times without checking terminated/truncated.
    obs, info = env.reset(seed=sim.seed)
    assert getattr(obs, "shape", None) == (2,)
    obs, r, term, trunc, info = step_once_safely(
        env, env.action_space.sample(), seed=sim.seed
    )
    assert getattr(obs, "shape", None) == (2,)
    obs, r, term, trunc, info = step_once_safely(
        env, env.action_space.sample(), seed=sim.seed
    )
    assert getattr(obs, "shape", None) == (2,)
