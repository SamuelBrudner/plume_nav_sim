import pytest

from plume_nav_sim.envs.plume_search_env import create_plume_search_env
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy


@pytest.mark.integration
def test_td_deterministic_policy_reaches_goal_simple():
    env = create_plume_search_env(
        grid_size=(5, 5),
        source_location=(1, 1),
        start_location=(0, 1),
        goal_radius=0.5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=50,
    )
    try:
        obs, info = env.reset(seed=321)
        policy = TemporalDerivativeDeterministicPolicy()
        policy.reset()

        terminated = False
        steps = 0
        for _ in range(5):
            action = policy.select_action(obs, explore=False)
            obs, reward, terminated, truncated, step_info = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        assert terminated is True
        assert steps <= 2
    finally:
        env.close()


@pytest.mark.integration
def test_td_deterministic_policy_forward_probe_after_turn():
    env = create_plume_search_env(
        grid_size=(5, 5),
        source_location=(1, 2),
        start_location=(1, 0),
        goal_radius=0.5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=50,
    )
    try:
        obs, info = env.reset(seed=42)
        policy = TemporalDerivativeDeterministicPolicy()
        policy.reset()

        a1 = policy.select_action(obs, explore=False)
        obs, *_ = env.step(a1)
        a2 = policy.select_action(obs, explore=False)
        obs, *_ = env.step(a2)
        a3 = policy.select_action(obs, explore=False)

        if a2 in (1, 2):
            assert a3 == 0
    finally:
        env.close()
