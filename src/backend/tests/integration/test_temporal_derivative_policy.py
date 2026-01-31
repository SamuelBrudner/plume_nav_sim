import pytest

from plume_nav_sim.envs.plume_search_env import create_plume_search_env
from plume_nav_sim.policies import TemporalDerivativePolicy


@pytest.mark.integration
def test_temporal_derivative_policy_reaches_goal_simple_greedy():
    """With exploration disabled, policy should solve a trivial straight case."""
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
        obs, info = env.reset(seed=123)
        policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
        policy.reset(seed=123)

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
def test_temporal_derivative_policy_after_turn_moves_forward_probe():
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
        policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
        policy.reset(seed=42)

        a1 = policy.select_action(obs, explore=False)
        obs, *_ = env.step(a1)
        a2 = policy.select_action(obs, explore=False)
        obs, *_ = env.step(a2)
        a3 = policy.select_action(obs, explore=False)

        if a2 in (1, 2):
            assert a3 == 0  # Forward probe after turn
    finally:
        env.close()
