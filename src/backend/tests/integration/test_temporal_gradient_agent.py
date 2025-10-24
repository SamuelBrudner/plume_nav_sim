import numpy as np
import pytest

from plume_nav_sim.envs.plume_search_env import create_plume_search_env


class TemporalGradientAgent:
    """Minimal oriented controller with 1-back odor derivative.

    Handles the zero-derivative-after-turn issue by forcing a forward probe
    after turns (since concentration doesn't change on pure rotations).
    """

    FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2

    def __init__(self, eps: float = 1e-6):
        self.prev_moving: float | None = None
        self.turn_right_next: bool = True
        self.eps = eps
        self.last_action: int | None = None

    def act(self, obs: np.ndarray) -> int:
        c = float(obs[0])

        if self.prev_moving is None:
            self.prev_moving = c
            self.last_action = self.FORWARD
            return self.FORWARD

        if self.last_action in (self.TURN_LEFT, self.TURN_RIGHT):
            self.last_action = self.FORWARD
            return self.FORWARD

        dc = c - self.prev_moving

        if dc >= self.eps:
            self.prev_moving = c
            self.last_action = self.FORWARD
            return self.FORWARD

        self.turn_right_next = not self.turn_right_next
        action = self.TURN_RIGHT if self.turn_right_next else self.TURN_LEFT
        self.last_action = action
        return action


@pytest.mark.integration
def test_temporal_gradient_agent_reaches_goal_quick_simple_layout():
    """
    With oriented control and a simple layout, a 1-back temporal-gradient
    agent should reach the goal quickly by walking straight.
    """
    env = create_plume_search_env(
        grid_size=(5, 5),
        source_location=(1, 1),
        start_location=(0, 1),  # Facing east by default; one forward reaches goal
        goal_radius=0.5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=50,
    )

    try:
        obs, info = env.reset(seed=123)
        agent = TemporalGradientAgent()

        total = 0.0
        steps = 0
        terminated = False
        truncated = False
        for _ in range(5):
            action = agent.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        assert terminated is True, "Agent should reach the goal in the simple layout"
        assert steps <= 2, f"Expected to finish in <=2 steps, took {steps}"
        assert total > 0.0, "Total reward should be positive upon reaching the goal"
    finally:
        env.close()


@pytest.mark.integration
def test_temporal_gradient_agent_does_not_spin_after_turn():
    """
    If the agent decides to turn, the next action must be FORWARD to probe the
    new heading (since concentration is unchanged by rotation). This guards
    against getting stuck spinning in place.
    """
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
        agent = TemporalGradientAgent()

        a1 = agent.act(obs)
        obs, *_ = env.step(a1)
        a2 = agent.act(obs)
        obs, *_ = env.step(a2)
        a3 = agent.act(obs)

        # If a2 was a turn, a3 must be FORWARD
        if a2 in (agent.TURN_LEFT, agent.TURN_RIGHT):
            assert (
                a3 == agent.FORWARD
            ), "Agent must move forward immediately after a turn"
    finally:
        env.close()
