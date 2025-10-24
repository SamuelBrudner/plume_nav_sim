"""
Temporal-gradient agent example using oriented actions (FORWARD / TURN_LEFT / TURN_RIGHT).

Agent keeps a one-back history of the odor concentration to estimate a
temporal derivative. It surges forward when concentration increases and
casts by turning when the signal decreases. The environment uses a
Gaussian plume with a step-penalty reward (sparse goal bonus plus small
per-step penalty).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import plume_nav_sim

# Oriented action ids from OrientedGridActions
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2


@dataclass
class TemporalGradientAgent:
    cast_right_first: bool = True
    eps: float = 1e-6  # small threshold to avoid noise-driven flips

    prev_moving: float | None = None  # last concentration after a FORWARD step
    cast_right: bool = True
    last_action: int | None = None

    def __post_init__(self) -> None:
        self.cast_right = self.cast_right_first

    def act(self, obs: np.ndarray) -> int:
        """Return oriented action based on temporal gradient of odor signal.

        Important: dC/dt is only meaningful across position changes. After a turn,
        the agent hasn't moved, so we force a FORWARD step to probe the new heading.
        """
        c = float(obs[0])

        # First step: move forward to obtain an initial moving sample
        if self.prev_moving is None:
            self.prev_moving = c
            self.last_action = FORWARD
            return FORWARD

        # If last action was a turn, we haven't moved; probe by going forward
        if self.last_action in (TURN_LEFT, TURN_RIGHT):
            self.last_action = FORWARD
            # Do not update prev_moving yet; we'll update after the move
            return FORWARD

        # Compute derivative relative to last moving concentration
        dc = c - self.prev_moving

        if dc >= self.eps:
            # Positive or flat trend: keep surging
            self.prev_moving = c
            self.last_action = FORWARD
            return FORWARD

        # Negative trend: cast by alternating turn direction
        self.cast_right = not self.cast_right
        action = TURN_RIGHT if self.cast_right else TURN_LEFT
        self.last_action = action
        # Do not update prev_moving (since no movement yet)
        return action


def run_episode(
    *,
    grid_size: tuple[int, int] = (64, 64),
    goal_radius: float = 1.0,
    plume_sigma: float = 20.0,
    max_steps: int = 500,
    seed: int | None = 123,
    render_mode: str | None = None,
) -> float:
    """Run a single episode and return total reward."""
    env = plume_nav_sim.make_env(
        grid_size=grid_size,
        goal_radius=goal_radius,
        plume_sigma=plume_sigma,
        max_steps=max_steps,
        action_type="oriented",  # enable FORWARD/LEFT/RIGHT control
        observation_type="concentration",  # 1-D odor reading
        reward_type="step_penalty",  # sparse goal + per-step penalty
        render_mode=render_mode,
    )

    try:
        obs, info = env.reset(seed=seed)
        agent = TemporalGradientAgent()

        total_reward = 0.0
        for _ in range(env.max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)

            if render_mode == "human":
                env.render()

            if terminated or truncated:
                break

        return total_reward
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal-gradient oriented agent demo"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid", type=str, default="64x64", help="WxH, e.g., 64x64")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--render",
        choices=["none", "human"],
        default="none",
        help="Enable human rendering",
    )
    args = parser.parse_args()

    try:
        w, h = (int(p) for p in args.grid.lower().split("x"))
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 64x64")

    total = run_episode(
        grid_size=(w, h),
        max_steps=args.max_steps,
        seed=args.seed,
        render_mode=(None if args.render == "none" else "human"),
    )
    print(f"Episode complete. Total reward = {total:.3f}")


if __name__ == "__main__":
    main()
