"""Demonstrate injecting custom components via dependency injection."""

from __future__ import annotations

import numpy as np

import plume_nav_sim as pns
from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import GridSize
from plume_nav_sim.envs.component_env import ComponentBasedEnvironment
from plume_nav_sim.interfaces import ObservationModel, RewardFunction
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.concentration_field import ConcentrationField


class DenseReward(RewardFunction):
    """Reward agent for moving toward the goal with a small step penalty."""

    def __init__(self, goal_radius: float) -> None:
        self._goal_radius = goal_radius

    def compute_reward(self, prev_state, action: int, next_state, plume_field) -> float:
        penalty = 0.01
        reward = -penalty
        if next_state.goal_reached:
            reward += 1.0
        return float(reward)

    def get_metadata(self) -> dict[str, float]:
        return {"type": "dense_reward", "goal_radius": self._goal_radius}


class NoisyConcentrationSensor(ObservationModel):
    """Wrap concentration sensor with Gaussian noise to showcase extensibility."""

    def __init__(self, sigma: float = 0.05) -> None:
        self._sigma = sigma
        self._base = ConcentrationSensor()

    @property
    def observation_space(self):  # type: ignore[override]
        return self._base.observation_space

    def get_observation(self, env_state):  # type: ignore[override]
        reading = self._base.get_observation(env_state)
        noise = np.random.normal(0.0, self._sigma, size=reading.shape)
        return np.clip(reading + noise, 0.0, 1.0)

    def get_metadata(self):  # type: ignore[override]
        meta = self._base.get_metadata()
        meta["noise_sigma"] = self._sigma
        return meta


def build_environment() -> ComponentBasedEnvironment:
    """Construct an environment using custom components."""

    wrapper = pns.make_env()
    try:
        base_env = getattr(wrapper, "_env", None)
        if not isinstance(base_env, ComponentBasedEnvironment):
            raise RuntimeError(
                "Unexpected environment type; expected ComponentBasedEnvironment"
            )

        grid_size: GridSize = base_env.grid_size
        goal_location = base_env.goal_location
        goal_radius = base_env.goal_radius
        max_steps = base_env.max_steps
    finally:
        wrapper.close()

    action_processor = DiscreteGridActions(step_size=1)
    observation_model = NoisyConcentrationSensor(sigma=0.1)

    field = ConcentrationField(grid_size)
    field.generate_field(goal_location)

    reward = DenseReward(goal_radius=goal_radius)

    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward,
        concentration_field=field,
        grid_size=grid_size,
        goal_location=goal_location,
        goal_radius=goal_radius,
        max_steps=max_steps,
    )


def main() -> None:
    env = build_environment()
    obs, info = env.reset(seed=0)
    print("metadata:", info)

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print({"step": step + 1, "reward": total_reward})
            break

    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
