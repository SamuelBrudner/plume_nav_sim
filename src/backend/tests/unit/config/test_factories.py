from __future__ import annotations

from plume_nav_sim.config import ObservationConfig, RewardConfig
from plume_nav_sim.config.factories import (
    create_observation_model,
    create_reward_function,
)
from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.observations import WindVectorSensor
from plume_nav_sim.rewards import SparseGoalReward, StepPenaltyReward


def test_create_reward_function_step_penalty_uses_config_fields():
    config = RewardConfig(
        type="step_penalty",
        goal_radius=2.5,
        goal_reward=3.0,
        step_penalty=0.2,
    )

    reward_fn = create_reward_function(config, Coordinates(1, 2))

    assert isinstance(reward_fn, StepPenaltyReward)
    assert reward_fn.goal_radius == 2.5
    assert reward_fn.goal_reward == 3.0
    assert reward_fn.step_penalty == 0.2


def test_create_reward_function_sparse_uses_goal_radius():
    config = RewardConfig(type="sparse", goal_radius=1.25)

    reward_fn = create_reward_function(config, Coordinates(0, 0))

    assert isinstance(reward_fn, SparseGoalReward)
    assert reward_fn.goal_radius == 1.25


def test_create_observation_model_for_wind_vector_respects_noise():
    config = ObservationConfig(type="wind_vector", noise_std=0.15)

    model = create_observation_model(config)

    assert isinstance(model, WindVectorSensor)
    assert model.noise_std == 0.15
