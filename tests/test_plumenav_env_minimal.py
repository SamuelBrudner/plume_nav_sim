import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np


def test_plumenav_env_registration_and_step():
    env = gym.make("PlumeNav-v1")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    step_out = env.step(action)
    assert len(step_out) == 5
    env.close()


def test_plumenav_env_checker():
    env = gym.make("PlumeNav-v1")
    check_env(env, warn=True)
    env.close()


def test_vectorized_reset_shape():
    vec_env = gym.vector.make("PlumeNav-v1", num_envs=2)
    obs, info = vec_env.reset(seed=[0, 1])
    assert obs.shape == (2, 2)
    vec_env.close()
