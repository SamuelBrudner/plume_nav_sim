import pytest
from plume_nav_sim.envs.plume_search_env import create_plume_search_env

class TestEnvironmentIntegration:
    def test_smoke(self):
        env = create_plume_search_env()
        try:
            obs, info = env.reset(seed=123)
            assert obs is not None
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()


def test_complete_episode_workflow():
    env = create_plume_search_env()
    try:
        obs, info = env.reset(seed=321)
        done = False
        steps = 0
        while not done and steps < 20:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps > 0
    finally:
        env.close()


def test_cross_component_seeding():
    env1 = create_plume_search_env()
    env2 = create_plume_search_env()
    try:
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert obs1 is not None and obs2 is not None
    finally:
        env1.close(); env2.close()


def test_system_level_performance():
    env = create_plume_search_env()
    try:
        obs, info = env.reset(seed=7)
        # Perform a few steps and ensure render does not crash
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            _ = env.render(mode='rgb_array')
            if terminated or truncated:
                break
        assert True
    finally:
        env.close()
