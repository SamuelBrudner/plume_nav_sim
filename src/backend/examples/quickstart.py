"""Quickstart example for `plume_nav_sim.make_env()`."""

import plume_nav_sim as pns


def main() -> None:
    """Run a short random episode with default settings."""

    env = pns.make_env()
    obs, info = env.reset(seed=42)
    print(f"Starting position: {info['agent_xy']}")

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished at step {step + 1} with reward {reward}")
            break

    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
