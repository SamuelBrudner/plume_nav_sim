"""Customize built-in components with string options."""

from plume_nav_sim.envs import create_component_environment


def main() -> None:
    """Showcase built-in configuration knobs."""

    env = create_component_environment(
        grid_size=(64, 64),
        goal_location=(16, 16),
        max_steps=500,
        goal_radius=2.5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
    )

    obs, info = env.reset(seed=7)
    print("Initial observation:", obs)
    print("Episode metadata:", info)

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished", {"step": step + 1, "reward": reward})
            break

    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
