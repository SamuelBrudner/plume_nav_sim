"""Customize built-in components with string options."""

import plume_nav_sim as pns


def main() -> None:
    """Showcase built-in configuration knobs."""

    env = pns.make_env(
        grid_size=(64, 64),
        source_location=(16, 16),
        max_steps=500,
        goal_radius=2.5,
        action_type="oriented",
        observation_type="antennae",
        reward_type="step_penalty",
        plume_model="static_gaussian",
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
