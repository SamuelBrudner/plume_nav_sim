"""Reproducibility tooling for plume_nav_sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plume_nav_sim as pns
from plume_nav_sim.registration import is_registered
from plume_nav_sim.utils.seeding import SeedManager, validate_seed

OUTPUT_PATH = Path("reproducibility_report.json")


def run_episode(env, *, seed: int, max_steps: int = 100) -> dict[str, Any]:
    """Roll out a short episode and capture observations."""

    obs, info = env.reset(seed=seed)
    trajectory = [info["agent_xy"]]
    total_reward = 0.0

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory.append(info["agent_position"])
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "trajectory": trajectory,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> None:
    seed_manager = SeedManager()
    base_seed = 2025

    is_valid, _, error = validate_seed(base_seed)
    if not is_valid:
        raise ValueError(f"Invalid base seed: {error}")

    env = pns.make_env()

    assert is_registered()  # sanity check that import side effects occurred

    summary = {
        "environment": "PlumeNav-v0",
        "base_seed": base_seed,
        "episodes": [],
    }

    for idx in range(3):
        child_seed = seed_manager.derive_seed(base_seed, f"episode-{idx}")
        episode = run_episode(env, seed=child_seed)
        summary["episodes"].append(episode)

    env.close()

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
