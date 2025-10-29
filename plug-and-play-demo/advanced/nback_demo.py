from __future__ import annotations

"""
Advanced demo: core ConcentrationNBackWrapper (n-back odor history)

- Wrap the env to expose an n-length history of concentration values
- Keep policy stateful or switch to a stateless variant if desired
"""

import argparse
from typing import List, Optional

import numpy as np


def run_demo(
    *,
    seed: Optional[int] = 123,
    grid: str = "128x128",
    max_steps: int = 300,
    n: int = 5,
    render: bool = False,
) -> None:
    from plume_nav_sim.compose import PolicySpec, SimulationSpec, prepare

    # Core wrapper
    from plume_nav_sim.observations.history_wrappers import ConcentrationNBackWrapper
    from plume_nav_sim.runner import runner

    try:
        w, h = (int(p) for p in grid.lower().split("x"))
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 128x128")

    sim = SimulationSpec(
        grid_size=(w, h),
        max_steps=max_steps,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=render,
        seed=seed,
        policy=PolicySpec(spec="plug_and_play_demo:DeltaBasedRunTumblePolicy"),
    )

    env, policy = prepare(sim)
    env = ConcentrationNBackWrapper(env, n=n)

    frames: List[np.ndarray] = []

    def on_step(ev: runner.StepEvent) -> None:
        if ev.frame is not None:
            frames.append(ev.frame)

    result = runner.run_episode(
        env, policy, seed=sim.seed, render=bool(sim.render), on_step=on_step
    )
    print(
        "N-back demo summary:",
        {
            "n": n,
            "seed": result.seed,
            "steps": result.steps,
            "total_reward": round(result.total_reward, 3),
            "frames": len(frames),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N-back concentration wrapper demo")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid", type=str, default="128x128")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    run_demo(
        seed=args.seed,
        grid=args.grid,
        max_steps=args.max_steps,
        n=args.n,
        render=(not args.no_render),
    )


if __name__ == "__main__":
    main()
