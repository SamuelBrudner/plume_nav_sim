from __future__ import annotations

"""
Advanced demo: showcase core benefits (determinism and subset validation)

- Spec-first composition with SimulationSpec + prepare
- Deterministic runs across resets with same seed
- Action-space subset safety checks (expected failure + allowed subset)
"""

import argparse
from typing import List, Optional

import numpy as np


def run_demo(
    *, seed: Optional[int] = 123, grid: str = "128x128", max_steps: int = 200
) -> None:
    from plume_nav_sim.compose import PolicySpec
    from plume_nav_sim.compose import PolicySpec as PolSpec
    from plume_nav_sim.compose import SimulationSpec
    from plume_nav_sim.compose import SimulationSpec as SimSpec
    from plume_nav_sim.compose import prepare
    from plume_nav_sim.compose import prepare as compose_prepare
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
        render=False,
        seed=seed,
        policy=PolicySpec(spec="plug_and_play_demo:DeltaBasedRunTumblePolicy"),
    )

    env, policy = prepare(sim)

    actions: List[int] = []

    def cap(ev: runner.StepEvent) -> None:
        actions.append(int(ev.action))

    res1 = runner.run_episode(env, policy, seed=sim.seed, render=False, on_step=cap)

    # Determinism check: re-compose and re-run with same seed
    env2, policy2 = prepare(sim)
    actions2: List[int] = []

    def cap2(ev: runner.StepEvent) -> None:
        actions2.append(int(ev.action))

    res2 = runner.run_episode(env2, policy2, seed=sim.seed, render=False, on_step=cap2)

    print(
        "Determinism:",
        {
            "same_steps": res1.steps == res2.steps,
            "same_total_reward": abs(res1.total_reward - res2.total_reward) < 1e-9,
            "same_action_sequence": actions == actions2,
        },
    )

    # Subset validation: oriented policy (n=3) on run_tumble (n=2) should be rejected
    bad_sim = SimSpec(
        grid_size=(w, h),
        max_steps=10,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
        seed=seed,
        policy=PolSpec(builtin="deterministic_td"),  # oriented policy
    )
    try:
        compose_prepare(bad_sim)
        print("Subset check: unexpected pass (should have raised)")
    except Exception as e:
        print("Subset check (expected failure):", type(e).__name__, str(e))

    # Subset allowed: run_tumble (n=2) policy on oriented (n=3) env
    ok_sim = SimSpec(
        grid_size=(w, h),
        max_steps=10,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
        seed=seed,
        policy=PolSpec(spec="plug_and_play_demo:DeltaBasedRunTumblePolicy"),
    )
    env_ok, pol_ok = compose_prepare(ok_sim)
    evs = list(runner.stream(env_ok, pol_ok, seed=ok_sim.seed, render=False))
    print("Subset check (subset allowed):", {"steps": len(evs)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced benefits demo")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid", type=str, default="128x128")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    run_demo(seed=args.seed, grid=args.grid, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
