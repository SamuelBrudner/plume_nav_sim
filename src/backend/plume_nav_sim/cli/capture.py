from __future__ import annotations

import argparse
from typing import Optional

import plume_nav_sim as pns
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.data_capture import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run plume-nav-sim episodes and capture analysis-ready data"
    )
    p.add_argument(
        "--output", type=str, default="results", help="Output root directory"
    )
    p.add_argument("--experiment", type=str, default="default", help="Experiment name")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    p.add_argument("--seed", type=int, default=123, help="Base seed for first episode")
    p.add_argument(
        "--grid", type=str, default="64x64", help="Grid size WxH, e.g., 64x64"
    )
    p.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode (override)"
    )
    p.add_argument(
        "--parquet",
        action="store_true",
        help="Export Parquet at end of run (if pyarrow available)",
    )
    p.add_argument(
        "--rotate-size",
        type=int,
        default=None,
        help="Rotate JSONL.gz after N bytes (compressed)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run episodes and capture analysis-ready artifacts via data_capture.

    Args:
        argv: Optional list of CLI arguments for testing

    Returns:
        Process exit code (0 on success)
    """
    args = _parse_args(argv)
    try:
        w, h = (int(p) for p in args.grid.lower().split("x"))
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 64x64")

    # Create env (use oriented+concentration defaults; reward step_penalty)
    env = pns.make_env(
        grid_size=(w, h),
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=args.max_steps,
    )
    try:
        rec = RunRecorder(
            args.output, experiment=args.experiment, rotate_size_bytes=args.rotate_size
        )
        cfg = EnvironmentConfig(grid_size=(w, h), source_location=(w // 2, h // 2))
        env = DataCaptureWrapper(env, rec, cfg)

        base_seed = int(args.seed)
        episodes = int(args.episodes)
        for i in range(episodes):
            seed = base_seed + i
            obs, info = env.reset(seed=seed)
            step_count = 0
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                if terminated or truncated:
                    break
            # next episode
        rec.finalize(export_parquet=bool(args.parquet))
    finally:
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
