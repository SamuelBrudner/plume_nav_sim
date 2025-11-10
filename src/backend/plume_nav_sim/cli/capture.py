from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import plume_nav_sim as pns
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.data_capture import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper


def _parse_args_and_overrides(
    argv: Optional[list[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI args, returning known args and passthrough overrides.

    Overrides are intended for Hydra, e.g. ["episodes=5", "env.max_steps=100"].
    """
    p = argparse.ArgumentParser(
        description="Run plume-nav-sim episodes and capture analysis-ready data",
        add_help=True,
    )
    # Legacy/basic args
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

    # Hydra integration (optional)
    p.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config to load (e.g., 'data_capture/config')",
    )
    p.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to Hydra config root (defaults to repository conf/)",
    )
    args, overrides = p.parse_known_args(argv)
    return args, overrides


def _load_hydra_config(
    *,
    config_name: str,
    config_path: Optional[str],
    overrides: List[str],
) -> Tuple[dict, str]:
    """Load and resolve Hydra config, returning (as_dict, config_hash)."""
    # Lazy import to avoid imposing Hydra on basic usage
    from hashlib import sha256
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Default config path to repository conf/
    if not config_path:
        conf_dir = pns.get_conf_dir()
        if conf_dir is None:
            # Fallback: relative to this file
            conf_dir = Path(__file__).resolve().parents[2] / "conf"
        config_path = str(conf_dir)

    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    # Stable hash of resolved config JSON
    import json

    cfg_bytes = json.dumps(cfg_resolved, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    cfg_hash = sha256(cfg_bytes).hexdigest()
    return cfg_resolved, cfg_hash


def main(argv: Optional[list[str]] = None) -> int:
    """Run episodes and capture analysis-ready artifacts via data_capture.

    Args:
        argv: Optional list of CLI arguments for testing

    Returns:
        Process exit code (0 on success)
    """
    args, overrides = _parse_args_and_overrides(argv)

    def _is_provided(flag: str) -> bool:
        """Return True if the given CLI flag was explicitly provided in argv."""
        if not argv:
            return False
        return any(tok == flag for tok in argv)

    env = None
    cfg_hash: Optional[str] = None
    # Hydra-driven configuration if requested
    if args.config_name:
        cfg, cfg_hash = _load_hydra_config(
            config_name=str(args.config_name),
            config_path=args.config_path,
            overrides=overrides,
        )
        env_cfg = cfg.get("env", {})
        grid = env_cfg.get("grid_size", [64, 64])
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise SystemExit("env.grid_size must be a 2-element list or tuple")
        w, h = int(grid[0]), int(grid[1])
        env = pns.make_env(
            grid_size=(w, h),
            action_type=env_cfg.get("action_type", "oriented"),
            observation_type=env_cfg.get("observation_type", "concentration"),
            reward_type=env_cfg.get("reward_type", "step_penalty"),
            max_steps=env_cfg.get("max_steps", None) or None,
        )
        # Sync top-level controls from cfg unless explicitly overridden via CLI
        args.output = (
            args.output if _is_provided("--output") else cfg.get("output", "results")
        )
        args.experiment = (
            args.experiment
            if _is_provided("--experiment")
            else cfg.get("experiment", "default")
        )
        args.episodes = (
            args.episodes if _is_provided("--episodes") else int(cfg.get("episodes", 1))
        )
        args.seed = args.seed if _is_provided("--seed") else int(cfg.get("seed", 123))
        args.rotate_size = (
            args.rotate_size
            if _is_provided("--rotate-size")
            else cfg.get("rotate_size", None)
        )
        args.parquet = bool(
            args.parquet if _is_provided("--parquet") else cfg.get("parquet", False)
        )
    else:
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
        env = DataCaptureWrapper(
            env,
            rec,
            cfg,
            meta_overrides={"config_hash": cfg_hash} if cfg_hash else None,
        )

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
