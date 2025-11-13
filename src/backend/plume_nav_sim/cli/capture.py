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
    from hydra.errors import MissingConfigException
    from omegaconf import OmegaConf

    # Default config path to repository conf/
    if not config_path:
        conf_dir = pns.get_conf_dir()
        if conf_dir is None:
            # Fallback: relative to this file
            conf_dir = Path(__file__).resolve().parents[2] / "conf"
        config_path = str(conf_dir)

    def _manual_compose_for_data_capture(conf_root: Path) -> dict:
        """Manual composition for data_capture/config to support mixed group locations.

        This handles the case where data_capture/config.yaml references groups
        that live at the repository conf/ root (e.g., movie/, env/plume/), while
        the experiment/ group lives under conf/data_capture/.
        """
        from omegaconf import OmegaConf

        # Locate the requested YAML file
        yaml_path = conf_root.joinpath(*config_name.split("/"))
        if not yaml_path.suffix:
            yaml_path = yaml_path.with_suffix(".yaml")
        if not yaml_path.exists():
            raise

        base_cfg = OmegaConf.load(yaml_path)
        defaults = list(base_cfg.get("defaults", []))

        composed = OmegaConf.create()

        for item in defaults:
            if item == "_self_" or (isinstance(item, dict) and "_self_" in item):
                # Merge base at the end to mirror Hydra's _self_ behavior
                composed = OmegaConf.merge(composed, base_cfg)
                continue

            if isinstance(item, dict):
                # Only single-key mappings are expected in defaults
                [(group_key, option)] = list(item.items())
                # Resolve group file path
                group_parts = str(group_key).split("/") if group_key else []
                option_name = str(option)

                # experiment group lives under data_capture/, others at root
                if group_parts and group_parts[0] == "experiment":
                    group_base = conf_root / "data_capture"
                else:
                    group_base = conf_root

                group_file = group_base.joinpath(*group_parts) / f"{option_name}.yaml"
                if not group_file.exists():
                    # If not found under root, attempt under data_capture/ as a fallback
                    alt = (conf_root / "data_capture").joinpath(
                        *group_parts
                    ) / f"{option_name}.yaml"
                    if alt.exists():
                        group_file = alt
                    else:
                        raise MissingConfigException(
                            missing_cfg_file=str(group_file),
                            message=f"Could not find '{group_key}/{option_name}' for data_capture composition",
                            options=None,
                        )
                part_cfg = OmegaConf.load(group_file)
                composed = OmegaConf.merge(composed, part_cfg)
            else:
                # Unsupported defaults entry format
                continue

        # Apply CLI overrides as a dotlist
        if overrides:
            try:
                composed = OmegaConf.merge(
                    composed, OmegaConf.from_dotlist(list(overrides))
                )
            except Exception:
                # If dotlist parsing fails, re-raise a helpful error
                raise SystemExit(
                    "Failed to parse Hydra overrides: " + ", ".join(overrides)
                )

        return OmegaConf.to_container(composed, resolve=True)  # type: ignore[arg-type]

    with initialize_config_dir(version_base=None, config_dir=config_path):
        try:
            cfg = compose(config_name=config_name, overrides=overrides or [])
            cfg_resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        except MissingConfigException:
            # Fallback: manually compose known data_capture config against repo conf/
            # This preserves CLI overrides and enables tests/CI to run regardless of
            # relative-default quirks in nested config directories.
            cfg_resolved = _manual_compose_for_data_capture(Path(config_path))
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
        plume_mode = env_cfg.get("plume", "static")

        # Build kwargs for make_env, handling movie mode specially
        make_kwargs = {
            "action_type": env_cfg.get("action_type", "oriented"),
            "observation_type": env_cfg.get("observation_type", "concentration"),
            "reward_type": env_cfg.get("reward_type", "step_penalty"),
            "max_steps": env_cfg.get("max_steps", None) or None,
        }

        if plume_mode == "movie":
            movie_cfg = cfg.get("movie", {})
            make_kwargs.update(
                {
                    "plume": "movie",
                    "movie_path": movie_cfg.get("path"),
                    "movie_fps": movie_cfg.get("fps"),
                    "movie_pixel_to_grid": (
                        tuple(movie_cfg.get("pixel_to_grid"))
                        if movie_cfg.get("pixel_to_grid")
                        else None
                    ),
                    "movie_origin": (
                        tuple(movie_cfg.get("origin"))
                        if movie_cfg.get("origin")
                        else None
                    ),
                    "movie_extent": (
                        tuple(movie_cfg.get("extent"))
                        if movie_cfg.get("extent")
                        else None
                    ),
                    "movie_step_policy": movie_cfg.get("step_policy", "wrap"),
                }
            )
            # In movie mode, grid_size derives from dataset; ignore env.grid_size if provided
            w = h = None  # filled after env constructed
            env = pns.make_env(**make_kwargs)
            # Try to discover grid from env for RunMeta
            gs = getattr(env, "grid_size", None)
            if gs is not None:
                # grid_size may be tuple or GridSize
                w = getattr(gs, "width", None) or int(gs[0])
                h = getattr(gs, "height", None) or int(gs[1])
            else:
                raise SystemExit(
                    "Failed to determine grid size from movie dataset/environment"
                )
        else:
            grid = env_cfg.get("grid_size", [64, 64])
            if not isinstance(grid, (list, tuple)) or len(grid) != 2:
                raise SystemExit("env.grid_size must be a 2-element list or tuple")
            w, h = int(grid[0]), int(grid[1])
            make_kwargs.update({"grid_size": (w, h)})
            env = pns.make_env(**make_kwargs)
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
        if w is None or h is None:
            # Fallback guard, should not happen
            raise SystemExit("Unable to resolve grid dimensions for EnvironmentConfig")
        cfg = EnvironmentConfig(
            grid_size=(int(w), int(h)), source_location=(int(w) // 2, int(h) // 2)
        )
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
