from __future__ import annotations

import argparse
from pathlib import Path
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


def _resolve_config_dir(config_path: Optional[str]) -> str:
    """Resolve the Hydra config directory, defaulting to the repo conf/ tree.

    Uses plume_nav_sim.get_conf_dir() when available, falling back to
    conf/ relative to this file. This mirrors the previous inline logic
    in _load_hydra_config.
    """
    if config_path:
        return config_path

    conf_dir = pns.get_conf_dir()
    if conf_dir is None:
        conf_dir = Path(__file__).resolve().parents[2] / "conf"
    return str(conf_dir)


def _manual_compose_for_data_capture(
    config_name: str,
    conf_root: Path,
    overrides: List[str],
) -> dict:
    """Manual composition for data_capture/config to support mixed group locations.

    This handles the case where data_capture/config.yaml references groups
    that live at the repository conf/ root (e.g., movie/, env/plume/), while
    the experiment/ group lives under conf/data_capture/.
    """
    from hydra.errors import MissingConfigException
    from omegaconf import OmegaConf

    # Locate the requested YAML file
    yaml_path = conf_root.joinpath(*config_name.split("/"))
    if not yaml_path.suffix:
        yaml_path = yaml_path.with_suffix(".yaml")
    if not yaml_path.exists():
        raise MissingConfigException(
            missing_cfg_file=str(yaml_path),
            message=f"Could not find config '{config_name}' for data_capture composition",
            options=None,
        )

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
                        message=(
                            f"Could not find '{group_key}/{option_name}' "
                            "for data_capture composition"
                        ),
                        options=None,
                    )
            part_cfg = OmegaConf.load(group_file)
            composed = OmegaConf.merge(composed, part_cfg)
        else:
            # Unsupported defaults entry format; ignore for now
            continue

    # Apply CLI overrides as a dotlist
    if overrides:
        try:
            composed = OmegaConf.merge(
                composed, OmegaConf.from_dotlist(list(overrides))
            )
        except Exception:
            # If dotlist parsing fails, re-raise a helpful error
            raise SystemExit("Failed to parse Hydra overrides: " + ", ".join(overrides))

    return OmegaConf.to_container(composed, resolve=True)  # type: ignore[arg-type]


def _stable_cfg_hash(cfg_resolved: dict) -> str:
    """Compute a stable hash for a resolved config dictionary."""
    import json
    from hashlib import sha256

    cfg_bytes = json.dumps(
        cfg_resolved,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(cfg_bytes).hexdigest()


def _load_hydra_config(
    *,
    config_name: str,
    config_path: Optional[str],
    overrides: List[str],
) -> Tuple[dict, str]:
    """Load and resolve Hydra config, returning (as_dict, config_hash)."""
    from hydra import compose, initialize_config_dir
    from hydra.errors import MissingConfigException
    from omegaconf import OmegaConf

    config_dir = _resolve_config_dir(config_path)

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        try:
            cfg = compose(config_name=config_name, overrides=overrides or [])
            cfg_resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        except MissingConfigException:
            # Fallback: manually compose known data_capture config against repo conf/
            # This preserves CLI overrides and enables tests/CI to run regardless of
            # relative-default quirks in nested config directories.
            cfg_resolved = _manual_compose_for_data_capture(
                config_name=config_name,
                conf_root=Path(config_dir),
                overrides=overrides or [],
            )

    cfg_hash = _stable_cfg_hash(cfg_resolved)
    return cfg_resolved, cfg_hash


def _is_flag_provided(argv: Optional[list[str]], flag: str) -> bool:
    if not argv:
        return False
    return any(tok == flag for tok in argv)


def _parse_grid_arg(grid: str) -> Tuple[int, int]:
    try:
        w, h = (int(p) for p in grid.lower().split("x"))
        return w, h
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 64x64")


def _env_from_cfg(cfg: dict) -> Tuple[object, int, int]:
    env_cfg = cfg.get("env", {})
    plume_mode = env_cfg.get("plume", "static")
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
                    tuple(movie_cfg.get("origin")) if movie_cfg.get("origin") else None
                ),
                "movie_extent": (
                    tuple(movie_cfg.get("extent")) if movie_cfg.get("extent") else None
                ),
                "movie_step_policy": movie_cfg.get("step_policy", "wrap"),
            }
        )
        env = pns.make_env(**make_kwargs)
        gs = getattr(env, "grid_size", None)
        if gs is None:
            raise SystemExit(
                "Failed to determine grid size from movie dataset/environment"
            )
        w = getattr(gs, "width", None) or int(gs[0])
        h = getattr(gs, "height", None) or int(gs[1])
        return env, w, h
    # static plume: require grid_size
    grid = env_cfg.get("grid_size", [64, 64])
    if not isinstance(grid, (list, tuple)) or len(grid) != 2:
        raise SystemExit("env.grid_size must be a 2-element list or tuple")
    w, h = int(grid[0]), int(grid[1])
    make_kwargs.update({"grid_size": (w, h)})
    env = pns.make_env(**make_kwargs)
    return env, w, h


def _merge_args_from_cfg(
    args: argparse.Namespace, cfg: dict, argv: Optional[list[str]]
) -> None:
    def provided(flag: str) -> bool:
        return _is_flag_provided(argv, flag)

    args.output = args.output if provided("--output") else cfg.get("output", "results")
    args.experiment = (
        args.experiment
        if provided("--experiment")
        else cfg.get("experiment", "default")
    )
    args.episodes = (
        args.episodes if provided("--episodes") else int(cfg.get("episodes", 1))
    )
    args.seed = args.seed if provided("--seed") else int(cfg.get("seed", 123))
    args.rotate_size = (
        args.rotate_size if provided("--rotate-size") else cfg.get("rotate_size", None)
    )
    args.parquet = bool(
        args.parquet if provided("--parquet") else cfg.get("parquet", False)
    )


def _run_capture(
    env: object, w: int, h: int, *, args: argparse.Namespace, cfg_hash: Optional[str]
) -> None:
    rec = RunRecorder(
        args.output, experiment=args.experiment, rotate_size_bytes=args.rotate_size
    )
    cfg = EnvironmentConfig(
        grid_size=(int(w), int(h)), source_location=(int(w) // 2, int(h) // 2)
    )
    wrapped = DataCaptureWrapper(
        env, rec, cfg, meta_overrides={"config_hash": cfg_hash} if cfg_hash else None
    )

    base_seed = int(args.seed)
    episodes = int(args.episodes)
    for i in range(episodes):
        seed = base_seed + i
        wrapped.reset(seed=seed)
        while True:
            _ = wrapped.action_space.sample()
            _, _, terminated, truncated, _ = wrapped.step(
                _.item() if hasattr(_, "item") else _
            )
            if terminated or truncated:
                break
    rec.finalize(export_parquet=bool(args.parquet))
    wrapped.close()


def main(argv: Optional[list[str]] = None) -> int:
    """Run episodes and capture analysis-ready artifacts via data_capture."""
    args, overrides = _parse_args_and_overrides(argv)

    env = None
    cfg_hash: Optional[str] = None
    if args.config_name:
        cfg, cfg_hash = _load_hydra_config(
            config_name=str(args.config_name),
            config_path=args.config_path,
            overrides=overrides,
        )
        env, w, h = _env_from_cfg(cfg)
        # Ensure flag detection works when invoked via `python -m` (argv is None)
        from sys import argv as sys_argv

        effective_argv: Optional[list[str]] = (
            argv if argv is not None else list(sys_argv[1:])
        )
        _merge_args_from_cfg(args, cfg, effective_argv)
    else:
        w, h = _parse_grid_arg(args.grid)
        env = pns.make_env(
            grid_size=(w, h),
            action_type="oriented",
            observation_type="concentration",
            reward_type="step_penalty",
            max_steps=args.max_steps,
        )

    try:
        if w is None or h is None:  # type: ignore[truthy-bool]
            raise SystemExit("Unable to resolve grid dimensions for EnvironmentConfig")
        _run_capture(env, w, h, args=args, cfg_hash=cfg_hash)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
