"""Minimal external-style demo runner for plume_nav_sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from plume_nav_sim.compose import PolicySpec, SimulationSpec, WrapperSpec, prepare
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.runner import runner

DEMO_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MOVIE_ZARR = DEMO_ASSETS_DIR / "gaussian_plume_demo.zarr"

# Quick reference for demo behavior:
# - Imports plume_nav_sim as a library only (policy is defined in this demo)
# - Uses compose.prepare to build (env, policy)
# - Runs one episode with the run–tumble temporal-derivative policy via runner
#
# Policy interface quick ref (must match plume_nav_sim Policy protocol):
# - property action_space -> gym.Space
# - def reset(*, seed: int | None = None) -> None
# - def select_action(observation: np.ndarray, *, explore: bool = True) -> int
#
# Observation setup
# - We apply the core ConcentrationNBackWrapper(n=2) via SimulationSpec so the
#   policy receives [c_prev, c_now] and can be stateless.
#
# Policy import
# - We load the stateless demo policy via short dotted path:
#   "plug_and_play_demo:DeltaBasedRunTumblePolicy".
#   The demo package is importable when running from this folder or after
#   installing it (e.g., `pip install -e plug-and-play-demo`).


def _build_movie_kwargs(
    plume: str,
    movie_path: Optional[str],
    movie_fps: Optional[float],
    movie_step_policy: Optional[str],
    movie_h5_dataset: Optional[str],
) -> Dict[str, Any]:
    """Resolve movie plume configuration (path, fps, step policy).

    Unit conventions
    ----------------

    - ``movie_fps`` is interpreted as frames per second (time unit = seconds).
    - Spatial calibration and physical spatial units for movie plumes are
      derived from the movie metadata sidecar (``spatial_unit`` and
      ``pixels_per_unit``); there are no separate CLI flags for spatial units.
    """

    if plume != "movie":
        return {}

    resolved_path = Path(movie_path) if movie_path else DEFAULT_MOVIE_ZARR
    if not resolved_path.exists():
        raise SystemExit(
            "Movie plume requested but dataset not found at "
            f"{resolved_path}. Regenerate via bead 214 or pass --movie-path."
        )

    movie_kwargs: Dict[str, Any] = {"movie_path": str(resolved_path)}
    suffix = resolved_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        if not movie_h5_dataset:
            raise SystemExit(
                "HDF5 movie plume requires --movie-h5-dataset to specify the dataset within the file"
            )
        movie_kwargs["movie_h5_dataset"] = movie_h5_dataset
    if movie_fps is not None:
        movie_kwargs["movie_fps"] = float(movie_fps)
    if movie_step_policy is not None:
        movie_kwargs["movie_step_policy"] = movie_step_policy
    return movie_kwargs


def _load_simulation_spec_from_config(path: str) -> Dict[str, Any]:
    """Load a SimulationSpec dictionary from a config file.

    Supports JSON by default. Attempts TOML (via tomllib) or YAML (via PyYAML)
    if available based on file extension.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Config file not found: {path}")

    ext = p.suffix.lower()

    if ext == ".json":
        import json

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext in (".toml",):
        try:
            import tomllib  # Python 3.11+
        except Exception as e:  # pragma: no cover - optional path
            raise SystemExit("TOML config requires Python 3.11+ (tomllib)") from e
        with p.open("rb") as f:
            data = tomllib.load(f)
    elif ext in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover - optional path
            raise SystemExit("YAML config requires PyYAML to be installed") from e
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise SystemExit(
            f"Unsupported config extension '{ext}'. Use .json, .toml, or .yaml"
        )

    # Allow top-level SimulationSpec, or nested under a 'simulation' key.
    if (
        isinstance(data, dict)
        and "simulation" in data
        and isinstance(data["simulation"], dict)
    ):
        return data["simulation"]
    if isinstance(data, dict):
        return data
    raise SystemExit("Config must parse into a dict or a dict with 'simulation' key")


def run_demo(
    *,
    seed: Optional[int] = 123,
    grid: str = "128x128",
    max_steps: int = 500,
    render: bool = True,
    policy_spec: str = "plug_and_play_demo:DeltaBasedRunTumblePolicy",
    save_gif: Optional[str] = None,
    plume: str = "static",
    movie_path: Optional[str] = None,
    movie_fps: Optional[float] = None,
    movie_step_policy: Optional[str] = None,
    movie_h5_dataset: Optional[str] = None,
) -> None:
    try:
        # Map human-friendly WxH string to SimulationSpec.grid_size=(W, H)
        w, h = (int(p) for p in grid.lower().split("x"))
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 128x128")

    # Compose a complete simulation (-> env, policy) from string options:
    # - action_type="run_tumble" selects OrientedRunTumbleActions (Discrete(2): 0=RUN, 1=TUMBLE)
    # - observation_type="concentration" selects ConcentrationSensor (Box(1,) with odor at agent)
    # - reward_type="step_penalty" selects StepPenaltyReward (sparse goal + step cost)
    # - render=True switches env.render_mode to "rgb_array" so runner can attach frames
    # - policy=PolicySpec(spec="plug_and_play_demo:DeltaBasedRunTumblePolicy") loads the demo policy
    #   via dotted-path import and instantiation
    # Specify an observation wrapper via dotted path so that the spec fully
    # describes the runtime behavior (no ad-hoc wrapping after creation).

    # To execute a run, parameters are given to main() and passed here.

    # Always apply the 1-back concentration wrapper to expose [c_prev, c_now]
    wrapper_specs: List[WrapperSpec] = [
        WrapperSpec(
            spec="plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
            kwargs={"n": 2},
        )
    ]

    movie_kwargs = _build_movie_kwargs(
        plume=plume,
        movie_path=movie_path,
        movie_fps=movie_fps,
        movie_step_policy=movie_step_policy,
        movie_h5_dataset=movie_h5_dataset,
    )

    sim = SimulationSpec(
        grid_size=(w, h),
        max_steps=max_steps,
        action_type="run_tumble",  # 2-action RUN/TUMBLE processor in the env
        observation_type="concentration",
        reward_type="step_penalty",
        render=render,  # enable rgb_array frames
        seed=seed,
        plume=plume,
        # Load policy via dotted path (short form by default).
        policy=PolicySpec(spec=policy_spec),
        observation_wrappers=wrapper_specs,
        **movie_kwargs,
    )

    env, policy = prepare(sim)

    frames: List[np.ndarray] = []

    def on_step(ev: runner.StepEvent) -> None:
        if ev.frame is not None:
            frames.append(ev.frame)

    # runner.run_episode resets env+policy with the given seed (deterministic),
    # enforces action-space subset checks,
    # and attaches RGB frames to events when render=True.
    result = runner.run_episode(
        env,
        policy,
        seed=sim.seed,
        render=bool(sim.render),
        on_step=on_step,
    )

    print(
        "Episode summary:",
        {
            "seed": result.seed,
            "steps": result.steps,
            "total_reward": round(result.total_reward, 3),
            "terminated": result.terminated,
            "truncated": result.truncated,
            "frames_captured": len(frames),
        },
    )

    # Optional: save frames to a video using library utility (imageio required)
    if save_gif and frames:
        try:
            from plume_nav_sim.utils.video import save_video_frames

            save_video_frames(frames, save_gif, fps=10)
            print(f"Saved video: {save_gif}")
        except ImportError as e:
            print("Could not save video (install imageio):", e)
        except Exception as e:
            print("Video export failed:", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plug-and-play demo: run–tumble TD with runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to SimulationSpec config (json|toml|yaml). "
            "If provided, CLI flags act as overrides when specified."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--grid", type=str, default="128x128", help="WxH, e.g., 128x128"
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rgb frame capture",
    )
    parser.add_argument(
        "--policy-spec",
        type=str,
        default="plug_and_play_demo:DeltaBasedRunTumblePolicy",
        help=(
            "Dotted path to policy class or callable, e.g., "
            "'plug_and_play_demo:DeltaBasedRunTumblePolicy' or 'my_pkg.mod:MyPolicy'"
        ),
    )
    parser.add_argument(
        "--plume",
        choices=["static", "movie"],
        default="static",
        help="Select the plume source (static Gaussian or bundled movie)",
    )
    parser.add_argument(
        "--movie-path",
        type=str,
        default=None,
        help=(
            "Path to movie plume dataset (defaults to bundled assets/gaussian_plume_demo.zarr)."
        ),
    )
    parser.add_argument(
        "--movie-fps",
        type=float,
        default=None,
        help=(
            "Expected movie fps (frames per second) when using plume='movie'. "
            "For raw media sources with a metadata sidecar this is a "
            "validation alias and must match the sidecar; it does not "
            "override sidecar-derived metadata."
        ),
    )
    parser.add_argument(
        "--movie-step-policy",
        choices=["wrap", "clamp"],
        default=None,
        help="Optional MoviePlumeField step policy override",
    )
    parser.add_argument(
        "--movie-h5-dataset",
        type=str,
        default=None,
        help=(
            "Required for this CLI when --movie-path points to an HDF5 file; "
            "gives the dataset path within the file (e.g., 'Plume Data/dataset_001') "
            "and must match the h5_dataset field in the movie metadata sidecar."
        ),
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default=None,
        help="Optional output GIF filepath to save rendered frames",
    )
    # Capture mode flags (optional). Capture is enabled when any of these are provided.
    parser.add_argument(
        "--capture-root",
        type=str,
        default="results",
        help="Enable capture mode and write under this root",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="demo",
        help="Experiment name (subfolder under capture root)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes when capture is enabled",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Export Parquet at end of run (if pyarrow available)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate captured artifacts and print a summary",
    )
    args = parser.parse_args()

    def _flag_provided(flag: str) -> bool:
        return flag in sys.argv[1:]

    capture_requested = any(
        _flag_provided(f)
        for f in (
            "--capture-root",
            "--experiment",
            "--episodes",
            "--parquet",
            "--validate",
        )
    )

    # Parse grid string early (shared across flows)
    try:
        w, h = (int(p) for p in args.grid.lower().split("x"))
    except Exception:
        raise SystemExit("--grid must be formatted as WxH, e.g., 128x128")

    # Build SimulationSpec: from config if provided, otherwise from CLI defaults
    sim: SimulationSpec
    if args.config:
        raw_cfg = _load_simulation_spec_from_config(args.config)
        # Track whether wrappers were specified explicitly in config
        wrappers_specified = (
            isinstance(raw_cfg, dict) and "observation_wrappers" in raw_cfg
        )

        # Apply targeted CLI overrides when flags were explicitly provided
        cfg = dict(raw_cfg)
        if _flag_provided("--seed"):
            cfg["seed"] = int(args.seed)
        if _flag_provided("--grid"):
            cfg["grid_size"] = (int(w), int(h))
        if _flag_provided("--max-steps"):
            cfg["max_steps"] = int(args.max_steps)
        if _flag_provided("--no-render"):
            cfg["render"] = not args.no_render
        if _flag_provided("--plume"):
            cfg["plume"] = args.plume
        # Movie-specific overrides (only meaningful for plume='movie')
        if _flag_provided("--movie-path"):
            cfg["movie_path"] = args.movie_path
        if _flag_provided("--movie-fps"):
            cfg["movie_fps"] = args.movie_fps
        if _flag_provided("--movie-step-policy"):
            cfg["movie_step_policy"] = args.movie_step_policy
        if _flag_provided("--movie-h5-dataset"):
            cfg["movie_h5_dataset"] = args.movie_h5_dataset
        # Policy override
        if _flag_provided("--policy-spec"):
            cfg["policy"] = {"spec": args.policy_spec}

        # Ensure baseline action/observation/reward if not specified at all
        cfg.setdefault("action_type", "run_tumble")
        cfg.setdefault("observation_type", "concentration")
        cfg.setdefault("reward_type", "step_penalty")

        # Default wrapper only if not provided by config
        if not wrappers_specified:
            cfg.setdefault(
                "observation_wrappers",
                [
                    {
                        "spec": "plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
                        "kwargs": {"n": 2},
                    }
                ],
            )

        # Validate via model
        sim = SimulationSpec.model_validate(cfg)
    else:
        # Build SimulationSpec consistently with run_demo defaults
        wrapper_specs: List[WrapperSpec] = [
            WrapperSpec(
                spec="plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
                kwargs={"n": 2},
            )
        ]
        movie_kwargs = _build_movie_kwargs(
            plume=args.plume,
            movie_path=args.movie_path,
            movie_fps=args.movie_fps,
            movie_step_policy=args.movie_step_policy,
            movie_h5_dataset=args.movie_h5_dataset,
        )

        sim = SimulationSpec(
            grid_size=(w, h),
            max_steps=args.max_steps,
            action_type="run_tumble",
            observation_type="concentration",
            reward_type="step_penalty",
            render=(not args.no_render),
            seed=args.seed,
            plume=args.plume,
            policy=PolicySpec(spec=args.policy_spec),
            observation_wrappers=wrapper_specs,
            **movie_kwargs,
        )

    # Preserve original behavior unless capture flags are provided
    if not capture_requested:
        # Non-capture: execute a single episode using the composed spec
        env, policy = prepare(sim)

        frames: List[np.ndarray] = []

        def on_step(ev: runner.StepEvent) -> None:
            if ev.frame is not None and sim.render:
                frames.append(ev.frame)

        result = runner.run_episode(
            env,
            policy,
            seed=sim.seed,
            render=bool(sim.render),
            on_step=on_step,
        )

        print(
            "Episode summary:",
            {
                "seed": result.seed,
                "steps": result.steps,
                "total_reward": round(result.total_reward, 3),
                "terminated": result.terminated,
                "truncated": result.truncated,
                "frames_captured": len(frames),
            },
        )

        if args.save_gif and frames:
            try:
                from plume_nav_sim.utils.video import save_video_frames

                save_video_frames(frames, args.save_gif, fps=10)
                print(f"Saved video: {args.save_gif}")
            except ImportError as e:
                print("Could not save video (install imageio):", e)
            except Exception as e:
                print("Video export failed:", e)
        return

    # Capture mode: wrap env and record
    env, policy = prepare(sim)
    from plume_nav_sim.data_capture import RunRecorder  # lazy import
    from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper  # lazy import

    rec = RunRecorder(args.capture_root, experiment=args.experiment)
    env_cfg = EnvironmentConfig(grid_size=(w, h), source_location=(w // 2, h // 2))
    env_wrapped = DataCaptureWrapper(env, rec, env_cfg)

    frames: List[np.ndarray] = []

    def on_step(ev: runner.StepEvent) -> None:
        if ev.frame is not None and sim.render:
            frames.append(ev.frame)

    base_seed = int(args.seed)
    for i in range(int(args.episodes)):
        frames.clear()
        ep_seed = base_seed + i
        _ = runner.run_episode(
            env_wrapped,
            policy,
            seed=ep_seed,
            render=bool(sim.render),
            on_step=on_step,
        )
        if args.save_gif and frames:
            try:
                from plume_nav_sim.utils.video import save_video_frames

                save_video_frames(frames, args.save_gif, fps=10)
                print(f"Saved video: {args.save_gif}")
            except ImportError as e:
                print("Could not save video (install imageio):", e)
            except Exception as e:
                print("Video export failed:", e)

    rec.finalize(export_parquet=bool(args.parquet))
    run_dir = rec.root
    print(f"Capture complete. Run directory: {run_dir}")

    if args.validate:
        try:
            from plume_nav_sim.data_capture.validate import validate_run_artifacts

            report = validate_run_artifacts(run_dir)
            print(
                "Validation:",
                {
                    "steps_ok": bool(report.get("steps", {}).get("ok")),
                    "episodes_ok": bool(report.get("episodes", {}).get("ok")),
                },
            )
        except Exception as e:
            print(f"Validation skipped or failed: {e}")


if __name__ == "__main__":
    main()
