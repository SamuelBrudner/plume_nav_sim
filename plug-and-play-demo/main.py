"""Minimal external-style demo runner for plume_nav_sim."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from plume_nav_sim.compose import PolicySpec, SimulationSpec, WrapperSpec, prepare
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.runner import runner

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


def run_demo(
    *,
    seed: Optional[int] = 123,
    grid: str = "128x128",
    max_steps: int = 500,
    render: bool = True,
    policy_spec: str = "plug_and_play_demo:DeltaBasedRunTumblePolicy",
    save_gif: Optional[str] = None,
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

    sim = SimulationSpec(
        grid_size=(w, h),
        max_steps=max_steps,
        action_type="run_tumble",  # 2-action RUN/TUMBLE processor in the env
        observation_type="concentration",
        reward_type="step_penalty",
        render=render,  # enable rgb_array frames
        seed=seed,
        # Load policy via dotted path (short form by default).
        policy=PolicySpec(spec=policy_spec),
        observation_wrappers=wrapper_specs,
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

    # Build SimulationSpec consistently with run_demo
    wrapper_specs: List[WrapperSpec] = [
        WrapperSpec(
            spec="plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
            kwargs={"n": 2},
        )
    ]
    sim = SimulationSpec(
        grid_size=(w, h),
        max_steps=args.max_steps,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render=(not args.no_render),
        seed=args.seed,
        policy=PolicySpec(spec=args.policy_spec),
        observation_wrappers=wrapper_specs,
    )

    # Preserve original behavior unless capture flags are provided
    if not capture_requested:
        run_demo(
            seed=args.seed,
            grid=args.grid,
            max_steps=args.max_steps,
            render=(not args.no_render),
            policy_spec=args.policy_spec,
            save_gif=args.save_gif,
        )
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
