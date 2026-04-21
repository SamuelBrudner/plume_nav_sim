from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from PySide6 import QtCore
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env."
    ) from e

try:
    from plume_nav_sim.data_capture.replay import ReplayEngine, ReplayLoadError
    from plume_nav_sim.data_capture.replay import ReplayConsistencyError
    from plume_nav_sim.data_capture.replay import ReplayStepEvent as _ReplayStepEvent
    from plume_nav_sim.data_capture.replay import load_replay_engine
    from plume_nav_sim.data_capture.schemas import RunMeta
except Exception as e:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "Replay support requires plume_nav_sim.data_capture.replay to be importable."
    ) from e


class ReplayDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(object)
    timeline_changed = QtCore.Signal(int, int)
    step_done = QtCore.Signal(object)
    episode_finished = QtCore.Signal()
    error_occurred = QtCore.Signal(str)
    replay_validation_failed = QtCore.Signal(object)  # payload: dict-like diff

    def __init__(self) -> None:
        super().__init__()
        self._run_dir: Path | None = None
        self._engine: Optional[ReplayEngine] = None
        self._events: list[_ReplayStepEvent] = []
        self._index = 0
        self._episode_starts: list[int] = []
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._running = False
        self._last_event: object | None = None
        self._last_validation_diff: object | None = None

    def _build_replay_env(self, *, render: bool) -> object:
        import plume_nav_sim as pns

        kwargs = self.get_resolved_env_kwargs(render=render)
        return pns.make_env(**kwargs)

    def _emit_validation_failure(self, exc: ReplayConsistencyError) -> None:
        self.error_occurred.emit(str(exc))
        diff = getattr(exc, "diff", None)
        if diff is None:
            return
        payload = diff.to_dict() if hasattr(diff, "to_dict") else diff
        self._last_validation_diff = payload
        self.replay_validation_failed.emit(payload)

    def _load_events(self) -> None:
        if self._engine is None:
            self._events = []
            self._episode_starts = []
            return
        self._events = list(self._engine.iter_events(start_index=0, render=True))
        self._episode_starts = list(self._engine.episode_starts)

    def _reset_playback_state(self) -> None:
        self._index = 0
        self._running = False
        self._timer.stop()

    def load_run(self, run_dir: str | Path) -> None:
        run_path = Path(run_dir)
        self._engine = load_replay_engine(str(run_path))
        self._run_dir = run_path
        self._last_validation_diff = None

        # Best-effort: validate by reconstructing and stepping the environment.
        try:
            self._engine.validate(
                env_factory=lambda _meta, render: self._build_replay_env(render=render),
                render=False,
            )
        except ReplayConsistencyError as exc:
            self._emit_validation_failure(exc)
        except Exception as exc:
            # Validation is optional; failure should be visible but not block playback.
            self.error_occurred.emit(f"Replay validation skipped: {exc}")

        self._load_events()
        self._reset_playback_state()
        self.seek_to(0, auto_emit=True)

    def is_loaded(self) -> bool:
        return self._engine is not None and bool(self._events)

    def total_steps(self) -> int:
        return len(self._events)

    def total_episodes(self) -> int:
        if self._engine is None:
            return len(self._episode_starts)
        return self._engine.total_episodes()

    def current_index(self) -> int:
        return self._index

    def current_episode_index(self) -> int:
        if not self._episode_starts:
            return 0
        idx = int(self._index)
        ep = 0
        for start in self._episode_starts:
            if start <= idx:
                ep += 1
        return max(0, ep - 1)

    def episode_starts(self) -> list[int]:
        return list(self._episode_starts)

    def get_timeline_markers(self) -> dict[str, list[int]]:
        """Return marker indices suitable for MarkerSlider."""

        if not self.is_loaded():
            return {}

        terminated: list[int] = []
        truncated: list[int] = []
        goal: list[int] = []
        for i, ev in enumerate(self._events):
            if bool(getattr(ev, "terminated", False)):
                terminated.append(i)
            if bool(getattr(ev, "truncated", False)):
                truncated.append(i)
            if self._is_goal_reached(ev):
                goal.append(i)

        return {
            "episode": list(self._episode_starts),
            "terminated": terminated,
            "truncated": truncated,
            "goal": goal,
        }

    def jump_prev_episode(self) -> bool:
        idx = self._find_prev_episode_start()
        if idx is None:
            return False
        self.seek_to(idx, auto_emit=True)
        return True

    def jump_next_episode(self) -> bool:
        idx = self._find_next_episode_start()
        if idx is None:
            return False
        self.seek_to(idx, auto_emit=True)
        return True

    def jump_next_done(self) -> bool:
        idx = self._find_next_index(self._is_done)
        if idx is None:
            return False
        self.seek_to(idx, auto_emit=True)
        return True

    def jump_next_goal(self) -> bool:
        idx = self._find_next_index(self._is_goal_reached)
        if idx is None:
            return False
        self.seek_to(idx, auto_emit=True)
        return True

    def _find_prev_episode_start(self) -> int | None:
        if not self.is_loaded():
            return None
        cur = int(self._index)
        for start in reversed(self._episode_starts):
            if start < cur:
                return int(start)
        return None

    def _find_next_episode_start(self) -> int | None:
        if not self.is_loaded():
            return None
        cur = int(self._index)
        for start in self._episode_starts:
            if start > cur:
                return int(start)
        return None

    def _find_next_index(self, pred: Callable[[object], bool]) -> int | None:
        if not self.is_loaded():
            return None
        start = min(max(int(self._index) + 1, 0), len(self._events))
        for i in range(start, len(self._events)):
            if pred(self._events[i]):
                return i
        return None

    @staticmethod
    def _coerce_int_pair(value: object) -> tuple[int, int] | None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None

    @staticmethod
    def _coerce_float_pair(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    def _run_meta(self) -> RunMeta | None:
        if self._engine is None:
            return None
        return self._engine.run_meta

    def _env_config(self) -> dict[str, Any]:
        meta = self._run_meta()
        if meta is None or not isinstance(meta.env_config, dict):
            return {}
        return meta.env_config

    def _extra_config(self) -> dict[str, Any]:
        meta = self._run_meta()
        if meta is None or not isinstance(meta.extra, dict):
            return {}
        return meta.extra

    def _infer_action_type(self) -> str:
        if self._engine is None:
            return "oriented"
        steps = self._engine.steps
        if not steps:
            return "oriented"
        max_action = max(int(rec.action) for rec in steps)
        if max_action >= 3:
            return "discrete"
        if max_action == 2:
            return "oriented"
        return "oriented"

    def _is_goal_reached(self, ev: object) -> bool:
        info = getattr(ev, "info", None)
        if isinstance(info, dict) and bool(info.get("goal_reached")):
            return True
        # Fallback: treat termination as goal reached for standard gym semantics.
        return bool(getattr(ev, "terminated", False)) and not bool(
            getattr(ev, "truncated", False)
        )

    def _is_done(self, ev: object) -> bool:
        return bool(getattr(ev, "terminated", False)) or bool(
            getattr(ev, "truncated", False)
        )

    def is_running(self) -> bool:
        return bool(self._running)

    def start(self, interval_ms: int = 50) -> None:
        if not self.is_loaded():
            return
        self._timer.start(max(1, interval_ms))
        self._running = True

    def pause(self) -> None:
        self._timer.stop()
        self._running = False

    def step_once(self) -> None:
        self._on_tick()

    def step_back(self) -> None:
        if not self.is_loaded():
            return
        if self._index <= 0:
            self._index = 0
        else:
            self._index -= 1
        self._emit_current(auto_emit_frame=True)

    def seek_to_episode(self, episode_index: int, *, auto_emit: bool = True) -> None:
        if not self.is_loaded():
            return
        if not self._episode_starts:
            self.seek_to(0, auto_emit=auto_emit)
            return
        ep = max(0, min(episode_index, len(self._episode_starts) - 1))
        self.seek_to(self._episode_starts[ep], auto_emit=auto_emit)

    def seek_to(self, step_index: int, *, auto_emit: bool = True) -> None:
        if not self.is_loaded():
            return
        max_idx = max(0, len(self._events) - 1)
        self._index = max(0, min(step_index, max_idx))
        if auto_emit:
            self._emit_current(auto_emit_frame=True)
        self.timeline_changed.emit(self._index, len(self._events))

    def get_action_names(self) -> list[str]:
        return []

    def get_grid_size(self) -> tuple[int, int]:
        grid = self._coerce_int_pair(self._env_config().get("grid_size"))
        return grid if grid is not None else (0, 0)

    def get_overlay_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        cfg = self._env_config()
        if "source_location" in cfg:
            ctx["source_xy"] = cfg.get("source_location")
        if "goal_radius" in cfg:
            ctx["goal_radius"] = cfg.get("goal_radius")
        return ctx

    def last_event(self):
        return self._last_event

    def get_resolved_env_kwargs(self, *, render: bool = False) -> dict[str, Any]:
        env_cfg = self._env_config()
        extra = self._extra_config()
        if not env_cfg and self._engine is None:
            return {}

        kwargs: dict[str, Any] = {}

        grid = self._coerce_int_pair(env_cfg.get("grid_size"))
        if grid is not None:
            kwargs["grid_size"] = grid

        src = self._coerce_int_pair(env_cfg.get("source_location"))
        if src is not None:
            kwargs["source_location"] = src

        max_steps = env_cfg.get("max_steps")
        if max_steps is not None:
            with contextlib.suppress(Exception):
                kwargs["max_steps"] = int(max_steps)
        goal_radius = env_cfg.get("goal_radius")
        if goal_radius is not None:
            with contextlib.suppress(Exception):
                kwargs["goal_radius"] = float(goal_radius)

        plume_params = env_cfg.get("plume_params")
        if isinstance(plume_params, dict):
            kwargs["plume_params"] = plume_params

        # Render configuration is always best-effort in replay mode.
        if render:
            kwargs["render_mode"] = "rgb_array"

        # Forward any richer capture config if present in RunMeta.extra.
        env_extra = extra.get("env")
        if isinstance(env_extra, dict):
            for key in ("action_type", "observation_type", "reward_type", "plume"):
                if key in env_extra and env_extra[key] is not None:
                    kwargs[key] = env_extra[key]
        else:
            # No captured env group: best-effort inference to recreate manual capture runs.
            kwargs.setdefault("action_type", self._infer_action_type())
            kwargs.setdefault("observation_type", "concentration")
            kwargs.setdefault("reward_type", "step_penalty")

        movie_extra = extra.get("movie")
        if isinstance(movie_extra, dict):
            # Hydra/capture config schema -> make_env kwargs
            mapping = {
                "path": "movie_path",
                "dataset_id": "movie_dataset_id",
                "auto_download": "movie_auto_download",
                "cache_root": "movie_cache_root",
                "fps": "movie_fps",
                "pixel_to_grid": "movie_pixel_to_grid",
                "origin": "movie_origin",
                "extent": "movie_extent",
                "step_policy": "movie_step_policy",
                "h5_dataset": "movie_h5_dataset",
                "normalize": "movie_normalize",
                "chunks": "movie_chunks",
            }
            for src_key, dst_key in mapping.items():
                val = movie_extra.get(src_key)
                if val is None:
                    continue
                if dst_key in {"movie_pixel_to_grid", "movie_origin", "movie_extent"}:
                    pair = self._coerce_float_pair(val)
                    if pair is not None:
                        kwargs[dst_key] = pair
                    continue
                kwargs[dst_key] = val

        # Also support flat keys in extra (e.g., demo capture mode).
        for key in (
            "plume",
            "movie_path",
            "movie_dataset_id",
            "movie_auto_download",
            "movie_cache_root",
            "movie_fps",
            "movie_step_policy",
            "movie_h5_dataset",
            "movie_normalize",
            "movie_chunks",
        ):
            if key in extra and extra[key] is not None:
                kwargs[key] = extra[key]

        return kwargs

    def get_resolved_replay_config(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        meta = getattr(self._engine, "run_meta", None)
        return {
            "run_dir": str(self._run_dir) if self._run_dir is not None else None,
            "run_meta": meta.to_dict() if hasattr(meta, "to_dict") else meta,
            "env_kwargs": self.get_resolved_env_kwargs(render=False),
        }

    @QtCore.Slot()
    def _on_tick(self) -> None:
        if not self.is_loaded():
            self.pause()
            return

        if self._index >= len(self._events):
            self.pause()
            return

        self._emit_current(auto_emit_frame=True)
        self._index += 1
        self.timeline_changed.emit(min(self._index, len(self._events)), len(self._events))

    def _emit_current(self, *, auto_emit_frame: bool) -> None:
        if not self.is_loaded():
            return
        idx = max(0, min(int(self._index), len(self._events) - 1))
        ev = self._events[idx]
        self._last_event = ev
        if auto_emit_frame and getattr(ev, "frame", None) is not None:
            self.frame_ready.emit(ev.frame)
        self.step_done.emit(ev)
        if getattr(ev, "terminated", False) or getattr(ev, "truncated", False):
            self.episode_finished.emit()
            if self.is_running():
                self.pause()


__all__ = ["ReplayDriver", "ReplayLoadError"]
