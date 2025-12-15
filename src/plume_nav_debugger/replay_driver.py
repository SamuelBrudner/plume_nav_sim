from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from PySide6 import QtCore
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

import plume_nav_sim as pns
from plume_nav_sim.data_capture import (
    ReplayArtifacts,
    ReplayEngine,
    ReplayLoadError,
    load_replay_artifacts,
)
from plume_nav_sim.data_capture.schemas import RunMeta


def _flatten_env_config(cfg: dict | None) -> dict[str, Any]:
    base = dict(cfg or {})
    nested = base.get("env")
    if isinstance(nested, dict):
        merged = {**base, **nested}
        merged.pop("env", None)
        return merged
    return base


def _infer_max_steps_from_artifacts(artifacts: ReplayArtifacts) -> int | None:
    cfg = _flatten_env_config(getattr(artifacts.run_meta, "env_config", {}))
    raw = cfg.get("max_steps")
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass

    limits: set[int] = set()
    for ep in artifacts.episodes:
        try:
            if ep.truncated:
                limits.add(int(ep.total_steps))
        except Exception:
            continue

    if not limits:
        for step in artifacts.steps:
            try:
                if step.truncated:
                    limits.add(int(step.step))
            except Exception:
                continue

    if len(limits) == 1:
        return limits.pop()
    return None


def _env_kwargs_from_meta(
    meta: RunMeta, *, render: bool, recorded_max_steps: int | None
) -> dict[str, Any]:
    raw_cfg = dict(meta.env_config or {})
    cfg = _flatten_env_config(raw_cfg)
    kwargs: dict[str, Any] = {}

    grid = cfg.get("grid_size")
    if grid:
        try:
            kwargs["grid_size"] = (int(grid[0]), int(grid[1]))
        except Exception:
            pass

    src = cfg.get("source_location")
    if src:
        try:
            kwargs["source_location"] = (int(src[0]), int(src[1]))
        except Exception:
            pass

    max_steps = cfg.get("max_steps")
    if max_steps is None:
        max_steps = recorded_max_steps
    if max_steps is not None:
        try:
            kwargs["max_steps"] = int(max_steps)
        except Exception:
            pass

    if cfg.get("goal_radius") is not None:
        try:
            kwargs["goal_radius"] = float(cfg["goal_radius"])
        except Exception:
            pass

    plume_params = cfg.get("plume_params")
    if plume_params:
        try:
            params = dict(plume_params)
            src_loc = params.get("source_location")
            if src_loc:
                params["source_location"] = (int(src_loc[0]), int(src_loc[1]))
            kwargs["plume_params"] = params
        except Exception:
            pass

    start_location = cfg.get("start_location")
    if start_location:
        try:
            kwargs["start_location"] = (int(start_location[0]), int(start_location[1]))
        except Exception:
            pass

    enable_rendering = bool(cfg.get("enable_rendering", True))
    if render and enable_rendering:
        kwargs["render_mode"] = "rgb_array"

    plume = cfg.get("plume")
    if isinstance(plume, str) and plume.strip().lower() == "movie":
        kwargs["plume"] = "movie"

        movie_group = raw_cfg.get("movie")
        movie_cfg = movie_group if isinstance(movie_group, dict) else {}

        movie_path = cfg.get("movie_path")
        if movie_path is None:
            movie_path = movie_cfg.get("path")
        if isinstance(movie_path, str) and movie_path.strip():
            kwargs["movie_path"] = movie_path

        movie_dataset_id = cfg.get("movie_dataset_id")
        if movie_dataset_id is None:
            movie_dataset_id = movie_cfg.get("dataset_id")
        if isinstance(movie_dataset_id, str) and movie_dataset_id.strip():
            kwargs["movie_dataset_id"] = movie_dataset_id

        movie_auto_download = cfg.get("movie_auto_download")
        if movie_auto_download is None:
            movie_auto_download = movie_cfg.get("auto_download")
        if movie_auto_download is not None:
            kwargs["movie_auto_download"] = bool(movie_auto_download)

        movie_cache_root = cfg.get("movie_cache_root")
        if movie_cache_root is None:
            movie_cache_root = movie_cfg.get("cache_root")
        if isinstance(movie_cache_root, str) and movie_cache_root.strip():
            kwargs["movie_cache_root"] = movie_cache_root

        movie_fps = cfg.get("movie_fps")
        if movie_fps is None:
            movie_fps = movie_cfg.get("fps")
        if movie_fps is not None:
            try:
                kwargs["movie_fps"] = float(movie_fps)
            except Exception:
                pass

        movie_step_policy = cfg.get("movie_step_policy")
        if movie_step_policy is None:
            movie_step_policy = movie_cfg.get("step_policy")
        if isinstance(movie_step_policy, str) and movie_step_policy.strip():
            kwargs["movie_step_policy"] = movie_step_policy

        movie_h5_dataset = cfg.get("movie_h5_dataset")
        if movie_h5_dataset is None:
            movie_h5_dataset = movie_cfg.get("h5_dataset")
        if isinstance(movie_h5_dataset, str) and movie_h5_dataset.strip():
            kwargs["movie_h5_dataset"] = movie_h5_dataset

        movie_normalize = cfg.get("movie_normalize")
        if movie_normalize is None:
            movie_normalize = movie_cfg.get("normalize")
        if movie_normalize is not None:
            kwargs["movie_normalize"] = movie_normalize

        movie_chunks = cfg.get("movie_chunks")
        if movie_chunks is None:
            movie_chunks = movie_cfg.get("chunks")
        if isinstance(movie_chunks, str) and movie_chunks.strip().lower() == "none":
            movie_chunks = None
        if movie_chunks is not None:
            kwargs["movie_chunks"] = movie_chunks

        pixel_to_grid = cfg.get("movie_pixel_to_grid")
        if pixel_to_grid is None:
            pixel_to_grid = movie_cfg.get("pixel_to_grid")
        if isinstance(pixel_to_grid, (list, tuple)) and len(pixel_to_grid) == 2:
            try:
                kwargs["movie_pixel_to_grid"] = (
                    float(pixel_to_grid[0]),
                    float(pixel_to_grid[1]),
                )
            except Exception:
                pass

        origin = cfg.get("movie_origin")
        if origin is None:
            origin = movie_cfg.get("origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            try:
                kwargs["movie_origin"] = (float(origin[0]), float(origin[1]))
            except Exception:
                pass

        extent = cfg.get("movie_extent")
        if extent is None:
            extent = movie_cfg.get("extent")
        if isinstance(extent, (list, tuple)) and len(extent) == 2:
            try:
                kwargs["movie_extent"] = (float(extent[0]), float(extent[1]))
            except Exception:
                pass

    defaults = {
        "action_type": "oriented",
        "observation_type": "concentration",
        "reward_type": "step_penalty",
    }
    return {**defaults, **kwargs}


class ReplayDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray)
    episode_finished = QtCore.Signal()
    step_done = QtCore.Signal(object)  # emits runner.StepEvent
    action_space_changed = QtCore.Signal(list)  # emits list[str]
    provider_mux_changed = QtCore.Signal(object)  # emits ProviderMux
    run_meta_changed = QtCore.Signal(int, object)  # (seed, start_xy tuple)
    timeline_changed = QtCore.Signal(int, int)  # (current_step, total_steps)

    def __init__(self) -> None:
        super().__init__()
        self._artifacts: Optional[ReplayArtifacts] = None
        self._engine: Optional[ReplayEngine] = None
        self._iter = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._running = False
        self._current_index: int = -1
        self._total_steps: int = 0
        self._mux = None
        self._probe_env = None
        self._run_dir: Optional[Path] = None
        self._last_event = None
        self._recorded_max_steps: Optional[int] = None
        self._cache: list[Any] = []
        self._episode_offsets: list[int] = []

    def load_run(self, run_dir: str | Path) -> None:
        artifacts = load_replay_artifacts(run_dir)
        self._artifacts = artifacts
        self._run_dir = Path(run_dir)
        self._recorded_max_steps = _infer_max_steps_from_artifacts(artifacts)
        self._engine = ReplayEngine(
            artifacts,
            env_factory=lambda meta, render: pns.make_env(
                **_env_kwargs_from_meta(
                    meta, render=render, recorded_max_steps=self._recorded_max_steps
                )
            ),
        )
        self._iter = None
        self._running = False
        self._current_index = -1
        self._total_steps = len(artifacts.steps)
        self._last_event = None
        self._cache = []
        self._episode_offsets = self._compute_episode_offsets(artifacts.steps)
        self._build_mux()
        self._emit_timeline()
        # Prime the first frame/event for immediate feedback
        self.seek_to(0, auto_emit=True)

    def is_loaded(self) -> bool:
        return self._artifacts is not None and self._engine is not None

    def total_steps(self) -> int:
        return self._total_steps

    def total_episodes(self) -> int:
        return len(self._episode_offsets)

    def current_index(self) -> int:
        return self._current_index

    def current_episode_index(self) -> int:
        return self._episode_for_step(self._current_index)

    def is_running(self) -> bool:
        return bool(self._running)

    def start(self, interval_ms: int = 50) -> None:
        if not self.is_loaded():
            return
        if self._total_steps <= 0:
            return
        if self._iter is None:
            next_idx = 0 if self._current_index < 0 else self._current_index + 1
            if next_idx >= self._total_steps:
                next_idx = 0
            self._reset_iterator(next_idx)
        self._timer.start(max(1, int(interval_ms)))
        self._running = True

    def pause(self) -> None:
        self._timer.stop()
        self._running = False

    def step_once(self) -> None:
        self._emit_next()

    def seek_to_episode(self, episode_index: int, *, auto_emit: bool = True) -> None:
        if not self._episode_offsets:
            return
        idx = max(0, min(int(episode_index), len(self._episode_offsets) - 1))
        target = self._episode_offsets[idx]
        self.seek_to(target, auto_emit=auto_emit)

    def seek_to(self, step_index: int, *, auto_emit: bool = True) -> None:
        if not self.is_loaded():
            return
        if self._total_steps == 0:
            return
        idx = int(step_index)
        idx = max(0, min(idx, self._total_steps - 1))
        was_running = self.is_running()
        if was_running:
            self.pause()
        self._reset_iterator(idx)
        if auto_emit:
            self._emit_next()
        if was_running:
            self.start(int(self._timer.interval()) or 50)

    def get_action_names(self) -> list[str]:
        if self._mux is not None:
            try:
                return self._mux.get_action_names()
            except Exception:
                return []
        return []

    def get_grid_size(self) -> tuple[int, int]:
        if self._artifacts is None:
            return (0, 0)
        cfg = self._artifacts.run_meta.env_config or {}
        grid = cfg.get("grid_size")
        try:
            if grid:
                return int(grid[0]), int(grid[1])
        except Exception:
            pass
        return (0, 0)

    def last_event(self):
        return self._last_event

    def _emit_timeline(self) -> None:
        self.timeline_changed.emit(self._current_index, self._total_steps)

    def _reset_iterator(self, start_step: int) -> None:
        if self._engine is None:
            return
        start = max(0, int(start_step))
        self._iter = self._cached_iter(start)
        self._current_index = start - 1
        self._emit_timeline()

    def _cached_iter(self, start_step: int):
        idx = max(0, int(start_step))
        # Reuse cached events for already replayed steps
        while idx < len(self._cache):
            yield self._cache[idx]
            idx += 1

        if self._engine is None:
            return

        engine_iter = self._engine.iter_events(
            render=True, start_step=idx, validate=False
        )
        for ev in engine_iter:
            self._cache.append(ev)
            yield ev

    def _emit_next(self) -> None:
        if self._iter is None:
            return
        try:
            ev = next(self._iter)
        except StopIteration:
            self.episode_finished.emit()
            self.pause()
            self._iter = None
            self._emit_timeline()
            return
        except Exception:
            self.pause()
            raise

        self._current_index += 1
        self._last_event = ev

        if isinstance(ev.frame, np.ndarray):
            self.frame_ready.emit(ev.frame)
        self.step_done.emit(ev)
        if ev.terminated or ev.truncated:
            self.episode_finished.emit()

        self._emit_timeline()

    def _on_tick(self) -> None:
        self._emit_next()

    def _first_seed(self) -> Optional[int]:
        if self._artifacts is None:
            return None
        meta = self._artifacts.run_meta
        if meta.episode_seeds:
            return int(meta.episode_seeds[0])
        if meta.base_seed is not None:
            return int(meta.base_seed)
        if self._artifacts.steps and self._artifacts.steps[0].seed is not None:
            return int(self._artifacts.steps[0].seed)
        return None

    def _build_mux(self) -> None:
        # Tear down previous probe env
        try:
            if self._probe_env is not None and hasattr(self._probe_env, "close"):
                self._probe_env.close()
        except Exception:
            pass
        self._probe_env = None
        self._mux = None

        if self._artifacts is None:
            self.action_space_changed.emit([])
            return

        try:
            env = pns.make_env(
                **_env_kwargs_from_meta(
                    self._artifacts.run_meta,
                    render=False,
                    recorded_max_steps=self._recorded_max_steps,
                )
            )
            self._probe_env = env
            seed = self._first_seed()
            try:
                env.reset(seed=seed)
            except Exception:
                pass
            from plume_nav_debugger.odc.mux import ProviderMux

            self._mux = ProviderMux(env, None)
            self.provider_mux_changed.emit(self._mux)
            self.action_space_changed.emit(self.get_action_names())
        except Exception:
            self._mux = None
            self._probe_env = None
            self.action_space_changed.emit([])

        start_xy = self._start_xy()
        seed = self._first_seed() or -1
        if start_xy is not None:
            self.run_meta_changed.emit(int(seed), start_xy)
        else:
            self.run_meta_changed.emit(int(seed), None)

    def _start_xy(self) -> Optional[tuple[int, int]]:
        if self._artifacts is None or not self._artifacts.steps:
            return None
        try:
            pos = self._artifacts.steps[0].agent_position
            return int(pos.x), int(pos.y)
        except Exception:
            return None

    def _episode_for_step(self, step_idx: int) -> int:
        if step_idx < 0 or not self._episode_offsets:
            return 0
        cur = 0
        for idx, start in enumerate(self._episode_offsets):
            if start > step_idx:
                break
            cur = idx
        return cur

    def _compute_episode_offsets(self, steps) -> list[int]:
        offsets: list[int] = []
        last_id = None
        last_step = None
        for idx, rec in enumerate(steps):
            is_new = False
            try:
                rec_id = rec.episode_id
                rec_step = int(rec.step)
            except Exception:
                rec_id = None
                rec_step = None
            if last_id is None:
                is_new = True
            elif rec_id != last_id:
                is_new = True
            elif (
                last_step is not None and rec_step is not None and rec_step <= last_step
            ):
                is_new = True
            if is_new:
                offsets.append(idx)
            last_id = rec_id
            last_step = rec_step
        return offsets


__all__ = ["ReplayDriver", "ReplayLoadError"]
