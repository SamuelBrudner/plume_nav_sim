from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Optional

try:
    from PySide6 import QtCore
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env."
    ) from e

try:
    from plume_nav_sim.data_capture.replay import ReplayEngine, ReplayLoadError
    from plume_nav_sim.data_capture.replay import ReplayStepEvent as _ReplayStepEvent
    from plume_nav_sim.data_capture.replay import load_replay_engine
except Exception as e:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "Replay support requires plume_nav_sim.data_capture.replay to be importable."
    ) from e


class ReplayDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(object)
    timeline_changed = QtCore.Signal(int, int)
    step_done = QtCore.Signal(object)
    episode_finished = QtCore.Signal()

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

    def load_run(self, run_dir: str | Path) -> None:
        run_path = Path(run_dir)
        self._engine = load_replay_engine(str(run_path))
        self._run_dir = run_path

        # Materialize events once so seeking and reverse stepping is trivial.
        self._events = list(self._engine.iter_events(start_index=0, render=True))
        self._index = 0
        self._running = False
        self._timer.stop()

        self._episode_starts = []
        try:
            # Derive episode boundaries from recorded episode_id transitions
            last_id: str | None = None
            for i, rec in enumerate(self._engine._artifacts.steps):  # type: ignore[attr-defined]
                if last_id is None or rec.episode_id != last_id:
                    self._episode_starts.append(i)
                    last_id = rec.episode_id
        except Exception:
            self._episode_starts = [0] if self._events else []

        self.seek_to(0, auto_emit=True)

    def is_loaded(self) -> bool:
        return self._engine is not None and bool(self._events)

    def total_steps(self) -> int:
        return len(self._events)

    def total_episodes(self) -> int:
        if self._engine is None:
            return 0
        try:
            return self._engine.total_episodes()
        except Exception:
            return 0

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
        if self._engine is None:
            return (0, 0)
        cfg = getattr(self._engine.run_meta, "env_config", None)
        if isinstance(cfg, dict):
            grid = cfg.get("grid_size")
            if isinstance(grid, (tuple, list)) and len(grid) == 2:
                try:
                    return int(grid[0]), int(grid[1])
                except Exception:
                    return (0, 0)
        return (0, 0)

    def get_overlay_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self._engine is None:
            return ctx
        cfg = getattr(self._engine.run_meta, "env_config", None)
        if isinstance(cfg, dict):
            with contextlib.suppress(Exception):
                if "source_location" in cfg:
                    ctx["source_xy"] = cfg.get("source_location")
            with contextlib.suppress(Exception):
                if "goal_radius" in cfg:
                    ctx["goal_radius"] = cfg.get("goal_radius")
        return ctx

    def last_event(self):
        return self._last_event

    def get_resolved_env_kwargs(self, *, render: bool = False) -> dict[str, Any]:
        _ = render
        return {}

    def get_resolved_replay_config(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        return {
            "run_dir": str(self._run_dir) if self._run_dir is not None else None,
            "env_config": getattr(self._engine.run_meta, "env_config", None),
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
