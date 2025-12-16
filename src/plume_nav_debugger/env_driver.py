from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


@dataclass
class DebuggerConfig:
    grid_size: tuple[int, int] = (64, 64)
    goal_radius: float = 1.0
    plume_sigma: float = 20.0
    max_steps: int = 500
    action_type: str = "oriented"
    seed: Optional[int] = 123
    start_location: Optional[tuple[int, int]] = None
    plume: str = "static"
    movie_path: Optional[str] = None
    movie_dataset_id: Optional[str] = None
    movie_auto_download: bool = False
    movie_cache_root: Optional[str] = None
    movie_fps: Optional[float] = None
    movie_step_policy: str = "wrap"
    movie_h5_dataset: Optional[str] = None
    movie_normalize: Optional[str] = None
    movie_chunks: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DebuggerConfig":
        def _get_str(key: str) -> Optional[str]:
            raw = os.environ.get(key)
            if raw is None:
                return None
            val = str(raw).strip()
            return val if val else None

        def _get_int(key: str) -> Optional[int]:
            raw = os.environ.get(key)
            if raw is None:
                return None
            txt = str(raw).strip()
            if not txt:
                return None
            try:
                return int(txt)
            except Exception:
                return None

        def _get_bool(key: str, default: bool) -> bool:
            raw = os.environ.get(key)
            if raw is None:
                return bool(default)
            val = str(raw).strip().lower()
            return val in {"1", "true", "yes", "y", "on"}

        def _get_float(key: str) -> Optional[float]:
            raw = os.environ.get(key)
            if raw is None:
                return None
            txt = str(raw).strip()
            if not txt:
                return None
            try:
                return float(txt)
            except Exception:
                return None

        plume = (_get_str("PLUME_DEBUGGER_PLUME") or "static").strip().lower()
        if plume not in {"static", "movie"}:
            plume = "static"

        movie_chunks = _get_str("PLUME_DEBUGGER_MOVIE_CHUNKS")
        if movie_chunks is not None and movie_chunks.strip().lower() == "none":
            movie_chunks = None

        step_policy = _get_str("PLUME_DEBUGGER_MOVIE_STEP_POLICY")
        if step_policy is None:
            step_policy = "wrap"
        step_policy = step_policy.strip().lower()
        if step_policy not in {"wrap", "clamp"}:
            step_policy = "wrap"

        action_type = (
            (_get_str("PLUME_DEBUGGER_ACTION_TYPE") or "oriented").strip().lower()
        )
        if action_type not in {"discrete", "oriented", "run_tumble"}:
            action_type = "oriented"

        seed_val = _get_int("PLUME_DEBUGGER_SEED")
        max_steps_val = _get_int("PLUME_DEBUGGER_MAX_STEPS")

        return cls(
            action_type=action_type,
            seed=seed_val if seed_val is not None else cls.seed,
            max_steps=max_steps_val if max_steps_val is not None else cls.max_steps,
            plume=plume,
            movie_path=_get_str("PLUME_DEBUGGER_MOVIE_PATH"),
            movie_dataset_id=_get_str("PLUME_DEBUGGER_MOVIE_DATASET_ID"),
            movie_auto_download=_get_bool("PLUME_DEBUGGER_MOVIE_AUTO_DOWNLOAD", False),
            movie_cache_root=_get_str("PLUME_DEBUGGER_MOVIE_CACHE_ROOT"),
            movie_fps=_get_float("PLUME_DEBUGGER_MOVIE_FPS"),
            movie_step_policy=step_policy,
            movie_h5_dataset=_get_str("PLUME_DEBUGGER_MOVIE_H5_DATASET"),
            movie_normalize=_get_str("PLUME_DEBUGGER_MOVIE_NORMALIZE"),
            movie_chunks=movie_chunks,
        )


class EnvDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray)
    episode_finished = QtCore.Signal()
    step_done = QtCore.Signal(object)  # emits runner.StepEvent
    action_space_changed = QtCore.Signal(list)  # emits list[str] action names
    policy_changed = QtCore.Signal(object)  # emits base policy for probing
    provider_mux_changed = QtCore.Signal(object)  # emits ProviderMux for introspection
    run_meta_changed = QtCore.Signal(int, object)  # (seed, start_xy tuple)
    error_occurred = QtCore.Signal(str)

    def __init__(self, config: DebuggerConfig) -> None:
        super().__init__()
        self.config = config
        self._env = None
        self._policy = None  # base policy or callable
        self._controller = None  # ControllablePolicy wrapper
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._iter = None  # iterator from runner.stream
        self._running = False
        self._last_event = None
        self._episode_seed: Optional[int] = self.config.seed
        self._mux = None  # ProviderMux
        self._last_start_xy: Optional[tuple[int, int]] = None

    def _make_default_policy(self):
        # Fallback policy: sample from env action_space each step
        class _Sampler:
            def __init__(self, env):
                self._env = env

            def __call__(self, _obs):
                return self._env.action_space.sample()

        return _Sampler(self._env)

    def initialize(self) -> None:
        import plume_nav_sim as pns
        from plume_nav_sim.runner import runner

        from .controllable_policy import ControllablePolicy

        action_type = str(getattr(self.config, "action_type", "oriented") or "oriented")
        action_type = action_type.strip().lower()
        if action_type not in {"discrete", "oriented", "run_tumble"}:
            action_type = "oriented"
        self.config.action_type = action_type

        kwargs = dict(
            grid_size=self.config.grid_size,
            goal_radius=self.config.goal_radius,
            plume_sigma=self.config.plume_sigma,
            max_steps=self.config.max_steps,
            render_mode="rgb_array",
            action_type=action_type,
            observation_type="concentration",
            reward_type="step_penalty",
        )
        if self.config.start_location is not None:
            kwargs["start_location"] = tuple(self.config.start_location)

        if (
            str(getattr(self.config, "plume", "static")).lower() == "movie"
            or self.config.movie_path
            or self.config.movie_dataset_id
        ):
            kwargs["plume"] = "movie"
            if self.config.movie_path:
                kwargs["movie_path"] = self.config.movie_path
            if self.config.movie_dataset_id:
                kwargs["movie_dataset_id"] = self.config.movie_dataset_id
            kwargs["movie_auto_download"] = bool(self.config.movie_auto_download)
            if self.config.movie_cache_root:
                kwargs["movie_cache_root"] = self.config.movie_cache_root
            if self.config.movie_fps is not None:
                kwargs["movie_fps"] = float(self.config.movie_fps)
            if self.config.movie_step_policy:
                kwargs["movie_step_policy"] = str(self.config.movie_step_policy)
            if self.config.movie_h5_dataset:
                kwargs["movie_h5_dataset"] = self.config.movie_h5_dataset
            if self.config.movie_normalize is not None:
                kwargs["movie_normalize"] = self.config.movie_normalize
            if self.config.movie_chunks is not None:
                kwargs["movie_chunks"] = self.config.movie_chunks

        try:
            self._env = pns.make_env(**kwargs)
        except Exception as exc:
            self._env = None
            self._policy = None
            self._controller = None
            self._iter = None
            try:
                self.error_occurred.emit(f"Env init failed: {exc}")
            except Exception:
                pass
            return

        # Choose a policy if available; otherwise fallback to sampler
        if action_type == "oriented":
            try:
                from plume_nav_sim.policies import TemporalDerivativePolicy

                self._policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
                try:
                    self._policy.reset(seed=self.config.seed)
                except Exception:
                    pass
            except Exception:
                self._policy = self._make_default_policy()
        else:
            self._policy = self._make_default_policy()

        # Wrap with ControllablePolicy for manual override
        self._controller = ControllablePolicy(self._policy)

        # Persist the episode seed actually in use
        self._episode_seed = self.config.seed

        # Eagerly reset env and controller so we can show the initial frame
        try:
            try:
                self._controller.reset(seed=self._episode_seed)  # type: ignore[attr-defined]
            except Exception:
                pass
            _obs0, _info0 = self._env.reset(seed=self._episode_seed)
            self._update_start_from_info(self._episode_seed or -1, _info0)
        except Exception:
            pass

        # Emit an initial frame (pre-step) for immediate visual feedback
        self._emit_frame_now()

        # Prime generator for stepping using runner.stream (will reset again with same seed)
        self._iter = runner.stream(
            self._env,
            self._controller if self._controller is not None else self._policy,
            seed=self._episode_seed,
            render=True,
        )
        # Build ProviderMux and announce action names after env/policy ready
        try:
            from plume_nav_debugger.odc.mux import ProviderMux

            self._mux = ProviderMux(self._env, self._policy)
            self.provider_mux_changed.emit(self._mux)
            # Structured one-time warning if no provider detected
            try:
                import warnings as _warnings

                if not getattr(self, "_warned_no_provider", False):
                    if hasattr(self._mux, "has_provider") and not self._mux.has_provider():  # type: ignore[attr-defined]
                        _warnings.warn(
                            "plume_nav_debugger:no_provider_detected provider is required for action labels, distributions, and pipeline",
                            RuntimeWarning,
                        )
                        self._warned_no_provider = True
            except Exception:
                pass
        except Exception:
            self._mux = None
        self._emit_action_space_changed()
        try:
            self.policy_changed.emit(self._policy)
        except Exception:
            pass
        # Observation pipeline (informational for inspector)
        try:
            # Notify inspector once with the wrapper chain
            self._notify_pipeline()
        except Exception:
            pass

    def start(self, interval_ms: int = 50) -> None:
        if self._env is None:
            self.initialize()
        if self._env is None or self._iter is None:
            self._running = False
            return
        self._timer.start(max(1, int(interval_ms)))
        self._running = True

    def pause(self) -> None:
        self._timer.stop()
        self._running = False

    def close(self) -> None:
        try:
            self.pause()
        except Exception:
            pass
        try:
            if self._env is not None and hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass
        self._env = None
        self._iter = None
        self._policy = None
        self._controller = None
        self._mux = None

    def step_once(self) -> None:
        self._on_tick()

    def reset(self, seed: Optional[int] = None) -> None:
        if self._env is None:
            return
        # Recreate the stream iterator with a fresh seed
        from plume_nav_sim.runner import runner

        # Choose seed: use provided, else last episode seed, else config
        eff_seed = (
            int(seed)
            if seed is not None
            else (
                self._episode_seed
                if self._episode_seed is not None
                else self.config.seed
            )
        )
        self._episode_seed = eff_seed

        # Ensure controller stays active for manual overrides
        if self._controller is None and self._policy is not None:
            try:
                from .controllable_policy import ControllablePolicy

                self._controller = ControllablePolicy(self._policy)
            except Exception:
                self._controller = None

        # Eagerly reset env and controller before building stream so the frame shows start state
        try:
            try:
                if self._controller is not None and hasattr(self._controller, "reset"):
                    self._controller.reset(seed=eff_seed)  # type: ignore[attr-defined]
                elif self._policy is not None and hasattr(self._policy, "reset"):
                    self._policy.reset(seed=eff_seed)  # type: ignore[attr-defined]
            except Exception:
                pass
            _obs0, _info0 = self._env.reset(seed=eff_seed)
            self._update_start_from_info(eff_seed or -1, _info0)
        except Exception:
            pass

        # Show first frame after reset
        self._emit_frame_now()

        # Build fresh stream (will reset again with same seed, preserving determinism)
        self._iter = runner.stream(
            self._env,
            (
                (self._controller if self._controller is not None else self._policy)
                if self._policy is not None
                else self._make_default_policy()
            ),
            seed=eff_seed,
            render=True,
        )
        # Announce action space after reset (safe no-op for same env)
        self._emit_action_space_changed()

    def is_running(self) -> bool:
        return bool(self._running)

    def get_interval_ms(self) -> int:
        try:
            return int(self._timer.interval())
        except Exception:
            return 50

    def get_observation_pipeline_names(self) -> list[str]:
        try:
            if self._mux is not None:
                return self._mux.get_pipeline()
        except Exception:
            pass
        return []

    def get_grid_size(self) -> tuple[int, int]:
        try:
            gs = getattr(self._env, "grid_size", None)
            if gs is None:
                return self.config.grid_size
            # gs may be (w,h) tuple or object with width/height
            w = getattr(gs, "width", None) or int(gs[0])
            h = getattr(gs, "height", None) or int(gs[1])
            return int(w), int(h)
        except Exception:
            return self.config.grid_size

    def set_policy(
        self, policy: object, *, seed: Optional[int] = None, resume: bool = True
    ) -> None:
        """Swap the active policy and reinitialize the stream.

        If the environment isn't initialized yet, just store the policy.
        """
        was_running = self.is_running()
        if was_running:
            self.pause()

        self._policy = policy

        if self._env is None:
            # Will be wired during initialize()
            if was_running and resume:
                self.start(self.get_interval_ms())
            return

        from plume_nav_sim.compose.policy_loader import reset_policy_if_possible
        from plume_nav_sim.runner import runner

        from .controllable_policy import ControllablePolicy

        # Reset policy deterministically if provided
        eff_seed = (
            seed
            if seed is not None
            else (
                self._episode_seed
                if self._episode_seed is not None
                else self.config.seed
            )
        )
        reset_policy_if_possible(self._policy, seed=eff_seed)

        # Recreate iterator
        self._controller = ControllablePolicy(self._policy)
        # Persist episode seed for reproducibility
        self._episode_seed = eff_seed

        self._iter = runner.stream(
            self._env,
            (
                self._controller
                if self._controller is not None
                else self._make_default_policy()
            ),
            seed=eff_seed,
            render=True,
        )

        # Emit immediate frame
        self._emit_frame_now()

        if was_running and resume:
            self.start(self.get_interval_ms())
        # Rebuild ProviderMux and announce action space after policy change
        try:
            from plume_nav_debugger.odc.mux import ProviderMux

            self._mux = ProviderMux(self._env, self._policy)
            self.provider_mux_changed.emit(self._mux)
        except Exception:
            self._mux = None
        self._emit_action_space_changed()
        try:
            self.policy_changed.emit(self._policy)
        except Exception:
            pass

    # --- Environment (re)creation with start override -------------------
    def apply_start_override(self, x: int, y: int, enabled: bool) -> None:
        was_running = self.is_running()
        if was_running:
            self.pause()

        self._recreate_env(start_location=(int(x), int(y)) if enabled else None)

        if was_running:
            self.start(self.get_interval_ms())

    def _recreate_env(self, start_location: Optional[tuple[int, int]]) -> None:
        import plume_nav_sim as pns
        from plume_nav_sim.runner import runner

        # Close old env if present
        try:
            if self._env is not None and hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass

        # Build new env with same config, injecting start_location when provided
        action_type = str(getattr(self.config, "action_type", "oriented") or "oriented")
        action_type = action_type.strip().lower()
        if action_type not in {"discrete", "oriented", "run_tumble"}:
            action_type = "oriented"
        self.config.action_type = action_type

        kwargs = dict(
            grid_size=self.config.grid_size,
            goal_radius=self.config.goal_radius,
            plume_sigma=self.config.plume_sigma,
            max_steps=self.config.max_steps,
            render_mode="rgb_array",
            action_type=action_type,
            observation_type="concentration",
            reward_type="step_penalty",
        )
        if start_location is not None:
            kwargs["start_location"] = tuple(start_location)
            self.config.start_location = tuple(start_location)
        else:
            self.config.start_location = None

        if (
            str(getattr(self.config, "plume", "static")).lower() == "movie"
            or self.config.movie_path
            or self.config.movie_dataset_id
        ):
            kwargs["plume"] = "movie"
            if self.config.movie_path:
                kwargs["movie_path"] = self.config.movie_path
            if self.config.movie_dataset_id:
                kwargs["movie_dataset_id"] = self.config.movie_dataset_id
            kwargs["movie_auto_download"] = bool(self.config.movie_auto_download)
            if self.config.movie_cache_root:
                kwargs["movie_cache_root"] = self.config.movie_cache_root
            if self.config.movie_fps is not None:
                kwargs["movie_fps"] = float(self.config.movie_fps)
            if self.config.movie_step_policy:
                kwargs["movie_step_policy"] = str(self.config.movie_step_policy)
            if self.config.movie_h5_dataset:
                kwargs["movie_h5_dataset"] = self.config.movie_h5_dataset
            if self.config.movie_normalize is not None:
                kwargs["movie_normalize"] = self.config.movie_normalize
            if self.config.movie_chunks is not None:
                kwargs["movie_chunks"] = self.config.movie_chunks

        try:
            self._env = pns.make_env(**kwargs)
        except Exception as exc:
            self._env = None
            self._iter = None
            try:
                self.error_occurred.emit(f"Env recreation failed: {exc}")
            except Exception:
                pass
            return

        # Reset controller/base policy deterministically and env
        eff_seed = (
            self._episode_seed if self._episode_seed is not None else self.config.seed
        )
        try:
            try:
                if self._controller is not None and hasattr(self._controller, "reset"):
                    self._controller.reset(seed=eff_seed)  # type: ignore[attr-defined]
                elif self._policy is not None and hasattr(self._policy, "reset"):
                    self._policy.reset(seed=eff_seed)  # type: ignore[attr-defined]
            except Exception:
                pass
            _obs0, _info0 = self._env.reset(seed=eff_seed)
            self._update_start_from_info(eff_seed or -1, _info0)
        except Exception:
            pass

        # Emit first frame and rebuild stream
        self._emit_frame_now()
        self._iter = runner.stream(
            self._env,
            (
                (self._controller if self._controller is not None else self._policy)
                if self._policy is not None
                else self._make_default_policy()
            ),
            seed=eff_seed,
            render=True,
        )
        # Rebuild ProviderMux and emit names
        try:
            from plume_nav_debugger.odc.mux import ProviderMux

            self._mux = ProviderMux(self._env, self._policy)
            self.provider_mux_changed.emit(self._mux)
        except Exception:
            self._mux = None
        self._emit_action_space_changed()

    def set_manual_action(self, action: int, *, sticky: bool = False) -> None:
        try:
            if self._controller is not None and hasattr(
                self._controller, "set_next_action"
            ):
                self._controller.set_next_action(int(action), sticky=sticky)  # type: ignore[attr-defined]
        except Exception:
            pass

    def clear_sticky_action(self) -> None:
        try:
            if self._controller is not None and hasattr(
                self._controller, "clear_sticky"
            ):
                self._controller.clear_sticky()  # type: ignore[attr-defined]
        except Exception:
            pass

    def last_event(self):
        return self._last_event

    def get_action_names(self) -> list[str]:
        # Provider-only: no heuristics or numeric fallbacks
        if self._mux is not None:
            try:
                return self._mux.get_action_names()
            except Exception:
                return []
        return []

    def _emit_action_space_changed(self) -> None:
        try:
            names = self.get_action_names()
            self.action_space_changed.emit(names)
        except Exception:
            pass

    def _notify_pipeline(self) -> None:
        # Deprecated; pipeline is provided via ProviderMux and Inspector introspection
        return

    def _update_start_from_info(self, seed_val: int, info: object) -> None:
        try:
            xy = None
            if isinstance(info, dict):
                if "agent_xy" in info:
                    xy = info.get("agent_xy")
                elif "agent_position" in info:
                    xy = info.get("agent_position")
            if xy is not None:
                x, y = int(xy[0]), int(xy[1])
                self._last_start_xy = (x, y)
                self.run_meta_changed.emit(int(seed_val), (x, y))
        except Exception:
            pass

    def _emit_frame_now(self) -> None:
        try:
            frame = None
            try:
                frame = self._env.render()
            except TypeError:
                try:
                    frame = self._env.render(mode="rgb_array")
                except Exception:
                    frame = None
            else:
                if not isinstance(frame, np.ndarray):
                    try:
                        frame = self._env.render(mode="rgb_array")
                    except Exception:
                        frame = None
            if isinstance(frame, np.ndarray):
                self.frame_ready.emit(frame)
        except Exception:
            pass

    @QtCore.Slot()
    def _on_tick(self) -> None:
        if self._iter is None:
            return
        try:
            ev = next(self._iter)
            # emit frame and step event for UI consumers
            if isinstance(ev.frame, np.ndarray):
                self.frame_ready.emit(ev.frame)
            self._last_event = ev
            self.step_done.emit(ev)
            if ev.terminated or ev.truncated:
                self.episode_finished.emit()
                self.pause()
        except StopIteration:
            self.episode_finished.emit()
            self.pause()
        except Exception:
            self.pause()
            raise
