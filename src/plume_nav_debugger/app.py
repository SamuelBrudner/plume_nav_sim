from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
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
    seed: Optional[int] = 123
    start_location: Optional[tuple[int, int]] = None
    strict_provider_only: bool = True


class EnvDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray)
    episode_finished = QtCore.Signal()
    step_done = QtCore.Signal(object)  # emits runner.StepEvent
    action_space_changed = QtCore.Signal(list)  # emits list[str] action names
    policy_changed = QtCore.Signal(object)  # emits base policy for probing
    provider_mux_changed = QtCore.Signal(object)  # emits ProviderMux for introspection
    run_meta_changed = QtCore.Signal(int, object)  # (seed, start_xy tuple)
    provider_mux_changed = QtCore.Signal(object)  # emits ProviderMux for introspection

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
        self._mux = None  # ProviderMux

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

        kwargs = dict(
            grid_size=self.config.grid_size,
            goal_radius=self.config.goal_radius,
            plume_sigma=self.config.plume_sigma,
            max_steps=self.config.max_steps,
            render_mode="rgb_array",
            action_type="oriented",
            observation_type="concentration",
            reward_type="step_penalty",
        )
        if self.config.start_location is not None:
            kwargs["start_location"] = tuple(self.config.start_location)
        self._env = pns.make_env(**kwargs)

        # Choose a policy if available; otherwise fallback to sampler
        try:
            from plume_nav_sim.policies import TemporalDerivativePolicy

            self._policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
            try:
                self._policy.reset(seed=self.config.seed)
            except Exception:
                pass
        except Exception:
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
        self._timer.start(max(1, int(interval_ms)))
        self._running = True

    def pause(self) -> None:
        self._timer.stop()
        self._running = False

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
        kwargs = dict(
            grid_size=self.config.grid_size,
            goal_radius=self.config.goal_radius,
            plume_sigma=self.config.plume_sigma,
            max_steps=self.config.max_steps,
            render_mode="rgb_array",
            action_type="oriented",
            observation_type="concentration",
            reward_type="step_penalty",
        )
        if start_location is not None:
            kwargs["start_location"] = tuple(start_location)
            self.config.start_location = tuple(start_location)
        else:
            self.config.start_location = None

        self._env = pns.make_env(**kwargs)

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
        # Prefer ProviderMux if available
        if self._mux is not None:
            try:
                return self._mux.get_action_names()
            except Exception:
                pass
        # Strict mode: avoid heuristic fallbacks
        if getattr(self, "config", None) is not None and getattr(
            self.config, "strict_provider_only", False
        ):
            return []
        # Non-strict fallback numeric labels
        try:
            n = getattr(getattr(self._env, "action_space", None), "n", 0) or 0
            n = int(n) if isinstance(n, (int, np.integer)) else 0
        except Exception:
            n = 0
        return [str(i) for i in range(max(0, n))]

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


class FrameView(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 320)
        self._last_image = None

    @QtCore.Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        # Expect HxWx3 uint8
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return
        h, w, c = frame.shape
        if c != 3:
            return
        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img.copy())
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio))
        self._last_image = pix

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_image is not None:
            self.setPixmap(
                self._last_image.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            )


class ControlBar(QtWidgets.QWidget):
    start = QtCore.Signal()
    pause = QtCore.Signal()
    step = QtCore.Signal()
    reset = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
        # Speed control
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(5, 2000)
        self.interval_spin.setSingleStep(5)
        self.interval_spin.setValue(50)
        # Policy selection
        self.policy_combo = QtWidgets.QComboBox()
        self.policy_combo.addItems(
            [
                "Greedy TD (bacterial)",
                "Stochastic TD",
                "Deterministic TD",
                "Random Sampler",
            ]
        )
        self.custom_policy_edit = QtWidgets.QLineEdit()
        self.custom_policy_edit.setPlaceholderText("custom.module:ClassOrCallable")
        self.custom_load_btn = QtWidgets.QPushButton("Load")
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.step_btn)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Policy:"))
        layout.addWidget(self.policy_combo)
        layout.addWidget(self.custom_policy_edit)
        layout.addWidget(self.custom_load_btn)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Interval (ms):"))
        layout.addWidget(self.interval_spin)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Seed:"))
        layout.addWidget(self.seed_edit)
        layout.addWidget(self.reset_btn)

        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.step_btn.clicked.connect(self.step)
        self.reset_btn.clicked.connect(self._emit_reset)
        # Expose additional signals
        # Note: MainWindow will connect signals directly to handlers

    @QtCore.Slot()
    def _emit_reset(self) -> None:
        text = self.seed_edit.text().strip()
        # If seed field is empty or not an int, emit a sentinel -1 and let driver reuse last seed
        if text.isdigit():
            self.reset.emit(int(text))
        else:
            # Use -1 to indicate "reuse last episode seed"
            self.reset.emit(-1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plume Nav Debugger (MVP)")
        self.resize(800, 700)

        self.frame_view = FrameView()
        self.controls = ControlBar()
        self.inspector = InspectorWidget()
        self._total_reward: float = 0.0

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.addWidget(self.frame_view, stretch=1)
        vbox.addWidget(self.controls, stretch=0)
        self.setCentralWidget(central)

        # Dockable inspector window (can float or dock)
        self.inspector_dock = QtWidgets.QDockWidget("Inspector", self)
        self.inspector_dock.setObjectName("InspectorDock")
        self.inspector_dock.setWidget(self.inspector)
        self.inspector_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.inspector_dock)

        # Menu to toggle inspector visibility
        view_menu = self.menuBar().addMenu("View")
        self.action_toggle_inspector = QtGui.QAction("Inspector", self, checkable=True)
        self.action_toggle_inspector.setChecked(True)
        self.action_toggle_inspector.toggled.connect(self.inspector_dock.setVisible)
        self.inspector_dock.visibilityChanged.connect(
            self.action_toggle_inspector.setChecked
        )
        view_menu.addAction(self.action_toggle_inspector)
        edit_menu = self.menuBar().addMenu("Edit")
        prefs_action = QtGui.QAction("Preferences…", self)
        prefs_action.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_action)

        # Restore UI layout/state
        self._restore_ui_state()

        # Load preferences and configure driver
        self.prefs = DebuggerPreferences.initial_load()
        cfg = DebuggerConfig()
        cfg.strict_provider_only = bool(self.prefs.strict_provider_only)
        self.driver = EnvDriver(cfg)
        self.driver.frame_ready.connect(self.frame_view.update_frame)
        self.driver.step_done.connect(self._on_step_event)
        self.driver.episode_finished.connect(self._on_episode_finished)
        self.driver.step_done.connect(self.inspector.on_step_event)
        self.driver.action_space_changed.connect(self.inspector.on_action_names)
        self.driver.policy_changed.connect(self.inspector.on_policy_changed)
        self.driver.provider_mux_changed.connect(self.inspector.on_mux_changed)
        self.driver.run_meta_changed.connect(self._on_run_meta)
        self.controls.start.connect(self._on_start_clicked)
        self.controls.pause.connect(self.driver.pause)
        self.controls.step.connect(self.driver.step_once)
        self.controls.reset.connect(self._on_reset_clicked)
        self.controls.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.controls.policy_combo.currentIndexChanged.connect(self._on_policy_selected)
        self.controls.custom_load_btn.clicked.connect(self._on_custom_policy_load)

        # Status bar showing step/total reward/flags and run meta
        self._status = QtWidgets.QLabel("ready")
        self._run_meta = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._status, 1)
        self.statusBar().addPermanentWidget(self._run_meta, 0)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Initialize once to show first frame
        QtCore.QTimer.singleShot(0, self.driver.initialize)
        # Also refresh action names shortly after init
        QtCore.QTimer.singleShot(
            50, lambda: self.inspector.on_action_names(self.driver.get_action_names())
        )
        # Initialize start override ranges based on grid size
        QtCore.QTimer.singleShot(
            60, lambda: self.inspector.set_grid_size(*self.driver.get_grid_size())
        )
        QtCore.QTimer.singleShot(
            70,
            lambda: self.inspector.set_observation_pipeline_from_env(
                getattr(self.driver, "_env", None)
            ),
        )
        # Configure inspector strictness from driver config
        QtCore.QTimer.singleShot(
            80,
            lambda: self.inspector.set_strict_provider_only(
                getattr(self.driver.config, "strict_provider_only", True)
            ),
        )
        # Apply inspector display prefs
        QtCore.QTimer.singleShot(90, lambda: self._apply_inspector_prefs())
        # Apply theme
        QtCore.QTimer.singleShot(100, lambda: self._apply_theme(self.prefs.theme))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            from PySide6 import QtCore as _QtCore

            settings = _QtCore.QSettings("plume-nav-sim", "Debugger")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_ui_state(self) -> None:
        try:
            from PySide6 import QtCore as _QtCore

            settings = _QtCore.QSettings("plume-nav-sim", "Debugger")
            geom = settings.value("geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            st = settings.value("windowState")
            if st is not None:
                self.restoreState(st)
        except Exception:
            pass

    # UI wiring --------------------------------------------------------------
    @QtCore.Slot()
    def _on_start_clicked(self) -> None:
        interval = int(self.controls.interval_spin.value())
        self.driver.start(interval)

    @QtCore.Slot(int)
    def _on_interval_changed(self, value: int) -> None:
        if self.driver.is_running():
            self.driver.start(int(value))

    def _setup_shortcuts(self) -> None:
        # Space: toggle start/pause
        space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        space.activated.connect(self._toggle_run)

        # N: step once
        step = QtGui.QShortcut(QtGui.QKeySequence("N"), self)
        step.activated.connect(self.driver.step_once)

        # R: reset with seed from edit (delegates to existing handler)
        reset = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        reset.activated.connect(self.controls._emit_reset)  # type: ignore[attr-defined]

    @QtCore.Slot()
    def _toggle_run(self) -> None:
        if self.driver.is_running():
            self.driver.pause()
        else:
            self._on_start_clicked()

    @QtCore.Slot(object)
    def _on_step_event(self, ev) -> None:
        try:
            self._total_reward += float(getattr(ev, "reward", 0.0))
        except Exception:
            pass
        t = getattr(ev, "t", "?")
        term = bool(getattr(ev, "terminated", False))
        trunc = bool(getattr(ev, "truncated", False))
        self._status.setText(
            f"t={t}  reward_total={self._total_reward:.2f}  term={term} trunc={trunc}"
        )

    @QtCore.Slot()
    def _on_episode_finished(self) -> None:
        self.statusBar().showMessage("Episode finished", 3000)
        # keep last metrics displayed; allow restart

    @QtCore.Slot(int)
    def _on_reset_clicked(self, seed: int) -> None:
        self._total_reward = 0.0
        self._status.setText("t=0  reward_total=0.00  term=False trunc=False")
        # Interpret -1 as "reuse last episode seed"
        self.driver.reset(None if seed == -1 else seed)

    @QtCore.Slot(int, object)
    def _on_run_meta(self, seed_val: int, start_xy: object) -> None:
        try:
            if isinstance(start_xy, tuple) and len(start_xy) == 2:
                x, y = int(start_xy[0]), int(start_xy[1])
                self._run_meta.setText(f"seed={seed_val} start=({x},{y})")
            else:
                self._run_meta.setText(f"seed={seed_val}")
        except Exception:
            self._run_meta.setText("")

    # Preferences ------------------------------------------------------------
    def _open_preferences(self) -> None:
        dlg = PreferencesDialog(self.prefs, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.prefs = dlg.get_prefs()
            self.prefs.to_qsettings()
            try:
                self.prefs.save_json_file()
            except Exception:
                pass
            # Apply to driver and inspector
            self.driver.config.strict_provider_only = bool(
                self.prefs.strict_provider_only
            )
            self.inspector.set_strict_provider_only(self.prefs.strict_provider_only)
            self._apply_inspector_prefs()
            self._apply_theme(self.prefs.theme)
            try:
                self.controls.interval_spin.setValue(
                    int(self.prefs.default_interval_ms)
                )
            except Exception:
                pass

    def _apply_inspector_prefs(self) -> None:
        try:
            self.inspector.set_show_pipeline(self.prefs.show_pipeline)
            self.inspector.set_show_preview(self.prefs.show_preview)
            self.inspector.set_show_sparkline(self.prefs.show_sparkline)
        except Exception:
            pass

    def _apply_theme(self, theme: str) -> None:
        try:
            if str(theme).lower() == "dark":
                self._enable_dark_palette()
            else:
                QtWidgets.QApplication.setPalette(
                    QtWidgets.QApplication.style().standardPalette()
                )
        except Exception:
            pass

    def _enable_dark_palette(self) -> None:
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        QtWidgets.QApplication.setPalette(pal)


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(
        self, prefs: DebuggerPreferences, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._prefs = DebuggerPreferences(**vars(prefs))
        layout = QtWidgets.QFormLayout(self)
        # Strict mode
        self.chk_strict = QtWidgets.QCheckBox("Strict provider-only mode")
        self.chk_strict.setChecked(self._prefs.strict_provider_only)
        # Inspector toggles
        self.chk_pipeline = QtWidgets.QCheckBox("Show pipeline")
        self.chk_pipeline.setChecked(self._prefs.show_pipeline)
        self.chk_preview = QtWidgets.QCheckBox("Show observation preview")
        self.chk_preview.setChecked(self._prefs.show_preview)
        self.chk_spark = QtWidgets.QCheckBox("Show sparkline for vectors")
        self.chk_spark.setChecked(self._prefs.show_sparkline)
        # Interval
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 5000)
        self.spin_interval.setValue(int(self._prefs.default_interval_ms))
        # Theme
        self.combo_theme = QtWidgets.QComboBox()
        self.combo_theme.addItems(["light", "dark"])
        idx = 0 if str(self._prefs.theme).lower() != "dark" else 1
        self.combo_theme.setCurrentIndex(idx)
        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout.addRow(self.chk_strict)
        layout.addRow(self.chk_pipeline)
        layout.addRow(self.chk_preview)
        layout.addRow(self.chk_spark)
        layout.addRow("Default interval (ms)", self.spin_interval)
        layout.addRow("Theme", self.combo_theme)
        layout.addRow(btns)

    def get_prefs(self) -> DebuggerPreferences:
        self._prefs.strict_provider_only = bool(self.chk_strict.isChecked())
        self._prefs.show_pipeline = bool(self.chk_pipeline.isChecked())
        self._prefs.show_preview = bool(self.chk_preview.isChecked())
        self._prefs.show_sparkline = bool(self.chk_spark.isChecked())
        self._prefs.default_interval_ms = int(self.spin_interval.value())
        self._prefs.theme = str(self.combo_theme.currentText())
        return self._prefs

    @QtCore.Slot(int)
    def _on_policy_selected(self, idx: int) -> None:
        # Map combo index to built-in policies
        label = self.controls.policy_combo.itemText(idx)
        try:
            if label == "Greedy TD (bacterial)":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(
                    eps=0.0, eps_after_turn=0.0, uniform_random_on_non_increase=True
                )
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage(
                    "Loaded Greedy TD (bacterial) policy", 1500
                )
            elif label == "Deterministic TD":
                from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

                policy = TemporalDerivativeDeterministicPolicy()
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Deterministic TD policy", 1500)
            elif label == "Stochastic TD":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(eps=0.05, eps_after_turn=0.0)
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Stochastic TD policy", 1500)
            elif label == "Random Sampler":
                # Use the driver's default random sampler
                policy = (
                    self.driver._make_default_policy()
                )  # noqa: SLF001 - internal is fine for UI wiring
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Random Sampler policy", 1500)
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Policy load failed: {e}", 3000)

    @QtCore.Slot()
    def _on_custom_policy_load(self) -> None:
        from plume_nav_sim.compose.policy_loader import load_policy

        spec = self.controls.custom_policy_edit.text().strip()
        if not spec:
            self.statusBar().showMessage("Enter custom policy as module:Attr", 2500)
            return
        try:
            loaded = load_policy(spec)
            self.driver.set_policy(loaded.obj, seed=self._current_seed_value())
            self.statusBar().showMessage(f"Loaded custom policy: {loaded.spec}", 2000)
            # Names may change with new policy/env combination
            self.inspector.on_action_names(self.driver.get_action_names())
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Custom policy load failed: {e}", 4000)

    def _current_seed_value(self) -> Optional[int]:
        text = self.controls.seed_edit.text().strip()
        return int(text) if text.isdigit() else self.driver.config.seed

    # No inspector-originating control handlers; inspector is read-only


class ActionPanelWidget(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        # Policy insight
        self.expected_action_label = QtWidgets.QLabel("expected action: -")
        self.distribution_label = QtWidgets.QLabel("distribution: N/A")
        self.source_label = QtWidgets.QLabel("source: heuristic")
        # Layout
        layout.addWidget(self.expected_action_label, 0, 0)
        layout.addWidget(self.distribution_label, 0, 1)
        layout.addWidget(self.source_label, 1, 0, 1, 2)

        self._action_names: list[str] = []

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        # Deprecated direct update; InspectorWidget now orchestrates via models
        pass

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        try:
            self._action_names = list(names)
            self.action_combo.clear()
            if self._action_names:
                self.action_combo.addItems(self._action_names)
        except Exception:
            pass

    def set_grid_size(self, w: int, h: int) -> None:
        pass


class ObservationPanelWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        self.obs_shape = QtWidgets.QLabel("shape: -")
        self.obs_stats = QtWidgets.QLabel("min/mean/max: -/-/-")
        self.pipeline_label = QtWidgets.QLabel("")
        self.preview_label = QtWidgets.QLabel("")
        self.sparkline = _SparklineWidget()
        self._show_pipeline = True
        self._show_preview = True
        self._show_sparkline = True
        layout.addWidget(QtWidgets.QLabel("Policy Observation"), 0, 0)
        layout.addWidget(self.obs_shape, 0, 1)
        layout.addWidget(self.obs_stats, 0, 2)
        layout.addWidget(self.pipeline_label, 1, 0, 1, 3)
        layout.addWidget(self.preview_label, 2, 0, 1, 1)
        layout.addWidget(self.sparkline, 2, 1, 1, 2)

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        # Deprecated direct update; InspectorWidget now orchestrates via models
        pass

    # Display toggles
    def set_show_pipeline(self, flag: bool) -> None:
        self._show_pipeline = bool(flag)
        self.pipeline_label.setVisible(self._show_pipeline)

    def set_show_preview(self, flag: bool) -> None:
        self._show_preview = bool(flag)
        self.preview_label.setVisible(self._show_preview)

    def set_show_sparkline(self, flag: bool) -> None:
        self._show_sparkline = bool(flag)
        self.sparkline.setVisible(self._show_sparkline)


class _SparklineWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._values: Optional[np.ndarray] = None
        self.setMinimumSize(160, 48)

    def set_values(self, values: Optional[np.ndarray]) -> None:
        if values is None:
            self._values = None
        else:
            try:
                arr = np.asarray(values, dtype=float).ravel()
                self._values = arr
            except Exception:
                self._values = None
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(200, 60)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, self.palette().window())

        if self._values is None or self._values.size == 0:
            painter.end()
            return

        pts = normalize_series_to_polyline(
            self._values, rect.width(), rect.height(), pad=3
        )
        if not pts:
            painter.end()
            return

        pen = QtGui.QPen(self.palette().highlight().color(), 2)
        painter.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        painter.drawPath(path)
        painter.end()


from plume_nav_debugger.config import DebuggerPreferences
from plume_nav_debugger.inspector.introspection import (
    format_pipeline,
    get_env_chain_names,
)
from plume_nav_debugger.inspector.models import ActionPanelModel, ObservationPanelModel
from plume_nav_debugger.inspector.plots import normalize_series_to_polyline
from plume_nav_debugger.odc.mux import ProviderMux


class InspectorWidget(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()
        vbox = QtWidgets.QVBoxLayout(self)
        # Strict-mode banner (hidden by default)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a60; padding: 4px;")
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setVisible(False)
        self.tabs = QtWidgets.QTabWidget()
        self.action_panel = ActionPanelWidget()
        self.obs_panel = ObservationPanelWidget()
        self.tabs.addTab(self.action_panel, "Action")
        self.tabs.addTab(self.obs_panel, "Observation")
        vbox.addWidget(self.info_label)
        vbox.addWidget(self.tabs)

        # No control signals: Inspector is information-only

        # Models for TDD-friendly logic
        self._obs_model = ObservationPanelModel()
        self._act_model = ActionPanelModel()
        self._policy_for_probe = None
        self._mux: Optional[ProviderMux] = None
        self._pipeline_text: str = ""
        self._strict_provider_only: bool = True

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        try:
            obs = getattr(ev, "obs", None)
            action = getattr(ev, "action", None)

            # Update models
            if isinstance(obs, np.ndarray):
                self._obs_model.update(obs)
            else:
                self._obs_model.summary = None

            self._act_model.update_event(
                action if isinstance(action, (int, np.integer)) else None
            )

            # Probe distribution (best effort)
            if isinstance(obs, np.ndarray):
                # Prefer ProviderMux for distribution
                dist = None
                if self._mux is not None:
                    try:
                        dist = self._mux.get_policy_distribution(obs)
                    except Exception:
                        dist = None
                if dist is not None:
                    self._act_model.state.distribution = [float(x) for x in dist]
                    self._act_model.state.distribution_source = "provider"
                elif (
                    not self._strict_provider_only
                ) and self._policy_for_probe is not None:
                    self._act_model.probe_distribution(self._policy_for_probe, obs)
                else:
                    self._act_model.state.distribution = None
                    self._act_model.state.distribution_source = None

            # Update UI: observation
            if self._obs_model.summary is not None:
                s = self._obs_model.summary
                self.obs_panel.obs_shape.setText(
                    f"shape: {'x'.join(str(d) for d in s.shape)}"
                )
                self.obs_panel.obs_stats.setText(
                    f"min/mean/max: {s.vmin:.3f}/{s.vmean:.3f}/{s.vmax:.3f}"
                )
                # Small preview for tiny vectors
                try:
                    if isinstance(obs, np.ndarray) and obs.size <= 8:
                        vals = ", ".join(f"{float(v):.3f}" for v in obs.ravel())
                        self.obs_panel.preview_label.setText(f"values: [{vals}]")
                    else:
                        self.obs_panel.preview_label.setText("")
                    # Sparkline for vector observations
                    if isinstance(obs, np.ndarray) and obs.size > 1:
                        self.obs_panel.sparkline.set_values(obs.astype(float))
                    else:
                        self.obs_panel.sparkline.set_values(None)
                except Exception:
                    self.obs_panel.preview_label.setText("")
                    self.obs_panel.sparkline.set_values(None)
            else:
                self.obs_panel.obs_shape.setText("shape: -")
                self.obs_panel.obs_stats.setText("min/mean/max: -/-/-")
                self.obs_panel.preview_label.setText("")
                self.obs_panel.sparkline.set_values(None)
            # Pipeline label if known
            if self._pipeline_text:
                if hasattr(self.obs_panel, "pipeline_label"):
                    self.obs_panel.pipeline_label.setText(self._pipeline_text)

            # Update UI: action
            self.action_panel.expected_action_label.setText(
                f"expected action: {self._act_model.state.action_label}"
            )
            if self._act_model.state.distribution is not None:
                src = self._act_model.state.distribution_source or "probs"
                p = self._act_model.state.distribution
                preview = ", ".join(f"{v:.2f}" for v in p[:6])
                self.action_panel.distribution_label.setText(
                    f"distribution ({src}): [{preview}]"
                )
            else:
                self.action_panel.distribution_label.setText("distribution: N/A")
        except Exception:
            pass

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        self._act_model.set_action_names(names)
        self.action_panel.on_action_names(names)

    def set_grid_size(self, w: int, h: int) -> None:
        self.action_panel.set_grid_size(w, h)

    @QtCore.Slot(object)
    def on_policy_changed(self, policy: object) -> None:
        self._policy_for_probe = policy

    def set_observation_pipeline_from_env(self, env: object) -> None:
        try:
            chain = get_env_chain_names(env)
            self._pipeline_text = f"Pipeline: {format_pipeline(chain)}"
        except Exception:
            self._pipeline_text = ""

    @QtCore.Slot(object)
    def on_mux_changed(self, mux: object) -> None:
        try:
            self._mux = mux  # type: ignore[assignment]
            # Update pipeline with provider if available
            if hasattr(self._mux, "get_pipeline"):
                try:
                    names = self._mux.get_pipeline()  # type: ignore[attr-defined]
                    self._pipeline_text = f"Pipeline: {format_pipeline(list(names))}"
                except Exception:
                    pass
            # Update source label to indicate provider
            try:
                self.action_panel.source_label.setText("source: provider")
            except Exception:
                pass
            self._update_strict_banner()
        except Exception:
            self._mux = None
            try:
                self.action_panel.source_label.setText("source: heuristic")
            except Exception:
                pass
            self._update_strict_banner()

    def set_strict_provider_only(self, flag: bool) -> None:
        self._strict_provider_only = bool(flag)
        self._update_strict_banner()

    # Inspector display preferences passthrough
    def set_show_pipeline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_pipeline(flag)
        except Exception:
            pass

    def set_show_preview(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_preview(flag)
        except Exception:
            pass

    def set_show_sparkline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_sparkline(flag)
        except Exception:
            pass

    def _update_strict_banner(self) -> None:
        try:
            show = bool(self._strict_provider_only) and (self._mux is None)
            if show:
                self.info_label.setText(
                    "Strict mode: no DebuggerProvider detected. Inspector shows limited information. "
                    "Implement a provider (ODC) to enable action labels, distributions, and pipeline details. "
                    '<a href="https://plume-nav-sim.dev/odc" style="color:#06c;">Read ODC docs</a>.'
                )
            self.info_label.setVisible(show)
        except Exception:
            self.info_label.setVisible(False)


def main(
    *, strict_provider_only: Optional[bool] = None
) -> None:  # pragma: no cover - UI entry point
    # macOS layer workaround for some terminals: set before QApplication
    if sys.platform == "darwin":
        import os

        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    # Optional CLI override for strict mode
    try:
        import argparse

        parser = argparse.ArgumentParser(add_help=True)
        # BooleanOptionalAction supports --strict-provider-only / --no-strict-provider-only
        parser.add_argument(
            "--strict-provider-only",
            dest="cli_strict_provider_only",
            action=getattr(argparse, "BooleanOptionalAction", "store_true"),
            default=None,
            help="Enable/disable strict provider-only mode (default: preference)",
        )
        # Only parse known args; ignore others to avoid conflicts in embedded contexts
        ns, _ = parser.parse_known_args(sys.argv[1:])
        if strict_provider_only is None:
            strict_provider_only = getattr(ns, "cli_strict_provider_only", None)
    except Exception:
        # Fall back to provided kwarg or preferences
        pass
    app = QtWidgets.QApplication(sys.argv)
    # QSettings identifiers for layout persistence
    QtCore.QCoreApplication.setOrganizationName("plume-nav-sim")
    QtCore.QCoreApplication.setApplicationName("Debugger")
    win = MainWindow()
    win.show()
    # Apply strict-provider-only override if provided
    try:
        if strict_provider_only is not None:
            win.prefs.strict_provider_only = bool(strict_provider_only)
            # Reflect into driver and inspector immediately
            win.driver.config.strict_provider_only = bool(strict_provider_only)
            win.inspector.set_strict_provider_only(bool(strict_provider_only))
    except Exception:
        pass
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    main()
