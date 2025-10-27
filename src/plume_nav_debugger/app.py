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


class EnvDriver(QtCore.QObject):
    frame_ready = QtCore.Signal(np.ndarray)
    episode_finished = QtCore.Signal()
    step_done = QtCore.Signal(object)  # emits runner.StepEvent

    def __init__(self, config: DebuggerConfig) -> None:
        super().__init__()
        self.config = config
        self._env = None
        self._policy = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._iter = None  # iterator from runner.stream
        self._running = False

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

        self._env = pns.make_env(
            grid_size=self.config.grid_size,
            goal_radius=self.config.goal_radius,
            plume_sigma=self.config.plume_sigma,
            max_steps=self.config.max_steps,
            render_mode="rgb_array",
            action_type="oriented",
            observation_type="concentration",
            reward_type="step_penalty",
        )

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

        # Prime generator for stepping using runner.stream
        self._iter = runner.stream(
            self._env,
            self._policy,
            seed=self.config.seed,
            render=True,
        )

        # Emit an initial frame (pre-step) for immediate visual feedback
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                self.frame_ready.emit(frame)
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

        self._iter = runner.stream(
            self._env,
            self._policy if self._policy is not None else self._make_default_policy(),
            seed=seed,
            render=True,
        )
        # Show first frame after reset
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                self.frame_ready.emit(frame)
        except Exception:
            pass

    def is_running(self) -> bool:
        return bool(self._running)

    def get_interval_ms(self) -> int:
        try:
            return int(self._timer.interval())
        except Exception:
            return 50

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

        # Reset policy deterministically if provided
        reset_policy_if_possible(
            self._policy, seed=seed if seed is not None else self.config.seed
        )

        # Recreate iterator
        self._iter = runner.stream(
            self._env,
            self._policy if self._policy is not None else self._make_default_policy(),
            seed=seed if seed is not None else self.config.seed,
            render=True,
        )

        # Emit immediate frame
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                self.frame_ready.emit(frame)
        except Exception:
            pass

        if was_running and resume:
            self.start(self.get_interval_ms())

    @QtCore.Slot()
    def _on_tick(self) -> None:
        if self._iter is None:
            return
        try:
            ev = next(self._iter)
            # emit frame and step event for UI consumers
            if isinstance(ev.frame, np.ndarray):
                self.frame_ready.emit(ev.frame)
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
        seed = int(text) if text.isdigit() else 123
        self.reset.emit(seed)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plume Nav Debugger (MVP)")
        self.resize(800, 700)

        self.frame_view = FrameView()
        self.controls = ControlBar()
        self._total_reward: float = 0.0

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.addWidget(self.frame_view, stretch=1)
        vbox.addWidget(self.controls, stretch=0)
        self.setCentralWidget(central)

        self.driver = EnvDriver(DebuggerConfig())
        self.driver.frame_ready.connect(self.frame_view.update_frame)
        self.driver.step_done.connect(self._on_step_event)
        self.driver.episode_finished.connect(self._on_episode_finished)
        self.controls.start.connect(self._on_start_clicked)
        self.controls.pause.connect(self.driver.pause)
        self.controls.step.connect(self.driver.step_once)
        self.controls.reset.connect(self._on_reset_clicked)
        self.controls.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.controls.policy_combo.currentIndexChanged.connect(self._on_policy_selected)
        self.controls.custom_load_btn.clicked.connect(self._on_custom_policy_load)

        # Status bar showing step/total reward/flags
        self._status = QtWidgets.QLabel("ready")
        self.statusBar().addPermanentWidget(self._status, 1)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Initialize once to show first frame
        QtCore.QTimer.singleShot(0, self.driver.initialize)

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
        self.driver.reset(seed)

    @QtCore.Slot(int)
    def _on_policy_selected(self, idx: int) -> None:
        # Map combo index to built-in policies
        label = self.controls.policy_combo.itemText(idx)
        try:
            if label == "Greedy TD (bacterial)":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(
                    eps=0.0, eps_after_turn=0.0, uniform_random_on_non_increase=False
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
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Custom policy load failed: {e}", 4000)

    def _current_seed_value(self) -> Optional[int]:
        text = self.controls.seed_edit.text().strip()
        return int(text) if text.isdigit() else self.driver.config.seed


def main() -> None:  # pragma: no cover - UI entry point
    app = QtWidgets.QApplication(sys.argv)
    # macOS layer workaround for some terminals
    if sys.platform == "darwin":
        import os

        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
