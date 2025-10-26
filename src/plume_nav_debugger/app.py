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

    def __init__(self, config: DebuggerConfig) -> None:
        super().__init__()
        self.config = config
        self._env = None
        self._running = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._policy = None

    def initialize(self) -> None:
        import plume_nav_sim as pns

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
        obs, _ = self._env.reset(seed=self.config.seed)
        # Optional: use temporal derivative policy if available
        try:
            from plume_nav_sim.policies import TemporalDerivativePolicy

            self._policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
            self._policy.reset(seed=self.config.seed)
        except Exception:
            self._policy = None

        frame = self._env.render()
        if isinstance(frame, np.ndarray):
            self.frame_ready.emit(frame)

    def start(self, interval_ms: int = 50) -> None:
        if not self._env:
            self.initialize()
        self._running = True
        self._timer.start(max(1, int(interval_ms)))

    def pause(self) -> None:
        self._running = False
        self._timer.stop()

    def step_once(self) -> None:
        self._on_tick()

    def reset(self, seed: Optional[int] = None) -> None:
        if self._env is None:
            return
        obs, _ = self._env.reset(seed=seed)
        if self._policy is not None:
            self._policy.reset(seed=seed)
        frame = self._env.render()
        if isinstance(frame, np.ndarray):
            self.frame_ready.emit(frame)

    @QtCore.Slot()
    def _on_tick(self) -> None:
        if self._env is None:
            return
        try:
            # Select action
            if self._policy is not None:
                # Observation reading for policy
                # We rely on last observation from env.step; for MVP, sample action_space
                action = None
            else:
                action = self._env.action_space.sample()

            # If we have a policy that needs observation, do a safe one-step with last obs
            if self._policy is not None:
                # Use previous rendered frame's concentration at agent pos is not trivial; fallback to random action when obs not at hand
                action = 0 if action is None else action

            obs, reward, terminated, truncated, _ = self._env.step(action)
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                self.frame_ready.emit(frame)
            if terminated or truncated:
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
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.step_btn)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Seed:"))
        layout.addWidget(self.seed_edit)
        layout.addWidget(self.reset_btn)

        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.step_btn.clicked.connect(self.step)
        self.reset_btn.clicked.connect(self._emit_reset)

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

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.addWidget(self.frame_view, stretch=1)
        vbox.addWidget(self.controls, stretch=0)
        self.setCentralWidget(central)

        self.driver = EnvDriver(DebuggerConfig())
        self.driver.frame_ready.connect(self.frame_view.update_frame)
        self.controls.start.connect(lambda: self.driver.start(50))
        self.controls.pause.connect(self.driver.pause)
        self.controls.step.connect(self.driver.step_once)
        self.controls.reset.connect(self.driver.reset)

        # Initialize once to show first frame
        QtCore.QTimer.singleShot(0, self.driver.initialize)


def main() -> None:  # pragma: no cover - UI entry point
    app = QtWidgets.QApplication(sys.argv)
    # macOS layer workaround for some terminals
    if sys.platform == "darwin":
        import os

        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
