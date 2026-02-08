from __future__ import annotations

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


class ControlBar(QtWidgets.QWidget):
    start = QtCore.Signal()
    pause = QtCore.Signal()
    step = QtCore.Signal()
    step_back = QtCore.Signal()
    reset = QtCore.Signal(int)
    mode_changed = QtCore.Signal(str)
    load_replay = QtCore.Signal()
    seek_requested = QtCore.Signal(int)
    episode_seek_requested = QtCore.Signal(int)
    explore_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        main_row = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
        self.step_back_btn = QtWidgets.QPushButton("Back")
        # Mode and replay loader
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Live", "Replay"])
        self.load_replay_btn = QtWidgets.QPushButton("Load Replay…")
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
        self.explore_check = QtWidgets.QCheckBox("Explore")
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        main_row.addWidget(self.start_btn)
        main_row.addWidget(self.pause_btn)
        main_row.addWidget(self.step_btn)
        main_row.addWidget(self.step_back_btn)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Mode:"))
        main_row.addWidget(self.mode_combo)
        main_row.addWidget(self.load_replay_btn)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Policy:"))
        main_row.addWidget(self.policy_combo)
        main_row.addWidget(self.custom_policy_edit)
        main_row.addWidget(self.custom_load_btn)
        main_row.addSpacing(8)
        main_row.addWidget(self.explore_check)
        main_row.addSpacing(8)
        main_row.addWidget(QtWidgets.QLabel("Interval (ms):"))
        main_row.addWidget(self.interval_spin)
        main_row.addStretch(1)
        main_row.addWidget(QtWidgets.QLabel("Seed:"))
        main_row.addWidget(self.seed_edit)
        main_row.addWidget(self.reset_btn)

        # Replay timeline row (hidden in live mode)
        self._timeline_row = QtWidgets.QWidget()
        replay_row = QtWidgets.QHBoxLayout(self._timeline_row)
        replay_row.setContentsMargins(0, 0, 0, 0)
        self.replay_label = QtWidgets.QLabel("Replay: none loaded")
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_spin = QtWidgets.QSpinBox()
        self.timeline_spin.setRange(0, 0)
        self.timeline_spin.setEnabled(False)
        self.episode_spin = QtWidgets.QSpinBox()
        self.episode_spin.setRange(1, 1)
        self.episode_spin.setEnabled(False)
        self.timeline_status = QtWidgets.QLabel("Step: -/-")
        self.episode_status = QtWidgets.QLabel("Episode: -/-")

        replay_row.addWidget(self.replay_label)
        replay_row.addSpacing(8)
        replay_row.addWidget(self.timeline_slider, stretch=1)
        replay_row.addWidget(self.timeline_spin)
        replay_row.addSpacing(6)
        replay_row.addWidget(QtWidgets.QLabel("Episode:"))
        replay_row.addWidget(self.episode_spin)
        replay_row.addWidget(self.timeline_status)
        replay_row.addWidget(self.episode_status)
        self._timeline_row.setVisible(False)

        layout.addLayout(main_row)
        layout.addWidget(self._timeline_row)

        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.step_btn.clicked.connect(self.step)
        self.step_back_btn.clicked.connect(self.step_back)
        self.reset_btn.clicked.connect(self._emit_reset)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        self.load_replay_btn.clicked.connect(self.load_replay)
        self.explore_check.toggled.connect(self.explore_toggled)
        self.timeline_slider.sliderReleased.connect(self._emit_seek_from_slider)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        self.timeline_spin.editingFinished.connect(self._emit_seek_from_spin)
        self.timeline_slider.valueChanged.connect(self._sync_slider_spin)
        self.episode_spin.editingFinished.connect(self._emit_seek_from_episode)
        # Expose additional signals
        # Note: MainWindow will connect signals directly to handlers

        self._updating_timeline = False

    @QtCore.Slot()
    def _emit_reset(self) -> None:
        text = self.seed_edit.text().strip()
        # If seed field is empty or not an int, emit a sentinel -1 and let driver reuse last seed
        if text.isdigit():
            self.reset.emit(int(text))
        else:
            # Use -1 to indicate "reuse last episode seed"
            self.reset.emit(-1)

    def set_mode(self, mode: str) -> None:
        is_replay = str(mode).lower().startswith("replay")
        self.policy_combo.setEnabled(not is_replay)
        self.custom_policy_edit.setEnabled(not is_replay)
        self.custom_load_btn.setEnabled(not is_replay)
        self.explore_check.setEnabled(not is_replay)
        self.seed_edit.setEnabled(not is_replay)
        self._timeline_row.setVisible(is_replay)

    def set_replay_label(self, text: str) -> None:
        self.replay_label.setText(f"Replay: {text}")

    def set_timeline(
        self,
        total_steps: int,
        current_step: int,
        *,
        total_episodes: int | None = None,
        current_episode: int | None = None,
    ) -> None:
        self._updating_timeline = True
        total = max(0, int(total_steps))
        cur = max(-1, int(current_step))
        ep_total = None if total_episodes is None else max(0, int(total_episodes))
        ep_cur = None if current_episode is None else max(-1, int(current_episode))
        if total <= 0:
            self.timeline_slider.setEnabled(False)
            self.timeline_spin.setEnabled(False)
            self.timeline_slider.setRange(0, 0)
            self.timeline_spin.setRange(0, 0)
            self.timeline_spin.setValue(0)
            self.timeline_status.setText("Step: -/-")
        else:
            max_idx = max(0, total - 1)
            cur = min(max_idx, max(cur, 0))
            self.timeline_slider.setEnabled(True)
            self.timeline_spin.setEnabled(True)
            self.timeline_slider.setRange(0, max_idx)
            self.timeline_spin.setRange(0, max_idx)
            self.timeline_slider.setValue(cur)
            self.timeline_spin.setValue(cur)
            self.timeline_status.setText(f"Step: {cur}/{max_idx}")
        if ep_total is None or ep_total <= 0:
            self.episode_spin.setEnabled(False)
            self.episode_spin.setRange(1, 1)
            self.episode_spin.setValue(1)
            self.episode_status.setText("Episode: -/-")
        else:
            max_ep = max(0, ep_total - 1)
            ep_cur = min(max_ep, max(ep_cur or 0, 0))
            # Display episodes as 1-based for UI
            self.episode_spin.setEnabled(True)
            self.episode_spin.setRange(1, ep_total)
            self.episode_spin.setValue(ep_cur + 1)
            self.episode_status.setText(f"Episode: {ep_cur + 1}/{ep_total}")
        self._updating_timeline = False

    @QtCore.Slot()
    def _emit_seek_from_slider(self) -> None:
        if self.timeline_slider.isEnabled():
            self.seek_requested.emit(int(self.timeline_slider.value()))

    @QtCore.Slot()
    def _emit_seek_from_spin(self) -> None:
        if self.timeline_spin.isEnabled():
            self.seek_requested.emit(int(self.timeline_spin.value()))

    @QtCore.Slot()
    def _emit_seek_from_episode(self) -> None:
        if self.episode_spin.isEnabled():
            self.episode_seek_requested.emit(int(self.episode_spin.value()) - 1)

    @QtCore.Slot(int)
    def _on_slider_moved(self, value: int) -> None:
        if not self._updating_timeline:
            self.timeline_spin.setValue(int(value))

    @QtCore.Slot(int)
    def _sync_slider_spin(self, value: int) -> None:
        if not self._updating_timeline:
            self.timeline_spin.setValue(int(value))

