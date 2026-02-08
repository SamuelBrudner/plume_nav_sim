from __future__ import annotations

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.widgets.sparkline_widget import _SparklineWidget


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

