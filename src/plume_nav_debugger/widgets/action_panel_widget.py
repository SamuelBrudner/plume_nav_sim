from __future__ import annotations

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


class ActionPanelWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        # Policy insight
        self.expected_action_label = QtWidgets.QLabel("action taken: -")
        self.distribution_label = QtWidgets.QLabel("distribution: N/A")
        self.source_label = QtWidgets.QLabel("source: none")
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

