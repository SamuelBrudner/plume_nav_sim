from __future__ import annotations

import sys

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume_nav_sim pip install PySide6)."
    ) from e

from plume_nav_debugger.env_driver import DebuggerConfig, EnvDriver
from plume_nav_debugger.main_window import (
    ActionPanelWidget,
    ControlBar,
    FrameView,
    InspectorWidget,
    MainWindow,
    ObservationPanelWidget,
    PreferencesDialog,
)
from plume_nav_debugger.replay_driver import ReplayDriver

__all__ = [
    "ActionPanelWidget",
    "ControlBar",
    "DebuggerConfig",
    "EnvDriver",
    "FrameView",
    "InspectorWidget",
    "MainWindow",
    "ObservationPanelWidget",
    "ReplayDriver",
    "PreferencesDialog",
    "main",
]


def main() -> None:  # pragma: no cover - UI entry point
    # macOS layer workaround for some terminals: set before QApplication
    if sys.platform == "darwin":
        import os

        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    app = QtWidgets.QApplication(sys.argv)
    # QSettings identifiers for layout persistence
    QtCore.QCoreApplication.setOrganizationName("plume_nav_sim")
    QtCore.QCoreApplication.setApplicationName("Debugger")
    win = MainWindow()
    win.show()
    # Strict provider-only is always enforced by the UI; no CLI or pref overrides
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    main()
