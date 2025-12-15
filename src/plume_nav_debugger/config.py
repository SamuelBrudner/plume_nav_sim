from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "plume_nav_sim" / "debugger.json"
LEGACY_CONFIG_PATH = Path.home() / ".config" / "plume-nav-sim" / "debugger.json"
QSETTINGS_ORG = "plume_nav_sim"
LEGACY_QSETTINGS_ORG = "plume-nav-sim"
QSETTINGS_APP = "Debugger"


@dataclass
class DebuggerPreferences:
    show_pipeline: bool = True
    show_preview: bool = True
    show_sparkline: bool = True
    default_interval_ms: int = 50
    theme: str = "light"  # or "dark"

    @classmethod
    def from_qsettings(cls) -> "DebuggerPreferences":  # pragma: no cover - Qt
        try:
            from PySide6 import QtCore

            s_new = QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            keys = (
                "prefs/show_pipeline",
                "prefs/show_preview",
                "prefs/show_sparkline",
                "prefs/default_interval_ms",
                "prefs/theme",
            )
            use_new = False
            try:
                use_new = any(s_new.contains(k) for k in keys)
            except Exception:
                use_new = False
            s = (
                s_new
                if use_new
                else QtCore.QSettings(LEGACY_QSETTINGS_ORG, QSETTINGS_APP)
            )
            return cls(
                show_pipeline=bool(s.value("prefs/show_pipeline", True, type=bool)),
                show_preview=bool(s.value("prefs/show_preview", True, type=bool)),
                show_sparkline=bool(s.value("prefs/show_sparkline", True, type=bool)),
                default_interval_ms=int(
                    s.value("prefs/default_interval_ms", 50, type=int)
                ),
                theme=str(s.value("prefs/theme", "light")),
            )
        except Exception:
            return cls()

    def to_qsettings(self) -> None:  # pragma: no cover - Qt
        try:
            from PySide6 import QtCore

            s = QtCore.QSettings(QSETTINGS_ORG, QSETTINGS_APP)
            s.setValue("prefs/show_pipeline", self.show_pipeline)
            s.setValue("prefs/show_preview", self.show_preview)
            s.setValue("prefs/show_sparkline", self.show_sparkline)
            s.setValue("prefs/default_interval_ms", int(self.default_interval_ms))
            s.setValue("prefs/theme", self.theme)
        except Exception:
            pass

    @classmethod
    def load_json_file(cls, path: Path = DEFAULT_CONFIG_PATH) -> "DebuggerPreferences":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(
                show_pipeline=bool(data.get("show_pipeline", True)),
                show_preview=bool(data.get("show_preview", True)),
                show_sparkline=bool(data.get("show_sparkline", True)),
                default_interval_ms=int(data.get("default_interval_ms", 50)),
                theme=str(data.get("theme", "light")),
            )
        except Exception:
            return cls()

    def save_json_file(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
        except Exception:
            pass

    @classmethod
    def initial_load(cls) -> "DebuggerPreferences":  # pragma: no cover - IO
        # Prefer JSON if present; otherwise use QSettings defaults
        path = None
        if DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        elif LEGACY_CONFIG_PATH.exists():
            path = LEGACY_CONFIG_PATH
        if path is not None:
            prefs = cls.load_json_file(path)
            if path != DEFAULT_CONFIG_PATH:
                try:
                    prefs.save_json_file(DEFAULT_CONFIG_PATH)
                except Exception:
                    pass
            prefs.to_qsettings()
            return prefs
        # Load from QSettings or defaults
        return cls.from_qsettings()
