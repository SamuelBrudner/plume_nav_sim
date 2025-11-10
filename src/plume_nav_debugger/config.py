from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "plume-nav-sim" / "debugger.json"


@dataclass
class DebuggerPreferences:
    strict_provider_only: bool = True
    show_pipeline: bool = True
    show_preview: bool = True
    show_sparkline: bool = True
    default_interval_ms: int = 50
    theme: str = "light"  # or "dark"

    @classmethod
    def from_qsettings(cls) -> "DebuggerPreferences":  # pragma: no cover - Qt
        try:
            from PySide6 import QtCore

            s = QtCore.QSettings("plume-nav-sim", "Debugger")
            return cls(
                strict_provider_only=bool(
                    s.value("prefs/strict_provider_only", True, type=bool)
                ),
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

            s = QtCore.QSettings("plume-nav-sim", "Debugger")
            s.setValue("prefs/strict_provider_only", self.strict_provider_only)
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
                strict_provider_only=bool(data.get("strict_provider_only", True)),
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
        if DEFAULT_CONFIG_PATH.exists():
            prefs = cls.load_json_file(DEFAULT_CONFIG_PATH)
            # sync into QSettings as the current baseline
            prefs.to_qsettings()
            return prefs
        # Load from QSettings or defaults
        return cls.from_qsettings()
