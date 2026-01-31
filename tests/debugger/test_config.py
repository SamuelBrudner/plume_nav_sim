from __future__ import annotations

from plume_nav_debugger.config import DebuggerPreferences


def test_debugger_preferences_json_roundtrip(tmp_path):
    prefs = DebuggerPreferences(
        show_pipeline=False,
        show_preview=False,
        show_sparkline=False,
        show_overlays=True,
        default_interval_ms=123,
        theme="dark",
    )
    path = tmp_path / "prefs.json"
    prefs.save_json_file(path)

    loaded = DebuggerPreferences.load_json_file(path)

    assert loaded == prefs
