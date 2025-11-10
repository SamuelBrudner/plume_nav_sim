from pathlib import Path

from plume_nav_debugger.config import DebuggerPreferences


def test_preferences_json_roundtrip(tmp_path: Path):
    p = DebuggerPreferences(
        strict_provider_only=False,
        show_pipeline=False,
        show_preview=True,
        show_sparkline=False,
        default_interval_ms=123,
        theme="dark",
    )
    cfg = tmp_path / "debugger.json"
    p.save_json_file(cfg)
    loaded = DebuggerPreferences.load_json_file(cfg)
    assert loaded == p
