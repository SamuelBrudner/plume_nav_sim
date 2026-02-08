# Debugger Configuration Reference

Source of truth:
- `src/plume_nav_debugger/config.py` (`DebuggerPreferences`, JSON + QSettings IO)
- `src/plume_nav_debugger/main_window.py` (`PreferencesDialog`, presets, overlay toggle)

## Preferences File Location

- Primary JSON file: `~/.config/plume_nav_sim/debugger.json`
- Legacy JSON file (read/migrated): `~/.config/plume-nav-sim/debugger.json`
- QSettings app scope:
  - org/app: `plume_nav_sim` / `Debugger`
  - legacy org fallback: `plume-nav-sim` / `Debugger`

## Preference Fields (Type + Default)

| Field | Type | Default | Notes |
|---|---|---|---|
| `show_overlays` | `bool` | `true` | Toggles frame overlays in `View -> Frame overlays`. |
| `show_pipeline` | `bool` | `true` | Shows/hides inspector pipeline text. |
| `show_preview` | `bool` | `true` | Shows/hides observation preview text. |
| `show_sparkline` | `bool` | `true` | Shows/hides vector sparkline in inspector. |
| `theme` | `str` | `"light"` | Allowed values: `"light"` or `"dark"`. |
| `default_interval_ms` | `int` | `50` | Default step interval applied to the control bar spinner. |
| `distribution_method` | `str` | `"provider"` | Include in JSON for builds that expose this preference. |

## Load Order, QSettings Fallback, and Migration

1. On startup, `DebuggerPreferences.initial_load()` checks:
   - `~/.config/plume_nav_sim/debugger.json`
   - then `~/.config/plume-nav-sim/debugger.json`
2. If a JSON file is found, it is loaded first.
3. If the JSON came from the legacy path, it is written back to `~/.config/plume_nav_sim/debugger.json` (path migration).
4. Loaded JSON preferences are mirrored to QSettings (`plume_nav_sim` / `Debugger`).
5. If no JSON file exists, preferences are loaded from QSettings.
6. QSettings read fallback: use new org if any `prefs/*` key exists there; otherwise read from legacy org.

## Built-in Presets (`LiveConfigWidget._build_presets`)

- Static quickstart: key `"Static quickstart"`
- Static small-grid: key `"Static small-grid deterministic"`
- Movie demo: key `"Movie demo (local zarr)"`
- Movie gaussian: key `"Movie gaussian (local zarr)"`

## Overlay Configuration (`show_overlays`)

When `show_overlays=true`, frame rendering draws:
- goal marker + goal radius ring + `GOAL` label
- agent marker + heading arrow
- HUD text: seed, step `t`, action/action name, reward, total reward, terminated/truncated, agent/goal coordinates

When `show_overlays=false`, the raw frame is shown without those overlay elements.

## Example `~/.config/plume_nav_sim/debugger.json`

```json
{
  "show_overlays": true,
  "show_pipeline": true,
  "show_preview": true,
  "show_sparkline": true,
  "theme": "light",
  "default_interval_ms": 50,
  "distribution_method": "provider"
}
```
