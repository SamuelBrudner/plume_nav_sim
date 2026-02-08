# Debugger Usage

The debugger is a Qt UI for running `plume_nav_sim` step-by-step, inspecting policy behavior, and replaying captured runs.

## Launch

From the repo root:

```bash
PYTHONPATH=src python -m plume_nav_debugger
```

If your local setup does not expose package module launching yet, use:

```bash
PYTHONPATH=src python -m plume_nav_debugger.app
```

## Core Controls

- `Start` / `Pause` run and stop stepping.
- `Step` advances one step.
- `Back` (`Step Back`) rewinds one step in live and replay modes.
- `Reset` restarts the current live episode.
- Keyboard shortcuts: `Space` (Start/Pause), `N` (Step), `B` (Step Back), `R` (Reset).
- `Interval (ms)` controls time between automatic steps.

## Seed Control

- Enter an integer in the `Seed` field, then click `Reset` (or press `R`) to reset with that seed.
- If the `Seed` field is empty (or invalid), reset reuses the last episode seed.

## Policy Selection

- Built-ins in the policy dropdown:
  - `Greedy TD (bacterial)`
  - `Stochastic TD`
  - `Deterministic TD`
  - `Random Sampler` (random policy)
- For custom policies, enter `module:ClassName` (or callable attribute) and click `Load`.
- `Explore` enables policy exploration (`explore=True`) in live mode.

## Live Config Panel

- Open `View -> Live Config`.
- Use presets to quickly switch common configurations.
- Edit `Plume`, `Action type`, `Seed`, `Max steps`, and movie inputs (especially `Movie path`).
- Click `Apply` to recreate the live environment with current draft settings.
- Click `Revert` to restore the last applied live configuration.

## Replay Mode

- Switch `Mode` to `Replay`, then click `Load Replay…` and choose a run directory.
- Use the timeline slider/spinbox to seek specific steps.
- Use the episode spinner to jump between episodes.
- `Back` steps backward through replayed events; `Step`/`Start` move forward.

## Inspector, Overlays, Preferences

- `Inspector` (View menu) has `Action` and `Observation` tabs.
- Action tab shows current action and policy distribution when provided.
- Observation tab shows shape/stats plus pipeline text, preview text, and sparkline (when enabled).
- Frame overlays show agent position, goal marker/radius, heading arrow, and HUD text; toggle with `View -> Frame overlays`.
- `Edit -> Preferences` controls theme (`light`/`dark`) and show/hide toggles for pipeline, preview, sparkline, and overlays.

## Source References

- Main UI and controls: `src/plume_nav_debugger/main_window.py`
- Live stepping, reset, policy/explore, seed behavior: `src/plume_nav_debugger/env_driver.py`
- Replay timeline and episode navigation: `src/plume_nav_debugger/replay_driver.py`
- Overlay metadata extraction: `src/plume_nav_debugger/frame_overlays.py`
- App entrypoint: `src/plume_nav_debugger/app.py`
- Preferences model: `src/plume_nav_debugger/config.py`
