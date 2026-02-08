# Rendering Semantics

## Overview

Rendering in `plume_nav_sim` is built directly into the environment classes.
There is **no separate `render/` package** — the legacy renderer was removed.

## Supported Modes

Both `PlumeEnv` and `ComponentBasedEnvironment` support the standard Gymnasium
`render()` interface:

| Mode | Behavior |
|------|----------|
| `"rgb_array"` | Returns an `np.ndarray` (H×W×3, dtype `uint8`) suitable for recording or headless use. |
| `"human"` | Displays a matplotlib window (requires a display or suitable backend). |

### Example

```python
env = PlumeEnv(render_mode="rgb_array", ...)
obs, info = env.reset()
frame = env.render()  # np.ndarray, shape (H, W, 3)
```

## DataCaptureWrapper Passthrough

`DataCaptureWrapper` delegates `render()` directly to the wrapped environment:

```python
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper

wrapped = DataCaptureWrapper(env, run_dir="/tmp/capture")
frame = wrapped.render()  # calls env.render() unchanged
```

The wrapper does not modify, intercept, or buffer frames.
It exists solely for step/episode data capture.

## Headless Rendering

For CI, notebooks, or server environments without a display:

```python
env = PlumeEnv(render_mode="rgb_array", ...)
```

This avoids any matplotlib GUI backend dependency.
Frames can be saved to disk or assembled into video with external tools.

## Implementation Locations

- `PlumeEnv.render()` → `src/backend/plume_nav_sim/envs/plume_env.py`
- `ComponentBasedEnvironment.render()` → `src/backend/plume_nav_sim/envs/component_env.py`
- `DataCaptureWrapper.render()` → `src/backend/plume_nav_sim/data_capture/wrapper.py`

## Recording Video

Use `save_video` to record frames as GIF or MP4:

```python
from plume_nav_sim.utils.video import save_video

# frames is a list of np.ndarray (H, W, 3) from env.render()
save_video(frames, "episode.gif", fps=10)
```

See `examples/save_video_example.py` for a complete runnable script.
