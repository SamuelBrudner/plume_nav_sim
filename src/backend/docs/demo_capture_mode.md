# Plug-and-Play Demo Capture Mode

Entrypoint: `plug-and-play-demo/main.py`

## Run the demo
```bash
python plug-and-play-demo/main.py
```
Expected terminal output includes `Episode summary: {...}`.

## Capture mode (`--capture`)
`plug-and-play-demo/main.py` does not expose a standalone `--capture` flag.
Capture mode is enabled when any capture flag is present:
`--capture-root`, `--experiment`, `--episodes`, `--parquet`, or `--validate`.

When enabled, the environment is wrapped by
`src/backend/plume_nav_sim/data_capture/wrapper.py:DataCaptureWrapper`, which
writes per-step and per-episode records using
`src/backend/plume_nav_sim/data_capture/recorder.py:RunRecorder`.

## GIF mode (`--save-gif`)
Use `--save-gif <path.gif>` to write rendered frames to a GIF via
`src/backend/plume_nav_sim/plume/video.py:save_video_frames`.
Expected terminal output includes `Saved video: <path>`.

## Output locations and saved files
Capture output directory:
`<capture-root>/<experiment>/<run-id>/`
(default example: `results/demo/run-YYYYMMDD-HHMMSS/`).

Saved artifacts:
- `run.json` (run metadata and env config)
- `steps.jsonl.gz` (step records; may rotate to `steps.partN.jsonl.gz`)
- `episodes.jsonl.gz` (episode summaries; may rotate to `episodes.partN.jsonl.gz`)
- validation summary in stdout when `--validate` is used

GIF output:
- exactly the file path passed to `--save-gif`

## Example commands and expected output
```bash
python plug-and-play-demo/main.py --save-gif /tmp/demo.gif
```
Expected: `Episode summary: {..., "frames_captured": <n>}` and
`Saved video: /tmp/demo.gif`.

```bash
python plug-and-play-demo/main.py --capture-root results --experiment demo --episodes 2 --validate
```
Expected: `Capture complete. Run directory: results/demo/run-<timestamp>` and
`Validation: {'steps_ok': True, 'episodes_ok': True}`.

```bash
python plug-and-play-demo/main.py --capture-root results --experiment demo --episodes 1 --save-gif /tmp/demo_capture.gif
```
Expected: `Saved video: /tmp/demo_capture.gif` and
`Capture complete. Run directory: results/demo/run-<timestamp>`.
