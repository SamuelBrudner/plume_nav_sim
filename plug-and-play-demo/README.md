Plug-and-play demo (external-style usage)

This folder mimics a separate project that imports `plume_nav_sim` as a library and runs a minimal demo using the run–tumble temporal-derivative policy with the runner.

How to run

- Prerequisite: install `plume_nav_sim` in your environment.
  - One‑liner (from repo root): `pip install -e src/backend`
  - Once published: `pip install plume_nav_sim`

- From repo root:
  - `python plug-and-play-demo/main.py`
  - With custom policy: `python plug-and-play-demo/main.py --policy-spec my_pkg.mod:MyPolicy`
  - Save frames: `python plug-and-play-demo/main.py --save-gif out.gif`

- From this folder:
  - `python main.py`

What it does

- Builds an environment configured for run–tumble actions
- Applies the core `ConcentrationNBackWrapper(n=2)` via SimulationSpec so the policy sees `[c_prev, c_now]`
- Instantiates the stateless run–tumble policy implemented in this demo via dotted-path
- Runs a single episode with `runner.run_episode`, collecting RGB frames via the per-step callback
- Prints a concise episode summary and the number of frames captured
Advanced demos available for determinism and subset validation

Policy interface (see inline comments in `plug_and_play_demo/stateless_policy.py`)

- Matches the `plume_nav_sim.interfaces.policy.Policy` protocol
- Required members:
  - `action_space` property (Gymnasium space)
  - `reset(seed=...)` (recommended for determinism)
  - `select_action(obs, explore=...) -> action`

Files

- `plug-and-play-demo/main.py` – standalone script importing only `plume_nav_sim`
- `plug-and-play-demo/plug_and_play_demo/stateless_policy.py` – stateless run–tumble policy (uses [c_prev, c_now])
- `plug-and-play-demo/plug_and_play_demo.ipynb` – notebook demo (spec-first + frames)

Advanced: custom observation space

- Use the `advanced/nback_demo.py` script to expose an n-back history via the core `ConcentrationNBackWrapper`.

```
PYTHONPATH=src/backend:plug-and-play-demo \
  python plug-and-play-demo/advanced/nback_demo.py --n 5
```

Showcase the benefits

Core benefits provided by the library:
- Spec-first composition: SimulationSpec + prepare() builds env+policy with one source of truth.
- Deterministic seeding: runner resets env and policy with the same seed.
- Safe wiring: action-space subset checks catch policy/env mismatches early.
- Plug-and-play policies: dotted-path loader keeps user policies outside the library.
- Frames and callbacks: runner emits StepEvent with RGB frames for visualization.

Examples:

```
# Determinism and subset validation (advanced)
PYTHONPATH=src/backend:plug-and-play-demo \
  python plug-and-play-demo/advanced/benefits_demo.py
```
