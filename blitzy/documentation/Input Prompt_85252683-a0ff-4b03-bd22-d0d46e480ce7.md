# Proof-of-Life Specification for `plume_nav_sim`

## 0. Goal

Deliver a minimal Gymnasium-compatible environment named `plume_nav_sim` that:

- Simulates a single agent in a static Gaussian plume.
- Provides the Gymnasium 5‑tuple API.
- Supports two render modes:
  
  - `rgb_array` (NumPy buffer).
  - `human` (Matplotlib window) — essential to visually confirm the environment is alive.
- Includes an example script and mirrored tests.

---

## 1. Repository Layout

```plaintext
plume-nav-sim/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ src/
│  └─ plume_nav_sim/
│     ├─ __init__.py
│     ├─ registration.py
│     ├─ envs/
│     │  └─ static_gaussian.py
│     ├─ plume/
│     │  └─ static_gaussian.py
│     ├─ render/
│     │  ├─ numpy_rgb.py
│     │  └─ matplotlib_viz.py
│     └─ utils/
│        └─ seeding.py
├─ tests/
│  └─ plume_nav_sim/
│     ├─ registration/test_registration.py
│     ├─ envs/test_static_gaussian.py
│     ├─ plume/test_static_gaussian.py
│     ├─ render/test_numpy_rgb.py
│     ├─ render/test_matplotlib_viz.py
│     └─ utils/test_seeding.py
└─ examples/
   ├─ quickstart_random_agent.py
   └─ visualization_demo.py
```

- `src/` layout ensures extensibility.
- `tests/` mirrors `src/` exactly.

---

## 2. Packaging

- **Project name:** `plume-nav-sim`
- **Import:** `plume_nav_sim`
- **Python:** &gt;=3.10
- **Runtime deps:** `gymnasium`, `numpy`, `matplotlib`
- **Dev deps:** `pytest`

**Minimal** `pyproject.toml`**:**

```toml
[build-system]
requires = ["hatchling>=1.21"]
build-backend = "hatchling.build"

[project]
name = "plume-nav-sim"
version = "0.0.1"
description = "Proof-of-life Gymnasium environment for plume navigation"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = [
  "gymnasium>=0.29",
  "numpy>=1.24",
  "matplotlib>=3.5",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

---

## 3. Environment API

- **Class:** `plume_nav_sim.envs.static_gaussian.PlumeSearchEnv`
- **Action space:** `Discrete(4)` (`0: up`, `1: right`, `2: down`, `3: left`).
- **Observation space:** `Box(low=0.0, high=1.0, shape=(1,), dtype=float32)` (odor concentration).
- **Reset:** returns `(obs, info)`.
- **Step:** returns `(obs, reward, terminated, truncated, info)`.
- **Render modes:** `rgb_array`, `human`.
- **Close:** no-op.

**Registration:**

```python
# src/plume_nav_sim/registration.py
from gymnasium.envs.registration import register

ENV_ID = "PlumeNav-StaticGaussian-v0"

def register_env() -> None:
    register(
        id=ENV_ID,
        entry_point="plume_nav_sim.envs.static_gaussian:PlumeSearchEnv",
    )
```

---

## 4. Environment Mechanics

### 4.1 Grid & Coordinates

- **Grid:** default `(128, 128)` (width, height).
- **Coordinate system:** `(x, y)` where `x` increases rightward, `y` increases downward.
- **Array indexing:** `plume_array[y, x]` (row=y, col=x).

### 4.2 Plume Model

- **Source:** default `(64, 64)` with Gaussian plume (σ=12.0).
- **Formula:** `C(x,y) = exp(-((x - sx)² + (y - sy)²) / (2 * σ²))`
- **Normalization:** Values clamped to `[0, 1]` with peak = 1.0 at source.
- **Observation:** `np.float32([plume_array[agent_y, agent_x]])`

### 4.3 Agent Dynamics

- **Agent state:** integer coordinates `(x, y)`.
- **Boundaries:** blocked (stay in place if move would exit grid).
- **Start position:** random (excluding source), seeded via `np_random`.

### 4.4 Rewards & Termination

- **Reward:** `+1.0` once upon first entry to goal region; `0.0` otherwise.
- **Termination:** when `distance_to_source <= goal_radius` (default: 0).
- **Truncation:** after `max_steps` (default: 1000).
- **Info:** `{"step": int, "distance_to_source": float, "agent_xy": tuple[int, int]}`.

### 4.5 Seeding

- Uses `gymnasium.utils.seeding.np_random(seed)` in `reset()`.
- All randomness (start position) uses `self.np_random`.
- Identical seeds produce identical episodes.

---

## 5. Rendering

### 5.1 `rgb_array` Mode

- Heatmap grayscale of plume `[0, 1] → [0, 255]`.
- Source marked with white cross (5×5 pixels).
- Agent marked red square (3×3 pixels).
- Returns `(H, W, 3)` uint8 array.

### 5.2 `human` Mode

- Matplotlib heatmap window with colorbar.
- Updates agent location on successive renders.
- Works headless using Agg backend.
- Graceful fallback to `rgb_array` if matplotlib unavailable.

---

## 6. Examples

### 6.1 Quickstart

`examples/quickstart_random_agent.py`

```python
import gymnasium as gym
from plume_nav_sim.registration import register_env, ENV_ID

register_env()
env = gym.make(ENV_ID, render_mode="rgb_array")
obs, info = env.reset(seed=123)

print(f"Agent starts at: {info['agent_xy']}")
print(f"Initial concentration: {obs[0]:.3f}")

total_reward = 0.0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

frame = env.render()
print(f"Episode: {info['step']} steps, reward: {total_reward}, frame: {frame.shape}")
env.close()
```

### 6.2 Visualization Demo

`examples/visualization_demo.py`

```python
import gymnasium as gym
from plume_nav_sim.registration import register_env, ENV_ID

register_env()
env = gym.make(ENV_ID, render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Updates matplotlib window
    
    if terminated or truncated:
        break

env.close()
```

---

## 7. Tests

All tests are **Gymnasium-aware** and must pass with `pytest -q`.

### 7.1 Registration

`tests/plume_nav_sim/registration/test_registration.py`

- `register_env()` completes without error.
- `gymnasium.make(ENV_ID)` constructs the environment.
- Action and observation spaces match specification.

### 7.2 Environment API

`tests/plume_nav_sim/envs/test_static_gaussian.py`

- `reset()` returns `(obs, info)` with correct shapes and ranges.
- `step()` returns 5-tuple with correct dtypes.
- Boundary enforcement: agent cannot move outside grid.
- Reward logic: exactly one `+1.0` reward when reaching goal.
- Termination and truncation work correctly.
- Seeding produces reproducible episodes.

### 7.3 Plume

`tests/plume_nav_sim/plume/test_static_gaussian.py`

- Plume array has shape `(H, W)` with values in `[0, 1]`.
- Maximum value (≈1.0) occurs at source location.
- Values decrease monotonically with distance from source.

### 7.4 Rendering

`tests/plume_nav_sim/render/test_numpy_rgb.py`

- `rgb_array` mode returns `(H, W, 3)` uint8 array.
- Agent pixel is red-dominant; source pixel is white-dominant.

`tests/plume_nav_sim/render/test_matplotlib_viz.py`

- `human` mode creates matplotlib figure without error.
- Graceful fallback when matplotlib unavailable.

### 7.5 Seeding

`tests/plume_nav_sim/utils/test_seeding.py`

- Two `reset(seed=42)` calls produce identical results.
- Different seeds produce different results.

---

## 8. Out of Scope (for PoL)

**Environment features:**

- Time-varying plumes, turbulence, or diffusion.
- Continuous agent kinematics, orientation, or sensors.
- Multiple agents or sources.
- Advanced sensors (binary encounters, noisy observations).
- Obstacles or complex boundaries.

**Infrastructure:**

- Vectorized/parallel environments.
- RL training utilities, baselines, or wrappers.
- Data logging, schemas, FAIR compliance.
- Unit conversions to physical units.
- PyPI distribution, CI/CD, Dockerization.
- Configuration management systems.

---

## 9. Acceptance Checklist

- [ ]  `pip install -e .` works from repository root.

- [ ]  `gymnasium.make(ENV_ID)` constructs the environment.

- [ ]  Reset reproducibility confirmed with seeding.

- [ ]  Rewards and terminations behave as specified.

- [ ]  Both render modes return/update visuals correctly.

- [ ]  `pytest -q` passes with full test coverage.

- [ ]  Example scripts run without errors and produce expected output.

- [ ]  Agent cannot move outside grid boundaries.

- [ ]  Plume visualization shows clear concentration gradient.

---

## Appendix A: Future Data Model

*This section documents the conceptual data model for future development beyond the PoL scope.*

### A.1 Core Entities

**SimulationConfiguration**: The immutable specification defining a type of simulation.

- Properties: `grid_size`, `cell_size_m`, `plume_type`, `plume_params`, `agent_type`, `reward_structure`, `termination_conditions`
- Purpose: Captures the "sameness" across all time points, episodes, and experimental runs
- Invariant: Never changes - represents the simulation "recipe" or "blueprint"

**Environment**: The stateful simulation world executing a configuration.

- **As object**: Gymnasium.Env instance created from SimulationConfiguration
- **As concept**: The "world state" at any given time (agent position, step count, etc.)
- Properties: `config` (fixed), `agent_position`, `step_count`, `plume_instance` (changing)
- Methods: `.reset()`, `.step()`, `.render()`, `.close()`
- Lifecycle: Created from configuration, state changes each step

**Agent**: The navigating entity seeking the plume source.

- State: `position_xy` (integer coordinates)
- Managed by: Environment (no separate object in PoL)
- Capabilities: Movement and sensing via Environment interface
- Constraint: Must remain within environment boundaries

**Episode**: A complete simulation run from reset to termination.

- Data: `episode_id`, `config_id`, `seed`, `start_xy`, `outcome`, `total_steps`, `total_reward`, `duration_sec`, `step_records`
- Lifecycle: `start` → `steps` → `end`
- Outcome: `success` (found source), `timeout`, or `error`

**Step**: A record of one agent-environment interaction.

- Data: `step_number`, `agent_x`, `agent_y`, `action`, `observation`, `reward`, `terminated`, `truncated`, `distance_to_source`, `timestamp_ns`
- Purpose: Captures complete state transition for analysis and replay
- Ordering: Sequential within episodes, deterministic given initial conditions

**Plume**: The chemical signal field the agent navigates.

- Type: Static Gaussian (PoL); extensible to dynamic models
- Properties: `source_location`, `concentration_field`
- Interface: `sample(position) → concentration`

### A.2 Relationships

```plaintext
SimulationConfiguration (1) ──defines────→ (*) Environment_instances
SimulationConfiguration (1) ──specifies──→ (1) Plume_type + params
Environment (1) ──instantiates──→ (1) Plume_instance
Environment (1) ──manages───────→ (1) Agent_State
Episode (1) ────references─────→ (1) SimulationConfiguration
Episode (1) ────contains───────→ (*) Step_Record  
Environment.step() ─────generates─────→ (1) Step_Record
```

### A.3 Data Flow

```plaintext
Experiment.setup()
    → SimulationConfiguration.create(grid_size, plume_params, ...)
    → Environment.from_config(configuration)

Episode.start() 
    → Environment.reset(seed)
    → Environment.initialize_agent_position()
    → Plume.instantiate(configuration.plume_params)

Environment.step(action)
    → Environment.move_agent(action)
    → Environment.check_boundaries(agent_position)
    → observation ← Plume.sample(agent_position)
    → reward ← Environment.check_goal(agent_position)
    → info ← Environment.get_status(agent_position)
    → Step_record ← {step_number, agent_x, agent_y, action, observation, 
                      reward, terminated, truncated, distance_to_source, timestamp_ns}
```

### A.4 Temporal Aspects

**Configuration vs. instances:**

- **SimulationConfiguration**: Immutable "recipe" - same across all time points, episodes, experimental runs
- **Environment object**: Stateful instance created from configuration, state changes each step
- **World state**: Conceptually different at each time step (different agent position, step count)
- **Episode boundary**: Environment object can run multiple episodes (via `.reset()`), but always with same configuration

**"Sameness" hierarchy:**

```plaintext
SimulationConfiguration (unchanging)
    ├── Episode₁ with seed=123 → Environment instance → t₀, t₁, t₂, ... states
    ├── Episode₂ with seed=456 → Environment instance → t₀, t₁, t₂, ... states  
    └── Episode₃ with seed=789 → Environment instance → t₀, t₁, t₂, ... states
```

**Experimental reproducibility:**

- Same SimulationConfiguration + same seed → identical Episode data
- Same SimulationConfiguration + different seeds → comparable but different Episodes
- Different SimulationConfiguration → not directly comparable

### A.5 Entity Categories & Interfaces

**Configuration objects** (SimulationConfiguration): Immutable specifications with validation methods

- Defines simulation "recipe" - same across all instances and time points
- Methods: `.create()`, `.validate()`, `.to_dict()`, `.from_dict()`

**Active objects** (Environment): Stateful instances created from configurations

- Behavioral methods `.reset()`, `.step()`, `.render()`, `.close()`
- Environment renders current simulation state (agent position, plume, real-time)

**Data structures** (Episode, Step): Information containers with fields, no behavioral methods

- Episode: Contains episode metadata and collection of Step records
- Step: Contains single interaction record (action, observation, reward, etc.)
- Note: Historical data can be visualized by separate analysis tools

**Functional objects** (Plume): Domain-specific methods like `.sample(position)`

**State representations** (Agent): Position and status, managed by Environment

### A.6 Data Persistence Implications

**Entity storage patterns:**

**SimulationConfiguration** (immutable, reused):

- PoL: Embedded in session metadata (simple but duplicated)
- Research scale: Separate configuration registry with stable IDs
- Impact: Cross-experiment comparison and reproducibility

**Episode** (metadata + Step collection):

- PoL: Session-level aggregation
- Research scale: Episode catalogs with trajectory pointers
- Impact: Query performance for "find episodes by config"

**Step** (high-volume time-series):

- PoL: Single file per session
- Research scale: Time/config partitioned files
- Impact: Analytics performance and storage efficiency

**Storage architecture evolution:**

- **PoL**: Low complexity, limited cross-session queries
- **Research Scale**: Medium complexity, good analytics performance
- **Production**: High complexity, optimized for all query patterns

This data model enables scientific workflows with proper experimental reproducibility, cross-study comparison, and FAIR data principles while starting from the simple PoL implementation.
