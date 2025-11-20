# Semantic Data Model Specification

**Version:** 1.0.0  
**Date:** 2025-09-30  
**Status:** CANONICAL - Defines meaning and relationships

> This document defines the semantic meaning of core abstractions, their relationships, and invariants that must hold. Complements `CONTRACTS.md` (which defines APIs).

---

## 🎯 Core Abstractions

### 1. **Environment**

**semantic Meaning:** A single instantiation of the plume navigation task.

**Lifecycle:**

```text
┌─────────┐     ┌───────┐     ┌──────────┐     ┌────────┐
│ Created │────▶│ Reset │◀───▶│ Stepping │────▶│ Closed │
└─────────┘     └───────┘     └──────────┘     └────────┘
                    │             │
                    └─────────────┘
                   (can reset again)
```

**States:**

- **Created**: Constructor called, components initialized
- **Reset**: Episode begins, agent at start position
- **Stepping**: Agent taking actions, receiving observations
- **Terminated**: Goal reached (success) or invalid state (failure)
- **Truncated**: Step limit reached without goal
- **Closed**: Resources released, no longer usable

**Invariants:**

- Can only `step()` after `reset()`
- Cannot `reset()` after `close()`
- Same seed → same sequence of `(obs, reward, terminated, truncated, info)`

---

### 2. **Agent**

**Semantic Meaning:** The entity navigating toward a goal in the environment.

**Properties:**

- Has a **position** in grid coordinates
- Accumulates **reward** over episode
- Tracks **step count** (age of episode)
- Has **goal status** (reached or not)

**Actions:** Movement in 8 directions + stay (9-dimensional discrete action space)

**Invariants:**

- Position always within grid bounds (enforced by boundary enforcer)
- Step count monotonically increases
- Total reward is cumulative sum of step rewards
- Goal status is idempotent (once true, stays true)

---

### 3. **Plume**

**Semantic Meaning:** A concentration field representing odor/chemical diffusion.

**Types:**

1. **Static Gaussian**: Fixed 2D Gaussian centered at source
2. **Dynamic**: Time-varying concentration (future extension)

**Properties:**

- **Source location**: Origin point of plume
- **Concentration field**: 2D array of concentration values
- **Range**: Values in `[0.0, 1.0]` (normalized)

**Invariants:**

- Concentration highest at source, decays with distance
- Field dimensions match grid dimensions
- Values never negative or > 1.0

---

### Video Plume Dataset Contract (Source of Truth)

See `src/backend/docs/contracts/video_plume_dataset.md` for the canonical, testable contract for video‑backed plume datasets (Zarr + xarray).

Summary (authoritative details in the contract document):

- Variable name: `concentration`
- Dims: `("t","y","x")`, declared via `_ARRAY_DIMENSIONS` on the array
- Array dtype: `float32`
- Required dataset/global attrs: `schema_version`, `fps`, `source_dtype`, `pixel_to_grid`, `origin`, `extent`
- `manifest.json` at dataset root, validating against `ProvenanceManifest`

Validators and writers in `plume_nav_sim` must conform to that contract; this semantic model references it as the source of truth.

#### Movie metadata sidecar (canonical movie metadata)

Movie‑backed plume fields created from raw movies (e.g., `.avi`, `.mp4`, `.h5`) use a **per‑movie YAML sidecar** as the *single* source of truth for movie metadata. The sidecar is loaded by `plume_nav_sim.media.sidecar.load_movie_sidecar` and drives ingest via `plume_nav_sim.plume.movie_field.resolve_movie_dataset_path`.

- Location: for a movie at `path/to/movie.ext`, the sidecar lives at `path/to/movie.ext.plume-movie.yaml` (see `get_default_sidecar_path`).
- Role: define canonical `fps` and spatial calibration for the movie; these are then transformed into `VideoPlumeAttrs` on the Zarr dataset.

**v1 sidecar schema and invariants** (see `MovieMetadataSidecar`):

- `version: int` – schema version (currently `1`); reserved for future changes.
- `path: Optional[str]` – optional original media path for provenance; does *not* affect ingest semantics.
- `fps: PositiveFloat` – frames per second of the movie; always interpreted as
  *frames per second* (time unit = seconds, no separate `time_unit` field).
- `spatial_unit: str` – unit label for the movie’s spatial coordinate system:
  - `"pixel"` → movie is in pixel space; `pixels_per_unit` **must be omitted**.
  - any other unit (e.g., `"mm"`, `"cm"`) → movie is in a physical unit; `pixels_per_unit` **must be provided**.
- `pixels_per_unit: Optional[Tuple[float, float]]` – number of pixels per one spatial unit `(y, x)`:
  - required and strictly positive when `spatial_unit` is not `"pixel"`.
  - must be omitted when `spatial_unit == "pixel"`.
- `h5_dataset: Optional[str]` – dataset path inside an HDF5 container:
  - required for `.h5` / `.hdf5` sources.
  - must be *absent* for non‑HDF5 sources (AVI/MP4/image directories, etc.).

**Mapping: sidecar → `VideoPlumeAttrs`** (applied by `resolve_movie_dataset_path`):

- `attrs.fps = sidecar.fps`.
- `attrs.pixel_to_grid` encodes grid units per pixel `(y, x)`:
  - if `spatial_unit == "pixel"`: `pixel_to_grid = (1.0, 1.0)` (one grid unit per pixel; grid
    units are pixels).
  - otherwise: `pixel_to_grid = (1.0 / pixels_per_unit_y, 1.0 / pixels_per_unit_x)`.
- `attrs.origin = (0.0, 0.0)` – the grid origin is fixed at the top‑left corner of the movie.
- `attrs.extent` is derived from array shape and `pixel_to_grid` (see `_resolve_extent`):
  - `extent_y = height * pixel_to_grid_y`
  - `extent_x = width * pixel_to_grid_x`

At the configuration level (e.g., `SimulationSpec`), movie plumes inherit these
units:

- The temporal unit is seconds via `fps` (frames per second); there is no
  separate `time_unit` field.
- The physical spatial unit of the plume field is determined by the
  sidecar’s `spatial_unit`; there is currently no separate
  `SimulationSpec.spatial_unit` override.

Once written, the dataset’s `VideoPlumeAttrs` are the canonical metadata used at runtime by `MoviePlumeField`. The sidecar is authoritative *only at ingest time*; after ingest, the semantic model for the movie plume is fully captured by the dataset attrs.

**Container metadata vs. sidecar:**

- Format‑specific metadata (e.g., `imagingParameters/frameRate` inside certain HDF5 files) may be consulted by ingestion utilities as a *validation* or best‑effort fallback when no sidecar is used directly.
- When a movie metadata sidecar is present (the default for `resolve_movie_dataset_path`), container metadata MUST NOT override sidecar values. Any discrepancies are treated as validation errors rather than alternative sources of truth.

In effect, there is a single canonical chain for movie‑backed plume fields:

`MovieMetadataSidecar` → `VideoPlumeAttrs` on the Zarr dataset → runtime `MoviePlumeField` behavior.

### 4. **Observation**

**Semantic Meaning:** What the agent perceives at current position.

**Structure:**

```python
observation = {
    'concentration': float,        # [0.0, 1.0] at agent position
    'position': Coordinates,       # Agent's current grid position
    'step_count': int,             # Current step in episode
    'goal_reached': bool,          # Whether goal has been reached
}
```

**Invariants:**

- Concentration matches plume field at agent position
- Position always valid coordinates within grid
- Step count matches environment's episode state
- Deterministic given environment state

---

### 5. **Reward**

**Semantic Meaning:** Scalar signal indicating progress toward goal.

**Structure:** Sparse binary reward

- `+1.0` when goal reached (distance to source ≤ goal_radius)
- `0.0` otherwise

**Rationale:** Sparse rewards force agent to learn exploration.

**Invariants:**

- Reward determined solely by distance to goal
- Goal detection is deterministic and consistent
- Reward structure unchanged within episode

---

### 6. **Episode**

**Semantic Meaning:** Single attempt from start to termination.

**Structure:**

```python
Episode = {
    seed: int,                    # RNG seed for reproducibility
    initial_state: State,         # Starting state
    trajectory: List[Transition], # Sequence of (s, a, r, s')
    outcome: Outcome,             # Success, failure, or timeout
    statistics: Statistics,       # Performance metrics
}
```

**Termination Conditions:**

1. **Terminated (Success)**: Agent reaches goal
2. **Terminated (Failure)**: Invalid state (should be rare)
3. **Truncated**: Step limit reached

**Invariants:**

- Each episode has unique seed (for tracking)
- Trajectory length ≤ max_steps
- Final state consistent with termination reason
- Reproducible: same seed → same episode

---

## 🏗️ Architecture Decisions

### Component-Based Design (Dependency Injection)

**Decision Date:** 2025-10-14  
**Status:** CANONICAL  

**Core Principle:** Environment is assembled from swappable components via dependency injection (DI).

**Rationale:**

- **Research tool extensibility**: Users need to inject novel algorithms without forking
- **Testing**: Easy to inject mocks and test components in isolation
- **Zero coupling**: Components communicate via Protocol interfaces only
- **Progressive disclosure**: Simple by default, powerful when needed

**Architecture:**

```text
                    Factory
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ActionProcessor  ObservationModel  RewardFunction
          │           │           │
          └───────────┴───────────┘
                      │
              ComponentBasedEnvironment
```

**Component Interfaces (Protocols):**

- `ActionProcessor`: Defines action space and action→movement mapping
- `ObservationModel`: Defines observation space and state→observation computation
- `RewardFunction`: Defines reward computation and goal detection
- `PlumeModel`: Defines concentration field dynamics

**Assembly Patterns:**

1. **Simple (Default):** Factory with defaults

   ```python
   env = make_env()  # Uses DiscreteGridActions, ConcentrationSensor, SparseGoalReward
   ```

2. **Customized (String-based):** Factory with built-in options

   ```python
   env = make_env(action_type='oriented', observation_type='antennae')
   ```

3. **Extended (Instance-based):** Factory with custom implementations

   ```python
   env = make_env(action_type=MyNovelActions(), observation_type=MyNovelSensor())
   ```

**See also (external plug‑and‑play example):**

- The `plug-and-play-demo` demonstrates DI usage from an external project, including applying the core `ConcentrationNBackWrapper(n=2)` via a spec-first flow and running a minimal episode.
  - Quick run from repo root: `python plug-and-play-demo/main.py`
  - Details: `plug-and-play-demo/README.md`
  - Config-based DI via `SimulationSpec`: see README section “Config‑based DI via SimulationSpec”; example spec at `plug-and-play-demo/configs/simulation_spec.json`

**Semantic Invariants:**

- Components MUST implement their Protocol completely
- Components MUST NOT directly access other components' internal state
- Factory MUST validate component compatibility before assembly
- Environment MUST work with any valid component combination

---

## 🔗 Relationships & Dependencies

### Dependency Graph

```text
Environment (ComponentBasedEnvironment)
    ├── ActionProcessor (injected)
    │   └── Defines: action_space, process_action()
    │
    ├── ObservationModel (injected)
    │   └── Defines: observation_space, compute_observation()
    │
    ├── RewardFunction (injected)
    │   └── Defines: compute_reward(), check_termination()
    │
    ├── PlumeModel (injected)
    │   └── ConcentrationField
    │
    ├── StateManager
    │   ├── AgentState
    │   └── BoundaryEnforcer
    │
    └── Renderer (optional)
        ├── ColorScheme
        └── VisualizationState
```

### Component Responsibilities

| Component | Owns | Decides | Cannot Access |
|-----------|------|---------|---------------|
| Environment | High-level API | Episode flow | Internal state of components |
| StateManager | Agent state | Position validity | Reward logic |
| EpisodeManager | Episode lifecycle | Termination | Plume dynamics |
| RewardCalculator | Reward logic | Goal detection | Agent actions |
| PlumeModel | Concentration field | Plume dynamics | Agent state |
| BoundaryEnforcer | Position validation | Wrapping/clipping | Episode state |

**Invariant:** Components communicate through defined interfaces, no direct state access.

---

## 🧮 Mathematical Model

### Coordinate System

```text
(0,0) ───────────────▶ x (width)
  │
  │    Grid coordinates
  │    Origin at top-left
  │    
  ▼
  y (height)
```

**Semantics:**

- `(0, 0)` is top-left corner
- `(width-1, height-1)` is bottom-right corner
- Negative coordinates are valid (off-grid)
- Boundary enforcer constrains agent to `[0, width) × [0, height)`

---

### Distance Metric

**Definition:** Euclidean distance (L2 norm)

```python
distance = sqrt((x₂ - x₁)² + (y₂ - y₁)²)
```

**Invariants:**

- `distance(a, b) = distance(b, a)` (symmetry)
- `distance(a, a) = 0` (identity)
- `distance(a, c) ≤ distance(a, b) + distance(b, c)` (triangle inequality)
- Non-negative always

**Goal Detection:**

```python
goal_reached = (distance_to_source ≤ goal_radius)
```

**Boundary Condition:** At `distance = goal_radius` exactly, goal IS reached (≤, not <).

---

### Action Space

**Definition:** Discrete(9)

```text
Action Mapping:
  7  0  1      ↖  ↑  ↗
  6  8  2   =  ←  ·  →
  5  4  3      ↙  ↓  ↘

Action 8 = Stay/No-op
```

**Movement Vectors:**

```python
MOVEMENTS = {
    0: (0, -1),   # North
    1: (1, -1),   # Northeast
    2: (1, 0),    # East
    3: (1, 1),    # Southeast
    4: (0, 1),    # South
    5: (-1, 1),   # Southwest
    6: (-1, 0),   # West
    7: (-1, -1),  # Northwest
    8: (0, 0),    # Stay
}
```

**Invariants:**

- All movements are unit steps (1 or √2 distance)
- Diagonal movements cost same as cardinal (no movement cost)
- Invalid actions cause `ValidationError` (fail-fast)

---

## 🎲 Randomness & Determinism

### Seeding Model

**Hierarchy:**

```text
Base Seed (user-provided or entropy)
    │
    ├─▶ Environment Seed
    │       │
    │       ├─▶ Episode 0 Seed
    │       ├─▶ Episode 1 Seed
    │       └─▶ Episode N Seed
    │
    ├─▶ Plume Model Seed
    └─▶ Agent Policy Seed (external)
```

**Derivation:**

```python
episode_seed = hash(base_seed, episode_number, experiment_id)
```

**Invariant:** Same base seed + episode number → same episode seed → same episode trajectory.

---

### Sources of Randomness

| Source | Seeded? | Affects |
|--------|---------|---------|
| Initial agent position | Yes | Starting state |
| Plume source location | Yes | Goal position |
| Plume field (if stochastic) | Yes | Observations |
| Agent policy (external) | Maybe | Actions |

**Invariant:** Given same seeds, environment behavior is deterministic.

---

## 🔍 Validation Model

### Two-Layer Validation

**Layer 1: Type Validation** (entry points)

- Check types: `isinstance(x, expected_type)`
- Check ranges: `min_value ≤ x ≤ max_value`
- Check special values: `math.isfinite(x)`, `x is not None`
- **Raises:** `ValidationError` immediately

**Layer 2: Semantic Validation** (deeper logic)

- Check state consistency: "can step() be called now?"
- Check component interactions: "is agent within grid?"
- Check invariants: "does total_reward match sum of step rewards?"
- **Raises:** `StateError` or `ComponentError`

**Principle:** Fail as early as possible, with context.

---

### Validation Contexts

**Strict Mode (Deprecated):**

- ❌ No longer used - removed in refactor
- All validation is now "strict" by default

**Validation Contexts:**

```python
@dataclass
class ValidationContext:
    operation_name: str
    component_name: str
    additional_constraints: Dict[str, Any] = field(default_factory=dict)
```

Used for debugging: "which component detected the error?"

---

## 📊 Performance Model

### Latency Targets

| Operation | Target | Measured By |
|-----------|--------|-------------|
| `env.step()` | < 1ms | `PerformanceMetrics` |
| `env.reset()` | < 10ms | `PerformanceMetrics` |
| `env.render()` | < 16ms (60 FPS) | Frame timing |
| Reward calculation | < 0.5ms | Internal timing |

**Invariant:** Performance does NOT affect correctness (semantic meaning unchanged if slower).

---

### Caching Strategy

**Cacheable:**

- Distance calculations (same coordinates → same distance)
- Plume field lookups (static plume)
- Validation results (same input → same result)

**Not Cacheable:**

- Agent state (changes every step)
- Episode statistics (accumulated)
- Random number generation (stateful)

**Cache Invalidation:**

- Distance cache: Cleared on `reset()`
- Plume cache: Never (static)
- State cache: No cache (always recomputed)

---

## 🔄 State Machine

### Environment State Machine

```text
┌──────────────────────────────────────────────────────┐
│                    UNINITIALIZED                     │
│  (object created but __init__ not complete)          │
└───────────────────────┬──────────────────────────────┘
                        │ __init__()
{{ ... }}
        │   └─────────────────────────────────┘
        │
        │ (terminated or truncated)
        ▼
┌──────────────────────────────────────────────────────┐
│                    EPISODE_END                       │
│  (one step after termination/truncation)            │
└───────────────────────┬──────────────────────────────┘
                        │ reset()
                        ▼
                 Back to ACTIVE

**Allowed Transitions:**
- `INITIALIZED → ACTIVE` via `reset()`
- `ACTIVE → ACTIVE` via `step()` (not terminated)
- `ACTIVE → TERMINATED` via `step()` (terminated/truncated)
- `TERMINATED → ACTIVE` via `reset()`
- `ANY → CLOSED` via `close()`

**Forbidden Transitions:**
- `CLOSED → anything` (cannot reopen)
- `UNINITIALIZED → ACTIVE` (must initialize first)
- `step()` on `TERMINATED` without `reset()` first

---

## 🧪 Semantic Invariants (Testable)

### Must Always Hold:

#### 1. **Position Invariant**
```python
assert env.agent_state.position.x >= 0
assert env.agent_state.position.y >= 0
assert env.agent_state.position.x < env.grid_size.width
assert env.agent_state.position.y < env.grid_size.height
```

#### 2. **Step Count Invariant**

```python
initial_steps = env.agent_state.step_count
env.step(action)
assert env.agent_state.step_count == initial_steps + 1
```

#### 3. **Reward Accumulation Invariant**

```python
initial_total = env.agent_state.total_reward
obs, reward, term, trunc, info = env.step(action)
assert env.agent_state.total_reward == initial_total + reward
```

#### 4. **Determinism Invariant**

```python
# Two environments with same seed
trajectory1 = run_episode(env1, seed=42)
trajectory2 = run_episode(env2, seed=42)
assert trajectory1 == trajectory2
```

#### 5. **Goal Detection Consistency**

```python
distance = calculate_distance(agent_pos, source_pos)
goal_reached = (distance <= goal_radius)
reward = (1.0 if goal_reached else 0.0)
# These three must be consistent
```

#### 6. **Termination Consistency**

```python
if terminated:
    assert goal_reached or invalid_state
if truncated:
    assert step_count >= max_steps
# Cannot be both terminated and truncated in same step (usually)
```

#### 7. **State Immutability (Configs)**

```python
config = RewardCalculatorConfig(...)
original_radius = config.goal_radius
# ... use config ...
assert config.goal_radius == original_radius  # unchanged
```

---

## 🎨 User-Facing API Design

### Progressive Disclosure Pattern

**Decision Date:** 2025-10-14  
**Status:** CANONICAL

**Principle:** Complexity should be opt-in, not opt-out.

**Three Levels of Interface:**

#### Level 1: Simple (Default)

**Target Users:** Students, quick experiments, tutorials  
**Complexity:** Minimal - just works

```python
import plume_nav_sim
env = plume_nav_sim.make_env()  # Uses sensible defaults
```

**Semantic Guarantee:** Returns a fully-functional environment with default configuration.

---

#### Level 2: Customized (String-based)

**Target Users:** Researchers comparing approaches, RL practitioners  
**Complexity:** Medium - choose from built-in options

```python
env = plume_nav_sim.make_env(
    action_type='oriented',       # Choose from: 'discrete', 'oriented'
    observation_type='antennae',  # Choose from: 'concentration', 'antennae'
    reward_type='step_penalty'    # Choose from: 'sparse', 'step_penalty'
)
```

**Semantic Guarantee:** String shortcuts map to validated built-in components.

---

#### Level 3: Extended (Instance-based)

**Target Users:** Novel algorithm developers, framework builders  
**Complexity:** Full - inject custom implementations

```python
from plume_nav_sim.interfaces import ActionProcessor

class MyNovelActions(ActionProcessor):
    # Custom implementation
    pass

env = plume_nav_sim.make_env(
    action_type=MyNovelActions()  # Inject custom component
)
```

**Semantic Guarantee:** Custom components must implement Protocol interface completely.

---

### API Design Invariants

**Must Hold:**

- Level 1 MUST work with zero configuration
- Level 2 MUST accept all Level 1 parameters
- Level 3 MUST accept all Level 2 parameters
- Each level MUST be independently documented
- Implementation details MUST NOT leak into simple levels

**Forbidden:**

- Requiring users to understand DI to use defaults
- Exposing internal caching/registration complexity
- Multiple ways to do the same simple thing
- Deprecation warnings for normal usage

---

## 📚 Glossary

### Key Terms

**Agent**: The learning entity navigating the environment.

**Plume**: Concentration field representing odor/chemical diffusion.

**Episode**: Single attempt from reset to termination.

**Trajectory**: Sequence of (state, action, reward, next_state) tuples.

**Seed**: Integer controlling randomness for reproducibility.

**Observation**: Agent's view of environment at current timestep.

**Reward**: Scalar feedback signal (sparse binary in this domain).

**Terminated**: Episode ended due to goal/failure (env-specific reason).

**Truncated**: Episode ended due to time limit (external constraint).

**Grid**: 2D discrete coordinate space containing agent and plume.

**Source**: Origin point of plume (goal location).

**Goal Radius**: Distance threshold for goal detection.

**Boundary Enforcer**: Component ensuring agent stays within grid.

**State Manager**: Component tracking agent state and episode progress.

**Episode Manager**: Component coordinating episode lifecycle.

**Reward Calculator**: Component computing rewards and goal detection.

---

## 🎭 Semantic Patterns

### Pattern 1: **Configuration-First**

All components use configuration objects, not constructor kwargs.

```python
# GOOD - explicit config
config = RewardCalculatorConfig(goal_radius=5.0, ...)
calculator = RewardCalculator(config)

# BAD - kwargs soup
calculator = RewardCalculator(goal_radius=5.0, reward=1.0, ...)
```

**Rationale:** Configs are testable, serializable, and self-documenting.

---

### Pattern 2: **Fail-Fast Validation**

Validate at entry, not deep in call stack.

```python
# GOOD - validate immediately
def step(self, action: int):
    if not isinstance(action, int):
        raise ValidationError(...)
    # Now proceed with logic

# BAD - validate late
def step(self, action):
    # ... many lines of logic ...
    result = self._compute_something(action)
    # ... action type error happens here
```

---

### Pattern 3: **Immutable Results**

Return frozen dataclasses, not mutable dicts.

```python
# GOOD - immutable result
@dataclass(frozen=True)
class RewardResult:
    reward: float
    goal_reached: bool

# BAD - mutable dict
result = {'reward': 1.0, 'goal_reached': True}
# Can be modified accidentally!
```

---

### Pattern 4: **Explicit State Machines**

Use enums for state, check transitions explicitly.

```python
class EnvironmentState(Enum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    TERMINATED = "terminated"
    CLOSED = "closed"

def step(self):
    if self.state != EnvironmentState.ACTIVE:
        raise StateError(f"Cannot step in {self.state} state")
```

---

### Pattern 5: **Semantic Exceptions**

Use specific exception types that convey meaning.

```python
# GOOD - semantic
raise ValidationError("seed must be non-negative", ...)
raise StateError("Cannot step after close", ...)
raise ComponentError("Reward calc failed", ...)

# BAD - generic
raise ValueError("Invalid seed")
raise RuntimeError("Error in step")
raise Exception("Something broke")
```

---

## 🔐 Backward Compatibility

### Semantic Versioning

- **Major version bump**: Breaking change to semantic meaning
  - Example: Change reward from sparse to dense
  - Example: Change coordinate system origin
  
- **Minor version bump**: Add new feature, preserve existing semantics
  - Example: Add new plume model type
  - Example: Add optional observation fields
  
- **Patch version bump**: Bug fixes, no semantic change
  - Example: Fix incorrect goal detection
  - Example: Fix memory leak

### Deprecation Policy

1. Mark old API as deprecated (add warning)
2. Document migration path
3. Maintain old API for 3 months minimum
4. Remove in next major version

---

## 📊 Test Coverage Requirements

### What Must Be Tested

**Unit Tests:**

- Each semantic invariant (1 test per invariant minimum)
- Each component's public API
- Each exception path (ValidationError, StateError, etc.)
- Edge cases (boundaries, special values)

**Integration Tests:**

- Component interactions (StateManager + EpisodeManager)
- Full episode workflow (reset → steps → termination)
- Cross-module contracts (RewardCalculator + AgentState)

**Property Tests:**

- Determinism (same seed → same trajectory)
- Invariants hold under random actions
- State transitions are valid

**Not Required:**

- Parametric tests of same logic with different values
- Performance tests in unit test suite (separate suite)
- Exhaustive combination testing (combinatorial explosion)
