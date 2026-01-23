# API Contracts & Data Model Specification

**Version:** 1.0.0  
**Date:** 2025-09-30  
**Status:** CANONICAL SOURCE OF TRUTH

> This document defines immutable contracts for plume_nav_sim APIs. Changes to signatures documented here require major version bump and deprecation cycle.

---

## 🎯 Core Principles

1. **Fail Fast, Fail Loud** - Invalid inputs raise exceptions immediately at entry points
2. **Explicit Over Implicit** - Required parameters have no defaults, optional parameters do
3. **Deterministic** - Given same inputs, produce same outputs (modulo RNG with same seed)
4. **Type-Safe** - Use dataclasses with type hints; validate at construction time
5. **Immutable Configs** - Configuration dataclasses are frozen after validation

---

## 📦 Exception Hierarchy

### Base Exception: `PlumeNavSimError`

**Signature (IMMUTABLE):**
```python
class PlumeNavSimError(Exception):
    def __init__(
        self,
        message: str,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    )
```

**Contract:**
- `message`: Human-readable error description (REQUIRED)
- `context`: Additional debugging information (OPTIONAL)
- `severity`: Error severity level from enum (OPTIONAL, default: MEDIUM)
- All exceptions store: `error_id`, `timestamp`, `recovery_suggestion`

---

### `ValidationError(PlumeNavSimError, ValueError)`

**Signature (IMMUTABLE):**
```python
class ValidationError(PlumeNavSimError, ValueError):
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        expected_format: Optional[str] = None,
        parameter_constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
    )
```

**Contract:**
- `message`: What validation failed (REQUIRED)
- `parameter_name`: Name of parameter that failed (OPTIONAL but RECOMMENDED)
- `parameter_value`: The value that was provided (OPTIONAL)
  - **NEVER USE:** `invalid_value` - this name is DEPRECATED
  - **RATIONALE:** Value isn't inherently "invalid", only invalid in context
- `expected_format`: Description of valid format (OPTIONAL)
- `parameter_constraints`: Dict of constraint name → value (OPTIONAL)
- `context`: Additional error context (OPTIONAL)

**Semantic Invariants:**
- Raised at entry points of functions/methods
- Never raised for internal logic errors (use `ComponentError` instead)
- Must include at least one of: `parameter_name`, `expected_format`, or `parameter_constraints`

**Example Usage:**
```python
if not isinstance(seed, int) or seed < 0:
    raise ValidationError(
        "Seed must be a non-negative integer",
        parameter_name="seed",
        parameter_value=seed,
        expected_format="int >= 0",
        parameter_constraints={"min": 0, "type": "int"}
    )
```

---

### `ComponentError(PlumeNavSimError)`

**Signature (IMMUTABLE):**
```python
class ComponentError(PlumeNavSimError):
    def __init__(
        self,
        message: str,
        component_name: str,
        operation_name: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
    )
```

**Contract:**
- `message`: What went wrong (REQUIRED)
- `component_name`: Which component failed (REQUIRED)
- `operation_name`: Which operation was being performed (OPTIONAL but RECOMMENDED)
- `underlying_error`: Wrapped exception if applicable (OPTIONAL)
- **NEVER USE:** `severity=` parameter - severity is always HIGH for ComponentError

**Semantic Invariants:**
- Raised for internal component failures (not input validation)
- Always includes component identification for debugging
- Used to wrap lower-level exceptions with context

**Example Usage:**
```python
try:
    result = complex_calculation()
except Exception as e:
    raise ComponentError(
        f"Reward calculation failed: {e}",
        component_name="RewardCalculator",
        operation_name="calculate_reward",
        underlying_error=e,
    ) from e
```

---

### `ConfigurationError(PlumeNavSimError)`

**Signature (IMMUTABLE):**
```python
class ConfigurationError(PlumeNavSimError):
    def __init__(
        self,
        message: str,
        config_parameter: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_options: Optional[Dict[str, Any]] = None,
    )
```

**Contract:**
- `message`: Configuration problem description (REQUIRED)
- `config_parameter`: Name of invalid config parameter (OPTIONAL)
- `parameter_value`: The provided value (OPTIONAL)
  - **NEVER USE:** `invalid_value` - DEPRECATED
- `valid_options`: Dict of valid options/ranges (OPTIONAL)

---

### `StateError(PlumeNavSimError)`

**Signature (IMMUTABLE):**
```python
class StateError(PlumeNavSimError):
    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        component_name: Optional[str] = None,
    )
```

**Contract:**
- Raised when operation called in invalid state
- Severity: HIGH
- Used for state machine violations

---

### `RenderingError(PlumeNavSimError)`

**Signature (IMMUTABLE):**
```python
class RenderingError(PlumeNavSimError):
    def __init__(
        self,
        message: str,
        render_mode: Optional[str] = None,
        backend_name: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
        context: Optional[Any] = None,
    )
```

**Contract:**
- Used for visualization failures
- Severity: MEDIUM
- Includes fallback suggestions

---

## 🏗️ Configuration Dataclasses

### General Contract for All Configs

**Required Properties:**
1. Use `@dataclass` decorator (or `@dataclass(frozen=True)` if immutable)
2. All parameters have type hints
3. Required parameters listed BEFORE optional parameters
4. Optional parameters have default values
5. Include `__post_init__()` for validation
6. Include `validate()` method returning `bool` or raising `ValidationError`

**Standard Structure:**
```python
@dataclass
class SomeConfig:
    # REQUIRED parameters (no defaults)
    required_param: int
    another_required: str
    
    # OPTIONAL parameters (with defaults)
    optional_param: bool = True
    another_optional: int = 100
    
    def __post_init__(self):
        """Validate immediately at construction."""
        self.validate()
    
    def validate(self, strict_mode: bool = False) -> bool:
        """Validate configuration parameters.
        
        Args:
            strict_mode: Enable additional validation checks
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Validation logic here
        return True
```

---

### `RewardCalculatorConfig`

**Signature (IMMUTABLE):**
```python
@dataclass
class RewardCalculatorConfig:
    # REQUIRED
    goal_radius: float
    reward_goal_reached: float
    reward_default: float
    
    # OPTIONAL (with defaults)
    distance_calculation_method: str = "euclidean"
    distance_precision: float = 1e-12
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
```

**Validation Rules:**
- `goal_radius >= 0` and `math.isfinite(goal_radius)`
- `reward_goal_reached != reward_default` (learning signal requirement)
- All reward values must be finite (no NaN, no infinity)
- `distance_calculation_method in ["euclidean", "manhattan", "chebyshev"]`
- `0 < distance_precision <= 1e-3`

---

### `EpisodeManagerConfig`

**Signature (IMMUTABLE):**
```python
@dataclass
class EpisodeManagerConfig:
    # REQUIRED
    max_steps: int
    
    # OPTIONAL (with defaults)
    enable_performance_monitoring: bool = True
    enable_state_validation: bool = True
    track_statistics: bool = True
    auto_reset: bool = False
```

**Validation Rules:**
- `max_steps > 0`
- Boolean flags accept only `True` or `False`

---

### `StateManagerConfig`

**Signature (IMMUTABLE):**
```python
@dataclass
class StateManagerConfig:
    # REQUIRED
    grid_size: GridDimensions
    max_steps: int
    
    # OPTIONAL (with defaults)
    enable_boundary_enforcement: bool = True
    enable_state_validation: bool = True
    track_history: bool = False
```

---

## 🔢 Core Data Types

### `Coordinates`

**Definition:**
```python
@dataclass(frozen=True)
class Coordinates:
    x: int
    y: int
```

**Contract:**
- Immutable (frozen dataclass)
- Both `x` and `y` must be integers
- Can be negative (coordinate system includes negative quadrants)
- Hashable (can be used as dict keys)

**Factory Function:**
```python
def create_coordinates(x: int, y: int) -> Coordinates:
    """Create validated Coordinates.
    
    Args:
        x: X-coordinate (integer)
        y: Y-coordinate (integer)
        
    Returns:
        Validated Coordinates instance
        
    Raises:
        ValidationError: If x or y are not integers
    """
```

**DEPRECATED Usage:**
- ❌ `create_coordinates(x=5, y=10)` - kwargs no longer supported
- ✅ `create_coordinates(5, 10)` - positional args only
- ✅ `Coordinates(5, 10)` - direct construction preferred

---

### `GridDimensions`

**Definition:**
```python
@dataclass(frozen=True)
class GridDimensions:
    width: int
    height: int
```

**Contract:**
- Immutable
- `width > 0` and `height > 0`
- Maximum practical size: 10,000 × 10,000
- Methods: `.area()`, `.to_tuple()`, `.contains(coords)`
- **DOES NOT HAVE:** `.to_dict()` method

---

### `AgentState`

**Definition:**
```python
@dataclass
class AgentState:
    position: Coordinates
    step_count: int
    goal_reached: bool = False
    total_reward: float = 0.0
```

**Contract:**
- `step_count >= 0`
- `position` must be valid Coordinates
- Methods: `.add_reward(amount: float)`, `.increment_step()`

---

### `RewardResult`

**Definition:**
```python
@dataclass
class RewardResult:
    reward: float
    goal_reached: bool
    distance_to_goal: float
    
    # Optional performance tracking
    calculation_time_ms: Optional[float] = None
    goal_achievement_reason: Optional[str] = None
```

**Contract:**
- All fields validated in `__post_init__()`
- `reward` must be finite
- `distance_to_goal >= 0`
- `goal_reached` is boolean

---

### `TerminationResult`

**Definition:**
```python
@dataclass
class TerminationResult:
    terminated: bool
    truncated: bool
    termination_reason: str
    
    # Optional final state info
    final_step_count: Optional[int] = None
    final_distance: Optional[float] = None
    termination_details: Dict[str, Any] = field(default_factory=dict)
```

**Contract:**
- Follows Gymnasium API: `terminated` XOR `truncated` (usually)
- `termination_reason` must be non-empty string
- `terminated=True` means goal achieved
- `truncated=True` means step limit reached

---

## 🎲 Seeding System

### Seed Values

**Contract:**
- Valid seeds: `int` in range `[0, 2**32 - 1]`
- `None` is valid (means "use entropy")
- Negative integers are INVALID
- Non-integers are INVALID

**Constants (IMMUTABLE):**
```python
SEED_MIN_VALUE = 0
SEED_MAX_VALUE = 2**32 - 1  # 4,294,967,295
```

---

### `validate_seed()` Function

**Signature (IMMUTABLE):**
```python
def validate_seed(seed: Optional[int]) -> tuple[Optional[int], str]:
    """Validate seed value.
    
    Args:
        seed: Seed to validate (None or int)
        
    Returns:
        Tuple of (normalized_seed, error_message)
        - normalized_seed: None if invalid, else the seed
        - error_message: Empty string if valid, else error description
        
    Raises:
        Never raises - returns error in tuple instead
    """
```

**Contract:**
- Returns `(seed, "")` if valid
- Returns `(None, "error message")` if invalid
- Never raises exceptions
- `None` input returns `(None, "")` (valid)

---

### `create_seeded_rng()` Function

**Signature (IMMUTABLE):**
```python
def create_seeded_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create seeded random number generator.
    
    Args:
        seed: Seed value (None for entropy-based)
        
    Returns:
        numpy.random.Generator instance
        
    Raises:
        ValidationError: If seed is invalid
    """
```

---

### `SeedManager`

**Constructor (IMMUTABLE):**
```python
class SeedManager:
    def __init__(
        self,
        default_seed: Optional[int] = None,
        enable_validation: bool = True,
        thread_safe: bool = False,
    ):
```

**Contract:**
- `default_seed`: Base seed for all generators (OPTIONAL)
- `enable_validation`: Validate seeds on creation (OPTIONAL, default True)
- `thread_safe`: Use locks for thread safety (OPTIONAL, default False)

---

## 📊 Performance & Metrics

- Timing data is recorded as simple dictionaries (operation → list of ms values).
- No `PerformanceMetrics` class; use component `get_performance_metrics()` summaries.

---

## 🎨 Constants

### Core Constants (IMMUTABLE)

**Defined in `core/constants.py`:**
```python
# Grid & Space
DEFAULT_GRID_WIDTH = 100
DEFAULT_GRID_HEIGHT = 100
DEFAULT_GRID_SIZE = GridDimensions(DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT)
MIN_GRID_SIZE = (1, 1)

# Rewards
# Use a tiny positive radius to avoid zero-radius edge cases while
# preserving the semantics of "exact/near-exact" goal detection.
import numpy as np
DEFAULT_GOAL_RADIUS = float(np.finfo(np.float32).eps)
REWARD_GOAL_REACHED = 1.0
REWARD_DEFAULT = 0.0

# Observation Space
CONCENTRATION_RANGE = (0.0, 1.0)  # Min, Max concentration values

# Action Space
ACTION_SPACE_SIZE = 9  # 8 directions + stay

# Performance
PERFORMANCE_TARGET_STEP_LATENCY_MS = 1.0
DISTANCE_PRECISION = 1e-10
```

**Contract:**
- These constants MUST exist
- Changes require major version bump
- Used across test suite and implementation

---

## 🔄 Validation Functions

### Pattern: Two-Stage Validation

All validation follows this pattern:

```python
def validate_thing(
    thing: Any,
    context: Optional[ValidationContext] = None,
) -> ValidationResult:
    """Validate thing with comprehensive checks.
    
    Args:
        thing: Object to validate
        context: Optional validation context for additional checks
        
    Returns:
        ValidationResult with errors/warnings
        
    Raises:
        Never raises - returns validation result
    """
```

### Validation Function Signatures (IMMUTABLE)

```python
def validate_coordinates(
    coordinates: Any,
    grid_size: Optional[GridDimensions] = None,
    context: Optional[ValidationContext] = None,
) -> Coordinates:
    """Returns validated Coordinates or raises ValidationError."""

def validate_grid_size(
    grid_size: Any,
    context: Optional[ValidationContext] = None,
) -> GridDimensions:
    """Returns validated GridDimensions or raises ValidationError."""

# NO LONGER HAS: strict_mode parameter (removed in refactor)
```

---

## 🚫 Removed APIs (Do Not Use)

These APIs were removed for simplicity. There is ONE correct way:

```python
# ❌ REMOVED - use parameter_value
ValidationError(..., invalid_value=x)

# ❌ REMOVED - severity not accepted
ComponentError(..., severity=ErrorSeverity.HIGH)

# ❌ REMOVED - use parameter_value
ConfigurationError(..., invalid_value=x)

# ❌ REMOVED - use to_tuple()
grid_size.to_dict()

# ❌ REMOVED - use get_performance_summary()
performance_metrics.get_summary()
performance_metrics.get_statistics()

# ❌ REMOVED - access .performance_metrics directly
episode_result.get_performance_metrics()

# ❌ REMOVED - use tuple: create_coordinates((x, y))
create_coordinates(x=5, y=10)

# ❌ REMOVED - strict_mode parameter eliminated
validate_*(..., strict_mode=True)

# ❌ REMOVED - session_id not needed
EpisodeStatistics(..., session_id="x")
```

### ✅ Correct API (Single Way):

```python
# Exceptions - clean, explicit
ValidationError("msg", parameter_name="x", parameter_value=42)
ComponentError("msg", component_name="Component", operation_name="op")
ConfigurationError("msg", config_parameter="x", parameter_value=val)

# Data structures - simple
grid_size.to_tuple()  # Returns (width, height)
metrics.get_performance_summary()  # Only method
episode_result.performance_metrics  # Direct access

# Functions - single signature
create_coordinates((x, y))  # Tuple only
validate_seed(seed)  # No strict_mode
```

---

## 🔐 Semantic Invariants

### Must Always Hold:

1. **Determinism**: `same inputs + same seed → same outputs`
2. **State Isolation**: No global state (except registry)
3. **Fail-Fast**: Invalid inputs caught at entry, not deep in call stack
4. **Type Safety**: Runtime type checks on untrusted boundaries
5. **Resource Cleanup**: Environments close cleanly even after errors
6. **Step Atomicity**: Either step completes fully or state unchanged

### Gymnasium Compliance:

**PlumeSearchEnv inherits from `gymnasium.Env`** and follows modern Gymnasium API (v1.0+):

```python
# Standard RL loop MUST work:
obs, info = env.reset(seed=42)  # Modern API: seed via reset()
for _ in range(max_steps):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

**Deprecated APIs:**
- ⚠️ `env.seed(seed)` — Deprecated. Accepted for backward compatibility in
  the public wrapper (calls `reset(seed=...)` under the hood). Prefer
  `env.reset(seed=seed)`.
- ❌ `done` flag — Use `terminated` and `truncated` instead

---

## 🔒 System Invariants

### Global Invariants (MUST always hold)

**G1: No Global Mutable State**
```python
# All state is encapsulated in objects
# No module-level mutable variables (except registry)
# Pure functions have no side effects
```

**G2: Determinism with Seed Control**
```python
∀ env₁, env₂, seed, actions:
  env₁.reset(seed) then apply(actions) ==
  env₂.reset(seed) then apply(actions)

# Same seed + actions → identical trajectories
```

**G3: Type Safety at Boundaries**
```python
# All public APIs validate input types
# Raise TypeError immediately for wrong types
# No silent coercion or implicit conversions
```

**G4: Fail Fast on Invalid Input**
```python
# Validation at entry point (not deep in call stack)
# ValidationError with clear message
# No partially-invalid states
```

### Component-Specific Invariants

See detailed specifications in `contracts/` directory:
- **Environment State Machine:** `contracts/environment_state_machine.md`
- **Core Data Types:** `contracts/core_types.md`
- **Reward Function:** `contracts/reward_function.md`
- **Concentration Field:** `contracts/concentration_field.md`

---

## 🔄 State Machines

### Environment Lifecycle

```
┌─────────┐  reset()   ┌───────┐  step()    ┌──────────┐
│ CREATED │──────────→ │ READY │──────────→ │ READY    │
└─────────┘            └───────┘            └──────────┘
                           │                      │
                           │ step()               │ step()
                           ↓                      ↓
                       ┌────────────┐      ┌──────────┐
                       │ TERMINATED │      │TRUNCATED │
                       └────────────┘      └──────────┘
                           │                      │
                           └──────┬───────────────┘
                                  │ reset()
                                  ↓
                              ┌───────┐
                              │ READY │
                              └───────┘
                                  │
                                  │ close()
                                  ↓
                              ┌────────┐
                              │ CLOSED │ (terminal)
                              └────────┘
```

**See full specification:** `contracts/environment_state_machine.md`

Key Rules:
- Cannot `step()` before `reset()`
- Cannot use after `close()`
- `CLOSED` is terminal (no transitions out)

---

## 📐 Mathematical Properties

### Distance Metric (Euclidean L2)

```
d: Coordinates × Coordinates → ℝ₊

Properties:
  1. Non-negativity: d(a, b) ≥ 0
  2. Identity: d(a, a) = 0
  3. Symmetry: d(a, b) = d(b, a)
  4. Triangle Inequality: d(a, c) ≤ d(a, b) + d(b, c)

Implementation:
  d(p₁, p₂) = √((x₂-x₁)² + (y₂-y₁)²)
```

### Reward Function

```
reward: Coordinates × Coordinates × ℝ₊ → {0.0, 1.0}

reward(agent, source, radius) = {
  1.0  if distance(agent, source) ≤ radius
  0.0  otherwise
}

Properties:
  1. Binary: result ∈ {0.0, 1.0}
  2. Deterministic: same inputs → same output
  3. Pure: no side effects
  4. Boundary Inclusive: d = radius → 1.0
  5. Symmetric: reward(a, b, r) = reward(b, a, r)
```

**Full specification:** `contracts/reward_function.md`

### Concentration Field (Gaussian)

```
C: Coordinates → [0, 1]

C(x, y) = exp(-d²(x, y, source) / (2σ²))

Physical Invariants:
  I1: ∀ p: 0 ≤ C(p) ≤ 1  (bounded)
  I2: C(source) ≥ C(p) ∀ p  (max at source)
  I3: d(p₁, source) ≤ d(p₂, source) ⇒ C(p₁) ≥ C(p₂)  (monotonic decay)
  I4: Radially symmetric around source
```

**Full specification:** `contracts/concentration_field.md`

---

## 🧪 Contract Testing

### Test Categories

All contracts MUST be verified by corresponding tests:

1. **Contract Guard Tests** (`tests/contracts/`)
   - Precondition validation
   - Postcondition verification
   - State transition enforcement

2. **Property Tests** (`tests/properties/`)
   - Mathematical properties (using Hypothesis)
   - Determinism, symmetry, monotonicity
   - Universal quantifiers (∀)

3. **Invariant Tests** (`tests/invariants/`)
   - Domain-specific invariants
   - Physical laws
   - Safety properties

4. **Schema Tests** (`tests/schemas/`)
   - Type safety
   - Data structure validation
   - API signature stability

**See full taxonomy:** `TEST_TAXONOMY.md`

### Contract Verification Checklist

When implementing or modifying a component:

- [ ] All preconditions enforced at entry
- [ ] All postconditions verified before return
- [ ] Class invariants hold before & after every method
- [ ] State transitions follow formal rules
- [ ] Property tests pass (50+ examples)
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Performance targets met

---

## 📚 Detailed Contract Specifications

For comprehensive contracts with mathematical precision:

### Public API (External Interface)

- **⭐ Gymnasium API:** `contracts/gymnasium_api.md`
  - Action space (current: Discrete(4))
  - Observation space (Dict structure)
  - Info dictionary (required/optional keys)
  - Method contracts (reset, step, close, render)
  - Determinism guarantees
  - Breaking change policy

### Internal Components

- **Environment:** `contracts/environment_state_machine.md`
  - Formal state definition (5 states)
  - Transition rules (inference notation)
  - 8 class invariants
  - Complete method contracts (reset, step, close, render)

- **Core Types:** `contracts/core_types.md`
  - `Coordinates`: Immutable 2D positions with distance metric
  - `GridSize`: Validated dimensions with bounds checking
  - `AgentState`: Mutable state with monotonic properties

- **Reward:** `contracts/reward_function.md`
  - Universal properties (purity, determinism)
  - Sparse binary model (current implementation)
  - Alternative reward models
  - Edge case specifications

- **Plume:** `contracts/concentration_field.md`
  - Universal physical laws (non-negativity, bounded, maximum at source)
  - Gaussian model properties (monotonic decay, radial symmetry, formula)
  - 7 invariants with classification
  - Performance requirements

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2025-09-30 | Added formal contracts, invariants, state machines, mathematical properties |
| 1.0.0 | 2025-09-30 | Initial canonical specification |

---

## ⚖️ Contract Enforcement

### How to Use This Document:

1. **When writing new code**: Check signatures here FIRST
2. **When tests fail**: Check if test or code violates contract
3. **When refactoring**: Update this doc BEFORE changing signatures
4. **When reviewing PRs**: Verify contract compliance

### Violation Policy:

- **Minor violations** (typos, missing optional params): Fix immediately
- **Major violations** (signature changes): Require architecture review
- **Breaking changes**: Major version bump + deprecation cycle (3 months minimum)

### Testing Contracts:

```python
# Example contract test
def test_validation_error_signature_is_stable():
    """Ensure ValidationError API matches CONTRACTS.md"""
    sig = inspect.signature(ValidationError.__init__)
    params = list(sig.parameters.keys())
    expected = ['self', 'message', 'parameter_name', 'parameter_value', 
                'expected_format', 'parameter_constraints', 'context']
    assert params == expected, "ValidationError signature changed - update CONTRACTS.md!"
```

---

**END OF CONTRACTS.MD**
