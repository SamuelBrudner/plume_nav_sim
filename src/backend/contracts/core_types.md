# Core Data Types Contract

**Components:** `Coordinates`, `GridSize`, `AgentState`  
**Version:** 1.1.0  
**Date:** 2025-09-30  
**Status:** CANONICAL - All implementations MUST conform

---

## 🎯 Purpose

Define formal contracts for core data types used throughout the system:
- **Coordinates:** Grid position representation
- **GridSize:** Grid dimensions with validation
- **AgentState:** Complete agent state tracking

---

## 📐 Type: Coordinates

### Mathematical Definition

```
Coordinates ⊂ ℤ²
Coordinates = {(x, y) | x ∈ ℤ, y ∈ ℤ}
```

A coordinate is an ordered pair of integers representing a position in 2D grid space.

### Type Specification

```python
@dataclass(frozen=True)
class Coordinates:
    """Immutable 2D integer coordinates.
    
    Invariants:
      I1: x ∈ ℤ (integer)
      I2: y ∈ ℤ (integer)
      I3: Immutable (frozen dataclass)
      I4: Hashable (can use in sets/dicts)
    """
    x: int
    y: int
```

### Constructor Contract

```python
def create_coordinates(position: tuple[int, int]) -> Coordinates:
    """Create Coordinates from tuple.
    
    Preconditions:
      P1: position = (x, y) where x, y ∈ ℤ
      P2: len(position) = 2
    
    Postconditions:
      C1: returns Coordinates(x, y)
      C2: result.x = position[0]
      C3: result.y = position[1]
      C4: isinstance(result, Coordinates)
    
    Raises:
      TypeError: If x or y not integers
      ValueError: If position wrong length
    
    Examples:
      create_coordinates((10, 20)) → Coordinates(x=10, y=20)
      create_coordinates((0, 0)) → Coordinates(x=0, y=0)
      create_coordinates((-5, -3)) → Coordinates(x=-5, y=-3)  # Negative OK
    """
```

### Properties

```python
# Equality
∀ c₁, c₂ ∈ Coordinates:
  c₁ = c₂ ⟺ (c₁.x = c₂.x) ∧ (c₁.y = c₂.y)

# Hashability
∀ c₁, c₂ ∈ Coordinates:
  c₁ = c₂ ⇒ hash(c₁) = hash(c₂)

# Immutability
∀ c ∈ Coordinates:
  c.x = x' raises AttributeError
  c.y = y' raises AttributeError
```

### Operations

#### `distance_to()`

```python
def distance_to(self, other: Coordinates) -> float:
    """Euclidean distance to another coordinate.
    
    Preconditions:
      P1: other is Coordinates
    
    Postconditions:
      C1: result ≥ 0 (non-negative)
      C2: result = √((other.x - self.x)² + (other.y - self.y)²)
      C3: isinstance(result, float)
    
    Mathematical Properties:
      1. Symmetry: a.distance_to(b) = b.distance_to(a)
      2. Identity: a.distance_to(a) = 0
      3. Positivity: a ≠ b ⇒ a.distance_to(b) > 0
      4. Triangle Inequality: 
         a.distance_to(c) ≤ a.distance_to(b) + b.distance_to(c)
    
    Determinism:
      Pure function, no side effects
    
    Examples:
      (0,0).distance_to((3,4)) = 5.0
      (10,10).distance_to((10,10)) = 0.0
      (0,0).distance_to((1,0)) = 1.0
    """
```

### Test Requirements

```python
# Property tests
@given(coords=st.tuples(st.integers(), st.integers()))
def test_coordinates_creation(coords):
    """Can create from any integer pair"""

@given(
    c1=coordinates_strategy(),
    c2=coordinates_strategy()
)
def test_distance_symmetry(c1, c2):
    """distance(a,b) = distance(b,a)"""
    assert c1.distance_to(c2) == c2.distance_to(c1)

@given(coords=coordinates_strategy())
def test_distance_identity(coords):
    """distance(a,a) = 0"""
    assert coords.distance_to(coords) == 0.0

@given(
    c1=coordinates_strategy(),
    c2=coordinates_strategy(),
    c3=coordinates_strategy()
)
def test_triangle_inequality(c1, c2, c3):
    """distance(a,c) ≤ distance(a,b) + distance(b,c)"""
    d_ac = c1.distance_to(c3)
    d_ab = c1.distance_to(c2)
    d_bc = c2.distance_to(c3)
    assert d_ac <= d_ab + d_bc + 1e-10  # tolerance for float

# Immutability
def test_coordinates_frozen():
    """Cannot modify after creation"""
    c = create_coordinates((5, 10))
    with pytest.raises(AttributeError):
        c.x = 20

# Hashability
def test_coordinates_hashable():
    """Can use in sets and dicts"""
    c1 = create_coordinates((5, 10))
    c2 = create_coordinates((5, 10))
    s = {c1, c2}
    assert len(s) == 1  # Same coordinates
```

---

## 📏 Type: GridSize

### Mathematical Definition

```
GridSize ⊂ ℕ₊ × ℕ₊
GridSize = {(w, h) | w, h ∈ ℕ₊, w ≤ MAX, h ≤ MAX}

where:
  MAX = MAX_GRID_DIMENSION = 10000
  ℕ₊ = {1, 2, 3, ...} (positive integers)
```

### Type Specification

```python
@dataclass(frozen=True)
class GridSize:
    """Immutable grid dimensions.
    
    Invariants:
      I1: width > 0 (positive)
      I2: height > 0 (positive)
      I3: width ≤ MAX_GRID_DIMENSION (10000)
      I4: height ≤ MAX_GRID_DIMENSION (10000)
      I5: Immutable (frozen dataclass)
      I6: Hashable
    """
    width: int
    height: int
    
    def __post_init__(self):
        """Validate invariants on construction."""
        if self.width <= 0 or self.height <= 0:
            raise ValidationError("Grid dimensions must be positive")
        if self.width > MAX_GRID_DIMENSION:
            raise ValidationError(f"Width exceeds maximum {MAX_GRID_DIMENSION}")
        if self.height > MAX_GRID_DIMENSION:
            raise ValidationError(f"Height exceeds maximum {MAX_GRID_DIMENSION}")
```

### Constructor Contract

```python
def create_grid_size(width: int, height: int) -> GridSize:
    """Create validated GridSize.
    
    Preconditions:
      P1: width > 0
      P2: height > 0
      P3: width ≤ MAX_GRID_DIMENSION
      P4: height ≤ MAX_GRID_DIMENSION
    
    Postconditions:
      C1: returns GridSize(width, height)
      C2: result.width = width
      C3: result.height = height
      C4: Invariants I1-I6 hold
    
    Raises:
      ValidationError: If any precondition violated
    
    Examples:
      create_grid_size(32, 32) → GridSize(32, 32)
      create_grid_size(1, 1) → GridSize(1, 1)  # Minimum
      create_grid_size(10000, 10000) → GridSize(10000, 10000)  # Maximum
      create_grid_size(0, 10) → ValidationError
      create_grid_size(10, -5) → ValidationError
    """
```

### Operations

#### `total_cells()`

```python
def total_cells(self) -> int:
    """Total number of grid cells.
    
    Preconditions:
      None (uses validated instance)
    
    Postconditions:
      C1: result = width × height
      C2: result > 0
      C3: result ≤ MAX_GRID_DIMENSION²
    
    Properties:
      - Deterministic: same grid → same result
      - Pure function: no side effects
    
    Examples:
      GridSize(32, 32).total_cells() = 1024
      GridSize(10, 20).total_cells() = 200
    """
```

#### `contains()`

```python
def contains(self, coord: Coordinates) -> bool:
    """Check if coordinate is within grid bounds.
    
    Preconditions:
      P1: coord is Coordinates
    
    Postconditions:
      C1: result = (0 ≤ coord.x < width) ∧ (0 ≤ coord.y < height)
      C2: result ∈ {True, False}
    
    Properties:
      - Pure function
      - Deterministic
      - Boundary: coord at (0,0) is inside
      - Boundary: coord at (width-1, height-1) is inside
      - Boundary: coord at (width, height) is outside
    
    Examples:
      GridSize(32, 32).contains((10, 20)) = True
      GridSize(32, 32).contains((0, 0)) = True
      GridSize(32, 32).contains((31, 31)) = True
      GridSize(32, 32).contains((32, 20)) = False
      GridSize(32, 32).contains((-1, 10)) = False
    """
```

#### `center()`

```python
def center(self) -> Coordinates:
    """Get center coordinate of grid.
    
    Preconditions:
      None
    
    Postconditions:
      C1: result.x = width // 2
      C2: result.y = height // 2
      C3: self.contains(result) = True
    
    Properties:
      - Deterministic
      - Pure function
      - Result always within grid
    
    Examples:
      GridSize(32, 32).center() = (16, 16)
      GridSize(10, 10).center() = (5, 5)
      GridSize(11, 11).center() = (5, 5)  # Integer division
    """
```

### Test Requirements

```python
# Validation tests
def test_grid_size_requires_positive():
    """Cannot create with zero or negative dimensions"""
    with pytest.raises(ValidationError):
        create_grid_size(0, 10)
    with pytest.raises(ValidationError):
        create_grid_size(10, -5)

def test_grid_size_maximum_enforced():
    """Cannot exceed maximum dimension"""
    with pytest.raises(ValidationError):
        create_grid_size(10001, 10)

# Property tests
@given(
    w=st.integers(1, MAX_GRID_DIMENSION),
    h=st.integers(1, MAX_GRID_DIMENSION)
)
def test_total_cells_equals_product(w, h):
    """total_cells = width * height"""
    grid = create_grid_size(w, h)
    assert grid.total_cells() == w * h

@given(
    grid=grid_size_strategy(),
    x=st.integers(),
    y=st.integers()
)
def test_contains_definition(grid, x, y):
    """contains ⟺ within bounds"""
    coord = Coordinates(x, y)
    expected = (0 <= x < grid.width) and (0 <= y < grid.height)
    assert grid.contains(coord) == expected

# Boundary tests
def test_contains_boundary_conditions():
    """Test exact boundaries"""
    grid = create_grid_size(10, 10)
    
    # Inside corners
    assert grid.contains(Coordinates(0, 0))
    assert grid.contains(Coordinates(9, 9))
    
    # Outside by 1
    assert not grid.contains(Coordinates(10, 5))
    assert not grid.contains(Coordinates(5, 10))
    assert not grid.contains(Coordinates(-1, 5))
```

---

## 🤖 Type: AgentState

### Mathematical Definition

```
AgentState = {
  position: Coordinates,
  step_count: ℕ,
  total_reward: ℝ₊,
  goal_reached: {True, False}
}

where:
  ℕ = {0, 1, 2, ...} (non-negative integers)
  ℝ₊ = {x ∈ ℝ | x ≥ 0} (non-negative reals)
```

### Type Specification

```python
@dataclass
class AgentState:
    """Mutable agent state within episode.
    
    Invariants:
      I1: position is Coordinates
      I2: step_count ∈ ℕ (step_count ≥ 0)
      I3: total_reward ∈ ℝ₊ (total_reward ≥ 0)
      I4: goal_reached ∈ {True, False}
      I5: step_count monotonically increases (never decreases)
      I6: total_reward monotonically increases (never decreases)
      I7: goal_reached is idempotent (once True, stays True)
    """
    position: Coordinates
    step_count: int = 0
    total_reward: float = 0.0
    goal_reached: bool = False
    
    def __post_init__(self):
        """Validate initial state."""
        if self.step_count < 0:
            raise ValidationError("step_count must be non-negative")
        if self.total_reward < 0:
            raise ValidationError("total_reward must be non-negative")
```

### Constructor Contract

```python
def create_agent_state(
    position: Coordinates,
    step_count: int = 0,
    total_reward: float = 0.0,
    goal_reached: bool = False
) -> AgentState:
    """Create validated AgentState.
    
    Preconditions:
      P1: position is Coordinates
      P2: step_count ≥ 0
      P3: total_reward ≥ 0
      P4: goal_reached ∈ {True, False}
    
    Postconditions:
      C1: returns AgentState with given values
      C2: All invariants I1-I7 hold
    
    Raises:
      ValidationError: If preconditions violated
    
    Examples:
      create_agent_state(Coordinates(0,0)) 
        → AgentState(position=(0,0), step_count=0, ...)
    """
```

### Operations

#### `increment_step()`

```python
def increment_step(self) -> None:
    """Increment step count by 1.
    
    Preconditions:
      P1: step_count < MAX_INT
    
    Postconditions:
      C1: step_count' = step_count + 1
      C2: All other fields unchanged
    
    Invariants Preserved:
      I2: step_count' ≥ 0 (still non-negative)
      I5: step_count' > step_count (monotonic)
    
    Modifies:
      step_count
    
    Side Effects:
      None
    
    Examples:
      state.step_count = 5
      state.increment_step()
      assert state.step_count == 6
    """
    old_count = self.step_count
    self.step_count += 1
    assert self.step_count == old_count + 1, "Atomic increment failed"
```

#### `update_position()`

```python
def update_position(self, new_position: Coordinates) -> None:
    """Update agent position.
    
    Preconditions:
      P1: new_position is Coordinates
      P2: new_position is valid (checked externally by BoundaryEnforcer)
    
    Postconditions:
      C1: position' = new_position
      C2: All other fields unchanged
    
    Modifies:
      position
    
    Side Effects:
      None
    
    Note:
      Position validation is BoundaryEnforcer's responsibility.
      This method trusts the caller.
    """
    self.position = new_position
```

#### `add_reward()`

```python
def add_reward(self, reward: float) -> None:
    """Add reward to cumulative total.
    
    Preconditions:
      P1: reward ≥ 0 (non-negative)
    
    Postconditions:
      C1: total_reward' = total_reward + reward
      C2: total_reward' ≥ total_reward (monotonic)
      C3: All other fields unchanged
    
    Invariants Preserved:
      I3: total_reward' ≥ 0 (still non-negative)
      I6: total_reward' ≥ total_reward (monotonic)
    
    Modifies:
      total_reward
    
    Raises:
      ValidationError: If reward < 0
    
    Examples:
      state.total_reward = 5.0
      state.add_reward(1.0)
      assert state.total_reward == 6.0
    """
    if reward < 0:
        raise ValidationError(f"Reward must be non-negative, got {reward}")
    
    old_reward = self.total_reward
    self.total_reward += reward
    assert self.total_reward >= old_reward, "Monotonicity violated"
```

#### `mark_goal_reached()`

```python
def mark_goal_reached(self) -> None:
    """Mark goal as reached (idempotent).
    
    Preconditions:
      None
    
    Postconditions:
      C1: goal_reached' = True
      C2: All other fields unchanged
    
    Invariants Preserved:
      I7: Idempotent - calling twice is safe
    
    Constraint:
      ¬∃ state: goal_reached = True ∧ goal_reached' = False
      (Cannot un-reach goal)
    
    Modifies:
      goal_reached
    
    Side Effects:
      None
    
    Idempotency:
      mark_goal_reached(); mark_goal_reached()
      has same effect as single call
    """
    self.goal_reached = True
```

#### `reset()`

```python
def reset(self, position: Coordinates) -> None:
    """Reset state for new episode.
    
    Preconditions:
      P1: position is Coordinates
    
    Postconditions:
      C1: position' = position
      C2: step_count' = 0
      C3: total_reward' = 0.0
      C4: goal_reached' = False
    
    Modifies:
      All fields
    
    Note:
      This is called by Environment.reset()
      Not for use during episode
    """
    self.position = position
    self.step_count = 0
    self.total_reward = 0.0
    self.goal_reached = False
```

### Test Requirements

```python
# Monotonicity tests
def test_step_count_monotonic():
    """Step count never decreases"""
    state = create_agent_state(Coordinates(0, 0))
    
    for i in range(10):
        old_count = state.step_count
        state.increment_step()
        assert state.step_count > old_count
        assert state.step_count == old_count + 1

def test_total_reward_monotonic():
    """Total reward never decreases"""
    state = create_agent_state(Coordinates(0, 0))
    
    for reward in [0.0, 0.5, 1.0, 0.0]:
        old_reward = state.total_reward
        state.add_reward(reward)
        assert state.total_reward >= old_reward

def test_negative_reward_rejected():
    """Cannot add negative reward"""
    state = create_agent_state(Coordinates(0, 0))
    with pytest.raises(ValidationError):
        state.add_reward(-1.0)

# Idempotency tests
def test_mark_goal_reached_idempotent():
    """Calling multiple times is safe"""
    state = create_agent_state(Coordinates(0, 0))
    
    assert state.goal_reached == False
    state.mark_goal_reached()
    assert state.goal_reached == True
    state.mark_goal_reached()
    assert state.goal_reached == True  # Still True

def test_cannot_unreach_goal():
    """Once reached, cannot un-reach"""
    state = create_agent_state(Coordinates(0, 0))
    state.mark_goal_reached()
    
    # No method exists to set goal_reached = False
    # (except reset, which is for new episodes)

# Property tests
@given(
    position=coordinates_strategy(),
    step_count=st.integers(0, 1000),
    total_reward=st.floats(0, 1000)
)
def test_agent_state_creation(position, step_count, total_reward):
    """Can create with valid non-negative values"""
    state = create_agent_state(position, step_count, total_reward)
    assert state.step_count >= 0
    assert state.total_reward >= 0
```

---

## 🔗 Relationships Between Types

### Type Hierarchy

```
Coordinates (primitive)
    ↓
GridSize (uses Coordinates for bounds checking)
    ↓
AgentState (has Coordinates position, validated by GridSize)
```

### Dependency Constraints

```python
# AgentState position should be valid for grid
∀ state: AgentState, grid: GridSize:
  (state used in env with grid) ⇒ grid.contains(state.position)

# This is enforced by BoundaryEnforcer, not AgentState itself
```

### Conversion Functions

```python
def grid_size_from_tuple(size: tuple[int, int]) -> GridSize:
    """Convert tuple to GridSize with validation."""
    return create_grid_size(size[0], size[1])

def coordinates_from_agent_state(state: AgentState) -> Coordinates:
    """Extract position from agent state."""
    return state.position
```

---

## 📊 Summary Table

| Type | Mutable | Hashable | Validated | Use Case |
|------|---------|----------|-----------|----------|
| Coordinates | ❌ Frozen | ✅ Yes | Minimal | Positions |
| GridSize | ❌ Frozen | ✅ Yes | ✅ Strict | Grid bounds |
| AgentState | ✅ Mutable | ❌ No | ✅ On create | Episode state |

---

## 🎯 Design Principles

1. **Coordinates are primitive** - No validation (can be negative)
2. **GridSize enforces physics** - Must be positive, bounded
3. **AgentState tracks episode** - Mutable but monotonic
4. **Separation of concerns** - Validation happens at boundaries
5. **Fail fast** - Invalid construction raises immediately
6. **Pure operations** - distance_to, contains, etc. have no side effects

---

**Last Updated:** 2025-09-30  
**Next Review:** After guard tests implemented
