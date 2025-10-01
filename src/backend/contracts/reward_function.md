# Reward Function Contract

**Component:** Reward Calculation  
**Version:** 1.1.0  
**Date:** 2025-09-30  
**Status:** CANONICAL - All implementations MUST conform

---

## 🎯 Purpose

Define reward function properties for reinforcement learning.

**IMPORTANT:** Distinguish between universal properties and model-specific design choices!

## 📊 Classification

### Universal Reward Properties
Properties ALL reward functions should satisfy:
- **Purity** - No side effects
- **Determinism** - Reproducible

### Sparse Binary Reward Model (Current Implementation)
Design choice for exploration-focused RL:
- Binary output {0.0, 1.0}
- Distance-based threshold
- No intermediate rewards

**Alternative Models:**
- Dense rewards: `1.0 / (1 + distance)`
- Potential shaping: `φ(s') - φ(s)`  
- Distance-based: `max(0, 1 - distance/max_distance)`
- Multi-objective: Vector rewards

---

## 📐 Mathematical Definition

### Function Signature

```
reward: Coordinates × Coordinates × ℝ₊ → {0.0, 1.0}
```
Where:
- **Domain:** All possible (agent_position, source_position, goal_radius)
- **Codomain:** Exactly two values: {0.0, 1.0}
- **ℝ₊** = positive real numbers

### Sparse Binary Model Specification

**Current Implementation:** Sparse binary reward for exploration-focused learning.

**Design Rationale:** Forces exploration, simple goal detection, no shaping bias.

Let d = distance(agent_pos, source_pos)
where distance(p₁, p₂) = √((x₂-x₁)² + (y₂-y₁)²)

reward(agent_pos, source_pos, goal_radius) = {
  1.0  if d ≤ goal_radius
  0.0  otherwise
}

### Boundary Condition
**CRITICAL:** The boundary is **inclusive** (≤, not <)

```
At exactly d = goal_radius:
  reward = 1.0 ✓

Just inside (d = goal_radius - ε):
  reward = 1.0 ✓

Just outside (d = goal_radius + ε):
  reward = 0.0 ✓
```

---

## 💻 Implementation Contract

```python
def calculate_reward(
    agent_position: Coordinates,
    source_position: Coordinates,
    goal_radius: float
) -> float:
    """Calculate sparse binary reward based on distance to goal.
    
    Preconditions:
      P1: agent_position is valid Coordinates
      P2: source_position is valid Coordinates
      P3: goal_radius > 0 (positive)
      P4: goal_radius is finite (not NaN, not inf)
    
    Postconditions:
      C1: result ∈ {0.0, 1.0} (exactly binary)
      C2: result = 1.0 ⟺ distance(agent, source) ≤ goal_radius
      C3: result = 0.0 ⟺ distance(agent, source) > goal_radius
      C4: isinstance(result, float)
    
    Properties:
      1. Purity: No side effects, no hidden state
      2. Determinism: Same inputs → same output (always)
      3. Binary: Output is exactly 0.0 or 1.0
      4. Boundary: Inclusive at goal_radius
      5. Symmetry: reward(a, b, r) = reward(b, a, r)
      6. Monotonic: If d₁ < d₂, then reward(d₁) ≥ reward(d₂)
    
    Raises:
      ValidationError: If goal_radius ≤ 0
      TypeError: If positions not Coordinates
    
    Time Complexity:
      O(1) - constant time
    
    Space Complexity:
      O(1) - no allocation
    
    Examples:
      # At source
      calculate_reward((0,0), (0,0), 5.0) = 1.0
      
      # Within radius
      calculate_reward((3,4), (0,0), 10.0) = 1.0  # d=5, r=10
      
      # At exact boundary
      calculate_reward((3,4), (0,0), 5.0) = 1.0  # d=5, r=5 ✓
      
      # Outside radius
      calculate_reward((10,0), (0,0), 5.0) = 0.0  # d=10, r=5
      
      # Just outside
      calculate_reward((3,4), (0,0), 4.999) = 0.0  # d=5, r<5
    """
```

---

---

## 🌍 Universal Reward Properties

These apply to ALL reward function implementations:

### Property 1: Purity (UNIVERSAL)

```python
∀ inputs: calculate_reward(inputs) has no side effects

No modification of:
  - Global variables
  - Input arguments
  - File system
  - Network
  - Randomness (no RNG calls)
```

**Test:**
```python
def test_reward_is_pure_function():
    """Reward has no side effects"""
    pos_a = Coordinates(10, 10)
    pos_b = Coordinates(15, 15)
    
    # Call multiple times
    r1 = calculate_reward(pos_a, pos_b, 5.0)
    r2 = calculate_reward(pos_a, pos_b, 5.0)
    r3 = calculate_reward(pos_a, pos_b, 5.0)
    
    # All identical
    assert r1 == r2 == r3
    
    # Inputs unchanged
    assert pos_a == Coordinates(10, 10)
    assert pos_b == Coordinates(15, 15)
```

### Property 2: Determinism (UNIVERSAL)

```python
∀ a, b, r: 
  calculate_reward(a, b, r) = calculate_reward(a, b, r)

No dependency on:
  - Time
  - Order of calls
  - External state
  - Random values
```

**Test:**
```python
@given(
    agent=coordinates_strategy(),
    source=coordinates_strategy(),
    radius=st.floats(0.1, 100.0)
)
def test_reward_deterministic(agent, source, radius):
    """Same inputs always give same output"""
    result1 = calculate_reward(agent, source, radius)
    result2 = calculate_reward(agent, source, radius)
    assert result1 == result2
```

---

## 🔬 Sparse Binary Model Properties

Model-specific properties for the current implementation:

### Property 3: Binary Output (MODEL-SPECIFIC)

```python
∀ a, b, r: calculate_reward(a, b, r) ∈ {0.0, 1.0}

No other values possible:
  - Not 0.5
  - Not -1
  - Not NaN
  - Not infinity
```

**Test:**
```python
@given(
    agent=coordinates_strategy(),
    source=coordinates_strategy(),
    radius=st.floats(0.1, 100.0)
)
def test_reward_is_binary(agent, source, radius):
    """Reward is always exactly 0.0 or 1.0"""
    reward = calculate_reward(agent, source, radius)
    assert reward in (0.0, 1.0), f"Expected 0.0 or 1.0, got {reward}"
```

⚠️ **Not universal** - Dense rewards would violate this.

### Property 4: Boundary Inclusivity (MODEL-SPECIFIC)

```python
At d = goal_radius:
  reward = 1.0

This is NOT:
  d < goal_radius  (exclusive)
  
This IS:
  d ≤ goal_radius  (inclusive)
```

**Test:**
```python
def test_boundary_is_inclusive():
    """At exactly goal_radius distance, reward is 1.0"""
    # Create positions exactly 5.0 units apart
    source = Coordinates(0, 0)
    agent = Coordinates(3, 4)  # 3² + 4² = 25, √25 = 5.0
    
    # At exact boundary
    reward = calculate_reward(agent, source, goal_radius=5.0)
    assert reward == 1.0, "Boundary should be inclusive"
    
    # Just inside
    reward = calculate_reward(agent, source, goal_radius=5.001)
    assert reward == 1.0
    
    # Just outside
    reward = calculate_reward(agent, source, goal_radius=4.999)
    assert reward == 0.0
```

Boundary definition specific to threshold-based model.

### Property 5: Symmetry (MODEL-SPECIFIC)

```python
∀ a, b, r:
  reward(a, b, r) = reward(b, a, r)

Distance is symmetric, so reward is too.
```

**Test:**
```python
@given(
    pos_a=coordinates_strategy(),
    pos_b=coordinates_strategy(),
    radius=st.floats(0.1, 100.0)
)
def test_reward_symmetric(pos_a, pos_b, radius):
    """reward(a,b,r) = reward(b,a,r)"""
    r1 = calculate_reward(pos_a, pos_b, radius)
    r2 = calculate_reward(pos_b, pos_a, radius)
    assert r1 == r2
```

Only holds for distance-based models. Directional rewards would violate.

### Property 6: Monotonicity (MODEL-SPECIFIC)

```python
∀ d₁, d₂:
  d₁ ≤ d₂ ⇒ reward(d₁) ≥ reward(d₂)

Closer or same distance → same or better reward
```

**Test:**
```python
def test_reward_monotonic_with_distance():
    """Closer positions get at least as much reward"""
    source = Coordinates(0, 0)
    goal_radius = 10.0
    
    # Positions at increasing distance
    close = Coordinates(2, 0)    # d = 2
    medium = Coordinates(5, 0)   # d = 5
    far = Coordinates(15, 0)     # d = 15
    
    r_close = calculate_reward(close, source, goal_radius)
    r_medium = calculate_reward(medium, source, goal_radius)
    r_far = calculate_reward(far, source, goal_radius)
    
    # Monotonic: closer ≥ farther
    assert r_close >= r_medium
    assert r_medium >= r_far
    
    # In this case: 1.0 ≥ 1.0 ≥ 0.0
    assert r_close == 1.0
    assert r_medium == 1.0
    assert r_far == 0.0
```

---

## 🧪 Edge Case Tests (MUST IMPLEMENT)

### Edge Case 1: Same Position (d = 0)

```python
def test_same_position_gives_reward():
    """Agent at source gets reward (if radius > 0)"""
    pos = Coordinates(10, 10)
    reward = calculate_reward(pos, pos, goal_radius=1.0)
    assert reward == 1.0, "Distance 0 ≤ any positive radius"
```

### Edge Case 2: Zero Radius

```python
def test_zero_radius_requires_exact_match():
    """Only at exact source with radius = 0"""
    source = Coordinates(10, 10)
    
    # At source
    reward = calculate_reward(source, source, goal_radius=0.0)
    assert reward == 1.0
    
    # One step away
    agent = Coordinates(11, 10)
    reward = calculate_reward(agent, source, goal_radius=0.0)
    assert reward == 0.0
```

### Edge Case 3: Very Large Radius

```python
def test_large_radius_includes_all():
    """Radius >> grid size → all positions get reward"""
    source = Coordinates(0, 0)
    agent = Coordinates(1000, 1000)  # Far away
    
    reward = calculate_reward(agent, source, goal_radius=10000.0)
    assert reward == 1.0, "Large radius includes distant positions"
```

### Edge Case 4: Floating Point Precision

```python
def test_floating_point_boundary():
    """Handle floating point precision at boundary"""
    source = Coordinates(0, 0)
    agent = Coordinates(3, 4)  # Distance = 5.0
    
    # Exact boundary
    reward = calculate_reward(agent, source, goal_radius=5.0)
    assert reward == 1.0
    
    # Slightly less (due to float precision)
    reward = calculate_reward(agent, source, goal_radius=4.999999999)
    assert reward == 0.0
    
    # Implementation should use proper comparison:
    # distance <= goal_radius
    # NOT: abs(distance - goal_radius) < epsilon
```

### Edge Case 5: Negative Coordinates

```python
def test_negative_coordinates_valid():
    """Distance works with negative coordinates"""
    agent = Coordinates(-10, -10)
    source = Coordinates(-5, -5)
    # Distance = √((5)² + (5)²) = √50 ≈ 7.07
    
    reward = calculate_reward(agent, source, goal_radius=10.0)
    assert reward == 1.0  # Within radius
    
    reward = calculate_reward(agent, source, goal_radius=5.0)
    assert reward == 0.0  # Outside radius
```

---

## ⚠️ Common Implementation Errors

### ❌ Wrong: Exclusive Boundary

```python
# WRONG - uses < instead of <=
def calculate_reward_wrong(agent, source, radius):
    distance = agent.distance_to(source)
    return 1.0 if distance < radius else 0.0
    # ❌ Excludes exact boundary!
```

### ❌ Wrong: Non-Binary Output

```python
# WRONG - returns proportional reward
def calculate_reward_wrong(agent, source, radius):
    distance = agent.distance_to(source)
    return max(0.0, 1.0 - distance / radius)
    # ❌ Not binary!
```

### ❌ Wrong: Side Effects

```python
# WRONG - has side effects
reward_count = 0
def calculate_reward_wrong(agent, source, radius):
    global reward_count
    reward_count += 1  # ❌ Side effect!
    return 1.0 if agent.distance_to(source) <= radius else 0.0
```

### ❌ Wrong: Caching Without Immutability

```python
# WRONG - cache can cause stale results
cache = {}
def calculate_reward_wrong(agent, source, radius):
    key = (agent, source, radius)
    if key not in cache:
        cache[key] = ...  # ❌ If inputs are mutable, cache is wrong
    return cache[key]
```

### ✅ Correct Implementation

```python
def calculate_reward(
    agent_position: Coordinates,
    source_position: Coordinates,
    goal_radius: float
) -> float:
    """Correct sparse binary reward."""
    # Validate inputs
    if goal_radius <= 0:
        raise ValidationError("goal_radius must be positive")
    
    # Calculate distance
    distance = agent_position.distance_to(source_position)
    
    # Binary reward with inclusive boundary
    return 1.0 if distance <= goal_radius else 0.0
```

---

## 📊 Test Coverage Requirements

### Minimum Test Suite

```python
# Property tests (50+ examples each)
test_reward_deterministic()
test_reward_is_binary()
test_reward_symmetric()
test_reward_pure_function()
test_reward_monotonic()

# Boundary tests
test_boundary_is_inclusive()
test_just_inside_boundary()
test_just_outside_boundary()

# Edge cases
test_same_position()
test_zero_radius()
test_large_radius()
test_floating_point_precision()
test_negative_coordinates()

# Error conditions
test_negative_radius_raises()
test_zero_radius_allowed()
test_invalid_position_raises()

# Integration
test_reward_with_episode()
test_reward_matches_termination()
```

---

## 🔗 Related Components

**Uses:**
- `Coordinates.distance_to()` for distance calculation
- Must satisfy distance metric properties

**Used By:**
- `RewardCalculator.calculate_reward_step()`
- `EpisodeManager` for termination checking
- `Environment.step()` for return value

**Affects:**
- Agent learning (RL signal)
- Episode termination (goal detection)
- Performance metrics

---

## 📐 Design Rationale

### Why Sparse Binary Reward? (Current Choice)

**Pros:**
- ✅ Clear goal definition
- ✅ Forces exploration (no gradient)
- ✅ Simple to understand
- ✅ Deterministic termination
- ✅ Easy to test (only 2 values)

**Cons:**
- ❌ Harder for RL agents to learn (no gradient)
- ❌ No partial credit for getting close

### Alternative Reward Models

**Dense Distance-Based:**
```python
reward = 1.0 / (1 + distance)
# Pros: Smooth gradient, easier to learn
# Cons: Never exactly zero, biases exploration
```

**Potential-Based Shaping:**
```python
reward = φ(s') - φ(s)  # where φ = -distance to goal
# Pros: Provably optimal policy-invariant
# Cons: Requires careful potential design
```

**Linear Shaping:**
```python
reward = max(0, 1 - distance/max_distance)
# Pros: Interpretable, bounded
# Cons: Arbitrary scaling
```

**Multi-Objective:**
```python
reward = [goal_reward, concentration_reward, efficiency_penalty]
# Pros: Rich feedback
# Cons: Complex, needs preference weighting
```

**Current Reason:** Simplicity and determinism prioritized. Future work may explore alternatives.

### Why Inclusive Boundary?

The boundary is inclusive (≤) rather than exclusive (<) because:
1. **Mathematical consistency:** Circle definition includes boundary
2. **Physical intuition:** "Within radius" means "up to and including"
3. **Numerical stability:** Avoids floating-point edge cases at boundary
4. **Testing clarity:** Easier to test exact boundary conditions

---

## 🎯 Verification Checklist

Implementation MUST satisfy:

- [ ] Returns only 0.0 or 1.0 (never other values)
- [ ] Deterministic (same inputs → same output)
- [ ] Pure function (no side effects)
- [ ] Symmetric (order of positions doesn't matter)
- [ ] Inclusive boundary (d = radius → 1.0)
- [ ] Handles d = 0 (same position)
- [ ] Validates goal_radius > 0
- [ ] Works with negative coordinates
- [ ] O(1) time complexity
- [ ] Property tests pass (100+ examples)
- [ ] All edge cases tested
- [ ] Boundary tests pass

---

**Last Updated:** 2025-09-30  
**Next Review:** After guard tests implemented
