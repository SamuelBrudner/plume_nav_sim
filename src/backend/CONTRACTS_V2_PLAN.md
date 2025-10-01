# Contract Formalization: Immediate Action Plan

**Date:** 2025-09-30  
**Phase:** 2 - Contract Formalization  
**Goal:** Create mathematically precise contracts for all core components

---

## 🎯 Current Status

**Completed:**
- ✅ SEMANTIC_MODEL.md exists (defines abstractions)
- ✅ CONTRACTS.md exists (basic API signatures)
- ✅ SEMANTIC_AUDIT.md created (identified gaps)
- ✅ TEST_TAXONOMY.md created (test strategy)

**Next:** Formalize contracts with pre/post/invariants

---

## 📋 Phase 2: Contract Formalization Tasks

### Task 1: Environment State Machine (2 hours) - PRIORITY 1

**File:** Create `contracts/environment_state_machine.md`

**What to Document:**

```python
# 1. Formal State Definition
class EnvironmentState(Enum):
    CREATED = "created"      # After __init__, before reset
    READY = "ready"          # After reset, can step
    TERMINATED = "terminated"  # Goal reached or failure
    TRUNCATED = "truncated"   # Max steps reached
    CLOSED = "closed"        # Resources released

# 2. Transition Rules (formal)
Γ ⊢ state : EnvironmentState

Rules:
  state = CREATED
  ─────────────────────  (RESET-FROM-CREATED)
  reset() → state' = READY

  state = READY ∧ ¬goal_reached ∧ steps < max_steps
  ────────────────────────────────────────────────  (STEP-CONTINUE)
  step(action) → state' = READY

  state = READY ∧ goal_reached
  ─────────────────────────  (STEP-TERMINATE)
  step(action) → state' = TERMINATED

  state = READY ∧ steps >= max_steps
  ─────────────────────────────  (STEP-TRUNCATE)
  step(action) → state' = TRUNCATED

  state ∈ {TERMINATED, TRUNCATED}
  ───────────────────────────────  (RESET-AFTER-END)
  reset() → state' = READY

  state ≠ CLOSED
  ────────────────  (CLOSE)
  close() → state' = CLOSED

  state = CLOSED
  ─────────────  (NO-OPS-AFTER-CLOSE)
  reset() → ERROR
  step() → ERROR

# 3. Invariants (must ALWAYS hold)
I1: state ∈ {CREATED, READY, TERMINATED, TRUNCATED, CLOSED}
I2: state = READY ⇒ agent_state ≠ null
I3: state = CLOSED ⇒ ∀ operations fail
I4: episode_count ≥ 0 ∧ monotonically increases
I5: step_count ≥ 0 ∧ resets with episode

# 4. Pre/Post Conditions
reset(seed: Optional[int]) → (Obs, Info):
  Preconditions:
    P1: state ≠ CLOSED
    P2: seed = None ∨ (seed ∈ ℕ ∧ 0 ≤ seed < 2³¹)
  
  Postconditions:
    C1: state' = READY
    C2: step_count' = 0
    C3: episode_count' = episode_count + 1
    C4: agent_state' initialized to start_position
    C5: returns (obs, info) where:
        - obs ∈ ObservationSpace
        - info['seed'] = seed (or auto-generated)
        - info['episode'] = episode_count'
  
  Modifies: {state, step_count, episode_count, agent_state, seed}

step(action: int) → (Obs, float, bool, bool, Info):
  Preconditions:
    P1: state ∈ {READY, TERMINATED, TRUNCATED}
    P2: action ∈ [0, 8]
  
  Postconditions:
    C1: returns (obs, reward, terminated, truncated, info)
    C2: obs ∈ ObservationSpace
    C3: reward ∈ {0.0, 1.0}
    C4: terminated, truncated ∈ {True, False}
    C5: info ∈ InfoSchema
    C6: step_count' = step_count + 1
    C7: state' ∈ {READY, TERMINATED, TRUNCATED}
    C8: terminated ⇒ info['termination_reason'] ≠ ∅
  
  Modifies: {agent_state, step_count, state (possibly)}
  
  Determinism:
    ∀ seed, actions: 
      run(seed, actions) = run(seed, actions)
```

**Deliverable:** Formal state machine with all transitions, invariants, pre/post conditions

---

### Task 2: Core Data Types (1.5 hours) - PRIORITY 2

**File:** Create `contracts/core_types.md`

**Components to Formalize:**

#### A. Coordinates

```python
Type: Coordinates = (x: int, y: int)

Constructor:
  create_coordinates(pos: tuple[int, int]) → Coordinates
  
  Preconditions:
    P1: pos = (x, y) where x, y ∈ ℤ
  
  Postconditions:
    C1: returns Coordinates(x, y)
    C2: result.x = pos[0]
    C3: result.y = pos[1]

Properties:
  1. Equality: (x₁, y₁) = (x₂, y₂) ⟺ x₁ = x₂ ∧ y₁ = y₂
  2. Hashable: hash(c₁) = hash(c₂) ⟺ c₁ = c₂
  3. Immutable: cannot modify x or y after creation

Operations:
  distance_to(other: Coordinates) → float
    Postconditions:
      C1: result ≥ 0
      C2: result = √((x₂-x₁)² + (y₂-y₁)²)
    
    Properties:
      - Symmetry: a.distance_to(b) = b.distance_to(a)
      - Identity: a.distance_to(a) = 0
      - Triangle: a.distance_to(c) ≤ a.distance_to(b) + b.distance_to(c)
```

#### B. GridSize

```python
Type: GridSize = {width: int, height: int}

Invariants:
  I1: width > 0 ∧ height > 0
  I2: width ≤ MAX_GRID_DIMENSION (10000)
  I3: height ≤ MAX_GRID_DIMENSION (10000)

Constructor:
  create_grid_size(w: int, h: int) → GridSize
  
  Preconditions:
    P1: w > 0 ∧ h > 0
    P2: w ≤ 10000 ∧ h ≤ 10000
  
  Raises:
    ValidationError if preconditions violated

Operations:
  total_cells() → int
    Postcondition: result = width × height
    Property: result > 0
  
  contains(coord: Coordinates) → bool
    Postcondition: result = (0 ≤ x < width) ∧ (0 ≤ y < height)
```

#### C. AgentState

```python
Type: AgentState = {
  position: Coordinates,
  step_count: int,
  total_reward: float,
  goal_reached: bool
}

Invariants:
  I1: step_count ≥ 0
  I2: total_reward ≥ 0 (assuming non-negative rewards)
  I3: goal_reached ⇒ goal_reached' (idempotent, write-once)
  I4: step_count' ≥ step_count (monotonic)

Operations:
  increment_step() → void
    Precondition: step_count < MAX_INT
    Postcondition: step_count' = step_count + 1
  
  update_position(new_pos: Coordinates) → void
    Precondition: new_pos is valid (checked by BoundaryEnforcer)
    Postcondition: position' = new_pos
  
  mark_goal_reached() → void
    Precondition: goal calculation performed
    Postcondition: goal_reached' = True
    Idempotency: calling twice has no effect
    
    Constraint: ¬(goal_reached = True ∧ goal_reached' = False)
                (cannot un-reach goal)
```

---

### Task 3: Reward Specification (1 hour) - PRIORITY 3

**File:** Create `contracts/reward_function.md`

```python
# Mathematical Definition

reward: Coordinates × Coordinates × ℝ₊ → {0.0, 1.0}

reward(agent_pos, source_pos, goal_radius) = {
  1.0  if distance(agent_pos, source_pos) ≤ goal_radius
  0.0  otherwise
}

where:
  distance(p₁, p₂) = √((x₂-x₁)² + (y₂-y₁)²)

# Function Signature

def calculate_reward(
    agent_position: Coordinates,
    source_position: Coordinates,
    goal_radius: float
) -> float:
    """
    Preconditions:
      P1: agent_position is valid Coordinates
      P2: source_position is valid Coordinates
      P3: goal_radius > 0
    
    Postconditions:
      C1: result ∈ {0.0, 1.0}
      C2: result = 1.0 ⟺ distance(agent, source) ≤ goal_radius
      C3: result = 0.0 ⟺ distance(agent, source) > goal_radius
    
    Properties:
      1. Purity: No side effects, no hidden state
      2. Determinism: Same inputs → same output
      3. Binary: Output is exactly 0.0 or 1.0
      4. Boundary: distance = goal_radius ⇒ reward = 1.0 (inclusive)
      5. Symmetry: reward(a, b, r) = reward(b, a, r)
    
    Test Coverage Required:
      - Boundary condition: d = goal_radius
      - Just inside: d = goal_radius - ε
      - Just outside: d = goal_radius + ε
      - Same position: d = 0
      - Far away: d >> goal_radius
      - Property test: ∀ valid inputs
    """

# Edge Cases

1. Exact boundary:
   distance = 5.0, goal_radius = 5.0 → reward = 1.0 ✓

2. Floating point precision:
   distance = 5.000000001, goal_radius = 5.0 → reward = 0.0 ✓
   (use proper comparison, not exact equality)

3. Zero radius:
   goal_radius = 0.0 → only at exact source position

4. Very large radius:
   goal_radius = 1000.0 → most positions get reward
```

---

### Task 4: Concentration Field (1.5 hours) - PRIORITY 4

**File:** Create `contracts/concentration_field.md`

```python
# Physical Model

Concentration Field: ℤ² → [0, 1]

Mathematical Form:
  C(x, y) = exp(-distance²(x, y, source) / (2σ²))

Normalized:
  C_norm(x, y) = C(x, y) / C(source)

# Type Specification

Type: ConcentrationField = {
  field: ndarray[height, width] of float64,
  source_location: Coordinates,
  sigma: float,
  grid_size: GridSize
}

# Invariants (Physical Laws)

I1: Non-negativity
    ∀ (x, y): field[x, y] ≥ 0

I2: Bounded
    ∀ (x, y): field[x, y] ≤ 1.0

I3: Maximum at source
    ∀ (x, y) ≠ source: field[source] ≥ field[x, y]

I4: Monotonic decay
    distance(p₁, source) ≤ distance(p₂, source) 
    ⇒ field[p₁] ≥ field[p₂]

I5: Symmetry (for Gaussian)
    field[source + Δ] ≈ field[source - Δ]
    (within numerical tolerance)

I6: Shape consistency
    field.shape = (grid_size.height, grid_size.width)

# Constructor Contract

def create_concentration_field(
    grid_size: GridSize,
    source_location: Coordinates,
    sigma: float
) -> ConcentrationField:
    """
    Preconditions:
      P1: grid_size valid
      P2: 0 ≤ source.x < width ∧ 0 ≤ source.y < height
      P3: sigma > 0
    
    Postconditions:
      C1: result.field.shape = (height, width)
      C2: ∀ values: 0 ≤ value ≤ 1
      C3: result.field[source] = max(result.field)
      C4: Gaussian distribution around source
    
    Properties:
      - Determinism: Same params → identical field
      - Physical: Satisfies invariants I1-I6
    """

# Sampling Contract

def sample(position: Coordinates) -> float:
    """
    Preconditions:
      P1: 0 ≤ position.x < width
      P2: 0 ≤ position.y < height
    
    Postconditions:
      C1: result ∈ [0, 1]
      C2: result = field[position.y, position.x]
    
    Properties:
      - Deterministic: same position → same value
      - Consistent: sample(p) = field[p.y, p.x]
    """
```

---

### Task 5: Update Main CONTRACTS.md (1 hour) - PRIORITY 5

**Enhancements needed:**

1. **Add Invariant Section**
   ```markdown
   ## 🔒 System Invariants
   
   ### Global Invariants (always true)
   - G1: No global mutable state
   - G2: Determinism with seed control
   - G3: Type safety at boundaries
   
   ### Per-Component Invariants
   - See individual contract files in contracts/
   ```

2. **Add State Machine References**
   ```markdown
   ## 🔄 State Machines
   
   See detailed specifications:
   - Environment: contracts/environment_state_machine.md
   - Episode: contracts/episode_lifecycle.md
   ```

3. **Add Mathematical Properties**
   ```markdown
   ## 📐 Mathematical Properties
   
   ### Distance Metric
   - Symmetry: d(a,b) = d(b,a)
   - Identity: d(a,a) = 0
   - Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
   
   ### Reward Function
   - Binary: R ∈ {0.0, 1.0}
   - Deterministic: Pure function
   - Boundary inclusive: d ≤ r (not d < r)
   ```

---

## 🎯 Immediate Next Actions

### Today (2-3 hours)

1. **Create Task 1: Environment State Machine** (Priority 1)
   - Create `contracts/` directory
   - Write `environment_state_machine.md`
   - Include formal transition rules
   - Document all invariants

2. **Start Task 2: Core Types** (Priority 2)
   - Create `core_types.md`
   - Document Coordinates, GridSize, AgentState
   - Include all properties and invariants

### Tomorrow (2-3 hours)

3. **Complete Task 2 & Start Task 3**
   - Finish core types
   - Write reward function specification

4. **Task 4: Concentration Field**
   - Physical laws as invariants
   - Mathematical specification

### Day 3 (1-2 hours)

5. **Task 5: Update CONTRACTS.md**
   - Add references to new files
   - Add invariant sections
   - Mathematical properties

6. **Review & Validation**
   - Check all contracts are consistent
   - No contradictions
   - Complete coverage

---

## ✅ Completion Criteria

**Phase 2 is complete when:**

- [ ] Environment state machine fully specified with formal rules
- [ ] All core data types have pre/post/invariants
- [ ] Reward function mathematically defined
- [ ] Concentration field physical laws documented
- [ ] CONTRACTS.md updated with references
- [ ] All invariants explicitly listed
- [ ] All pre/postconditions documented
- [ ] Mathematical properties proven/testable
- [ ] No ambiguities in specifications
- [ ] Peer review complete

**Success Metrics:**
- Can generate test cases mechanically from contracts
- Every contract has corresponding guard test (Phase 3)
- Zero ambiguity in expected behavior
- All edge cases identified and documented

---

## 🚀 After Phase 2

**Phase 3 Begins:**
- Write guard tests for each contract
- One test file per contract document
- 100% coverage of state transitions
- Property tests for mathematical properties

**Expected Timeline:**
- Phase 2: 6-8 hours (spread over 3 days)
- Phase 3: 8-10 hours (guard test writing)
- Phase 4: 4-6 hours (align existing tests)
- Phase 5: 10-15 hours (fix implementations)

**Total:** ~30-40 hours = 1 work week

---

**REMEMBER:**
- Contracts first, tests second, code last
- Be mathematically precise
- Every statement must be testable
- No implementation details in contracts
- Focus on "what", not "how"
