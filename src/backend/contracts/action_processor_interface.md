# Action Processor Interface Contract

**Component:** Action Processor Abstraction  
**Version:** 1.0.0  
**Date:** 2025-10-01  
**Status:** CANONICAL - All implementations MUST conform

---

## 📦 Type Dependencies

This contract references types defined in other contracts:
- `AgentState`: See `core_types.md` - Contains position and orientation
- `Coordinates`: See `core_types.md` - 2D integer grid position
- `GridSize`: See `core_types.md` - Grid dimensions
- `ActionType`: `Union[int, np.ndarray]` depending on action space

---

## 🎯 Purpose

Define the **universal interface** for action processing, enabling:
- Diverse action spaces (discrete, continuous, hybrid, multi-agent)
- Pluggable movement models without environment modification
- Research flexibility for different control paradigms
- Clean separation between action representation and state transition

---

## 📐 Interface Definition

### Type Signature

```python
ActionProcessor: (ActionType, AgentState, GridSize) → AgentState

Where:
  - Defines: action_space (Gymnasium Space)
  - Processes: action + current_state → new_state
  - Updates: position and potentially orientation
  - Enforces: Boundary constraints
```

### Protocol Specification

```python
class ActionProcessor(Protocol):
    """Protocol defining action processor interface.
    
    All action processors must conform to this interface to be
    compatible with the environment and config system.
    """
    
    @property
    def action_space(self) -> gym.Space:
        """Gymnasium action space definition.
        
        Postconditions:
          C1: Returns valid gym.Space instance
          C2: Space is immutable (same instance every call)
          C3: Space defines valid action representation
        
        Returns:
            Gymnasium Space defining valid actions
        """
        ...
    
    def process_action(
        self,
        action: ActionType,
        current_state: AgentState,
        grid_size: GridSize
    ) -> AgentState:
        """Process action to compute new agent state.
        
        Preconditions:
          P1: action ∈ self.action_space
          P2: current_state is valid AgentState
          P3: grid_size.contains(current_state.position) = True
        
        Postconditions:
          C1: result is valid AgentState
          C2: grid_size.contains(result.position) = True (stays in bounds)
          C3: Result is deterministic (same inputs → same output)
          C4: result is new instance (not mutated current_state)
        
        Properties:
          1. Boundary Safety: Result position always within grid
          2. Determinism: Same (action, state, grid) → same result
          3. Purity: No side effects, no mutation of current_state
          4. Updates: May update position and/or orientation
        
        Args:
            action: Action from action_space
            current_state: Agent's current state (position, orientation, etc.)
            grid_size: Grid bounds for boundary enforcement
        
        Returns:
            New AgentState after action (position within grid bounds)
        """
        ...
    
    def validate_action(self, action: ActionType) -> bool:
        """Check if action is valid for this processor.
        
        Args:
            action: Action to validate
        
        Returns:
            True if action is valid for this action_space
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return action processor metadata.
        
        Returns:
            Dictionary containing:
            - 'type': str - Action processor type
            - 'parameters': dict - Configuration
            - 'movement_model': Description of movement semantics
        """
        ...
```

---

## 🌍 Universal Properties

### Property 1: Boundary Safety (UNIVERSAL)

```python
∀ action, state, grid:
  grid.contains(state.position) = True
  ⇒ grid.contains(process_action(action, state, grid).position) = True

Agent NEVER leaves grid bounds.
```

**Test:**
```python
@given(
    action=action_strategy(),
    state=agent_state_strategy(),
    grid_size=grid_size_strategy()
)
def test_boundary_safety(action_proc, action, state, grid_size):
    """Processed action always stays within bounds."""
    # Ensure starting position is valid
    assume(grid_size.contains(state.position))
    assume(action_proc.validate_action(action))
    
    new_state = action_proc.process_action(action, state, grid_size)
    
    assert grid_size.contains(new_state.position), \
        f"Position {new_state.position} outside grid {grid_size}"
```

### Property 2: Determinism (UNIVERSAL)

```python
∀ action, state, grid:
  process_action(action, state, grid) = process_action(action, state, grid)
```

**Test:**
```python
@given(action=action_strategy(), state=agent_state_strategy(), grid=grid_size_strategy())
def test_determinism(action_proc, action, state, grid):
    """Same inputs produce same output."""
    assume(grid.contains(state.position))
    assume(action_proc.validate_action(action))
    
    result1 = action_proc.process_action(action, state, grid)
    result2 = action_proc.process_action(action, state, grid)
    
    assert result1 == result2, "Action processing must be deterministic"
```

### Property 3: Purity (UNIVERSAL)

```python
∀ inputs: process_action(inputs) has no side effects

No modification of:
  - action
  - current_state (returns new instance)
  - grid_size
  - Global state
```

**Test:**
```python
def test_purity(action_proc):
    """Action processing has no side effects."""
    action = 0
    state = AgentState(position=Coordinates(10, 10), orientation=0.0)
    grid = GridSize(128, 128)
    
    orig_state = copy.copy(state)
    
    new_state = action_proc.process_action(action, state, grid)
    
    assert state.position == orig_state.position, "current_state.position modified"
    assert state.orientation == orig_state.orientation, "current_state.orientation modified"
    assert new_state is not state, "Should return new instance"
```

---

## 📊 Implementation Examples

### Discrete Grid Actions (Current Default)

```python
class DiscreteGridActions:
    """Standard 4-directional discrete movement.
    
    Current default action model matching existing behavior.
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    """
    
    def __init__(self, step_size: int = 1):
        self.step_size = step_size
        self._movements = {
            0: (0, step_size),   # UP
            1: (step_size, 0),   # RIGHT
            2: (0, -step_size),  # DOWN
            3: (-step_size, 0)   # LEFT
        }
    
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(4)
    
    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize
    ) -> AgentState:
        """Process discrete action to new state."""
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get movement vector
        dx, dy = self._movements[action]
        
        # Compute new position
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy
        
        # Enforce boundaries (clamp to grid)
        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))
        
        # Return new state (orientation unchanged for absolute movement)
        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=current_state.orientation
        )
    
    def validate_action(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and 0 <= action < 4
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'discrete_grid',
            'parameters': {'step_size': self.step_size, 'n_actions': 4},
            'movement_model': 'cardinal_directions'
        }
```

### Continuous Actions

```python
@dataclass
class ContinuousActions:
    """Continuous velocity-based control.
    
    Actions are (vx, vy) velocities in [-1, 1]^2.
    Useful for smooth control and optimal control research.
    """
    max_speed: float = 2.0
    
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
    
    def process_action(
        self,
        action: np.ndarray,
        current_state: AgentState,
        grid_size: GridSize
    ) -> AgentState:
        """Process continuous velocity action."""
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        vx, vy = action
        
        # Scale by max_speed
        dx = int(round(vx * self.max_speed))
        dy = int(round(vy * self.max_speed))
        
        # Compute new position
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy
        
        # Clamp to bounds
        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))
        
        # Return new state (orientation unchanged)
        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=current_state.orientation
        )
    
    def validate_action(self, action: np.ndarray) -> bool:
        if not isinstance(action, np.ndarray):
            return False
        if action.shape != (2,):
            return False
        return np.all(np.abs(action) <= 1.0)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'continuous',
            'parameters': {'max_speed': self.max_speed},
            'movement_model': 'velocity_control'
        }
```

### Oriented Grid Actions (Uses Orientation!)

```python
class OrientedGridActions:
    """3-action surge/turn control with orientation tracking.
    
    Actions: 0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT
    Movement is relative to current orientation.
    """
    
    def __init__(self, step_size: int = 1):
        self.step_size = step_size
    
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(3)
    
    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize
    ) -> AgentState:
        """Process oriented action."""
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        new_orientation = current_state.orientation
        dx, dy = 0, 0
        
        if action == 0:  # FORWARD
            # Move in current heading direction
            rad = np.radians(current_state.orientation)
            dx = int(round(self.step_size * np.cos(rad)))
            dy = int(round(self.step_size * np.sin(rad)))
        elif action == 1:  # TURN_LEFT
            new_orientation = (current_state.orientation + 90.0) % 360.0
        elif action == 2:  # TURN_RIGHT
            new_orientation = (current_state.orientation - 90.0) % 360.0
        
        # Compute new position
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy
        
        # Clamp to bounds
        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))
        
        # Return new state with updated position and orientation
        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=new_orientation
        )
    
    def validate_action(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and 0 <= action < 3
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'oriented_grid',
            'parameters': {'step_size': self.step_size, 'n_actions': 3},
            'movement_model': 'forward_turn_left_turn_right'
        }
```

### 8-Direction + Stay

```python
class EightDirectionActions:
    """8-directional movement plus stay action.
    
    Actions: 0-7 = 8 directions, 8 = stay
    Useful for richer movement without continuous control.
    """
    
    def __init__(self, step_size: int = 1):
        self.step_size = step_size
        s = step_size
        self._movements = {
            0: (0, s),    # N
            1: (s, s),    # NE
            2: (s, 0),    # E
            3: (s, -s),   # SE
            4: (0, -s),   # S
            5: (-s, -s),  # SW
            6: (-s, 0),   # W
            7: (-s, s),   # NW
            8: (0, 0)     # STAY
        }
    
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(9)
    
    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize
    ) -> AgentState:
        """Process 8-direction + stay action."""
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        dx, dy = self._movements[action]
        
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy
        
        # Clamp to bounds
        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))
        
        # Return new state (orientation unchanged)
        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=current_state.orientation
        )
    
    def validate_action(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and 0 <= action < 9
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'type': 'eight_direction_stay',
            'parameters': {'step_size': self.step_size, 'n_actions': 9},
            'movement_model': 'eight_directions_plus_stay'
        }
```

---

## 🧪 Required Test Suite

```python
class TestActionProcessorInterface:
    """Universal test suite for action processors."""
    
    @pytest.fixture
    def action_processor(self):
        """Override in concrete test classes."""
        raise NotImplementedError
    
    @given(action=action_strategy(), state=agent_state_strategy(), grid=grid_size_strategy())
    def test_boundary_safety(self, action_processor, action, state, grid):
        """P1: Result always within bounds."""
        
    @given(action=action_strategy(), state=agent_state_strategy(), grid=grid_size_strategy())
    def test_determinism(self, action_processor, action, state, grid):
        """P2: Deterministic processing."""
        
    def test_purity(self, action_processor):
        """P3: No side effects."""
        
    def test_space_immutability(self, action_processor):
        """action_space returns same instance."""
        space1 = action_processor.action_space
        space2 = action_processor.action_space
        assert space1 is space2
    
    def test_validate_action_consistency(self, action_processor):
        """validate_action matches action_space.contains()."""
        valid_action = action_processor.action_space.sample()
        assert action_processor.validate_action(valid_action)
```

### Edge Case Tests

```python
def test_corner_actions(action_proc):
    """Test actions at grid corners."""
    grid = GridSize(10, 10)
    corner_positions = [
        Coordinates(0, 0),
        Coordinates(9, 0),
        Coordinates(0, 9),
        Coordinates(9, 9)
    ]
    for pos in corner_positions:
        state = AgentState(position=pos, orientation=0.0)
        for action in range(action_proc.action_space.n):
            new_state = action_proc.process_action(action, state, grid)
            assert grid.contains(new_state.position)

def test_edge_actions(action_proc):
    """Test actions at grid edges."""
    grid = GridSize(10, 10)
    edge_positions = [
        Coordinates(5, 0),   # Top edge
        Coordinates(9, 5),   # Right edge
        Coordinates(5, 9),   # Bottom edge
        Coordinates(0, 5)    # Left edge
    ]
    for pos in edge_positions:
        state = AgentState(position=pos, orientation=0.0)
        for action in range(action_proc.action_space.n):
            new_state = action_proc.process_action(action, state, grid)
            assert grid.contains(new_state.position)
```

---

## 🔗 Integration Requirements

### Environment Integration

```python
class PlumeSearchEnv:
    def __init__(
        self,
        action_processor: Optional[ActionProcessor] = None,
        **kwargs
    ):
        # Default to current behavior
        self.action_proc = action_processor or DiscreteGridActions()
        
        # Action space from processor
        self.action_space = self.action_proc.action_space
    
    def step(self, action):
        # Validate action
        if not self.action_proc.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Process action to get new state
        self.agent_state = self.action_proc.process_action(
            action,
            self.agent_state,
            self.grid_size
        )
        
        # ... rest of step logic ...
```

---

## ⚠️ Common Implementation Errors

### ❌ Wrong: Violates Boundaries

```python
class BadActionProc:
    def process_action(self, action, state, grid):
        dx, dy = self._movements[action]
        new_x = state.position.x + dx
        new_y = state.position.y + dy
        return AgentState(
            position=Coordinates(new_x, new_y),  # ❌ Can go out of bounds!
            orientation=state.orientation
        )
```

### ❌ Wrong: Non-Deterministic

```python
class BadActionProc:
    def process_action(self, action, state, grid):
        dx, dy = self._movements[action]
        noise = np.random.randint(-1, 2)  # ❌ Random!
        new_x = state.position.x + dx + noise
        return AgentState(
            position=Coordinates(new_x, state.position.y),
            orientation=state.orientation
        )
```

### ✅ Correct: Safe and Deterministic

```python
class GoodActionProc:
    def process_action(self, action, state, grid):
        dx, dy = self._movements[action]
        new_x = state.position.x + dx
        new_y = state.position.y + dy
        
        # Always clamp to bounds
        clamped_x = max(0, min(new_x, grid.width - 1))
        clamped_y = max(0, min(new_y, grid.height - 1))
        
        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=state.orientation
        )
```

---

## 📊 Verification Checklist

Implementation MUST satisfy:

- [ ] Implements ActionProcessor protocol
- [ ] action_space is valid gym.Space
- [ ] action_space is immutable
- [ ] process_action() always returns valid position
- [ ] Boundary safety (never out of bounds)
- [ ] Deterministic processing
- [ ] Pure (no side effects)
- [ ] validate_action() matches action_space
- [ ] Passes all property tests
- [ ] Handles edge and corner cases

---

**Last Updated:** 2025-10-01  
**Related Contracts:**
- `core_types.md` - Coordinates, GridSize definitions
- `environment_state_machine.md` - Environment step() integration
