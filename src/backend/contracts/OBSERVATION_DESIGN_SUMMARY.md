# Observation Model Design Summary

**Date:** 2025-10-01  
**Status:** Design Decision Record

---

## ✅ Design Principles

### 1. **Leverage Gymnasium, Don't Reinvent**
- Use `gym.spaces.Dict` for multi-modal composition ✓
- Use `gym.spaces.Tuple` for ordered composition ✓
- Use `gym.spaces.Box` for vector observations ✓
- No custom composition logic needed

### 2. **General Abstraction, Not Olfactory-Specific**
- Sensors observe `env_state: Dict[str, Any]` (generic)
- Not hardcoded to `(agent_state, plume_field)`
- Supports any sensory modality: olfactory, mechanosensory, visual, temporal
- Users can add custom state fields for novel sensors

### 3. **Make Simple Things Simple**
- `ConcentrationSensor()` as default - convenient odor sampling (95% use case)
- Single line: `env = PlumeSearchEnv()` just works
- Complex multi-modal sensing is possible but not required

---

## 🏗️ Architecture

### Signature Change

**Before (too specific):**
```python
def get_observation(
    self,
    agent_state: AgentState,
    plume_field: ConcentrationField,
    **context
) -> np.ndarray:
```

**After (general, extensible):**
```python
def get_observation(
    self,
    env_state: Dict[str, Any]
) -> Union[np.ndarray, Dict, Tuple]:
    """
    env_state contains:
      - 'agent_state': AgentState (required)
      - 'plume_field': ConcentrationField (for olfactory sensors)
      - 'wind_field': WindField (for mechanosensory)
      - Custom fields added by users
    """
```

### Environment Integration

```python
class PlumeSearchEnv:
    def _get_env_state(self) -> Dict[str, Any]:
        """Extensibility point for novel sensors."""
        state = {
            'agent_state': self.agent_state,
            'plume_field': self.plume_field,
            'time_step': self.step_count
        }
        
        # Optional fields
        if self.wind_model:
            state['wind_field'] = self.wind_model.get_field()
        
        return state
```

---

## 📊 Usage Patterns

### Pattern 1: Simple Odor Sensing (Default)
```python
env = PlumeSearchEnv()  # Uses ConcentrationSensor by default
# observation: array([0.42], dtype=float32)
```

### Pattern 2: Multiple Olfactory Sensors
```python
env = PlumeSearchEnv(
    observation_model=AntennaeArraySensor([(0,0), (1,0), (-1,0)])
)
# observation: array([0.42, 0.38, 0.45], dtype=float32)
```

### Pattern 3: Multi-Modal (Odor + Wind)
```python
env = PlumeSearchEnv(
    observation_model=MultiModalSensor({
        'odor': ConcentrationSensor(),
        'wind': WindSensor()
    }),
    wind_model=ConstantWindModel()
)
# observation: {'odor': array([0.42]), 'wind': array([0.3, -0.8])}
```

### Pattern 4: Flattened Multi-Modal (For RL)
```python
env = PlumeSearchEnv(
    observation_model=FlattenedMultiSensor([
        ConcentrationSensor(),
        WindSensor()
    ])
)
# observation: array([0.42, 0.3, -0.8], dtype=float32)
```

---

## 🎯 Benefits

1. **Not reinventing composition** - Gymnasium's Dict/Tuple spaces do this
2. **Convenient odor sampling** - `ConcentrationSensor()` default is simple
3. **Extensible to any modality** - wind, obstacles, time, user-defined fields
4. **Composable** - build complex observations from simple sensors
5. **RL-friendly** - flattened option for algorithms expecting vectors

---

## 🔄 Comparison to Gymnasium Standards

### Gymnasium Dict Space (Built-in)
```python
gym.spaces.Dict({
    'position': Box(low=0, high=1, shape=(2,)),
    'velocity': Box(low=-1, high=1, shape=(2,))
})
```

### Our MultiModalSensor (Uses Dict Space)
```python
MultiModalSensor({
    'odor': ConcentrationSensor(),
    'wind': WindSensor()
})
# observation_space is gym.spaces.Dict - standard Gymnasium!
```

**We're not reinventing - just providing convenient sensor abstractions that compose via standard Gymnasium spaces.**

---

## 📝 Implementation Checklist

- [ ] Create `ConcentrationSensor` class (default, simple)
- [ ] Create `WindSensor` class (example non-olfactory)
- [ ] Create `AntennaeArraySensor` class (spatial olfactory)
- [ ] Create `MultiModalSensor` class (uses `gym.spaces.Dict`)
- [ ] Create `FlattenedMultiSensor` class (uses `gym.spaces.Box`)
- [ ] Update `PlumeSearchEnv` to use `_get_env_state()` pattern
- [ ] Document extension pattern for custom sensors
- [ ] Add examples showing all patterns

---

**Approved:** 2025-10-01  
**Rationale:** Balances simplicity (convenient odor sampling) with extensibility (any sensor modality) while leveraging Gymnasium's existing composition primitives.
