# API Simplification & DI Architecture Plan

**Date:** 2025-10-14  
**Status:** APPROVED  
**Goal:** Keep DI architecture for extensibility while simplifying user-facing complexity

---

## 🎯 Core Decision

**KEEP:** Component-based DI architecture (competitive advantage for research)  
**REMOVE:** Dual architecture confusion (legacy vs components)  
**SIMPLIFY:** Registration complexity and user-facing API  
**PATTERN:** Progressive disclosure (simple → customized → extended)

---

## 📋 Action Items

### Phase 1: Unify on DI (Remove Dual Architecture)

#### 1.1 Make Factory the Default

- [ ] Remove `PlumeSearchEnv` as separate class (or make it thin wrapper over factory)
- [ ] Eliminate `ENV_ID` vs `COMPONENT_ENV_ID` distinction → Single `PlumeNav-v0`
- [ ] Remove `PLUMENAV_DEFAULT` environment variable (always use DI internally)
- [ ] Update `plume_search_env.py` to always delegate to `create_component_environment`

**Files to modify:**

```
src/backend/plume_nav_sim/envs/plume_search_env.py    # Make wrapper over factory
src/backend/plume_nav_sim/registration/register.py     # Single env_id, single entry point
src/backend/plume_nav_sim/registration/__init__.py     # Remove dual-ID exports
```

#### 1.2 Keep Factory String-Only (Opinionated Design)

- [x] **Decision**: Factory accepts ONLY string shortcuts, NOT component instances
- [ ] Add better error messages for invalid `action_type`, `observation_type`, etc.
- [ ] Keep existing signature (already excellent with Literal types)

**Rationale:**

- Clear separation: Factory = convenience, ComponentBasedEnvironment = power
- Simpler mental model: strings for built-ins, direct instantiation for custom
- Better error messages: "Unknown action_type" vs "Expected str or ActionProcessor"
- Explicit progression: `make_env()` → `make_env(action_type='oriented')` → `ComponentBasedEnvironment(...)`

**No changes needed to factory signature** - it already uses `Literal` types!

---

### Phase 2: Simplify Registration

#### 2.1 Reduce Registration Module Complexity

- [ ] Move complex caching/status functions to `registration/_internal.py` (testing only)
- [ ] Reduce `registration/__init__.py` from 1046 lines → ~100 lines
- [ ] Auto-register on import (no manual `register_env()` call needed)
- [ ] Keep only essential public API: `register()`, `is_registered()`

**Current bloat to remove from public API:**

```python
# Remove from __all__ and public docs
_module_initialized          # Internal state
_registration_cache          # Internal caching
_cache_lock                  # Internal threading
get_registration_status()    # Debugging, not user-facing
clear_registration_cache()   # Testing utility
initialize_package()         # Should be automatic
get_package_info()          # Redundant with metadata
```

**Target public API:**

```python
# plume_nav_sim/registration/__init__.py
__all__ = [
    'register',      # Simple registration
    'is_registered'  # Check if registered
]
```

#### 2.2 Simplify Package Init

- [ ] Add `make_env()` convenience function to `plume_nav_sim/__init__.py`
- [ ] Auto-register on import (remove manual registration burden)
- [ ] Reduce exports from 40+ → ~15 essential items

**Target package API:**

```python
# plume_nav_sim/__init__.py
__all__ = [
    # Environment creation
    "make_env",              # NEW: Simple entry point
    
    # Constants (most commonly needed)
    "DEFAULT_GRID_SIZE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_GOAL_RADIUS",
    
    # Types (for type hints)
    "Action",
    "AgentState",
    "EnvironmentConfig",
    "GridSize",
    "Coordinates",
    
    # Metadata
    "__version__",
]
```

---

### Phase 3: Documentation Overhaul

#### 3.1 Update README.md

- [ ] Lead with simplest API: `plume_nav_sim.make_env()`
- [ ] Show customization with string parameters
- [ ] Move custom component examples to `docs/extending/`
- [ ] Remove dual-architecture explanations

**New README structure:**

```markdown
## Quick Start

### Basic Usage
```python
import plume_nav_sim
env = plume_nav_sim.make_env()
```

### Customize Components

```python
env = plume_nav_sim.make_env(
    action_type='oriented',
    observation_type='antennae',
    reward_type='step_penalty'
)
```

### Gymnasium Integration

```python
import gymnasium as gym
import plume_nav_sim

plume_nav_sim.register()  # Auto-called, but explicit OK
env = gym.make('PlumeNav-v0', action_type='oriented')
```

**Want to build custom components?** → [Extending Guide](docs/extending/)

```

#### 3.2 Reorganize Examples
- [ ] Delete empty example files (`advanced_usage.py`, `performance_benchmark.py`, etc.)
- [ ] Split oversized files (>10k LOC) into focused examples
- [ ] Create clear progression:
  - `examples/quickstart.py` - 5 minute intro
  - `examples/custom_configuration.py` - Built-in customization
  - `examples/custom_components.py` - Inject custom classes

**Target examples structure:**
```

examples/
├── quickstart.py              (50 lines: basic usage)
├── custom_configuration.py    (100 lines: customize via strings)
├── custom_components.py       (150 lines: inject custom classes)
└── reproducibility.py         (80 lines: seeding examples)

```

#### 3.3 Update docs/extending/
- [ ] Create clear landing page explaining DI architecture
- [ ] Show custom component template for each interface
- [ ] Explain when/why to extend (research scenarios)
- [ ] Keep technical details here (not in main README)

**New extending docs structure:**
```markdown
# Extending PlumeNav

PlumeNav uses **dependency injection** so you can plug in custom components.

## Why Extend?

- Novel action spaces (e.g., continuous control)
- New observation types (e.g., egocentric vision)
- Custom reward shaping (e.g., curriculum learning)
- Different plume models (e.g., turbulent flow)

## Quick Example

```python
from plume_nav_sim.interfaces import ObservationModel

class EgocentricVision(ObservationModel):
    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, shape=(64, 64, 3))
    
    def compute_observation(self, state):
        # Your implementation
        return render_egocentric_view(state)

# Use it!
env = plume_nav_sim.make_env(
    observation_type=EgocentricVision()  # Custom component
)
```

## Component Interfaces

- [ActionProcessor](custom_actions.md) - Define action spaces
- [ObservationModel](custom_observations.md) - Design observations  
- [RewardFunction](custom_rewards.md) - Shape rewards
- [PlumeModel](custom_plumes.md) - Simulate environments

```

---

### Phase 4: Cleanup & Testing

#### 4.1 Remove Deprecated Patterns
- [ ] Delete `component_di_usage.py` (merge into main examples)
- [ ] Remove environment variable toggles (`PLUMENAV_DEFAULT`)
- [ ] Remove `ensure_component_env_registered()` (redundant)
- [ ] Consolidate deprecation warnings

#### 4.2 Update Tests
- [ ] Ensure tests use `make_env()` pattern
- [ ] Test both string-based and instance-based component injection
- [ ] Remove tests for dual-architecture logic
- [ ] Add tests for simplified registration

#### 4.3 Update Type Stubs
- [ ] Add `py.typed` marker
- [ ] Provide type stubs for public API
- [ ] Use Python 3.10+ syntax (`tuple[...]` not `Tuple[...]`)

---

## 📊 Success Metrics

**Before (Current State):**
- Two architectures visible to users
- 1046 lines in registration init
- 40+ exports in package init
- 3 ways to register environment
- 12 example files (many empty)

**After (Target State):**
- One architecture (DI, transparent)
- ~100 lines in registration init
- ~15 exports in package init
- One primary way (`make_env()`)
- 4 focused example files

**User Experience:**
```python
# From this (confusing)
from plume_nav_sim.registration import register_env, COMPONENT_ENV_ID
env_id = register_env(env_id=COMPONENT_ENV_ID, force_reregister=True)
env = gym.make(env_id)

# To this (simple)
import plume_nav_sim
env = plume_nav_sim.make_env()
```

---

## 🚦 Implementation Order

1. **Week 1:** Phase 1 (Unify on DI)
   - Remove dual architecture
   - Enhance factory for progressive disclosure

2. **Week 2:** Phase 2 (Simplify Registration)
   - Reduce registration complexity
   - Streamline package API

3. **Week 3:** Phase 3 (Documentation)
   - Rewrite README
   - Reorganize examples
   - Update extending guide

4. **Week 4:** Phase 4 (Cleanup & Testing)
   - Remove deprecated code
   - Update test suite
   - Polish public API

---

## 🎓 Architectural Rationale

### Why Keep DI?

**For research tools, extensibility is critical:**

- Users need to test novel algorithms
- Standard components should be swappable
- Custom implementations shouldn't require forking

**DI provides:**

- Clean interfaces (Protocols)
- Easy testing (inject mocks)
- Zero coupling (components don't know about each other)

### Why Hide Complexity?

**Most users don't need customization:**

- 80% use case: "just give me an environment"
- 15% use case: "customize with built-in options"
- 5% use case: "inject my custom research code"

**Progressive disclosure:**

- Level 1: `make_env()` → works immediately
- Level 2: `make_env(action_type='oriented')` → built-in options
- Level 3: `make_env(action_type=MyActions())` → full power

### Why Single Architecture?

**Maintaining two implementations is costly:**

- Double testing burden
- Confusing for users ("which one?")
- Creates technical debt
- No clear migration path

**DI handles all use cases:**

- Simple: factory with defaults
- Custom: factory with string options
- Extended: factory with custom instances

---

## 📝 Migration Guide (for existing users)

### From Legacy to Unified

**Old way:**

```python
from plume_nav_sim.envs import PlumeSearchEnv
env = PlumeSearchEnv(grid_size=(128, 128))
```

**New way:**

```python
import plume_nav_sim
env = plume_nav_sim.make_env(grid_size=(128, 128))
```

### From Component-Specific to Unified

**Old way:**

```python
from plume_nav_sim.registration import register_env, COMPONENT_ENV_ID
env_id = register_env(env_id=COMPONENT_ENV_ID)
```

**New way:**

```python
import plume_nav_sim
env = plume_nav_sim.make_env()  # Uses DI internally
```

---

**END OF SIMPLIFICATION PLAN**
