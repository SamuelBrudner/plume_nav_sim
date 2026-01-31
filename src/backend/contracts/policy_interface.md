# Policy Interface Contract

**Component:** Policy Abstraction  
**Version:** 1.0.0  
**Date:** 2025-10-24  
**Status:** CANONICAL - All implementations MUST conform

---

## 📦 Type Dependencies

This contract references types defined in other contracts:

- `ActionType`: Action representation compatible with `gymnasium` spaces. Matches `action_space` (e.g., `int` for `Discrete`, `np.ndarray` for `Box/MultiDiscrete`, dict/tuple for composite spaces).
- `ObservationType`: Matches observation model outputs: `np.ndarray` (Box), `dict` (Dict), or `tuple` (Tuple) depending on the configured observation space.

---

## 🎯 Purpose

Define the universal interface for policies (controllers) that map observations to actions. This enables:

- Plugging in rule-based and learning-based policies interchangeably
- Clean separation between decision logic and environment dynamics
- Reproducible stochastic policies via explicit seeding

---

## 📐 Interface Definition

### Type Signature

```python
Policy: (Observation, explore: bool=True) → Action

Where:
  - Defines: action_space (Gymnasium Space)
  - Exposes: reset(seed) for RNG control and internal state reset
  - Computes: select_action(observation, explore) → action ∈ action_space
```

### Protocol Specification

```python
class Policy(Protocol):
    """Protocol defining a stochastic or deterministic policy.

    Minimal interface:
      - action_space: Gymnasium space for actions
      - reset(seed): optional seeding hook
      - select_action(observation, explore): returns an action compatible with action_space
    """

    @property
    def action_space(self) -> gym.Space:
        """Gymnasium action space definition.

        Postconditions:
          C1: Returns valid gym.Space instance
          C2: Space is immutable (same instance every call)

        Returns:
            Gymnasium Space defining valid actions.
        """
        ...

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal state and (optionally) RNG with a seed.

        Effects:
          - Clears any internal policy memory/state used across steps
          - If seed is provided, reinitializes RNG deterministically

        Postconditions:
          C1: After reset with the same seed, subsequent action sequences are reproducible given the same observation sequence and explore flag.
        """
        ...

    def select_action(self, observation: ObservationType, *, explore: bool = True) -> ActionType:
        """Select an action given the current observation.

        Preconditions:
          P1: observation conforms to the environment's observation model (type/shape)

        Postconditions:
          C1: action ∈ self.action_space
          C2: Does not mutate the input observation
          C3: For a fixed observation sequence and seed, produces the same action sequence (seed-determinism)

        Notes:
          - explore controls whether exploratory stochasticity is enabled; policies may still be stochastic when explore=False if their greedy tie-breaking is randomized. Seed-determinism applies in all cases when seeds and inputs are identical.
        """
        ...
```

---

## 🌍 Universal Properties

### Property 1: Action-Space Containment (UNIVERSAL)

```python
∀ observation:
  action = select_action(observation)
  ⇒ action_space.contains(action) = True
```

**Test:**

```python
def test_policy_action_in_space(policy, observation):
    action = policy.select_action(observation, explore=True)
    assert policy.action_space.contains(action)
```

### Property 2: Seed Determinism (UNIVERSAL)

```python
∀ seed, observations:
  reset(seed); a1 = [select_action(o) for o in observations]
  reset(seed); a2 = [select_action(o) for o in observations]
  ⇒ a1 == a2
```

**Test:**

```python
def test_policy_seed_determinism(policy, observations):
    policy.reset(seed=123)
    a1 = [policy.select_action(o, explore=True) for o in observations]
    policy.reset(seed=123)
    a2 = [policy.select_action(o, explore=True) for o in observations]
    assert a1 == a2
```

### Property 3: Purity w.r.t. Inputs (UNIVERSAL)

```python
∀ observation: select_action(observation) does not mutate observation
```

**Test:**

```python
def test_policy_does_not_mutate_observation(policy):
    obs = np.array([0.5], dtype=np.float32)
    obs_copy = obs.copy()
    _ = policy.select_action(obs, explore=True)
    np.testing.assert_array_equal(obs, obs_copy)
```

### Property 4: Space Immutability (UNIVERSAL)

```python
action_space is stable across calls
```

**Test:**

```python
def test_policy_action_space_immutable(policy):
    assert policy.action_space is policy.action_space
```

---

## 📊 Implementation Notes

- Policies may maintain internal state (e.g., previous observation) as long as `reset()` restores a known baseline.
- Stochasticity must be entirely derived from internal RNG state controlled by `reset(seed=...)` to satisfy seed determinism.
- Policies should accept observations by read-only contract; do not write into provided arrays or objects.
