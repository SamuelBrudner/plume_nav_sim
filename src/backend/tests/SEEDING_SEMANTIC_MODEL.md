# Seeding Validation Semantic Model

**Version**: 1.0  
**Status**: Canonical Design Specification  
**Last Updated**: 2025-09-29

## Purpose

Define the complete, unambiguous contract for seed validation in `plume_nav_sim` to ensure:
- Scientific reproducibility
- Gymnasium API compatibility
- FAIR data principles (traceable, explicit)
- Fail-loud behavior (no silent transformations)

## Core Contract: `validate_seed(seed: Any) -> Tuple[bool, Optional[int], str]`

**SIMPLIFIED**: `strict_mode` parameter **eliminated** (vestigial complexity, no useful semantic distinction).

### Return Signature
```python
(is_valid: bool, normalized_seed: Optional[int], error_message: str)
```

### Validation Rules (Canonical)

#### ✅ VALID Inputs

| Input Type | Example | Normalized To | Rationale |
|------------|---------|---------------|-----------|
| `None` | `None` | `None` | Explicit request for random seed generation (Gymnasium standard) |
| Non-negative int in range | `42` | `42` | Direct use; within [0, 2³²-1] |
| Zero | `0` | `0` | Valid seed; edge case |
| Max value | `2**32-1` | `2**32-1` | Valid seed; edge case |
| numpy.integer | `np.int64(42)` | `42` | Converted to native int for compatibility |

#### ❌ INVALID Inputs

| Input Type | Example | Reason | Error Message Pattern |
|------------|---------|--------|----------------------|
| Negative int | `-1` | No silent normalization; likely user error | "Seed must be non-negative..." |
| Out-of-range int | `2**32` | Exceeds max; platform incompatibility risk | "Seed {value} out of range..." |
| Float | `3.14` | No silent truncation; type mismatch | "Seed must be integer type..." |
| String | `"invalid"` | Type error | "Seed must be integer type..." |
| Other types | `[1,2,3]` | Type error | "Seed must be integer type..." |

### Type Conversion Policy

**Single, consistent behavior** (no mode flags):
- Accept `None` → pass through
- Accept native Python `int` → pass through
- Accept `numpy.integer` → convert to native `int`
- Reject `float` → no truncation
- Reject `string` → no parsing
- Reject all other types → type error

### Normalization Policy

**NO NORMALIZATION** is performed. The function validates only.

- ❌ No modulo operation on negative seeds
- ❌ No truncation of floats
- ❌ No string-to-int conversion
- ✅ Only type conversion: `numpy.integer → int`

**Rationale**: Silent transformations violate "fail loud and fast" principle and obscure user intent in scientific code.

## Test Classification

### Valid Seed Test Cases
```python
VALID_SEEDS = [
    None,           # Random seed request
    0,              # Boundary: minimum
    1,              # Boundary: smallest positive
    42,             # Standard test seed
    123, 456, 789, 2023,  # Additional test seeds
    SEED_MAX_VALUE - 1,   # Near maximum
    SEED_MAX_VALUE,       # Boundary: maximum (2**32-1)
]
```

### Invalid Seed Test Cases
```python
INVALID_SEEDS = [
    -1,                    # Negative (no normalization)
    -100,                  # Negative
    SEED_MAX_VALUE + 1,    # Out of range (too large)
    2**33,                 # Way out of range
    3.14,                  # Float (no truncation)
    0.0,                   # Float zero
    "invalid",             # String
    "42",                  # String digit
    [42],                  # List
    {"seed": 42},          # Dict
]
```

### Numpy Integer Test Cases
```python
NUMPY_INTEGER_SEEDS = [
    np.int32(42),    # Valid, converts to 42
    np.int64(123),   # Valid, converts to 123
    np.uint32(456),  # Valid, converts to 456
]
# All numpy integer types are VALID and converted to native int
```

## Error Message Requirements

Error messages MUST:
1. Be descriptive and actionable
2. Include the actual invalid value (when safe to display)
3. Specify the valid range or type
4. Contain keywords: "seed", "type", "range", "invalid", or "value"

### Required Error Message Patterns

| Failure Type | Required Keywords | Example |
|--------------|-------------------|---------|
| Type error | "type", "invalid" | "Seed must be integer type, got str" |
| Negative | "range", "negative" | "Seed -1 out of range [0, 4294967295]" |
| Out of range | "range", "maximum" | "Seed 4294967296 exceeds maximum 4294967295" |
| Float | "type", "integer" | "Seed must be integer type, got float (no truncation)" |

## Integration with Other Components

### `create_seeded_rng(seed: Optional[int], validate_input: bool = True) -> Tuple[Generator, int]`
- MUST call `validate_seed(seed)` if `validate_input=True`
- If validation fails, MUST raise `ValidationError` with context
- MUST handle `seed=None` as "generate random" (pass to Gymnasium)
- MUST NOT assume transformation occurred (identity for valid seeds)

### `SeedManager.seed(seed: Optional[int], context_id: str)`
- MUST validate seed using `validate_seed()` if `enable_validation=True`
- MUST track actual seed used (after generation or normalization)
- MUST NOT silently modify seeds

### `get_random_seed(use_system_entropy: bool) -> int`
- Generates random seeds when explicitly requested
- MUST return value in valid range [0, SEED_MAX_VALUE]
- MUST use system entropy when available

## Versioning & Compatibility

- **Current Version**: 1.0
- **Breaking Changes**: Any modification to validation rules requires major version bump
- **Gymnasium Compatibility**: Must remain compatible with `gymnasium.Env.reset(seed=...)` semantics

## Testing Requirements

All tests MUST:
1. Use constants from this specification
2. Test both `strict_mode=True` and `strict_mode=False` where applicable
3. Verify error message content matches requirements
4. Test boundary values (0, SEED_MAX_VALUE)
5. Ensure no silent transformations occur

## References

- Gymnasium seeding: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
- NumPy random: https://numpy.org/doc/stable/reference/random/generator.html
- FAIR principles: https://www.go-fair.org/fair-principles/
