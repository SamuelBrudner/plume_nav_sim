# Concentration Field Contract

**Component:** `ConcentrationField`  
**Version:** 1.0.0  
**Date:** 2025-09-30  
**Status:** CANONICAL - All implementations MUST conform

---

## 🎯 Purpose

Define the mathematical and physical properties of concentration fields representing chemical/odor diffusion. The field must satisfy physical laws and provide deterministic, efficient sampling.

---

## 📐 Mathematical Model

### Type Definition

```
ConcentrationField: GridSize → (Coordinates → [0, 1])

A concentration field is a function that maps every grid coordinate to a 
normalized concentration value between 0 and 1.
```

### Gaussian Model

For static Gaussian plume:

```
C(x, y) = exp(-d²(x, y, source) / (2σ²))

where:
  d(p₁, p₂) = √((x₂-x₁)² + (y₂-y₁)²)  (Euclidean distance)
  σ = sigma parameter (dispersion width)
  source = (x_s, y_s) source location
```

### Normalization

```
C_norm(x, y) = C(x, y) / C(source)
             = C(x, y) / 1.0
             = C(x, y)

Since C(source) = exp(0) = 1.0, the field is naturally normalized.
```

---

## 🔬 Invariants Classification

**IMPORTANT:** Distinguish between universal laws and model-specific properties!

### Universal Physical Laws (I1-I2)

These apply to **all** concentration field models:

- **I1: Non-negativity** - Concentration cannot be negative
- **I2: Bounded** - Normalized concentration ≤ 1.0 (after normalization)

### Quasi-Universal Laws (I3)

Usually true but has exceptions:

- **I3: Maximum at source** - Source has highest concentration (usually)

### Gaussian Model Properties (I4-I7)

These apply **only** to static Gaussian fields:

- **I4: Monotonic radial decay** - Closer → higher concentration
- **I5: Radial symmetry** - Symmetric around source
- **I7: Gaussian formula** - Follows exp(-d²/(2σ²))

⚠️ **NOT universal physical laws!**

- Real plumes with turbulence violate monotonic decay
- Wind breaks radial symmetry
- Other models (advection-diffusion, turbulent) use different equations

---

## 🌍 Universal Physical Laws

### I1: Non-Negativity (UNIVERSAL)

```
∀ (x, y) ∈ Grid: field[x, y] ≥ 0

Concentration cannot be negative.
```

**Test:**

```python
@given(grid=grid_size_strategy(), source=coordinates_strategy(), sigma=st.floats(0.1, 5.0))
def test_field_non_negative(grid, source, sigma):
    field = create_concentration_field(grid, source, sigma)
    assert np.all(field.field >= 0)
```

### I2: Bounded (UNIVERSAL)

```
∀ (x, y) ∈ Grid: field[x, y] ≤ 1.0

Normalized concentration cannot exceed 1.0.
Applies after normalization.
```

**Test:**

```python
@given(grid=grid_size_strategy(), source=coordinates_strategy(), sigma=st.floats(0.1, 5.0))
def test_field_bounded(grid, source, sigma):
    field = create_concentration_field(grid, source, sigma)
    assert np.all(field.field <= 1.0)
```

### I3: Maximum at Source (QUASI-UNIVERSAL)

```
∀ (x, y) ∈ Grid, (x, y) ≠ source:
  field[source] ≥ field[x, y]

The concentration is highest at the source location.
```

**Assumptions:**

- Passive tracer (no reactions)
- Continuous emission
- Steady-state (no transients)
- Time-averaged (for stochastic models)

⚠️ **Can be violated by:**

- Turbulent accumulation zones (eddies, recirculation)
- Reactive chemistry (products form downstream)
- Stochastic particle models (instantaneous snapshots)
- Boundary reflection effects (corners, walls)
- Particle coalescence/condensation downstream
- Distributed sources (nominal vs actual emission point)

**Holds for:** Most passive tracer plume models in steady-state

**Test:**

```python
@given(
    grid_size=st.integers(10, 50),
    sigma=st.floats(0.5, 5.0)
)
def test_maximum_at_source(grid_size, sigma):
    source = (grid_size // 2, grid_size // 2)
    field = create_concentration_field(
        grid_size=(grid_size, grid_size),
        source_location=source,
        sigma=sigma
    )
    
    source_value = field.sample(source)
    
    # Check all other positions
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) != source:
                value = field.sample((x, y))
                assert value <= source_value, \
                    f"Field at ({x},{y})={value} > source {source_value}"
```

---

## 🔬 Gaussian Model Properties

### I4: Monotonic Radial Decay (GAUSSIAN-SPECIFIC)

```
∀ p₁, p₂ ∈ Grid:
  d(p₁, source) ≤ d(p₂, source) ⇒ field[p₁] ≥ field[p₂]

Positions closer to source have higher or equal concentration.

⚠️ GAUSSIAN-SPECIFIC: NOT a universal physical law!
Real plumes with turbulence, wind, obstacles violate this.
Only holds for radially symmetric models.
```

**Test:**

```python
@given(
    grid=grid_size_strategy(),
    source=coordinates_in_grid_strategy(),
    sigma=st.floats(0.5, 5.0),
    samples=st.lists(coordinates_in_grid_strategy(), min_size=10, max_size=20)
)
def test_monotonic_decay(grid, source, sigma, samples):
    field = create_concentration_field(grid, source, sigma)
    
    # Sort samples by distance to source
    samples_with_distance = [
        (s, source.distance_to(s)) for s in samples
    ]
    samples_with_distance.sort(key=lambda x: x[1])
    
    # Check concentrations are monotonically decreasing
    prev_concentration = float('inf')
    for sample, distance in samples_with_distance:
        concentration = field.sample(sample)
        assert concentration <= prev_concentration, \
            f"Concentration increased with distance"
        prev_concentration = concentration
```

### I5: Radial Symmetry (GAUSSIAN-SPECIFIC)

```
∀ Δx, Δy:
  field[source + (Δx, Δy)] ≈ field[source + (-Δx, -Δy)]
  (within numerical tolerance)

Concentration is radially symmetric around source.

⚠️ GAUSSIAN-SPECIFIC: NOT a universal physical law!
Real plumes with wind, anisotropic diffusion violate this.
Only holds for isotropic models (no wind, uniform diffusion).
```

**Test:**

```python
def test_radial_symmetry():
    """Concentration symmetric around source"""
    grid_size = 32
    source = (16, 16)  # Center
    field = create_concentration_field((grid_size, grid_size), source, sigma=2.0)
    
    # Test several symmetric pairs
    deltas = [(1, 0), (0, 1), (2, 3), (3, 2)]
    
    for dx, dy in deltas:
        pos1 = (source[0] + dx, source[1] + dy)
        pos2 = (source[0] - dx, source[1] - dy)
        
        # Both positions in bounds
        if all(0 <= p[i] < grid_size for p in [pos1, pos2] for i in [0, 1]):
            c1 = field.sample(pos1)
            c2 = field.sample(pos2)
            assert np.isclose(c1, c2, rtol=1e-5), \
                f"Asymmetry: {c1} vs {c2} at ±({dx},{dy})"
```

### I6: Shape Consistency

```
field.shape = (grid_size.height, grid_size.width)

Array dimensions match grid dimensions.
```

**Test:**

```python
@given(
    width=st.integers(1, 100),
    height=st.integers(1, 100)
)
def test_shape_consistency(width, height):
    grid = create_grid_size(width, height)
    field = create_concentration_field(grid, (width//2, height//2), sigma=2.0)
    assert field.field.shape == (height, width)
```

### I7: Gaussian Formula (GAUSSIAN-SPECIFIC)

```
∀ (x, y) ∈ Grid:
  field[x, y] = exp(-distance²(x, y, source) / (2σ²))
  (within numerical tolerance)

Field follows Gaussian distribution formula.

⚠️ GAUSSIAN-SPECIFIC: Obviously model-specific!
Other plume models (advection-diffusion, turbulent, etc.) use different equations.
```

**Test:**

```python
def test_gaussian_form():
    """Field values match Gaussian formula"""
    grid = create_grid_size(32, 32)
    source = Coordinates(16, 16)
    sigma = 3.0
    field = create_concentration_field(grid, source, sigma)
    
    # Check random positions
    for _ in range(20):
        x = np.random.randint(0, 32)
        y = np.random.randint(0, 32)
        
        pos = Coordinates(x, y)
        distance = pos.distance_to(source)
        
        # Expected Gaussian value
        expected = np.exp(-distance**2 / (2 * sigma**2))
        
        # Actual field value
        actual = field.sample(pos)
        
        assert np.isclose(actual, expected, rtol=1e-6), \
            f"Gaussian mismatch at ({x},{y}): {actual} vs {expected}"
```

---

## 💻 Type Specification

```python
@dataclass
class ConcentrationField:
    """Immutable concentration field with Gaussian distribution.
    
    Attributes:
      field: 2D ndarray of shape (height, width) with values in [0, 1]
      source_location: Coordinates of plume source
      sigma: Gaussian dispersion parameter
      grid_size: Grid dimensions
    
    Invariants:
      I1: np.all(field >= 0)
      I2: np.all(field <= 1.0)
      I3: field[source] >= field[any_other]
      I4: Monotonic decay with distance
      I5: Radially symmetric (Gaussian)
      I6: field.shape == (grid_size.height, grid_size.width)
      I7: field[x,y] = exp(-d²/(2σ²))
    """
    field: np.ndarray
    source_location: Coordinates
    sigma: float
    grid_size: GridSize
    
    def __post_init__(self):
        """Validate field invariants."""
        # Shape check
        expected_shape = (self.grid_size.height, self.grid_size.width)
        if self.field.shape != expected_shape:
            raise ValidationError(
                f"Field shape {self.field.shape} != expected {expected_shape}"
            )
        
        # Value bounds
        if not np.all(self.field >= 0):
            raise ValidationError("Field contains negative values")
        if not np.all(self.field <= 1.0):
            raise ValidationError("Field contains values > 1.0")
        
        # Source validation
        if not self.grid_size.contains(self.source_location):
            raise ValidationError(
                f"Source {self.source_location} outside grid {self.grid_size}"
            )
```

---

## 🏗️ Constructor Contract

```python
def create_concentration_field(
    grid_size: GridSize | tuple[int, int],
    source_location: Coordinates | tuple[int, int],
    sigma: float,
    validate: bool = True
) -> ConcentrationField:
    """Create Gaussian concentration field.
    
    Preconditions:
      P1: grid_size is valid GridSize or (width, height) with w, h > 0
      P2: source_location within grid bounds
      P3: sigma > 0 (positive dispersion)
      P4: sigma is finite (not NaN, not inf)
    
    Postconditions:
      C1: returns ConcentrationField satisfying I1-I7
      C2: field.shape == (height, width)
      C3: All invariants hold
      C4: field is normalized (max value at source ≈ 1.0)
    
    Properties:
      - Deterministic: same inputs → identical field
      - Efficient: O(width × height) time complexity
      - Memory: O(width × height) space
    
    Raises:
      ValidationError: If preconditions violated
      ValueError: If sigma ≤ 0
    
    Examples:
      # Standard field
      field = create_concentration_field(
          grid_size=(32, 32),
          source_location=(16, 16),
          sigma=2.0
      )
      
      # Check source concentration
      assert field.sample((16, 16)) ≈ 1.0
      
      # Check decay
      assert field.sample((16, 16)) > field.sample((20, 16))
    """
```

### Field Generation Algorithm

```python
def _generate_field(
    grid_size: GridSize,
    source: Coordinates,
    sigma: float
) -> np.ndarray:
    """Generate Gaussian field (internal).
    
    Algorithm:
      1. Create meshgrid of coordinates
      2. Calculate distance from each point to source
      3. Apply Gaussian formula: exp(-d²/(2σ²))
      4. Return normalized field
    
    Time Complexity: O(width × height)
    Space Complexity: O(width × height)
    
    Vectorized for performance (no explicit loops).
    """
    height, width = grid_size.height, grid_size.width
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing='ij'
    )
    
    # Calculate squared distance from source
    dx = x_coords - source.x
    dy = y_coords - source.y
    distance_squared = dx**2 + dy**2
    
    # Apply Gaussian formula
    field = np.exp(-distance_squared / (2 * sigma**2))
    
    # Postconditions
    assert field.shape == (height, width)
    assert np.all(field >= 0)
    assert np.all(field <= 1.0)
    assert field[source.y, source.x] == np.max(field)  # Max at source
    
    return field
```

---

## 📍 Sampling Contract

```python
def sample(self, position: Coordinates | tuple[int, int]) -> float:
    """Sample concentration at position.
    
    Preconditions:
      P1: position is valid Coordinates or (x, y) tuple
      P2: 0 ≤ position.x < grid_size.width
      P3: 0 ≤ position.y < grid_size.height
    
    Postconditions:
      C1: result ∈ [0, 1]
      C2: result = field[position.y, position.x]
      C3: isinstance(result, (float, np.floating))
    
    Properties:
      - Deterministic: same position → same value
      - Pure function: no side effects
      - O(1) time complexity (array lookup)
    
    Raises:
      IndexError: If position out of bounds
      TypeError: If position wrong type
    
    Examples:
      field = create_concentration_field((32, 32), (16, 16), 2.0)
      
      # Sample at source
      c = field.sample((16, 16))
      assert c == 1.0
      
      # Sample nearby
      c = field.sample((17, 16))
      assert 0.5 < c < 1.0  # High but less than source
      
      # Sample far away
      c = field.sample((0, 0))
      assert 0 < c < 0.1  # Very low
    """
    # Convert to Coordinates if tuple
    if isinstance(position, tuple):
        position = Coordinates(position[0], position[1])
    
    # Validate bounds
    if not self.grid_size.contains(position):
        raise IndexError(
            f"Position {position} outside grid {self.grid_size}"
        )
    
    # Sample (note: numpy indexing is [row, col] = [y, x])
    value = self.field[position.y, position.x]
    
    # Postconditions
    assert 0 <= value <= 1.0, f"Invalid concentration {value}"
    
    return float(value)
```

---

## 🧪 Property Tests (MUST IMPLEMENT)

### Test Suite Structure

```python
# tests/properties/test_concentration_field_properties.py

class TestConcentrationFieldInvariants:
    """Test physical invariants."""
    
    @given(
        grid=grid_size_strategy(),
        source=coordinates_in_grid_strategy(),
        sigma=st.floats(0.1, 10.0)
    )
    def test_non_negativity(self, grid, source, sigma):
        """I1: All values non-negative"""
    
    @given(...)
    def test_bounded(self, ...):
        """I2: All values ≤ 1.0"""
    
    @given(...)
    def test_maximum_at_source(self, ...):
        """I3: Source has maximum concentration"""
    
    @given(...)
    def test_monotonic_decay(self, ...):
        """I4: Concentration decreases with distance"""
    
    @given(...)
    def test_shape_consistency(self, ...):
        """I6: Field shape matches grid"""
    
    @given(...)
    def test_gaussian_form(self, ...):
        """I7: Values follow Gaussian formula"""


class TestConcentrationFieldDeterminism:
    """Test deterministic generation."""
    
    @given(
        grid=grid_size_strategy(),
        source=coordinates_in_grid_strategy(),
        sigma=st.floats(0.1, 10.0)
    )
    def test_field_generation_deterministic(self, grid, source, sigma):
        """Same params → identical fields"""
        field1 = create_concentration_field(grid, source, sigma)
        field2 = create_concentration_field(grid, source, sigma)
        
        assert np.allclose(field1.field, field2.field)
    
    @given(...)
    def test_sampling_deterministic(self, ...):
        """Same position → same value"""


class TestConcentrationFieldEdgeCases:
    """Test edge cases and boundaries."""
    
    def test_source_at_corner(self):
        """Source at grid corner"""
    
    def test_source_at_edge(self):
        """Source at grid edge"""
    
    def test_minimum_grid_1x1(self):
        """1×1 grid (single cell)"""
    
    def test_large_sigma(self):
        """σ >> grid size (uniform field)"""
    
    def test_small_sigma(self):
        """σ << 1 (sharp peak)"""
    
    def test_maximum_grid_size(self):
        """Performance at max grid size"""
```

---

## 🎯 Performance Requirements

### Time Complexity

```text
Field Generation: O(width × height)
  - Single pass through grid
  - Vectorized numpy operations
  - No explicit loops

Sampling: O(1)
  - Direct array access
  - No computation
```

### Space Complexity

```text
Field Storage: O(width × height)
  - Single float64 array
  - Memory = width × height × 8 bytes
  
For 32×32 grid: 32 × 32 × 8 = 8,192 bytes ≈ 8 KB
For 100×100 grid: 100 × 100 × 8 = 80,000 bytes ≈ 80 KB
```

### Performance Targets

```
Field Generation:
  - 32×32 grid: < 1 ms
  - 100×100 grid: < 10 ms
  - 1000×1000 grid: < 500 ms

Sampling:
  - Any grid size: < 1 μs (microsecond)
  - O(1) array lookup
```

### Memory Limits

```
Maximum Field Size:
  - 10,000 × 10,000 = 100M cells
  - 100M × 8 bytes = 800 MB
  - Should fit comfortably in RAM
```

---

## ⚠️ Common Implementation Errors

### ❌ Wrong: Index Confusion

```python
# WRONG - x/y vs row/col confusion
def sample_wrong(self, position):
    return self.field[position.x, position.y]  # ❌ Swapped!
    # Correct: field[y, x] (numpy indexing is [row, col])
```

### ❌ Wrong: Not Vectorized

```python
# WRONG - explicit loops (slow)
def generate_field_wrong(grid_size, source, sigma):
    field = np.zeros((grid_size.height, grid_size.width))
    for y in range(grid_size.height):
        for x in range(grid_size.width):
            distance = sqrt((x - source.x)**2 + (y - source.y)**2)
            field[y, x] = exp(-distance**2 / (2 * sigma**2))
    return field  # ❌ 100x slower than vectorized
```

### ❌ Wrong: No Validation

```python
# WRONG - no bounds checking
def sample_wrong(self, position):
    return self.field[position.y, position.x]
    # ❌ Can raise IndexError without helpful message
```

### ❌ Wrong: Mutable Field

```python
# WRONG - field can be modified externally
@dataclass
class ConcentrationField:
    field: np.ndarray  # ❌ Mutable!
    
    # External code can do:
    # field.field[0, 0] = 999.0  # Breaks invariants!
```

### ✅ Correct: Immutable Field

```python
@dataclass(frozen=True)
class ConcentrationField:
    field: np.ndarray
    
    def __post_init__(self):
        # Make field read-only
        self.field.flags.writeable = False
```

---

## 🔗 Integration with Other Components

### Used By

- **Environment:** Generates field during reset
- **Observation:** Samples concentration at agent position
- **Rendering:** Visualizes field as heatmap

### Dependencies

- **GridSize:** Defines field dimensions
- **Coordinates:** Specifies source and sample positions
- **numpy:** Array operations

### Performance Impact

- **Episode Reset:** Field generation is one-time cost
- **Step Execution:** Sampling is O(1), negligible
- **Memory:** Field stored throughout episode

---

## 📊 Test Coverage Requirements

**Minimum Test Suite:**

```python
# Property tests (50+ examples each)
✓ test_non_negativity
✓ test_bounded
✓ test_maximum_at_source
✓ test_monotonic_decay
✓ test_radial_symmetry
✓ test_shape_consistency
✓ test_gaussian_form
✓ test_deterministic_generation
✓ test_deterministic_sampling

# Edge cases
✓ test_source_at_corner
✓ test_source_at_edge
✓ test_minimum_grid
✓ test_large_sigma
✓ test_small_sigma
✓ test_various_grid_shapes

# Performance
✓ test_generation_performance
✓ test_sampling_performance
✓ test_memory_usage

# Error conditions
✓ test_negative_sigma_raises
✓ test_zero_sigma_raises
✓ test_source_outside_grid_raises
✓ test_sample_outside_bounds_raises
```

**Coverage Goal:** 95%+ of field generation and sampling code

---

## 🎯 Verification Checklist

Implementation MUST satisfy:

- [ ] All values in [0, 1] (I1, I2)
- [ ] Maximum at source (I3)
- [ ] Monotonic decay with distance (I4)
- [ ] Radially symmetric (I5)
- [ ] Correct shape (I6)
- [ ] Gaussian formula (I7)
- [ ] Deterministic generation
- [ ] O(1) sampling
- [ ] Proper bounds checking
- [ ] Immutable field
- [ ] Property tests pass (100+ examples)
- [ ] Performance targets met
- [ ] Memory limits respected

---

**Last Updated:** 2025-09-30  
**Next Review:** After guard tests implemented
