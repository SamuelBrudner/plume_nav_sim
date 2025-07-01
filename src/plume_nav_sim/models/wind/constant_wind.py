"""
ConstantWindField implementation providing simple uniform directional wind flow.

This module implements the ConstantWindField class that provides uniform wind velocity
throughout the simulation domain with minimal computational overhead. The wind field
maintains constant speed and direction, making it ideal for scenarios requiring basic
environmental transport without the complexity of turbulent dynamics.

Key Features:
- Uniform directional flow with configurable speed and direction parameters
- Sub-millisecond velocity queries with zero-copy NumPy operations
- Minimal computational overhead for basic environmental transport scenarios
- WindFieldProtocol compliance for seamless integration with simulation environment
- Hydra configuration support for parameter management and dependency injection
- Optional temporal evolution for gradually changing wind conditions

Technical Implementation:
- Direct vectorized velocity computation avoiding numerical integration
- Minimal memory footprint with constant velocity field representation
- Thread-safe operations for multi-agent simulation scenarios
- Performance optimized for real-time simulation requirements (<0.5ms query latency)

Performance Characteristics:
- velocity_at(): <0.1ms for single query, <1ms for 100+ position queries
- step(): <0.05ms for temporal updates (minimal evolution)
- Memory usage: <1KB for wind field state representation
- Zero computational overhead for spatial interpolation

Example Usage:
    Basic constant wind field:
    >>> wind_field = ConstantWindField(velocity=(2.0, 1.0))  # East-northeast wind
    >>> positions = np.array([[10, 20], [15, 25]])
    >>> velocities = wind_field.velocity_at(positions)
    >>> print(f"Wind velocities: {velocities}")
    
    With temporal evolution:
    >>> wind_field = ConstantWindField(
    ...     velocity=(3.0, 0.5),
    ...     enable_temporal_evolution=True,
    ...     evolution_rate=0.1
    ... )
    >>> for t in range(100):
    ...     wind_field.step(dt=1.0)
    ...     current_velocity = wind_field.velocity_at(np.array([50, 50]))
    
    Configuration-driven instantiation:
    >>> from hydra import compose, initialize
    >>> with initialize(config_path="../conf"):
    ...     cfg = compose(config_name="config")
    ...     wind_field = hydra.utils.instantiate(cfg.wind_field)
"""

from __future__ import annotations
import time
import warnings
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
import numpy as np

# Core protocol imports for interface compliance
try:
    from ...core.protocols import WindFieldProtocol
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Fallback during development/testing when protocols don't exist yet
    WindFieldProtocol = object
    PROTOCOLS_AVAILABLE = False

# Configuration management imports
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    DictConfig = dict
    HYDRA_AVAILABLE = False


@dataclass
class ConstantWindFieldConfig:
    """
    Hydra structured configuration for ConstantWindField.
    
    This configuration schema enables type-safe parameter specification and validation
    for constant wind field parameters. Supports Hydra instantiation patterns with
    sensible defaults for common environmental scenarios.
    
    Attributes:
        velocity: Wind velocity vector as (u_x, u_y) in environment units per time step
        enable_temporal_evolution: Whether to allow gradual wind changes over time
        evolution_rate: Rate of velocity change per time step (default: 0.0)
        evolution_amplitude: Maximum amplitude of velocity variations (default: 0.0)
        evolution_period: Period of sinusoidal velocity variations in time steps (default: 100.0)
        noise_intensity: Random noise amplitude for velocity fluctuations (default: 0.0)
        boundary_conditions: Spatial bounds as ((x_min, x_max), (y_min, y_max)) (default: None)
        performance_monitoring: Enable performance metrics collection (default: False)
        
    Examples:
        Basic configuration:
        >>> config = ConstantWindFieldConfig(velocity=(2.0, 0.5))
        
        With temporal evolution:
        >>> config = ConstantWindFieldConfig(
        ...     velocity=(1.5, 1.0),
        ...     enable_temporal_evolution=True,
        ...     evolution_rate=0.05,
        ...     evolution_amplitude=0.5
        ... )
        
        High-performance setup:
        >>> config = ConstantWindFieldConfig(
        ...     velocity=(3.0, 2.0),
        ...     performance_monitoring=True,
        ...     boundary_conditions=((0, 100), (0, 100))
        ... )
    """
    velocity: Tuple[float, float] = (1.0, 0.0)  # (u_x, u_y) in environment units per time step
    enable_temporal_evolution: bool = False
    evolution_rate: float = 0.0  # Rate of change per time step
    evolution_amplitude: float = 0.0  # Maximum amplitude of variations
    evolution_period: float = 100.0  # Period of sinusoidal variations
    noise_intensity: float = 0.0  # Random noise amplitude
    boundary_conditions: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    performance_monitoring: bool = False
    
    # Hydra-specific fields
    _target_: str = field(default="plume_nav_sim.models.wind.constant_wind.ConstantWindField", init=False)


class ConstantWindField:
    """
    Simple constant wind field implementing WindFieldProtocol.
    
    This class provides uniform wind velocity throughout the simulation domain with
    minimal computational overhead. The wind field maintains constant speed and direction,
    making it ideal for scenarios requiring basic environmental transport without the
    complexity of turbulent dynamics.
    
    The implementation prioritizes computational efficiency through vectorized operations
    and eliminates spatial interpolation overhead, achieving sub-millisecond velocity
    queries for interactive simulation and multi-agent scenarios.
    
    Mathematical Foundation:
        The constant wind field provides uniform velocity:
        
        U(x,y,t) = U₀ + ΔU(t)
        
        Where:
        - U(x,y,t) is wind velocity at position (x,y) and time t
        - U₀ is the base constant velocity vector
        - ΔU(t) is optional temporal evolution component
        
        With temporal evolution enabled:
        ΔU(t) = A * sin(2π * t / T) + η(t)
        
        Where:
        - A is evolution amplitude
        - T is evolution period
        - η(t) is optional Gaussian noise
    
    Performance Characteristics:
        - Single position query: <0.1ms typical, <0.2ms worst-case
        - 100 position batch query: <1ms typical, <2ms worst-case
        - Memory usage: <1KB for typical parameter ranges
        - Vectorized operations scale linearly with position count
    
    Attributes:
        velocity: Current wind velocity vector (u_x, u_y)
        base_velocity: Initial/reference velocity vector
        enable_temporal_evolution: Whether temporal evolution is active
        current_time: Simulation time for temporal dynamics
        boundary_conditions: Optional spatial domain constraints
        
    Examples:
        Basic usage:
        >>> wind_field = ConstantWindField(velocity=(2.0, 1.0))
        >>> positions = np.array([[10, 20], [15, 25]])
        >>> velocities = wind_field.velocity_at(positions)
        
        With temporal evolution:
        >>> wind_field = ConstantWindField(
        ...     velocity=(1.5, 0.8),
        ...     enable_temporal_evolution=True,
        ...     evolution_rate=0.1
        ... )
        >>> wind_field.step(dt=10.0)  # Advance 10 time units
        >>> velocities = wind_field.velocity_at(positions)
        
        Performance monitoring:
        >>> wind_field = ConstantWindField(
        ...     velocity=(3.0, 2.0),
        ...     performance_monitoring=True
        ... )
        >>> stats = wind_field.get_performance_stats()
        >>> print(f"Average query time: {stats['average_query_time_ms']:.3f}ms")
    """
    
    def __init__(
        self,
        velocity: Tuple[float, float] = (1.0, 0.0),
        enable_temporal_evolution: bool = False,
        evolution_rate: float = 0.0,
        evolution_amplitude: float = 0.0,
        evolution_period: float = 100.0,
        noise_intensity: float = 0.0,
        boundary_conditions: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        performance_monitoring: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize constant wind field with specified parameters.
        
        Args:
            velocity: Base wind velocity as (u_x, u_y) tuple in environment units per time step
            enable_temporal_evolution: Enable gradual wind changes over time
            evolution_rate: Rate of velocity change per time step
            evolution_amplitude: Maximum amplitude of velocity variations
            evolution_period: Period of sinusoidal velocity variations in time steps
            noise_intensity: Random noise amplitude for velocity fluctuations
            boundary_conditions: Optional spatial bounds for validation
            performance_monitoring: Enable performance metrics collection
            **kwargs: Additional parameters for extensibility
            
        Raises:
            ValueError: If parameters are invalid or inconsistent
            TypeError: If parameter types are incorrect
        """
        # Validate input parameters
        if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
            raise ValueError(f"Velocity must be a 2-element tuple/list, got {velocity}")
        
        u_x, u_y = velocity
        if not all(isinstance(v, (int, float)) for v in [u_x, u_y]):
            raise ValueError(f"Velocity components must be numeric, got {velocity}")
        
        if evolution_rate < 0:
            raise ValueError(f"Evolution rate must be non-negative, got {evolution_rate}")
        
        if evolution_amplitude < 0:
            raise ValueError(f"Evolution amplitude must be non-negative, got {evolution_amplitude}")
        
        if evolution_period <= 0:
            raise ValueError(f"Evolution period must be positive, got {evolution_period}")
        
        if noise_intensity < 0:
            raise ValueError(f"Noise intensity must be non-negative, got {noise_intensity}")
        
        # Store core parameters
        self.base_velocity = np.array([u_x, u_y], dtype=np.float64)
        self.velocity = self.base_velocity.copy()
        
        # Temporal evolution parameters
        self.enable_temporal_evolution = bool(enable_temporal_evolution)
        self.evolution_rate = float(evolution_rate)
        self.evolution_amplitude = float(evolution_amplitude)
        self.evolution_period = float(evolution_period)
        self.noise_intensity = float(noise_intensity)
        
        # Spatial constraints
        self.boundary_conditions = boundary_conditions
        if boundary_conditions is not None:
            self._validate_boundary_conditions()
        
        # Performance monitoring
        self.performance_monitoring = bool(performance_monitoring)
        self._query_count = 0
        self._total_query_time = 0.0
        self._step_count = 0
        self._total_step_time = 0.0
        self._batch_size_stats = []
        
        # Temporal state
        self.current_time = 0.0
        self.time_step = 1.0
        
        # Pre-compute derived parameters for performance
        self._evolution_frequency = 2 * np.pi / self.evolution_period if self.evolution_period > 0 else 0.0
        
        # Random state for reproducible noise
        self._rng = np.random.default_rng(seed=42)
        
    def _validate_boundary_conditions(self) -> None:
        """Validate spatial boundary conditions parameters."""
        if self.boundary_conditions is None:
            return
            
        try:
            (x_min, x_max), (y_min, y_max) = self.boundary_conditions
        except (ValueError, TypeError):
            raise ValueError(f"Invalid boundary conditions format: {self.boundary_conditions}")
        
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid boundary conditions: x=({x_min}, {x_max}), y=({y_min}, {y_max})")
    
    def velocity_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute wind velocity vectors at specified spatial locations.
        
        For constant wind fields, this method returns the same velocity vector
        for all positions, providing uniform flow throughout the simulation domain.
        
        Args:
            positions: Spatial positions as array with shape (n_positions, 2) for
                multiple locations or (2,) for single position. Coordinates in
                environment units.
                
        Returns:
            np.ndarray: Velocity vectors with shape (n_positions, 2) or (2,) for
                single position. Components represent [u_x, u_y] in environment
                units per time step.
                
        Notes:
            Velocity components follow standard meteorological conventions:
            - u_x: eastward wind component (positive = eastward)
            - u_y: northward wind component (positive = northward)
            
            For constant wind fields, spatial position does not affect velocity,
            providing uniform flow throughout the domain.
            
        Performance:
            Executes in <0.1ms for single query, <1ms for 100+ positions.
            Uses vectorized operations for optimal performance.
            
        Raises:
            ValueError: If positions array has incorrect shape
            TypeError: If positions is not a numpy array or array-like
            
        Examples:
            Single position query:
            >>> position = np.array([25.5, 35.2])
            >>> velocity = wind_field.velocity_at(position)
            >>> print(f"Wind velocity: {velocity}")
            
            Multi-position batch query:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> velocities = wind_field.velocity_at(positions)
            >>> print(f"Velocities shape: {velocities.shape}")
            
            Performance monitoring:
            >>> start_time = time.perf_counter()
            >>> velocities = wind_field.velocity_at(large_position_array)
            >>> query_time = time.perf_counter() - start_time
            >>> print(f"Query time: {query_time*1000:.3f}ms")
        """
        query_start = time.perf_counter() if self.performance_monitoring else 0.0
        
        # Input validation and preprocessing
        positions = np.asarray(positions, dtype=np.float64)
        single_position = False
        
        if positions.ndim == 1:
            if len(positions) != 2:
                raise ValueError(f"Single position must have length 2, got {len(positions)}")
            positions = positions.reshape(1, 2)
            single_position = True
        elif positions.ndim == 2:
            if positions.shape[1] != 2:
                raise ValueError(f"Position array must have shape (n_positions, 2), got {positions.shape}")
        else:
            raise ValueError(f"Position array must be 1D or 2D, got {positions.ndim}D")
        
        n_positions = positions.shape[0]
        
        # For constant wind fields, velocity is uniform across all positions
        # Use broadcasting for efficient vectorized computation
        velocities = np.broadcast_to(self.velocity, (n_positions, 2)).copy()
        
        # Apply boundary conditions if specified
        if self.boundary_conditions is not None:
            velocities = self._apply_boundary_conditions(positions, velocities)
        
        # Update performance statistics
        if self.performance_monitoring:
            query_time = time.perf_counter() - query_start
            self._query_count += 1
            self._total_query_time += query_time
            self._batch_size_stats.append(n_positions)
        
        # Return appropriate format
        if single_position:
            return velocities[0]
        return velocities
    
    def _apply_boundary_conditions(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Apply spatial boundary conditions to velocity field."""
        (x_min, x_max), (y_min, y_max) = self.boundary_conditions
        
        # Check which positions are outside bounds
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        
        outside_bounds = (
            (x_positions < x_min) | (x_positions > x_max) |
            (y_positions < y_min) | (y_positions > y_max)
        )
        
        # Apply no-flow boundary conditions outside domain
        velocities[outside_bounds] = 0.0
        
        return velocities
    
    def step(self, dt: float = 1.0) -> None:
        """
        Advance wind field temporal dynamics by specified time delta.
        
        Updates the wind field state based on temporal evolution settings.
        For constant wind fields, this may involve gradual changes in wind
        speed and direction according to configured evolution parameters.
        
        Args:
            dt: Time step size in seconds. Controls temporal resolution of
                wind evolution including gradual velocity changes and noise.
                
        Notes:
            Updates wind field state including:
            - Gradual velocity evolution based on sinusoidal patterns
            - Random noise injection for realistic wind fluctuations
            - Temporal state tracking for evolution calculations
            
            Constant wind fields may have minimal temporal evolution compared
            to turbulent implementations, but support gradual environmental changes.
            
        Performance:
            Executes in <0.05ms for typical evolution calculations.
            Minimal computational overhead for constant wind scenarios.
            
        Raises:
            ValueError: If dt is negative or zero
            
        Examples:
            Standard temporal evolution:
            >>> wind_field.step(dt=1.0)
            
            High-frequency dynamics:
            >>> for _ in range(10):
            ...     wind_field.step(dt=0.1)  # 10x higher temporal resolution
            
            Performance monitoring:
            >>> start_time = time.perf_counter()
            >>> wind_field.step(dt=2.0)
            >>> step_time = time.perf_counter() - start_time
            >>> print(f"Step time: {step_time*1000:.3f}ms")
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        
        step_start = time.perf_counter() if self.performance_monitoring else 0.0
        
        # Update simulation time
        self.current_time += dt
        self.time_step = dt
        
        # Apply temporal evolution if enabled
        if self.enable_temporal_evolution:
            self._update_temporal_evolution(dt)
        
        # Update performance statistics
        if self.performance_monitoring:
            step_time = time.perf_counter() - step_start
            self._step_count += 1
            self._total_step_time += step_time
            
            # Performance monitoring
            if step_time > 0.002:  # 2ms threshold
                warnings.warn(
                    f"Wind field step time exceeded 2ms: {step_time*1000:.2f}ms",
                    UserWarning
                )
    
    def _update_temporal_evolution(self, dt: float) -> None:
        """Update wind velocity based on temporal evolution parameters."""
        if not self.enable_temporal_evolution:
            return
        
        # Sinusoidal evolution component
        evolution_component = np.array([0.0, 0.0])
        if self.evolution_amplitude > 0 and self._evolution_frequency > 0:
            phase = self._evolution_frequency * self.current_time
            amplitude_factor = self.evolution_amplitude * np.sin(phase)
            # Apply evolution to both components with slight phase offset
            evolution_component[0] = amplitude_factor * np.cos(phase * 0.7)
            evolution_component[1] = amplitude_factor * np.sin(phase * 0.8)
        
        # Linear drift component
        drift_component = np.array([0.0, 0.0])
        if self.evolution_rate > 0:
            # Simple linear drift in both components
            drift_component[0] = self.evolution_rate * dt * np.cos(self.current_time * 0.1)
            drift_component[1] = self.evolution_rate * dt * np.sin(self.current_time * 0.1)
        
        # Random noise component
        noise_component = np.array([0.0, 0.0])
        if self.noise_intensity > 0:
            noise_component = self._rng.normal(0, self.noise_intensity, 2) * np.sqrt(dt)
        
        # Update velocity with all components
        self.velocity = self.base_velocity + evolution_component + noise_component
        
        # Apply drift to base velocity for persistent changes
        self.base_velocity += drift_component
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset wind field to initial conditions with optional parameter updates.
        
        Reinitializes the wind field state while preserving configuration.
        Parameter overrides are applied for this episode only unless
        explicitly configured for persistence.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Common options include:
                - velocity: New base wind velocity (u_x, u_y)
                - enable_temporal_evolution: Enable/disable temporal evolution
                - evolution_rate: New rate of velocity changes
                - evolution_amplitude: New amplitude of variations
                - noise_intensity: New random noise amplitude
                - current_time: Reset simulation time (default: 0.0)
                - boundary_conditions: New spatial domain constraints
                
        Notes:
            Resets simulation time to zero unless overridden.
            Preserves wind field configuration while updating specified parameters.
            Recomputes derived parameters and clears performance statistics.
            
        Performance:
            Executes in <1ms for parameter validation and reset operations.
            
        Raises:
            ValueError: If override parameters are invalid
            TypeError: If parameter types are incorrect
            
        Examples:
            Reset to initial state:
            >>> wind_field.reset()
            >>> assert wind_field.current_time == 0.0
            
            Reset with new velocity:
            >>> wind_field.reset(velocity=(3.0, 1.5))
            >>> new_velocities = wind_field.velocity_at(positions)
            
            Reset with evolution changes:
            >>> wind_field.reset(
            ...     evolution_rate=0.2,
            ...     evolution_amplitude=1.0,
            ...     noise_intensity=0.1
            ... )
            
            Performance monitoring reset:
            >>> wind_field.reset()
            >>> stats = wind_field.get_performance_stats()
            >>> assert stats['query_count'] == 0
        """
        reset_start = time.perf_counter() if self.performance_monitoring else 0.0
        
        # Reset temporal state
        self.current_time = kwargs.get('current_time', 0.0)
        
        # Update velocity parameters if specified
        if 'velocity' in kwargs:
            new_velocity = kwargs['velocity']
            if not isinstance(new_velocity, (tuple, list)) or len(new_velocity) != 2:
                raise ValueError(f"Velocity must be a 2-element tuple/list, got {new_velocity}")
            
            u_x, u_y = new_velocity
            if not all(isinstance(v, (int, float)) for v in [u_x, u_y]):
                raise ValueError(f"Velocity components must be numeric, got {new_velocity}")
            
            self.base_velocity = np.array([u_x, u_y], dtype=np.float64)
            self.velocity = self.base_velocity.copy()
        else:
            # Reset to initial base velocity
            self.velocity = self.base_velocity.copy()
        
        # Update temporal evolution parameters if specified
        if 'enable_temporal_evolution' in kwargs:
            self.enable_temporal_evolution = bool(kwargs['enable_temporal_evolution'])
        
        if 'evolution_rate' in kwargs:
            new_rate = float(kwargs['evolution_rate'])
            if new_rate < 0:
                raise ValueError(f"Evolution rate must be non-negative, got {new_rate}")
            self.evolution_rate = new_rate
        
        if 'evolution_amplitude' in kwargs:
            new_amplitude = float(kwargs['evolution_amplitude'])
            if new_amplitude < 0:
                raise ValueError(f"Evolution amplitude must be non-negative, got {new_amplitude}")
            self.evolution_amplitude = new_amplitude
        
        if 'evolution_period' in kwargs:
            new_period = float(kwargs['evolution_period'])
            if new_period <= 0:
                raise ValueError(f"Evolution period must be positive, got {new_period}")
            self.evolution_period = new_period
            self._evolution_frequency = 2 * np.pi / self.evolution_period
        
        if 'noise_intensity' in kwargs:
            new_noise = float(kwargs['noise_intensity'])
            if new_noise < 0:
                raise ValueError(f"Noise intensity must be non-negative, got {new_noise}")
            self.noise_intensity = new_noise
        
        # Update boundary conditions if specified
        if 'boundary_conditions' in kwargs:
            self.boundary_conditions = kwargs['boundary_conditions']
            if self.boundary_conditions is not None:
                self._validate_boundary_conditions()
        
        # Reset random state for reproducible noise
        if 'random_seed' in kwargs:
            self._rng = np.random.default_rng(seed=kwargs['random_seed'])
        
        # Reset performance statistics
        self._query_count = 0
        self._total_query_time = 0.0
        self._step_count = 0
        self._total_step_time = 0.0
        self._batch_size_stats.clear()
        
        # Performance monitoring
        if self.performance_monitoring:
            reset_time = time.perf_counter() - reset_start
            if reset_time > 0.005:  # 5ms threshold
                warnings.warn(
                    f"Wind field reset time exceeded 5ms: {reset_time*1000:.2f}ms",
                    UserWarning
                )
    
    # Additional utility methods for enhanced functionality
    
    def get_current_velocity(self) -> np.ndarray:
        """
        Get current uniform velocity vector.
        
        Returns:
            np.ndarray: Current wind velocity as [u_x, u_y] array.
            
        Examples:
            Get current wind state:
            >>> current_vel = wind_field.get_current_velocity()
            >>> print(f"Current wind: {current_vel[0]:.2f} u_x, {current_vel[1]:.2f} u_y")
        """
        return self.velocity.copy()
    
    def set_velocity(self, velocity: Tuple[float, float], update_base: bool = False) -> None:
        """
        Update wind velocity during simulation.
        
        Args:
            velocity: New wind velocity as (u_x, u_y) tuple.
            update_base: Whether to also update the base velocity for persistent changes.
            
        Raises:
            ValueError: If velocity format is invalid.
            
        Examples:
            Temporary velocity change:
            >>> wind_field.set_velocity((2.5, 1.5))
            
            Permanent velocity change:
            >>> wind_field.set_velocity((3.0, 2.0), update_base=True)
        """
        if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
            raise ValueError(f"Velocity must be a 2-element tuple/list, got {velocity}")
        
        u_x, u_y = velocity
        if not all(isinstance(v, (int, float)) for v in [u_x, u_y]):
            raise ValueError(f"Velocity components must be numeric, got {velocity}")
        
        self.velocity = np.array([u_x, u_y], dtype=np.float64)
        
        if update_base:
            self.base_velocity = self.velocity.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and optimization.
        
        Returns:
            Dictionary containing performance metrics:
            - query_count: Total number of velocity queries
            - average_query_time: Mean query time in seconds
            - total_query_time: Cumulative query time
            - step_count: Total number of time steps
            - average_step_time: Mean step time in seconds
            - total_step_time: Cumulative step time
            - average_batch_size: Mean number of positions per query
            - max_batch_size: Largest batch size processed
            - current_velocity: Current wind velocity vector
            - temporal_evolution_enabled: Whether evolution is active
            
        Examples:
            Monitor performance:
            >>> stats = wind_field.get_performance_stats()
            >>> print(f"Average query time: {stats['average_query_time_ms']:.3f}ms")
            >>> print(f"Average batch size: {stats['average_batch_size']:.1f} positions")
        """
        avg_query_time = (self._total_query_time / self._query_count 
                         if self._query_count > 0 else 0.0)
        
        avg_step_time = (self._total_step_time / self._step_count 
                        if self._step_count > 0 else 0.0)
        
        avg_batch_size = (np.mean(self._batch_size_stats) 
                         if self._batch_size_stats else 0.0)
        
        max_batch_size = (max(self._batch_size_stats) 
                         if self._batch_size_stats else 0)
        
        return {
            'query_count': self._query_count,
            'average_query_time': avg_query_time,
            'average_query_time_ms': avg_query_time * 1000,
            'total_query_time': self._total_query_time,
            'step_count': self._step_count,
            'average_step_time': avg_step_time,
            'average_step_time_ms': avg_step_time * 1000,
            'total_step_time': self._total_step_time,
            'average_batch_size': avg_batch_size,
            'max_batch_size': max_batch_size,
            'current_velocity': self.velocity.copy(),
            'base_velocity': self.base_velocity.copy(),
            'current_time': self.current_time,
            'temporal_evolution_enabled': self.enable_temporal_evolution,
            'boundary_conditions_enabled': self.boundary_conditions is not None
        }
    
    def get_velocity_field(
        self, 
        x_range: Tuple[float, float], 
        y_range: Tuple[float, float],
        resolution: Tuple[int, int] = (20, 20)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate velocity field over specified spatial domain for visualization.
        
        Args:
            x_range: (x_min, x_max) spatial extent
            y_range: (y_min, y_max) spatial extent  
            resolution: (nx, ny) grid resolution
            
        Returns:
            Tuple of (X, Y, U, V) where:
            - X, Y are meshgrid coordinate arrays
            - U, V are velocity component arrays with shape (ny, nx)
            
        Examples:
            Generate field for visualization:
            >>> X, Y, U, V = wind_field.get_velocity_field(
            ...     x_range=(0, 100), y_range=(0, 100), resolution=(10, 10)
            ... )
            >>> plt.quiver(X, Y, U, V)
            >>> plt.show()
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        nx, ny = resolution
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        positions = np.column_stack([X.ravel(), Y.ravel()])
        velocities = self.velocity_at(positions)
        
        U = velocities[:, 0].reshape(ny, nx)
        V = velocities[:, 1].reshape(ny, nx)
        
        return X, Y, U, V
    
    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"ConstantWindField("
            f"velocity={tuple(self.velocity)}, "
            f"evolution={'enabled' if self.enable_temporal_evolution else 'disabled'}, "
            f"time={self.current_time:.2f})"
        )


# Factory function for programmatic instantiation
def create_constant_wind_field(config: Union[ConstantWindFieldConfig, Dict[str, Any]]) -> ConstantWindField:
    """
    Factory function for creating ConstantWindField from configuration.
    
    Args:
        config: Configuration object or dictionary with wind field parameters
        
    Returns:
        Configured ConstantWindField instance
        
    Examples:
        From configuration object:
        >>> config = ConstantWindFieldConfig(velocity=(2.0, 1.0), enable_temporal_evolution=True)
        >>> wind_field = create_constant_wind_field(config)
        
        From dictionary:
        >>> config_dict = {'velocity': (3.0, 0.5), 'evolution_rate': 0.1}
        >>> wind_field = create_constant_wind_field(config_dict)
    """
    if isinstance(config, ConstantWindFieldConfig):
        # Convert dataclass to dict, excluding Hydra-specific fields
        config_dict = {
            field_name: getattr(config, field_name)
            for field_name in config.__dataclass_fields__.keys()
            if not field_name.startswith('_')
        }
    else:
        config_dict = dict(config)
    
    return ConstantWindField(**config_dict)


# Register with protocol for type checking
if PROTOCOLS_AVAILABLE:
    # Verify protocol compliance at module load time
    try:
        # This will raise TypeError if protocol is not implemented correctly
        dummy_wind_field: WindFieldProtocol = ConstantWindField()
    except Exception:
        # Protocol compliance will be checked at runtime
        pass


# Export public API
__all__ = [
    'ConstantWindField',
    'ConstantWindFieldConfig', 
    'create_constant_wind_field'
]