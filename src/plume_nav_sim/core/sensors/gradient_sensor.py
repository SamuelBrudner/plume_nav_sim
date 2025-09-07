"""
GradientSensor implementation providing spatial gradient computation for directional navigation cues.

This module implements a high-performance gradient sensor that computes spatial derivatives of
concentration fields using configurable finite difference algorithms. The sensor provides
directional navigation cues through gradient magnitude and direction calculations with
comprehensive metadata reporting, optimized for sub-10ms step execution targets.

Key Features:
- Configurable finite difference algorithms (central, forward, backward) with adaptive order selection
- Multi-point sampling strategies with configurable spatial resolution for gradient quality control
- Optimized computational kernels maintaining performance targets despite complexity
- Vectorized operations for efficient multi-agent scenarios with linear scaling
- Comprehensive error handling and edge case management
- Integration with BaseSensor infrastructure for monitoring and configuration

Performance Requirements:
- Gradient computation: <0.2ms per agent due to multi-point sampling requirements
- Multi-agent scaling: Linear performance with agent count
- Memory efficiency: <1KB per agent for gradient computation state
- Numerical stability: Robust gradient estimation in noisy concentration fields

Examples:
    Basic gradient sensor with default configuration:
        >>> sensor = GradientSensor()
        >>> positions = np.array([[15, 25], [20, 30]])
        >>> gradients = sensor.compute_gradient(plume_state, positions)

    Custom configuration for high-accuracy gradients:
        >>> sensor = GradientSensor(
        ...     method='central',
        ...     order=4,
        ...     spatial_resolution=(0.1, 0.1),
        ...     adaptive_step_size=True
        ... )
        >>> gradients = sensor.compute_gradient(plume_state, positions)

    Gradient magnitude and direction analysis:
        >>> gradient_data = sensor.compute_gradient_with_metadata(plume_state, positions)
        >>> magnitudes = gradient_data['magnitude']
        >>> directions = gradient_data['direction']
        >>> confidence = gradient_data['confidence']

Notes:
    The GradientSensor uses finite difference methods to approximate spatial derivatives
    of concentration fields. Multi-point sampling and adaptive step sizing ensure accurate
    gradient estimation even in regions with sharp concentration variations or noise.

    Performance is optimized through vectorized NumPy operations and efficient memory
    management. The sensor maintains sub-10ms execution targets even for complex
    gradient calculations in multi-agent scenarios.
"""
from loguru import logger
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

# Core protocol imports
from plume_nav_sim.protocols.sensor import SensorProtocol
from .base_sensor import BaseSensor

try:  # Fail fast if Loguru is missing
    from loguru import logger
except ImportError as exc:  # pragma: no cover - executed only when Loguru is absent
    logger.error(
        "loguru is required for GradientSensor but is not installed. "
        "Install loguru to enable advanced logger."
    )
    raise

try:  # Fail fast if Hydra is missing
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover - executed only when Hydra is absent
    logger.error(
        "Hydra's omegaconf is required for GradientSensor but is not installed. "
        "Install hydra-core to enable configuration management."
    )
    raise


class FiniteDifferenceMethod(Enum):
    """Enumeration of supported finite difference methods for gradient computation."""

    CENTRAL = "central"
    FORWARD = "forward"
    BACKWARD = "backward"
    ADAPTIVE = "adaptive"


@dataclass
class GradientSensorConfig:
    """
    Configuration dataclass for GradientSensor parameters with validation.

    This dataclass provides type-safe configuration for gradient computation
    parameters, enabling Hydra integration and runtime validation.

    Attributes:
        method: Finite difference method selection
        order: Derivative approximation order (2, 4, 6, 8)
        spatial_resolution: Step size for finite differences (dx, dy)
        adaptive_step_size: Enable adaptive step sizing for accuracy
        min_step_size: Minimum step size for adaptive methods
        max_step_size: Maximum step size for adaptive methods
        noise_threshold: Concentration threshold for noise detection
        edge_handling: Method for handling domain boundaries
        enable_caching: Enable gradient computation caching
        enable_metadata: Enable comprehensive metadata collection

    Examples:
        High-accuracy configuration:
            >>> config = GradientSensorConfig(
            ...     method=FiniteDifferenceMethod.CENTRAL,
            ...     order=4,
            ...     spatial_resolution=(0.1, 0.1),
            ...     adaptive_step_size=True
            ... )

        Performance-optimized configuration:
            >>> config = GradientSensorConfig(
            ...     method=FiniteDifferenceMethod.FORWARD,
            ...     order=2,
            ...     spatial_resolution=(0.5, 0.5),
            ...     adaptive_step_size=False,
            ...     enable_caching=True
            ... )
    """

    method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL
    order: int = 2
    spatial_resolution: Tuple[float, float] = (0.5, 0.5)
    adaptive_step_size: bool = True
    min_step_size: float = 0.1
    max_step_size: float = 2.0
    noise_threshold: float = 1e-6
    edge_handling: str = "zero_padding"
    enable_caching: bool = True
    enable_metadata: bool = True
    max_cache_size: int = 1000
    cache_ttl_seconds: float = 60.0

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Convert string method to enum if needed
        if isinstance(self.method, str):
            self.method = FiniteDifferenceMethod(self.method)
        elif not isinstance(self.method, FiniteDifferenceMethod):
            raise ValueError(
                f"method must be FiniteDifferenceMethod or string, got {type(self.method)}"
            )

        if self.order not in [2, 4, 6, 8]:
            raise ValueError(f"Order must be in [2, 4, 6, 8], got {self.order}")

        if self.order > 2 and self.method == FiniteDifferenceMethod.ADAPTIVE:
            warnings.warn(
                "High-order derivatives with adaptive method may be unstable. "
                "Consider using central differences for orders > 2.",
                UserWarning,
            )

        if self.min_step_size >= self.max_step_size:
            raise ValueError(
                f"min_step_size ({self.min_step_size}) must be < max_step_size ({self.max_step_size})"
            )

        if any(r <= 0 for r in self.spatial_resolution):
            raise ValueError(
                f"spatial_resolution components must be positive, got {self.spatial_resolution}"
            )


@dataclass
class GradientResult:
    """
    Comprehensive gradient computation result with metadata.

    This dataclass encapsulates gradient computation results with comprehensive
    metadata for navigation support, debugging, and performance analysis.

    Attributes:
        gradient: Spatial gradient vectors [∂c/∂x, ∂c/∂y]
        magnitude: Gradient magnitude values
        direction: Gradient direction in degrees (0° = east, 90° = north)
        confidence: Computation confidence based on numerical stability
        metadata: Additional computation metadata and diagnostics

    Examples:
        Accessing gradient components:
            >>> result = sensor.compute_gradient_with_metadata(plume_state, positions)
            >>> dx_values = result.gradient[:, 0]  # ∂c/∂x components
            >>> dy_values = result.gradient[:, 1]  # ∂c/∂y components

        Navigation decision support:
            >>> uphill_direction = result.direction  # Direction of steepest ascent
            >>> gradient_strength = result.magnitude  # Gradient strength indicator
            >>> reliable_gradients = result.confidence > 0.8  # High-confidence readings
    """

    gradient: np.ndarray
    magnitude: np.ndarray
    direction: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result array shapes for consistency."""
        shapes = [
            self.gradient.shape[0],
            self.magnitude.shape[0],
            self.direction.shape[0],
            self.confidence.shape[0],
        ]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent result array shapes: {shapes}")


class GradientSensor(BaseSensor):
    """
    High-performance gradient sensor for spatial derivative computation.

    This sensor computes spatial gradients of concentration fields using configurable
    finite difference algorithms with adaptive step sizing and multi-point sampling.
    Optimized for sub-10ms step execution targets while providing accurate directional
    navigation cues through gradient magnitude and direction calculations.

    The implementation supports vectorized operations for efficient multi-agent scenarios
    and includes comprehensive error handling, edge case management, and performance
    monitoring integration with the BaseSensor infrastructure.

    Key Design Principles:
    - Numerical accuracy through adaptive finite difference methods
    - Performance optimization via vectorized NumPy operations
    - Robustness through comprehensive error handling and validation
    - Extensibility via configurable computation algorithms and metadata collection
    - Integration with existing sensor infrastructure for monitoring and logging

    Performance Characteristics:
    - Gradient computation: <0.2ms per agent for multi-point sampling
    - Memory usage: <1KB per agent for computation state
    - Multi-agent scaling: Linear performance with agent count
    - Numerical stability: Robust in noisy and discontinuous fields

    Examples:
        Basic gradient computation:
            >>> sensor = GradientSensor()
            >>> positions = np.array([[10, 20], [15, 25]])
            >>> gradients = sensor.compute_gradient(plume_state, positions)

        Custom configuration with high accuracy:
            >>> config = GradientSensorConfig(
            ...     method=FiniteDifferenceMethod.CENTRAL,
            ...     order=4,
            ...     spatial_resolution=(0.1, 0.1)
            ... )
            >>> sensor = GradientSensor(config=config)

        Runtime configuration updates:
            >>> sensor.configure(
            ...     spatial_resolution=(0.2, 0.2),
            ...     adaptive_step_size=True,
            ...     noise_threshold=1e-5
            ... )
    """

    def __init__(
        self,
        config: Optional[GradientSensorConfig] = None,
        enable_logging: bool = True,
        sensor_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize GradientSensor with comprehensive configuration and monitoring.

        Args:
            config: GradientSensorConfig instance with computation parameters
            enable_logging: Enable structured logging and performance monitoring
            sensor_id: Unique sensor identifier for logging correlation
            **kwargs: Additional configuration options for extensibility

        Raises:
            ValueError: If configuration parameters are invalid
            TypeError: If config is not a GradientSensorConfig instance
        """
        # Initialize configuration with validation
        if config is None:
            self.config = GradientSensorConfig()
        elif isinstance(config, dict):
            self.config = GradientSensorConfig(**config)
        elif isinstance(config, GradientSensorConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be GradientSensorConfig or dict, got {type(config)}"
            )

        super().__init__(
            sensor_type="GradientSensor",
            enable_logging=enable_logging,
            sensor_id=sensor_id,
            **kwargs,
        )

        self._performance_metrics.update(
            {
                "computation_times": [],
                "total_computations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "numerical_warnings": 0,
                "edge_case_count": 0,
            }
        )

        self._gradient_cache = {} if self.config.enable_caching else None
        self._cache_timestamps = {} if self.config.enable_caching else None

        self._fd_coefficients = self._precompute_fd_coefficients()

        if self._logger:
            self._logger = self._logger.bind(
                method=self.config.method.value,
                order=self.config.order,
                spatial_resolution=self.config.spatial_resolution,
            )
            self._logger.info(
                "GradientSensor initialized with optimized configuration",
                config_summary={
                    "method": self.config.method.value,
                    "order": self.config.order,
                    "spatial_resolution": self.config.spatial_resolution,
                    "adaptive_step_size": self.config.adaptive_step_size,
                    "caching_enabled": self.config.enable_caching,
                },
            )

    def _precompute_fd_coefficients(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Pre-compute finite difference coefficients for all supported orders.

        Returns:
            Dict[int, Dict[str, np.ndarray]]: Coefficients indexed by order and method

        Notes:
            Pre-computation eliminates runtime coefficient calculation overhead,
            improving performance for repeated gradient computations.
        """
        coefficients = {}

        # Central difference coefficients
        central_coeffs = {
            2: np.array([-0.5, 0.0, 0.5]),  # 2nd order: [-1/2, 0, 1/2]
            4: np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),  # 4th order
            6: np.array(
                [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]
            ),  # 6th order
            8: np.array(
                [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
            ),  # 8th order
        }

        # Forward difference coefficients
        forward_coeffs = {
            2: np.array([-1.5, 2.0, -0.5]),  # 2nd order: [-3/2, 2, -1/2]
            4: np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]),  # 4th order
            6: np.array(
                [-147 / 60, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
            ),  # 6th order
            8: np.array(
                [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
            ),  # 8th order
        }

        # Backward difference coefficients (reverse of forward)
        backward_coeffs = {}
        for order, coeffs in forward_coeffs.items():
            backward_coeffs[order] = -np.flip(coeffs)

        # Store all coefficient sets
        coefficients = {
            order: {
                "central": central_coeffs[order],
                "forward": forward_coeffs[order],
                "backward": backward_coeffs[order],
            }
            for order in [2, 4, 6, 8]
        }

        return coefficients

    def _get_cache_key(self, positions: np.ndarray, plume_state_id: Any) -> str:
        """
        Generate cache key for gradient computation results.

        Args:
            positions: Agent positions for gradient computation
            plume_state_id: Identifier for plume state (e.g., frame number, hash)

        Returns:
            str: Cache key for storing/retrieving computation results
        """
        # Create deterministic hash from positions and configuration
        pos_hash = hash(positions.tobytes())
        config_hash = hash(
            (
                self.config.method.value,
                self.config.order,
                self.config.spatial_resolution,
                self.config.adaptive_step_size,
            )
        )

        return f"{plume_state_id}_{pos_hash}_{config_hash}"

    def _sample_concentration_at_offset(
        self, plume_state: Any, positions: np.ndarray, offset: Tuple[float, float]
    ) -> np.ndarray:
        """
        Sample concentration at positions with spatial offset.

        Args:
            plume_state: Plume model state providing concentration_at() method
            positions: Base positions for sampling with shape (n_agents, 2)
            offset: Spatial offset (dx, dy) to apply to positions

        Returns:
            np.ndarray: Concentration values at offset positions with shape (n_agents,)

        Notes:
            This method abstracts plume model interaction, supporting both
            PlumeModelProtocol instances and raw concentration arrays.
        """
        offset_positions = positions + np.array(offset)

        # Handle different plume state types
        if hasattr(plume_state, "concentration_at"):
            # PlumeModelProtocol implementation
            return plume_state.concentration_at(offset_positions)
        elif isinstance(plume_state, np.ndarray):
            # Raw concentration array - use bilinear interpolation
            return self._sample_array_at_positions(plume_state, offset_positions)
        else:
            # Fallback for other types
            try:
                return np.array(
                    [plume_state[int(pos[1]), int(pos[0])] for pos in offset_positions]
                )
            except (IndexError, TypeError):
                # Return zeros for out-of-bounds or invalid access
                return np.zeros(positions.shape[0])

    def _sample_array_at_positions(
        self, array: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        Sample 2D array at specified positions using bilinear interpolation.

        Args:
            array: 2D concentration array with shape (height, width)
            positions: Positions for sampling with shape (n_positions, 2)

        Returns:
            np.ndarray: Interpolated values at positions with shape (n_positions,)
        """
        if array.ndim != 2:
            raise ValueError(f"Array must be 2D, got shape {array.shape}")

        height, width = array.shape
        x_coords = np.clip(positions[:, 0], 0, width - 1)
        y_coords = np.clip(positions[:, 1], 0, height - 1)

        # Integer coordinates for array indexing
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)

        # Interpolation weights
        wx = x_coords - x0
        wy = y_coords - y0

        # Bilinear interpolation
        c00 = array[y0, x0]
        c01 = array[y1, x0]
        c10 = array[y0, x1]
        c11 = array[y1, x1]

        interpolated = (
            c00 * (1 - wx) * (1 - wy)
            + c10 * wx * (1 - wy)
            + c01 * (1 - wx) * wy
            + c11 * wx * wy
        )

        return interpolated

    def _compute_adaptive_step_size(
        self, plume_state: Any, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute adaptive step sizes based on local concentration variation.

        Args:
            plume_state: Plume model state for concentration sampling
            positions: Agent positions for step size computation

        Returns:
            Tuple[np.ndarray, np.ndarray]: Adaptive step sizes (dx_steps, dy_steps)

        Notes:
            Adaptive step sizing improves gradient accuracy in regions with
            sharp concentration variations while maintaining computational efficiency
            in smoother regions.
        """
        base_dx, base_dy = self.config.spatial_resolution

        # Sample concentrations in multiple directions to assess variation
        test_offsets = [
            (base_dx, 0),
            (-base_dx, 0),
            (0, base_dy),
            (0, -base_dy),
            (base_dx, base_dy),
            (-base_dx, -base_dy),
        ]

        concentration_samples = []
        for offset in test_offsets:
            samples = self._sample_concentration_at_offset(
                plume_state, positions, offset
            )
            concentration_samples.append(samples)

        concentration_samples = np.array(
            concentration_samples
        )  # Shape: (n_offsets, n_agents)

        # Compute local variation metrics
        center_concentration = self._sample_concentration_at_offset(
            plume_state, positions, (0, 0)
        )
        variation = np.std(
            concentration_samples - center_concentration[np.newaxis, :], axis=0
        )

        # Adaptive step size based on local variation
        # High variation -> smaller steps for accuracy
        # Low variation -> larger steps for efficiency
        adaptation_factor = np.clip(
            1.0 / (1.0 + 10.0 * variation),
            self.config.min_step_size / base_dx,
            self.config.max_step_size / base_dx,
        )

        adaptive_dx = base_dx * adaptation_factor
        adaptive_dy = base_dy * adaptation_factor

        return adaptive_dx, adaptive_dy

    def _compute_finite_difference_gradient(
        self, plume_state: Any, positions: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute spatial gradients using finite difference methods.

        Args:
            plume_state: Plume model state for concentration sampling
            positions: Agent positions for gradient computation

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Gradients and computation metadata

        Notes:
            This is the core gradient computation method using pre-computed
            finite difference coefficients for optimal performance.
        """
        start_time = time.perf_counter() if self._enable_logging else None

        n_agents = positions.shape[0]
        gradients = np.zeros((n_agents, 2))
        metadata = {
            "method_used": [],
            "step_sizes_used": [],
            "numerical_stability": [],
            "edge_corrections": [],
        }

        # Determine step sizes (adaptive or fixed)
        if self.config.adaptive_step_size:
            dx_steps, dy_steps = self._compute_adaptive_step_size(
                plume_state, positions
            )
        else:
            dx_steps = np.full(n_agents, self.config.spatial_resolution[0])
            dy_steps = np.full(n_agents, self.config.spatial_resolution[1])

        # Get finite difference coefficients
        coeffs = self._fd_coefficients[self.config.order]

        # Determine method for each position
        methods_used = self._select_fd_methods(
            plume_state, positions, dx_steps, dy_steps
        )

        # Compute gradients using vectorized operations where possible
        for method in set(methods_used):
            method_mask = methods_used == method
            if not np.any(method_mask):
                continue

            method_positions = positions[method_mask]
            method_dx = dx_steps[method_mask]
            method_dy = dy_steps[method_mask]

            # Compute x-derivatives
            x_derivs = self._compute_directional_derivative(
                plume_state, method_positions, method_dx, coeffs[method], axis=0
            )

            # Compute y-derivatives
            y_derivs = self._compute_directional_derivative(
                plume_state, method_positions, method_dy, coeffs[method], axis=1
            )

            # Store results
            gradients[method_mask, 0] = x_derivs
            gradients[method_mask, 1] = y_derivs

            # Store metadata
            for i, (pos_idx, _) in enumerate(
                zip(np.where(method_mask)[0], method_positions)
            ):
                metadata["method_used"].append(method)
                metadata["step_sizes_used"].append((method_dx[i], method_dy[i]))

                # Assess numerical stability
                grad_magnitude = np.sqrt(x_derivs[i] ** 2 + y_derivs[i] ** 2)
                stability = 1.0 / (1.0 + grad_magnitude) if grad_magnitude > 0 else 1.0
                metadata["numerical_stability"].append(stability)

                # Note any edge corrections applied
                edge_correction = self._detect_edge_effects(
                    plume_state, method_positions[i : i + 1]
                )
                metadata["edge_corrections"].append(edge_correction)

        # Track performance metrics
        if self._enable_logging:
            computation_time = (time.perf_counter() - start_time) * 1000
            self._performance_metrics["computation_times"].append(computation_time)
            self._performance_metrics["total_computations"] += 1

            # Log performance warnings if computation exceeds targets
            if computation_time > 0.2 * n_agents:  # 0.2ms per agent target
                if self._logger:
                    self._logger.warning(
                        "Gradient computation exceeded performance target",
                        computation_time_ms=computation_time,
                        target_time_ms=0.2 * n_agents,
                        n_agents=n_agents,
                        method=self.config.method.value,
                        adaptive_steps=self.config.adaptive_step_size,
                    )

        return gradients, metadata

    def _select_fd_methods(
        self,
        plume_state: Any,
        positions: np.ndarray,
        dx_steps: np.ndarray,
        dy_steps: np.ndarray,
    ) -> List[str]:
        """
        Select appropriate finite difference method for each position.

        Args:
            plume_state: Plume model state for boundary detection
            positions: Agent positions
            dx_steps: X-direction step sizes
            dy_steps: Y-direction step sizes

        Returns:
            List[str]: Selected method for each position
        """
        n_agents = positions.shape[0]
        methods = []

        for i in range(n_agents):
            pos = positions[i]
            dx = dx_steps[i]
            dy = dy_steps[i]

            if self.config.method == FiniteDifferenceMethod.ADAPTIVE:
                # Choose method based on position and boundary proximity
                near_boundary = self._check_boundary_proximity(plume_state, pos, dx, dy)

                if near_boundary["left"] or near_boundary["right"]:
                    methods.append("central")  # Safest for boundaries
                elif near_boundary["top"] or near_boundary["bottom"]:
                    methods.append("central")
                else:
                    methods.append("central")  # Default to central for best accuracy
            else:
                methods.append(self.config.method.value)

        return methods

    def _check_boundary_proximity(
        self, plume_state: Any, position: np.ndarray, dx: float, dy: float
    ) -> Dict[str, bool]:
        """
        Check proximity to domain boundaries for method selection.

        Args:
            plume_state: Plume model state for boundary information
            position: Single agent position
            dx: X-direction step size
            dy: Y-direction step size

        Returns:
            Dict[str, bool]: Boundary proximity flags
        """
        # Default implementation - can be overridden for specific plume models
        if hasattr(plume_state, "shape"):
            height, width = plume_state.shape[:2]
            x, y = position

            return {
                "left": x - dx < 0,
                "right": x + dx >= width,
                "top": y - dy < 0,
                "bottom": y + dy >= height,
            }
        else:
            # Conservative assumption if boundary information unavailable
            return {"left": False, "right": False, "top": False, "bottom": False}

    def _compute_directional_derivative(
        self,
        plume_state: Any,
        positions: np.ndarray,
        step_sizes: np.ndarray,
        coefficients: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """
        Compute directional derivative using finite difference coefficients.

        Args:
            plume_state: Plume model state for concentration sampling
            positions: Positions for derivative computation
            step_sizes: Step sizes for each position
            coefficients: Finite difference coefficients
            axis: Derivative axis (0=x, 1=y)

        Returns:
            np.ndarray: Directional derivatives
        """
        n_positions = positions.shape[0]
        n_points = len(coefficients)
        derivatives = np.zeros(n_positions)

        # Create offset points for finite difference stencil
        center_idx = n_points // 2

        for i, coeff in enumerate(coefficients):
            if coeff == 0:
                continue

            # Calculate offset from center
            offset_distance = i - center_idx

            # Create offset vector
            offset = np.zeros(2)
            offset[axis] = offset_distance

            # Scale by step size for each position
            offsets = offset[np.newaxis, :] * step_sizes[:, np.newaxis]

            # Sample concentrations at offset positions
            offset_positions = positions + offsets
            concentrations = self._sample_concentration_at_offset(
                plume_state, positions, (0, 0)  # This will be corrected below
            )

            # Correct implementation: sample at each offset position individually
            concentrations = np.array(
                [
                    self._sample_concentration_at_offset(
                        plume_state, pos.reshape(1, -1), offset[:2]
                    )[0]
                    for pos, offset in zip(positions, offsets)
                ]
            )

            # Accumulate weighted contributions
            derivatives += coeff * concentrations / step_sizes

        return derivatives

    def _detect_edge_effects(self, plume_state: Any, positions: np.ndarray) -> bool:
        """
        Detect potential edge effects in gradient computation.

        Args:
            plume_state: Plume model state
            positions: Positions to check for edge effects

        Returns:
            bool: True if edge effects detected
        """
        # Simple edge detection based on zero concentrations in neighborhood
        if hasattr(plume_state, "shape"):
            height, width = plume_state.shape[:2]

            for pos in positions:
                x, y = int(pos[0]), int(pos[1])

                # Check if position is near domain boundary
                margin = max(self.config.spatial_resolution)
                if (
                    x < margin
                    or x >= width - margin
                    or y < margin
                    or y >= height - margin
                ):
                    return True

        return False

    def _compute_gradient_metadata(
        self, gradients: np.ndarray, computation_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute comprehensive gradient metadata for navigation support.

        Args:
            gradients: Computed gradient vectors
            computation_metadata: Metadata from gradient computation

        Returns:
            Dict[str, Any]: Comprehensive gradient metadata
        """
        n_agents = gradients.shape[0]

        # Handle case where gradients array is empty
        if n_agents == 0:
            return {
                "magnitude": np.array([]),
                "direction": np.array([]),
                "confidence": np.array([]),
                "computation_method": computation_metadata.get("method_used", []),
                "step_sizes": computation_metadata.get("step_sizes_used", []),
                "edge_effects_detected": np.array([], dtype=bool),
                "noise_level": self.config.noise_threshold,
                "sensor_id": self._sensor_id,
                "timestamp": time.time(),
            }

        # Compute gradient magnitudes and directions
        magnitudes = np.sqrt(np.sum(gradients**2, axis=1))

        # Compute directions in degrees (0° = east, 90° = north)
        directions = np.degrees(np.arctan2(gradients[:, 1], gradients[:, 0]))
        directions = (directions + 360) % 360  # Normalize to [0, 360)

        # Compute confidence metrics based on numerical stability
        numerical_stability = np.array(
            computation_metadata.get("numerical_stability", [1.0] * n_agents)
        )
        edge_effects = np.array(
            computation_metadata.get("edge_corrections", [False] * n_agents), dtype=bool
        )

        # Ensure all arrays have consistent size
        if len(numerical_stability) != n_agents:
            numerical_stability = np.array([1.0] * n_agents)
        if len(edge_effects) != n_agents:
            edge_effects = np.array([False] * n_agents, dtype=bool)

        # Confidence calculation considering multiple factors
        confidence = numerical_stability.copy()
        confidence[edge_effects] *= 0.5  # Reduce confidence near edges
        confidence[
            magnitudes < self.config.noise_threshold
        ] *= 0.3  # Low confidence for weak gradients

        metadata = {
            "magnitude": magnitudes,
            "direction": directions,
            "confidence": confidence,
            "computation_method": computation_metadata.get("method_used", []),
            "step_sizes": computation_metadata.get("step_sizes_used", []),
            "edge_effects_detected": edge_effects,
            "noise_level": self.config.noise_threshold,
            "sensor_id": self._sensor_id,
            "timestamp": time.time(),
        }

        return metadata

    # SensorProtocol interface implementation

    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Binary detection based on gradient magnitude threshold.

        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent

        Returns:
            np.ndarray: Boolean detection results - True indicates significant gradient detected

        Notes:
            Detection is based on gradient magnitude exceeding a threshold,
            indicating presence of spatial concentration variation suitable for navigation.
        """
        gradients = self.compute_gradient(plume_state, positions)
        magnitudes = np.sqrt(np.sum(gradients**2, axis=1))

        # Use noise threshold as detection threshold
        detection_threshold = (
            self.config.noise_threshold * 10
        )  # Scale for detection sensitivity
        detections = magnitudes > detection_threshold

        return detections

    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Quantitative gradient magnitude measurement.

        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent

        Returns:
            np.ndarray: Gradient magnitude values in concentration units per distance unit

        Notes:
            Provides quantitative gradient strength measurements for
            navigation algorithms requiring gradient magnitude information.
        """
        gradients = self.compute_gradient(plume_state, positions)
        magnitudes = np.sqrt(np.sum(gradients**2, axis=1))

        return magnitudes

    def _compute_gradient_raw(
        self, plume_state: Any, positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute spatial gradients at specified agent positions.

        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent

        Returns:
            np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single agent.
                Components represent [∂c/∂x, ∂c/∂y] spatial derivatives.

        Notes:
            This is the core method for spatial gradient computation using
            configurable finite difference algorithms with adaptive step sizing.

        Performance:
            Optimized for <0.2ms per agent execution time through vectorized
            operations and pre-computed coefficients.

        Examples:
            Single agent gradient:
                >>> position = np.array([15, 25])
                >>> gradient = sensor.compute_gradient(plume_state, position)
                >>> dx, dy = gradient[0], gradient[1]  # Gradient components

            Multi-agent batch gradients:
                >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
                >>> gradients = sensor.compute_gradient(plume_state, positions)
                >>> uphill_directions = np.arctan2(gradients[:, 1], gradients[:, 0])
        """
        start_time = (
            time.perf_counter() if self._enable_performance_monitoring else None
        )

        # Input validation and normalization
        if positions.ndim == 1 and positions.shape[0] == 2:
            # Single agent case - reshape to (1, 2)
            positions = positions.reshape(1, -1)
            single_agent = True
        elif (
            positions.ndim == 2 and positions.shape[0] == 1 and positions.shape[1] == 2
        ):
            # Single agent case - already in (1, 2) format
            single_agent = True
        else:
            single_agent = False

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"positions must have shape (n_agents, 2), got {positions.shape}"
            )

        # Check cache if enabled
        cache_key = None
        if self.config.enable_caching:
            plume_state_id = getattr(plume_state, "state_id", id(plume_state))
            cache_key = self._get_cache_key(positions, plume_state_id)

            # Check cache validity
            if (
                cache_key in self._gradient_cache
                and time.time() - self._cache_timestamps[cache_key]
                < self.config.cache_ttl_seconds
            ):

                self._performance_metrics["cache_hits"] += 1
                cached_result = self._gradient_cache[cache_key]

                if self._logger:
                    self._logger.debug(
                        "Retrieved gradient computation from cache",
                        cache_key=cache_key,
                        n_agents=positions.shape[0],
                    )
                if self._enable_performance_monitoring and start_time is not None:
                    comp_time = (time.perf_counter() - start_time) * 1000
                    self._performance_metrics["computation_times"].append(comp_time)
                self._performance_metrics["total_computations"] += positions.shape[0]
                return cached_result if not single_agent else cached_result[0]
            else:
                self._performance_metrics["cache_misses"] += 1

        # Compute gradients using finite difference methods
        try:
            gradients, metadata = self._compute_finite_difference_gradient(
                plume_state, positions
            )

            # Store in cache if enabled
            if self.config.enable_caching and cache_key is not None:
                # Maintain cache size limits
                if len(self._gradient_cache) >= self.config.max_cache_size:
                    # Remove oldest entries
                    oldest_key = min(
                        self._cache_timestamps.keys(),
                        key=lambda k: self._cache_timestamps[k],
                    )
                    del self._gradient_cache[oldest_key]
                    del self._cache_timestamps[oldest_key]

                self._gradient_cache[cache_key] = gradients.copy()
                self._cache_timestamps[cache_key] = time.time()

            if self._enable_performance_monitoring and start_time is not None:
                comp_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics["computation_times"].append(comp_time)

            self._performance_metrics["total_computations"] += positions.shape[0]

            # Log computation summary
            if (
                self._logger
                and self._performance_metrics["total_computations"] % 100 == 0
            ):
                recent_times = self._performance_metrics["computation_times"][-50:]
                self._logger.debug(
                    "Gradient computation performance summary",
                    total_computations=self._performance_metrics["total_computations"],
                    avg_computation_time_ms=(
                        np.mean(recent_times) if recent_times else 0
                    ),
                    cache_hit_rate=(
                        self._performance_metrics["cache_hits"]
                        / max(
                            1,
                            self._performance_metrics["cache_hits"]
                            + self._performance_metrics["cache_misses"],
                        )
                    ),
                    n_agents=positions.shape[0],
                )

            # Return appropriate format
            return gradients if not single_agent else gradients[0]

        except Exception as e:
            self._performance_metrics["numerical_warnings"] += 1

            if self._logger:
                self._logger.error(
                    "Gradient computation failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    positions=positions.tolist(),
                    config=self.config.__dict__,
                )
            raise

    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        return self._execute_with_monitoring(
            self._compute_gradient_raw, "gradient", plume_state, positions
        )

    def compute_gradient_with_metadata(
        self, plume_state: Any, positions: np.ndarray
    ) -> GradientResult:
        """
        Compute gradients with comprehensive metadata for advanced navigation support.

        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent

        Returns:
            GradientResult: Comprehensive gradient result with metadata including:
                - gradient: Spatial gradient vectors [∂c/∂x, ∂c/∂y]
                - magnitude: Gradient magnitude values
                - direction: Gradient direction in degrees (0° = east, 90° = north)
                - confidence: Computation confidence based on numerical stability
                - metadata: Additional computation diagnostics and performance data

        Notes:
            This method provides the most comprehensive gradient information,
            including navigation-relevant derived metrics and quality assessments.

        Examples:
            Complete gradient analysis:
                >>> result = sensor.compute_gradient_with_metadata(plume_state, positions)
                >>> strong_gradients = result.magnitude > 0.1
                >>> reliable_directions = result.direction[result.confidence > 0.8]
                >>> uphill_directions = result.direction[strong_gradients]

            Navigation decision making:
                >>> result = sensor.compute_gradient_with_metadata(plume_state, positions)
                >>> if result.confidence[0] > 0.7 and result.magnitude[0] > 0.05:
                ...     navigation_direction = result.direction[0]
                ...     navigation_strength = result.magnitude[0]
        """
        # Input validation and normalization
        if positions.ndim == 1 and positions.shape[0] == 2:
            positions = positions.reshape(1, -1)
            single_agent = True
        else:
            single_agent = False

        # Compute gradients with detailed metadata
        gradients, computation_metadata = self._compute_finite_difference_gradient(
            plume_state, positions
        )

        # Compute comprehensive metadata
        full_metadata = self._compute_gradient_metadata(gradients, computation_metadata)

        # Create structured result
        result = GradientResult(
            gradient=gradients,
            magnitude=full_metadata["magnitude"],
            direction=full_metadata["direction"],
            confidence=full_metadata["confidence"],
            metadata=full_metadata,
        )

        # Adjust for single agent case
        if single_agent:
            result.gradient = result.gradient[0]
            result.magnitude = result.magnitude[0]
            result.direction = result.direction[0]
            result.confidence = result.confidence[0]

        return result

    def get_sensor_info(self) -> Dict[str, Any]:
        info = {
            "sensor_type": "GradientSensor",
            "sensor_id": self._sensor_id,
            "capabilities": [
                "gradient_computation",
                "vectorized_operations",
            ],
            "configuration": {
                "method": self.config.method.value,
                "order": self.config.order,
                "spatial_resolution": self.config.spatial_resolution,
                "adaptive_step_size": self.config.adaptive_step_size,
                "caching_enabled": self.config.enable_caching,
            },
        }
        if self._logger:
            self._logger.debug("GradientSensor info requested", info=info)
        return info

    def get_metadata(self) -> Dict[str, Any]:
        metadata = {
            "sensor_type": "GradientSensor",
            "config": self.get_sensor_info()["configuration"],
            "performance": self.get_performance_metrics(),
        }
        if self._logger:
            self._logger.debug("GradientSensor metadata requested", metadata=metadata)
        return metadata

    def get_observation_space_info(self) -> Dict[str, Any]:
        return {"shape": (2,), "dtype": np.float64}

    def configure(self, **kwargs: Any) -> None:
        """
        Update sensor configuration parameters during runtime.

        Args:
            **kwargs: Configuration parameters to update. Supported options:
                - method: Finite difference method (FiniteDifferenceMethod enum or string)
                - order: Derivative approximation order (2, 4, 6, 8)
                - spatial_resolution: Step size tuple (dx, dy)
                - adaptive_step_size: Enable adaptive step sizing (bool)
                - min_step_size: Minimum step size for adaptive methods (float)
                - max_step_size: Maximum step size for adaptive methods (float)
                - noise_threshold: Concentration threshold for noise detection (float)
                - enable_caching: Enable gradient computation caching (bool)
                - enable_metadata: Enable comprehensive metadata collection (bool)

        Notes:
            Configuration updates apply immediately to subsequent gradient computations.
            Cache is cleared when configuration changes to ensure consistency.

        Examples:
            Update spatial resolution for higher accuracy:
                >>> sensor.configure(spatial_resolution=(0.1, 0.1))

            Switch to adaptive method with custom thresholds:
                >>> sensor.configure(
                ...     method='adaptive',
                ...     adaptive_step_size=True,
                ...     min_step_size=0.05,
                ...     max_step_size=1.0
                ... )

            Optimize for performance:
                >>> sensor.configure(
                ...     method='forward',
                ...     order=2,
                ...     adaptive_step_size=False,
                ...     enable_caching=True
                ... )
        """
        config_changed = False

        # Update configuration parameters with validation
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)

                # Special handling for method enum
                if key == "method":
                    if isinstance(value, str):
                        value = FiniteDifferenceMethod(value)
                    elif not isinstance(value, FiniteDifferenceMethod):
                        raise ValueError(
                            f"method must be FiniteDifferenceMethod or string, got {type(value)}"
                        )

                # Validate specific parameters
                if key == "order" and value not in [2, 4, 6, 8]:
                    raise ValueError(f"order must be in [2, 4, 6, 8], got {value}")

                if key == "spatial_resolution":
                    if not (isinstance(value, (tuple, list)) and len(value) == 2):
                        raise ValueError(
                            f"spatial_resolution must be 2-element tuple, got {value}"
                        )
                    if any(r <= 0 for r in value):
                        raise ValueError(
                            f"spatial_resolution components must be positive, got {value}"
                        )

                if key in ["min_step_size", "max_step_size"] and value <= 0:
                    raise ValueError(f"{key} must be positive, got {value}")

                # Update configuration
                setattr(self.config, key, value)

                if old_value != value:
                    config_changed = True
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Clear cache if configuration changed
        if config_changed and self.config.enable_caching:
            self._gradient_cache.clear()
            self._cache_timestamps.clear()

        # Re-compute finite difference coefficients if method or order changed
        if any(key in kwargs for key in ["method", "order"]):
            self._fd_coefficients = self._precompute_fd_coefficients()

        # Log configuration update
        if self._logger and config_changed:
            self._logger.info(
                "GradientSensor configuration updated",
                updated_parameters=list(kwargs.keys()),
                new_config_summary={
                    "method": self.config.method.value,
                    "order": self.config.order,
                    "spatial_resolution": self.config.spatial_resolution,
                    "adaptive_step_size": self.config.adaptive_step_size,
                },
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.

        Returns:
            Dict[str, Any]: Performance metrics including computation times,
                cache statistics, error counts, and efficiency indicators

        Examples:
            Performance monitoring:
                >>> metrics = sensor.get_performance_metrics()
                >>> avg_computation_time = metrics['avg_computation_time_ms']
                >>> cache_efficiency = metrics['cache_hit_rate']
                >>> error_rate = metrics['error_rate']
        """
        base_metrics = super().get_performance_metrics()

        metrics = {
            **base_metrics,
            "total_computations": self._performance_metrics["total_computations"],
            "cache_enabled": self.config.enable_caching,
        }

        # Computation time statistics
        if self._performance_metrics["computation_times"]:
            times = np.array(self._performance_metrics["computation_times"])
            metrics.update(
                {
                    "avg_computation_time_ms": float(np.mean(times)),
                    "max_computation_time_ms": float(np.max(times)),
                    "computation_time_std_ms": float(np.std(times)),
                    "performance_violations": int(
                        np.sum(times > 0.2)
                    ),  # >0.2ms per agent
                }
            )

        # Cache statistics
        if self.config.enable_caching:
            total_requests = (
                self._performance_metrics["cache_hits"]
                + self._performance_metrics["cache_misses"]
            )
            if total_requests > 0:
                metrics.update(
                    {
                        "cache_hit_rate": self._performance_metrics["cache_hits"]
                        / total_requests,
                        "cache_size": (
                            len(self._gradient_cache) if self._gradient_cache else 0
                        ),
                        "cache_efficiency": self._performance_metrics["cache_hits"]
                        / max(1, total_requests),
                    }
                )

        # Error and stability metrics
        metrics.update(
            {
                "numerical_warnings": self._performance_metrics["numerical_warnings"],
                "edge_case_count": self._performance_metrics["edge_case_count"],
                "error_rate": (
                    self._performance_metrics["numerical_warnings"]
                    / max(1, self._performance_metrics["total_computations"])
                ),
            }
        )

        # Configuration summary
        metrics.update(
            {
                "config_method": self.config.method.value,
                "config_order": self.config.order,
                "config_adaptive": self.config.adaptive_step_size,
                "config_resolution": self.config.spatial_resolution,
            }
        )

        return metrics

    def reset(self, **kwargs: Any) -> None:
        """
        Reset sensor to initial state, clearing cache and performance metrics.

        Args:
            **kwargs: Optional parameters for sensor-specific reset behavior.
                Common options include:
                - clear_cache: Whether to reset computation cache (default: True)
                - clear_metrics: Whether to reset performance metrics (default: True)
                - reset_config: Whether to reset sensor configuration (default: False)

        Notes:
            This method provides comprehensive sensor reset functionality by clearing
            both computation cache and performance metrics. Sensor configuration
            is preserved unless explicitly requested via reset_config parameter.

        Examples:
            Complete sensor reset:
                >>> sensor.reset()

            Reset only cache:
                >>> sensor.reset(clear_cache=True, clear_metrics=False)
        """
        clear_cache = kwargs.get("clear_cache", True)
        clear_metrics = kwargs.get("clear_metrics", True)

        if clear_cache and hasattr(self, "_gradient_cache"):
            self._gradient_cache.clear()

        if clear_metrics:
            self.reset_performance_metrics()

        if self._logger:
            self._logger.debug(f"Sensor {self._sensor_id} reset completed")

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics for new monitoring period."""
        super().reset_performance_metrics()
        if not self._enable_performance_monitoring:
            return

        self._performance_metrics.update(
            {
                "computation_times": [],
                "total_computations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "numerical_warnings": 0,
                "edge_case_count": 0,
            }
        )

        if self._logger:
            self._logger.debug("Performance metrics reset", sensor_id=self._sensor_id)


# Export public API
__all__ = [
    "GradientSensor",
    "GradientSensorConfig",
    "GradientResult",
    "FiniteDifferenceMethod",
]
