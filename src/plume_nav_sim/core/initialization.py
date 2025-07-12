"""
Agent initialization strategies for configurable starting position generation.

This module implements the AgentInitializer abstraction for creating diverse experimental 
setups with deterministic seeding and validation for multi-agent scenarios. The 
initialization system supports four core strategies for flexible agent placement:

- UniformRandomInitializer: Random placement within domain boundaries
- GridInitializer: Systematic grid-based positioning patterns  
- FixedListInitializer: Predefined positions from configuration
- FromDatasetInitializer: Dataset-driven position loading

Key Features:
- Protocol-based design for extensible initialization patterns
- Deterministic seeding for reproducible experimental setups
- Multi-agent validation ensuring domain constraint compliance
- Performance optimization for large-scale scenarios (≤1ms for 100 agents)
- Seamless integration with PlumeNavigationEnv.reset() method
- Hydra configuration support via conf/base/agent_init/ group

The initialization framework follows the v1.0 architecture patterns with strict interface
compliance and vectorized operations for efficient multi-agent position generation.

Performance Requirements:
- Initialization time: <1ms for 100 agents
- Memory efficiency: O(n) scaling with agent count
- Deterministic behavior with identical seeding
- Domain validation: <0.1ms per agent validation

Examples:
    Basic uniform random initialization:
    >>> initializer = UniformRandomInitializer(bounds=(100, 100), seed=42)
    >>> positions = initializer.initialize_positions(num_agents=10)
    
    Grid-based systematic placement:
    >>> initializer = GridInitializer(grid_size=(5, 2), domain_bounds=(100, 100))
    >>> positions = initializer.initialize_positions(num_agents=10)
    
    Configuration-driven factory creation:
    >>> config = {'_target_': 'UniformRandomInitializer', 'bounds': (100, 100)}
    >>> initializer = create_agent_initializer(config)

Authors: Blitzy Platform v1.0 Development Team
"""

from __future__ import annotations
import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import warnings

# Import the protocol from the protocols module once it's defined
# For now, we'll define it locally as per the schema requirements
try:
    from .protocols import AgentInitializerProtocol
except ImportError:
    # Define the protocol locally since it may not exist yet in protocols.py
    @runtime_checkable
    class AgentInitializerProtocol(Protocol):
        """
        Protocol defining the interface for agent initialization strategies.
        
        This protocol ensures consistent initialization behavior across different
        strategies while maintaining type safety and extensibility. All concrete
        initialization strategies must implement these core methods.
        
        The protocol supports deterministic seeding, domain validation, and
        flexible position generation for diverse experimental requirements.
        """
        
        def initialize_positions(
            self, 
            num_agents: int, 
            **kwargs: Any
        ) -> np.ndarray:
            """
            Generate initial agent positions based on strategy configuration.
            
            Args:
                num_agents: Number of agents to initialize
                **kwargs: Additional strategy-specific parameters
                
            Returns:
                np.ndarray: Agent positions with shape (num_agents, 2)
                    Each row contains [x, y] coordinates for one agent
                    
            Notes:
                - Must be deterministic when seeded
                - Positions should be within domain bounds
                - Performance target: <1ms for 100 agents
            """
            ...
        
        def validate_domain(self, positions: np.ndarray) -> bool:
            """
            Validate that generated positions comply with domain constraints.
            
            Args:
                positions: Agent positions to validate with shape (n_agents, 2)
                
            Returns:
                bool: True if all positions are valid, False otherwise
                
            Notes:
                - Checks domain bounds compliance
                - No collision detection required per specification
                - Should be called after position generation
            """
            ...
        
        def reset(self, seed: Optional[int] = None) -> None:
            """
            Reset initializer state with optional deterministic seeding.
            
            Args:
                seed: Random seed for deterministic behavior (optional)
                
            Notes:
                - Reinitializes internal random state
                - Ensures reproducible position generation
                - Should be called before each episode
            """
            ...


@dataclass
class InitializationConfig:
    """
    Base configuration class for agent initialization strategies.
    
    Provides common configuration parameters shared across all initialization
    strategies with type validation and default value management.
    """
    domain_bounds: Tuple[float, float] = (100.0, 100.0)
    seed: Optional[int] = None
    validation_enabled: bool = True


class UniformRandomInitializer:
    """
    Uniform random agent initialization within rectangular domain boundaries.
    
    This strategy places agents uniformly at random within the specified domain
    bounds, ensuring even spatial distribution for unbiased experimental setups.
    Supports deterministic seeding for reproducible position generation.
    
    The implementation uses NumPy's vectorized random number generation for
    efficient initialization of large agent populations while maintaining
    performance requirements.
    
    Performance Characteristics:
    - Time complexity: O(n) where n is number of agents
    - Memory complexity: O(n) for position storage
    - Target latency: <1ms for 100 agents
    
    Examples:
        Basic random initialization:
        >>> initializer = UniformRandomInitializer(bounds=(100, 100))
        >>> positions = initializer.initialize_positions(num_agents=50)
        
        With deterministic seeding:
        >>> initializer = UniformRandomInitializer(bounds=(100, 100), seed=42)
        >>> positions = initializer.initialize_positions(num_agents=50)
        # Repeated calls with same seed produce identical results
    """
    
    def __init__(
        self, 
        bounds: Tuple[float, float] = (100.0, 100.0),
        seed: Optional[int] = None,
        margin: float = 0.0
    ):
        """
        Initialize uniform random strategy with domain configuration.
        
        Args:
            bounds: Domain dimensions as (width, height) tuple
            seed: Random seed for deterministic behavior
            margin: Safety margin from domain edges (default: 0.0)
        """
        self.bounds = bounds
        self.margin = margin
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Validate configuration parameters
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError(f"Domain bounds must be positive: {bounds}")
        if margin < 0:
            raise ValueError(f"Margin must be non-negative: {margin}")
        if 2 * margin >= min(bounds):
            raise ValueError(f"Margin {margin} too large for domain bounds {bounds}")
    
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """
        Generate uniformly random agent positions within domain bounds.
        
        Args:
            num_agents: Number of agent positions to generate
            **kwargs: Additional parameters (unused for uniform random)
            
        Returns:
            np.ndarray: Random positions with shape (num_agents, 2)
                Each row contains [x, y] coordinates within domain bounds
                
        Raises:
            ValueError: If num_agents is invalid or domain constraints violated
        """
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive: {num_agents}")
        
        # Calculate effective bounds accounting for margin
        effective_width = self.bounds[0] - 2 * self.margin
        effective_height = self.bounds[1] - 2 * self.margin
        
        # Generate random positions within effective bounds
        x_positions = self._rng.uniform(
            self.margin, 
            self.margin + effective_width, 
            size=num_agents
        )
        y_positions = self._rng.uniform(
            self.margin, 
            self.margin + effective_height, 
            size=num_agents
        )
        
        # Combine into position array
        positions = np.column_stack((x_positions, y_positions)).astype(np.float32)
        
        return positions
    
    def validate_domain(self, positions: np.ndarray) -> bool:
        """
        Validate that positions are within domain boundaries.
        
        Args:
            positions: Agent positions to validate with shape (n_agents, 2)
            
        Returns:
            bool: True if all positions are within bounds, False otherwise
        """
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        # Check x-coordinate bounds
        x_valid = np.all((positions[:, 0] >= 0) & (positions[:, 0] <= self.bounds[0]))
        
        # Check y-coordinate bounds  
        y_valid = np.all((positions[:, 1] >= 0) & (positions[:, 1] <= self.bounds[1]))
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset random number generator with optional new seed.
        
        Args:
            seed: New random seed (uses instance seed if None)
        """
        reset_seed = seed if seed is not None else self.seed
        self._rng = np.random.RandomState(reset_seed)
    
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        return "uniform_random"
    
    def set_domain_bounds(self, bounds: Tuple[float, float]) -> None:
        """Update domain bounds configuration."""
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError(f"Domain bounds must be positive: {bounds}")
        self.bounds = bounds
    
    def set_seed(self, seed: int) -> None:
        """Update random seed and reset generator."""
        self.seed = seed
        self.reset(seed)


class GridInitializer:
    """
    Grid-based systematic agent initialization for controlled spatial arrangements.
    
    This strategy arranges agents in a regular grid pattern within the domain,
    providing systematic spatial distribution for controlled experimental conditions.
    Supports flexible grid dimensions and spacing configurations.
    
    The grid layout automatically adjusts spacing to fit the specified number of
    agents within domain bounds while maintaining even distribution patterns.
    
    Performance Characteristics:
    - Time complexity: O(n) where n is number of agents
    - Memory complexity: O(n) for position storage  
    - Target latency: <1ms for 100 agents
    
    Examples:
        Square grid arrangement:
        >>> initializer = GridInitializer(domain_bounds=(100, 100))
        >>> positions = initializer.initialize_positions(num_agents=25)  # 5x5 grid
        
        Custom grid dimensions:
        >>> initializer = GridInitializer(
        ...     domain_bounds=(100, 50), 
        ...     grid_size=(10, 5)
        ... )
        >>> positions = initializer.initialize_positions(num_agents=50)
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float] = (100.0, 100.0),
        grid_size: Optional[Tuple[int, int]] = None,
        spacing: Optional[Tuple[float, float]] = None,
        offset: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize grid strategy with layout configuration.
        
        Args:
            domain_bounds: Domain dimensions as (width, height) tuple
            grid_size: Explicit grid dimensions as (cols, rows) tuple (optional)
            spacing: Grid spacing as (dx, dy) tuple (optional)
            offset: Grid offset from origin as (x_offset, y_offset)
        """
        self.domain_bounds = domain_bounds
        self.grid_size = grid_size
        self.spacing = spacing
        self.offset = offset
        
        # Validate configuration parameters
        if domain_bounds[0] <= 0 or domain_bounds[1] <= 0:
            raise ValueError(f"Domain bounds must be positive: {domain_bounds}")
        if grid_size and (grid_size[0] <= 0 or grid_size[1] <= 0):
            raise ValueError(f"Grid size must be positive: {grid_size}")
        if spacing and (spacing[0] <= 0 or spacing[1] <= 0):
            raise ValueError(f"Grid spacing must be positive: {spacing}")
    
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """
        Generate grid-based agent positions within domain bounds.
        
        Args:
            num_agents: Number of agent positions to generate
            **kwargs: Additional parameters (e.g., 'grid_size_override')
            
        Returns:
            np.ndarray: Grid positions with shape (num_agents, 2)
                Each row contains [x, y] coordinates in grid arrangement
                
        Raises:
            ValueError: If num_agents is invalid or grid doesn't fit domain
        """
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive: {num_agents}")
        
        # Determine grid dimensions
        grid_cols, grid_rows = self._calculate_grid_dimensions(num_agents, **kwargs)
        
        # Calculate grid spacing
        spacing_x, spacing_y = self._calculate_grid_spacing(grid_cols, grid_rows)
        
        # Generate grid positions
        positions = []
        agent_count = 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if agent_count >= num_agents:
                    break
                    
                x = self.offset[0] + col * spacing_x
                y = self.offset[1] + row * spacing_y
                positions.append([x, y])
                agent_count += 1
            
            if agent_count >= num_agents:
                break
        
        return np.array(positions, dtype=np.float32)
    
    def validate_domain(self, positions: np.ndarray) -> bool:
        """
        Validate that grid positions are within domain boundaries.
        
        Args:
            positions: Agent positions to validate with shape (n_agents, 2)
            
        Returns:
            bool: True if all positions are within bounds, False otherwise
        """
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        # Check domain bounds
        x_valid = np.all(
            (positions[:, 0] >= 0) & 
            (positions[:, 0] <= self.domain_bounds[0])
        )
        y_valid = np.all(
            (positions[:, 1] >= 0) & 
            (positions[:, 1] <= self.domain_bounds[1])
        )
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset grid initializer (no random state to reset).
        
        Args:
            seed: Unused for deterministic grid initialization
        """
        # Grid initialization is deterministic, no state to reset
        pass
    
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        return "grid"
    
    def set_grid_parameters(
        self, 
        grid_size: Optional[Tuple[int, int]] = None,
        spacing: Optional[Tuple[float, float]] = None
    ) -> None:
        """Update grid layout parameters."""
        if grid_size:
            if grid_size[0] <= 0 or grid_size[1] <= 0:
                raise ValueError(f"Grid size must be positive: {grid_size}")
            self.grid_size = grid_size
        
        if spacing:
            if spacing[0] <= 0 or spacing[1] <= 0:
                raise ValueError(f"Grid spacing must be positive: {spacing}")
            self.spacing = spacing
    
    def calculate_grid_spacing(self, grid_cols: int, grid_rows: int) -> Tuple[float, float]:
        """
        Calculate optimal grid spacing for given dimensions.
        
        Args:
            grid_cols: Number of grid columns
            grid_rows: Number of grid rows
            
        Returns:
            Tuple[float, float]: Grid spacing as (dx, dy)
        """
        return self._calculate_grid_spacing(grid_cols, grid_rows)
    
    def _calculate_grid_dimensions(
        self, 
        num_agents: int, 
        **kwargs: Any
    ) -> Tuple[int, int]:
        """Calculate optimal grid dimensions for agent count."""
        # Check for explicit override
        if 'grid_size_override' in kwargs:
            return kwargs['grid_size_override']
        
        # Use configured grid size if available
        if self.grid_size:
            return self.grid_size
        
        # Calculate square-ish grid dimensions
        grid_cols = int(np.ceil(np.sqrt(num_agents)))
        grid_rows = int(np.ceil(num_agents / grid_cols))
        
        return grid_cols, grid_rows
    
    def _calculate_grid_spacing(
        self, 
        grid_cols: int, 
        grid_rows: int
    ) -> Tuple[float, float]:
        """Calculate grid spacing to fit domain bounds."""
        if self.spacing:
            return self.spacing
        
        # Calculate spacing to fit domain with margin
        spacing_x = self.domain_bounds[0] / max(1, grid_cols - 1) if grid_cols > 1 else 0
        spacing_y = self.domain_bounds[1] / max(1, grid_rows - 1) if grid_rows > 1 else 0
        
        return spacing_x, spacing_y


class FixedListInitializer:
    """
    Fixed list agent initialization from predefined position configurations.
    
    This strategy uses predetermined agent positions from configuration data,
    enabling precise control over initial spatial arrangements for specific
    experimental scenarios and reproducible test conditions.
    
    Supports position lists in various formats (lists, arrays, tuples) with
    automatic validation and type conversion for seamless integration.
    
    Performance Characteristics:
    - Time complexity: O(n) where n is number of agents  
    - Memory complexity: O(n) for position storage
    - Target latency: <1ms for 100 agents
    
    Examples:
        From list of coordinates:
        >>> positions_list = [[10, 20], [30, 40], [50, 60]]
        >>> initializer = FixedListInitializer(positions=positions_list)
        >>> positions = initializer.initialize_positions(num_agents=3)
        
        From NumPy array:
        >>> positions_array = np.array([[0, 0], [50, 50], [100, 100]])
        >>> initializer = FixedListInitializer(positions=positions_array)
        >>> positions = initializer.initialize_positions(num_agents=3)
    """
    
    def __init__(
        self,
        positions: Union[List[List[float]], List[Tuple[float, float]], np.ndarray],
        domain_bounds: Tuple[float, float] = (100.0, 100.0),
        cycling_enabled: bool = True
    ):
        """
        Initialize fixed list strategy with predefined positions.
        
        Args:
            positions: Predefined agent positions as list or array
            domain_bounds: Domain dimensions for validation
            cycling_enabled: Whether to cycle through positions for excess agents
        """
        self.domain_bounds = domain_bounds
        self.cycling_enabled = cycling_enabled
        
        # Convert and validate positions
        self.positions = self._validate_and_convert_positions(positions)
        
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """
        Generate positions from predefined list.
        
        Args:
            num_agents: Number of agent positions to generate
            **kwargs: Additional parameters (unused for fixed list)
            
        Returns:
            np.ndarray: Fixed positions with shape (num_agents, 2)
                Uses cycling if num_agents > available positions
                
        Raises:
            ValueError: If num_agents is invalid or insufficient positions
        """
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive: {num_agents}")
        
        if len(self.positions) == 0:
            raise ValueError("No positions available in fixed list")
        
        if num_agents > len(self.positions) and not self.cycling_enabled:
            raise ValueError(
                f"Requested {num_agents} agents but only {len(self.positions)} "
                f"positions available and cycling is disabled"
            )
        
        # Select positions (with cycling if needed)
        if num_agents <= len(self.positions):
            selected_positions = self.positions[:num_agents]
        else:
            # Cycle through available positions
            selected_positions = []
            for i in range(num_agents):
                pos_index = i % len(self.positions)
                selected_positions.append(self.positions[pos_index])
            selected_positions = np.array(selected_positions, dtype=np.float32)
        
        return selected_positions.copy()
    
    def validate_domain(self, positions: np.ndarray) -> bool:
        """
        Validate that fixed positions are within domain boundaries.
        
        Args:
            positions: Agent positions to validate with shape (n_agents, 2)
            
        Returns:
            bool: True if all positions are within bounds, False otherwise
        """
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        # Check domain bounds
        x_valid = np.all(
            (positions[:, 0] >= 0) & 
            (positions[:, 0] <= self.domain_bounds[0])
        )
        y_valid = np.all(
            (positions[:, 1] >= 0) & 
            (positions[:, 1] <= self.domain_bounds[1])
        )
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset fixed list initializer (no state to reset).
        
        Args:
            seed: Unused for deterministic fixed list initialization
        """
        # Fixed list initialization is deterministic, no state to reset
        pass
    
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        return "fixed_list"
    
    def set_positions(
        self, 
        positions: Union[List[List[float]], List[Tuple[float, float]], np.ndarray]
    ) -> None:
        """Update predefined positions list."""
        self.positions = self._validate_and_convert_positions(positions)
    
    def get_position_count(self) -> int:
        """Get number of available predefined positions."""
        return len(self.positions)
    
    def _validate_and_convert_positions(
        self, 
        positions: Union[List[List[float]], List[Tuple[float, float]], np.ndarray]
    ) -> np.ndarray:
        """Validate and convert positions to standardized format."""
        if isinstance(positions, np.ndarray):
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError(
                    f"Position array must have shape (n, 2), got {positions.shape}"
                )
            converted = positions.astype(np.float32)
        elif isinstance(positions, (list, tuple)):
            try:
                converted = np.array(positions, dtype=np.float32)
                if converted.ndim != 2 or converted.shape[1] != 2:
                    raise ValueError(
                        f"Position list must contain (x, y) pairs, "
                        f"got shape {converted.shape}"
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid position format: {e}")
        else:
            raise TypeError(
                f"Positions must be list, tuple, or numpy array, "
                f"got {type(positions)}"
            )
        
        # Validate that positions contain valid numbers
        if np.any(np.isnan(converted)) or np.any(np.isinf(converted)):
            raise ValueError("Positions contain invalid values (NaN or Inf)")
        
        return converted


class FromDatasetInitializer:
    """
    Dataset-driven agent initialization from experimental data files.
    
    This strategy loads agent positions from external datasets (CSV, JSON, etc.)
    enabling initialization from real experimental data or pre-computed position
    sets for systematic comparison studies.
    
    Supports multiple file formats with flexible column mapping and data filtering
    capabilities for integration with diverse experimental workflows.
    
    Performance Characteristics:
    - Time complexity: O(n) where n is dataset size
    - Memory complexity: O(n) for cached dataset
    - Target latency: <1ms for 100 agents (after initial load)
    
    Examples:
        From CSV file:
        >>> initializer = FromDatasetInitializer(
        ...     dataset_path="experiments/initial_positions.csv",
        ...     x_column="x_pos", 
        ...     y_column="y_pos"
        ... )
        >>> positions = initializer.initialize_positions(num_agents=50)
        
        From JSON with sampling:
        >>> initializer = FromDatasetInitializer(
        ...     dataset_path="data/positions.json",
        ...     sampling_mode="random",
        ...     seed=42
        ... )
        >>> positions = initializer.initialize_positions(num_agents=25)
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        x_column: str = "x",
        y_column: str = "y", 
        domain_bounds: Tuple[float, float] = (100.0, 100.0),
        sampling_mode: str = "sequential",
        seed: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dataset strategy with file and column configuration.
        
        Args:
            dataset_path: Path to dataset file (CSV, JSON, etc.)
            x_column: Column name for x-coordinates
            y_column: Column name for y-coordinates  
            domain_bounds: Domain dimensions for validation
            sampling_mode: How to sample positions ("sequential", "random", "stratified")
            seed: Random seed for sampling reproducibility
            filter_conditions: Optional data filtering conditions
        """
        self.dataset_path = Path(dataset_path)
        self.x_column = x_column
        self.y_column = y_column
        self.domain_bounds = domain_bounds
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.filter_conditions = filter_conditions or {}
        
        # Initialize random number generator
        self._rng = np.random.RandomState(seed)
        
        # Cache for loaded dataset
        self._dataset: Optional[pd.DataFrame] = None
        self._positions_cache: Optional[np.ndarray] = None
        
        # Validate configuration
        self._validate_configuration()
        
        # Load dataset if pandas is available
        try:
            self._load_dataset()
        except ImportError:
            warnings.warn(
                "pandas not available - dataset loading will be delayed until first use",
                UserWarning
            )
    
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """
        Generate positions from dataset with configured sampling strategy.
        
        Args:
            num_agents: Number of agent positions to generate
            **kwargs: Additional parameters (e.g., 'sampling_override')
            
        Returns:
            np.ndarray: Dataset positions with shape (num_agents, 2)
                Sampled according to configured sampling mode
                
        Raises:
            ValueError: If num_agents is invalid or dataset insufficient
            ImportError: If pandas is not available for dataset loading
        """
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive: {num_agents}")
        
        # Ensure dataset is loaded
        if self._positions_cache is None:
            self._load_dataset()
        
        if len(self._positions_cache) == 0:
            raise ValueError(f"No valid positions found in dataset: {self.dataset_path}")
        
        if num_agents > len(self._positions_cache):
            warnings.warn(
                f"Requested {num_agents} agents but dataset only contains "
                f"{len(self._positions_cache)} positions. Will cycle through dataset.",
                UserWarning
            )
        
        # Apply sampling strategy
        sampling_mode = kwargs.get('sampling_override', self.sampling_mode)
        
        if sampling_mode == "sequential":
            indices = np.arange(num_agents) % len(self._positions_cache)
        elif sampling_mode == "random":
            indices = self._rng.choice(
                len(self._positions_cache), 
                size=num_agents, 
                replace=True
            )
        elif sampling_mode == "stratified":
            # Implement stratified sampling (simplified version)
            indices = self._stratified_sampling(num_agents)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        
        return self._positions_cache[indices].copy()
    
    def validate_domain(self, positions: np.ndarray) -> bool:
        """
        Validate that dataset positions are within domain boundaries.
        
        Args:
            positions: Agent positions to validate with shape (n_agents, 2)
            
        Returns:
            bool: True if all positions are within bounds, False otherwise
        """
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        # Check domain bounds
        x_valid = np.all(
            (positions[:, 0] >= 0) & 
            (positions[:, 0] <= self.domain_bounds[0])
        )
        y_valid = np.all(
            (positions[:, 1] >= 0) & 
            (positions[:, 1] <= self.domain_bounds[1])
        )
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset dataset sampler with optional new seed.
        
        Args:
            seed: New random seed (uses instance seed if None)
        """
        reset_seed = seed if seed is not None else self.seed
        self._rng = np.random.RandomState(reset_seed)
    
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        return "from_dataset"
    
    def load_dataset(self, force_reload: bool = False) -> None:
        """
        Load or reload dataset from file.
        
        Args:
            force_reload: Whether to force reloading even if cached
        """
        if force_reload or self._dataset is None:
            self._load_dataset()
    
    def set_sampling_mode(self, mode: str) -> None:
        """
        Update sampling strategy.
        
        Args:
            mode: New sampling mode ("sequential", "random", "stratified")
        """
        valid_modes = ["sequential", "random", "stratified"]
        if mode not in valid_modes:
            raise ValueError(f"Sampling mode must be one of {valid_modes}, got {mode}")
        self.sampling_mode = mode
    
    def _validate_configuration(self) -> None:
        """Validate initialization configuration parameters."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        if self.sampling_mode not in ["sequential", "random", "stratified"]:
            raise ValueError(f"Invalid sampling mode: {self.sampling_mode}")
        
        if self.domain_bounds[0] <= 0 or self.domain_bounds[1] <= 0:
            raise ValueError(f"Domain bounds must be positive: {self.domain_bounds}")
    
    def _load_dataset(self) -> None:
        """Load dataset from file and extract position data."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for dataset loading. "
                "Install with: pip install pandas>=1.5.0"
            )
        
        # Load data based on file extension
        file_suffix = self.dataset_path.suffix.lower()
        
        if file_suffix == '.csv':
            self._dataset = pd.read_csv(self.dataset_path)
        elif file_suffix == '.json':
            self._dataset = pd.read_json(self.dataset_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_suffix}. "
                f"Supported formats: .csv, .json"
            )
        
        # Validate required columns exist
        if self.x_column not in self._dataset.columns:
            raise ValueError(f"X column '{self.x_column}' not found in dataset")
        if self.y_column not in self._dataset.columns:
            raise ValueError(f"Y column '{self.y_column}' not found in dataset")
        
        # Apply filtering conditions if specified
        filtered_data = self._dataset
        for column, condition in self.filter_conditions.items():
            if column in filtered_data.columns:
                if isinstance(condition, dict):
                    # Handle range conditions
                    if 'min' in condition:
                        filtered_data = filtered_data[
                            filtered_data[column] >= condition['min']
                        ]
                    if 'max' in condition:
                        filtered_data = filtered_data[
                            filtered_data[column] <= condition['max']
                        ]
                else:
                    # Handle equality conditions
                    filtered_data = filtered_data[filtered_data[column] == condition]
        
        # Extract position data
        x_data = filtered_data[self.x_column].values
        y_data = filtered_data[self.y_column].values
        
        # Validate position data
        if len(x_data) == 0:
            raise ValueError("No data remaining after filtering")
        
        # Remove invalid entries (NaN, Inf)
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) == 0:
            raise ValueError("No valid positions found in dataset")
        
        # Store positions cache
        self._positions_cache = np.column_stack((x_data, y_data)).astype(np.float32)
    
    def _stratified_sampling(self, num_agents: int) -> np.ndarray:
        """
        Implement stratified sampling for representative position selection.
        
        Args:
            num_agents: Number of agents to sample
            
        Returns:
            np.ndarray: Stratified sample indices
        """
        # Simple stratified sampling based on spatial bins
        if self._positions_cache is None or len(self._positions_cache) == 0:
            return np.array([], dtype=np.int32)
        
        # Create spatial bins
        n_bins = min(10, int(np.sqrt(len(self._positions_cache))))
        x_bins = np.linspace(
            self._positions_cache[:, 0].min(),
            self._positions_cache[:, 0].max(),
            n_bins + 1
        )
        y_bins = np.linspace(
            self._positions_cache[:, 1].min(),
            self._positions_cache[:, 1].max(),
            n_bins + 1
        )
        
        # Assign positions to bins
        x_indices = np.digitize(self._positions_cache[:, 0], x_bins) - 1
        y_indices = np.digitize(self._positions_cache[:, 1], y_bins) - 1
        
        # Sample from each bin proportionally
        bin_indices = x_indices * n_bins + y_indices
        unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
        
        selected_indices = []
        remaining_agents = num_agents
        
        for bin_id, count in zip(unique_bins, bin_counts):
            if remaining_agents <= 0:
                break
            
            # Calculate samples for this bin
            bin_mask = bin_indices == bin_id
            bin_positions = np.where(bin_mask)[0]
            
            samples_per_bin = max(1, int(remaining_agents * count / len(self._positions_cache)))
            samples_per_bin = min(samples_per_bin, len(bin_positions), remaining_agents)
            
            # Sample from this bin
            if samples_per_bin > 0:
                sampled = self._rng.choice(
                    bin_positions, 
                    size=samples_per_bin, 
                    replace=False
                )
                selected_indices.extend(sampled)
                remaining_agents -= samples_per_bin
        
        # Fill remaining slots with random sampling if needed
        if remaining_agents > 0:
            available_indices = np.setdiff1d(
                np.arange(len(self._positions_cache)), 
                selected_indices
            )
            if len(available_indices) > 0:
                additional = self._rng.choice(
                    available_indices,
                    size=min(remaining_agents, len(available_indices)),
                    replace=False
                )
                selected_indices.extend(additional)
        
        return np.array(selected_indices[:num_agents], dtype=np.int32)


def create_agent_initializer(config: Dict[str, Any]) -> AgentInitializerProtocol:
    """
    Factory function to create agent initializer from configuration.
    
    This function provides a centralized entry point for creating initialization
    strategies based on configuration dictionaries, supporting both explicit
    class instantiation and Hydra-based dependency injection.
    
    The factory handles type validation, parameter checking, and error handling
    to ensure robust initialization strategy creation across different usage patterns.
    
    Args:
        config: Configuration dictionary containing initializer type and parameters.
            Must include either '_target_' key for Hydra instantiation or 'type' key
            for factory-based creation. Additional keys are passed as initialization
            parameters.
            
    Returns:
        AgentInitializerProtocol: Configured initializer instance implementing the
            protocol interface with all specified parameters applied.
            
    Raises:
        ValueError: If configuration is invalid or initializer type unknown
        TypeError: If configuration format is incorrect
        ImportError: If required dependencies are missing (e.g., pandas for dataset)
        
    Examples:
        Uniform random initializer:
        >>> config = {
        ...     'type': 'uniform_random',
        ...     'bounds': (100, 100),
        ...     'seed': 42
        ... }
        >>> initializer = create_agent_initializer(config)
        
        Grid initializer with Hydra target:
        >>> config = {
        ...     '_target_': 'plume_nav_sim.core.initialization.GridInitializer',
        ...     'domain_bounds': (200, 150),
        ...     'grid_size': (10, 5)
        ... }
        >>> initializer = create_agent_initializer(config)
        
        Dataset initializer:
        >>> config = {
        ...     'type': 'from_dataset',
        ...     'dataset_path': 'data/positions.csv',
        ...     'sampling_mode': 'random',
        ...     'seed': 123
        ... }
        >>> initializer = create_agent_initializer(config)
    """
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be dict, got {type(config)}")
    
    if '_target_' in config:
        # Hydra-style instantiation
        try:
            # Import hydra utils for instantiation
            from hydra.utils import instantiate
            return instantiate(config)
        except ImportError:
            # Fallback to manual instantiation if Hydra not available
            target_class = config['_target_']
            if target_class.endswith('UniformRandomInitializer'):
                init_class = UniformRandomInitializer
            elif target_class.endswith('GridInitializer'):
                init_class = GridInitializer
            elif target_class.endswith('FixedListInitializer'):
                init_class = FixedListInitializer
            elif target_class.endswith('FromDatasetInitializer'):
                init_class = FromDatasetInitializer
            else:
                raise ValueError(f"Unknown initializer target: {target_class}")
            
            # Extract parameters (excluding _target_)
            params = {k: v for k, v in config.items() if k != '_target_'}
            return init_class(**params)
    
    elif 'type' in config:
        # Factory-style instantiation
        initializer_type = config['type']
        params = {k: v for k, v in config.items() if k != 'type'}
        
        if initializer_type in ['uniform_random', 'UniformRandomInitializer']:
            return UniformRandomInitializer(**params)
        elif initializer_type in ['grid', 'GridInitializer']:
            return GridInitializer(**params)
        elif initializer_type in ['fixed_list', 'FixedListInitializer']:
            return FixedListInitializer(**params)
        elif initializer_type in ['from_dataset', 'FromDatasetInitializer']:
            return FromDatasetInitializer(**params)
        else:
            raise ValueError(
                f"Unknown initializer type: {initializer_type}. "
                f"Supported types: uniform_random, grid, fixed_list, from_dataset"
            )
    
    else:
        raise ValueError(
            "Configuration must contain either '_target_' or 'type' key. "
            f"Got keys: {list(config.keys())}"
        )


# Export public interface
__all__ = [
    # Core protocol interface
    "AgentInitializerProtocol",
    
    # Concrete initializer implementations  
    "UniformRandomInitializer",
    "GridInitializer", 
    "FixedListInitializer",
    "FromDatasetInitializer",
    
    # Factory function
    "create_agent_initializer",
    
    # Configuration support
    "InitializationConfig",
]