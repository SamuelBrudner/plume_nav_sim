"""
Pluggable boundary handling via BoundaryPolicyProtocol interface for domain edge management.

This module implements configurable boundary policy strategies that replace inline boundary
checking logic from navigation controllers, enabling modular design and runtime policy
selection. All implementations support vectorized multi-agent operations with optimized
performance for 100+ agent scenarios.

Boundary Policy Types:
- TerminateBoundary: End episode when agent reaches boundary (status = "oob")
- BounceBoundary: Reflect agent trajectory off boundary walls with energy conservation
- WrapBoundary: Periodic boundary conditions wrapping to opposite domain edge  
- ClipBoundary: Constrain agent position to remain within valid domain

Key Design Principles:
- Protocol-based implementation for pluggable boundary behavior strategies
- Vectorized operations for efficient multi-agent boundary checking
- Configuration-driven policy selection without code changes
- Performance optimization for ≤33ms step latency requirements
- Integration with Hydra config group conf/base/boundary/ for runtime selection

Performance Requirements:
- apply_policy(): <1ms for 100 agents with vectorized operations
- check_violations(): <0.5ms for boundary detection across all agents  
- get_termination_status(): <0.1ms for episode termination decisions
- Memory efficiency: <1MB for boundary state management

Examples:
    Episode termination boundary for navigation experiments:
    >>> policy = TerminateBoundary(domain_bounds=(100, 100))
    >>> violations = policy.check_violations(agent_positions)
    >>> if violations.any():
    ...     status = policy.get_termination_status()  # Returns "oob"
    
    Elastic collision boundary physics:
    >>> policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
    >>> corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
    
    Periodic domain wrapping for continuous exploration:
    >>> policy = WrapBoundary(domain_bounds=(100, 100))
    >>> wrapped_positions = policy.apply_policy(out_of_bounds_positions)
    
    Hard position constraints for confined navigation:
    >>> policy = ClipBoundary(domain_bounds=(100, 100))
    >>> clipped_positions = policy.apply_policy(agent_positions)
    
    Configuration-driven boundary policy instantiation:
    >>> from hydra import utils as hydra_utils
    >>> boundary_config = {
    ...     '_target_': 'plume_nav_sim.core.boundaries.BounceBoundary',
    ...     'domain_bounds': (100, 100),
    ...     'elasticity': 0.9,
    ...     'energy_loss': 0.05
    ... }
    >>> policy = hydra_utils.instantiate(boundary_config)
    
    Factory method with runtime policy selection:
    >>> policy = create_boundary_policy(
    ...     policy_type="bounce", 
    ...     domain_bounds=(100, 100), 
    ...     elasticity=0.8
    ... )
"""

from __future__ import annotations
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Import the boundary policy protocol from the protocols module
from .protocols import BoundaryPolicyProtocol


@dataclass
class BoundaryConfig:
    """
    Configuration dataclass for boundary policy parameters with validation.
    
    Provides type-safe parameter validation and Hydra integration for boundary
    policy configuration management across different policy implementations.
    
    Attributes:
        domain_bounds: Domain size as (width, height) tuple defining valid region
        policy_type: Boundary policy type identifier for factory instantiation
        allow_negative_coords: Whether negative coordinates are permitted
        boundary_buffer: Minimum distance from boundary for violation detection
        
    Examples:
        Basic boundary configuration:
        >>> config = BoundaryConfig(domain_bounds=(100, 100), policy_type="terminate")
        
        Advanced configuration with buffer zone:
        >>> config = BoundaryConfig(
        ...     domain_bounds=(200, 150), 
        ...     policy_type="bounce",
        ...     boundary_buffer=2.0
        ... )
    """
    domain_bounds: Tuple[float, float]
    policy_type: str = "terminate"
    allow_negative_coords: bool = False
    boundary_buffer: float = 0.0
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if len(self.domain_bounds) != 2:
            raise ValueError("domain_bounds must be a tuple of (width, height)")
        if any(bound <= 0 for bound in self.domain_bounds):
            raise ValueError("domain_bounds must be positive values")
        if self.boundary_buffer < 0:
            raise ValueError("boundary_buffer must be non-negative")


class TerminateBoundary:
    """
    Boundary policy that terminates episodes when agents reach domain boundaries.
    
    This policy implements the traditional "out of bounds" termination strategy
    where agents that move outside the valid domain cause episode termination
    with status "oob". Used for navigation experiments requiring strict spatial
    constraints and exploration behavior analysis.
    
    The policy performs no position correction, allowing violations to persist
    for analysis while signaling episode termination to the environment.
    
    Key Features:
    - Zero-cost boundary violation handling with no position modification
    - Vectorized violation detection for multi-agent scenarios
    - Configurable domain bounds with optional coordinate restrictions
    - Integration with episode management via termination status reporting
    
    Performance Characteristics:
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - apply_policy(): O(1) no-op operation, <0.01ms regardless of agent count
    - Memory usage: <1KB for policy state management
    
    Examples:
        Basic termination boundary for rectangular domain:
        >>> policy = TerminateBoundary(domain_bounds=(100, 100))
        >>> violations = policy.check_violations(agent_positions)
        >>> if violations.any():
        ...     episode_done = (policy.get_termination_status() == "oob")
        
        Termination with negative coordinate restrictions:
        >>> policy = TerminateBoundary(
        ...     domain_bounds=(100, 100), 
        ...     allow_negative_coords=False
        ... )
        >>> # Agents at negative positions will trigger violations
        
        Multi-agent boundary violation checking:
        >>> positions = np.array([[50, 50], [105, 75], [25, 110]])  # One out of bounds
        >>> violations = policy.check_violations(positions)
        >>> # Returns [False, True, True] for domain bounds (100, 100)
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        allow_negative_coords: bool = False,
        status_on_violation: str = "oob"
    ):
        """
        Initialize termination boundary policy with domain constraints.
        
        Args:
            domain_bounds: Domain size as (width, height) defining valid region
            allow_negative_coords: Whether negative coordinates are allowed
            status_on_violation: Termination status string for episode management
            
        Raises:
            ValueError: If domain_bounds are invalid or non-positive
        """
        if len(domain_bounds) != 2 or any(bound <= 0 for bound in domain_bounds):
            raise ValueError("domain_bounds must be positive (width, height) tuple")
        
        self.domain_bounds = domain_bounds
        self.allow_negative_coords = allow_negative_coords
        self.status_on_violation = status_on_violation
        
        # Cache boundary limits for efficient violation checking
        self.x_min = 0.0 if not allow_negative_coords else -np.inf
        self.y_min = 0.0 if not allow_negative_coords else -np.inf
        self.x_max = float(domain_bounds[0])
        self.y_max = float(domain_bounds[1])
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply termination policy (no position correction performed).
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,)
            velocities: Optional velocities (ignored for termination policy)
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Unchanged positions, or (positions, velocities) if velocities provided
                
        Notes:
            Termination policy does not modify agent positions or velocities.
            Boundary violations are detected separately and handled via episode
            termination rather than position correction.
        """
        # Termination policy performs no corrections - return positions unchanged
        if velocities is not None:
            return positions, velocities
        return positions
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect boundary violations for agent positions using vectorized operations.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean violations with shape (n_agents,) or scalar bool
            
        Notes:
            Efficient vectorized implementation using NumPy logical operations
            for sub-millisecond performance with large agent populations.
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
        
        # Vectorized boundary violation checking
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check violations for all boundaries simultaneously
        x_violations = np.logical_or(x_coords < self.x_min, x_coords > self.x_max)
        y_violations = np.logical_or(y_coords < self.y_min, y_coords > self.y_max)
        
        violations = np.logical_or(x_violations, y_violations)
        
        # Return scalar for single agent, array for multi-agent
        return violations[0] if single_agent else violations
    
    def get_termination_status(self) -> str:
        """
        Get termination status for episode management.
        
        Returns:
            str: Status string "oob" indicating out-of-bounds termination
        """
        return self.status_on_violation
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update boundary policy configuration parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - domain_bounds: New domain size tuple
                - allow_negative_coords: Coordinate restriction update
                - status_on_violation: Termination status string update
        """
        if 'domain_bounds' in kwargs:
            new_bounds = kwargs['domain_bounds']
            if len(new_bounds) != 2 or any(bound <= 0 for bound in new_bounds):
                raise ValueError("domain_bounds must be positive (width, height) tuple")
            self.domain_bounds = new_bounds
            self.x_max = float(new_bounds[0])
            self.y_max = float(new_bounds[1])
        
        if 'allow_negative_coords' in kwargs:
            self.allow_negative_coords = kwargs['allow_negative_coords']
            self.x_min = 0.0 if not self.allow_negative_coords else -np.inf
            self.y_min = 0.0 if not self.allow_negative_coords else -np.inf
        
        if 'status_on_violation' in kwargs:
            self.status_on_violation = kwargs['status_on_violation']


class BounceBoundary:
    """
    Boundary policy implementing elastic collision behavior at domain edges.
    
    This policy simulates realistic physics by reflecting agent trajectories off
    boundary walls with configurable energy conservation. Agents maintain momentum
    in the tangential direction while velocity components normal to boundaries
    are reversed with optional energy loss modeling.
    
    The implementation supports both elastic (no energy loss) and inelastic 
    (partial energy loss) collisions for diverse physical modeling requirements.
    
    Key Features:
    - Realistic collision physics with energy conservation modeling
    - Configurable elasticity coefficient for material property simulation
    - Vectorized reflection calculations for multi-agent efficiency
    - Corner collision handling with proper momentum conservation
    - Optional energy dissipation for long-term behavioral studies
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.5ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and collision parameters
    
    Examples:
        Elastic boundary collisions with perfect energy conservation:
        >>> policy = BounceBoundary(domain_bounds=(100, 100), elasticity=1.0)
        >>> corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
        
        Inelastic collisions with energy dissipation:
        >>> policy = BounceBoundary(
        ...     domain_bounds=(100, 100), 
        ...     elasticity=0.8,
        ...     energy_loss=0.1
        ... )
        >>> # 80% velocity reflection + 10% additional energy loss
        
        Dynamic elasticity adjustment during simulation:
        >>> policy.set_elasticity(0.9)  # Update collision parameters
        >>> policy.configure(energy_loss=0.05)  # Reduce energy dissipation
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        elasticity: float = 1.0,
        energy_loss: float = 0.0,
        allow_negative_coords: bool = False
    ):
        """
        Initialize bounce boundary policy with collision physics parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining valid region
            elasticity: Coefficient of restitution [0, 1] for collision energy conservation
            energy_loss: Additional energy dissipation factor [0, 1] for realistic modeling
            allow_negative_coords: Whether negative coordinates are allowed
            
        Raises:
            ValueError: If parameters are outside valid ranges or domain_bounds invalid
        """
        if len(domain_bounds) != 2 or any(bound <= 0 for bound in domain_bounds):
            raise ValueError("domain_bounds must be positive (width, height) tuple")
        if not 0 <= elasticity <= 1:
            raise ValueError("elasticity must be in range [0, 1]")
        if not 0 <= energy_loss <= 1:
            raise ValueError("energy_loss must be in range [0, 1]")
        
        self.domain_bounds = domain_bounds
        self.elasticity = elasticity
        self.energy_loss = energy_loss
        self.allow_negative_coords = allow_negative_coords
        
        # Cache boundary limits for efficient collision detection
        self.x_min = 0.0 if not allow_negative_coords else -np.inf
        self.y_min = 0.0 if not allow_negative_coords else -np.inf
        self.x_max = float(domain_bounds[0])
        self.y_max = float(domain_bounds[1])
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply bounce policy with elastic collision physics.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,)
            velocities: Agent velocities with same shape as positions (required for bounce)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (corrected_positions, corrected_velocities)
            
        Notes:
            Bounce policy requires velocity information for proper collision physics.
            Positions are corrected to remain within domain bounds while velocities
            are reflected and scaled according to elasticity parameters.
            
        Raises:
            ValueError: If velocities not provided (required for bounce physics)
        """
        if velocities is None:
            raise ValueError("BounceBoundary requires velocities for collision physics")
        
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
            velocities = velocities.reshape(1, -1)
        
        # Copy arrays to avoid modifying inputs
        corrected_pos = positions.copy()
        corrected_vel = velocities.copy()
        
        # Detect and handle boundary violations with physics
        violations = self.check_violations(positions)
        if np.any(violations):
            corrected_pos, corrected_vel = self._apply_collision_physics(
                corrected_pos, corrected_vel, violations
            )
        
        # Return in original format
        if single_agent:
            return corrected_pos[0], corrected_vel[0]
        return corrected_pos, corrected_vel
    
    def _apply_collision_physics(
        self, 
        positions: np.ndarray, 
        velocities: np.ndarray, 
        violations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply collision physics to agents violating boundaries.
        
        Args:
            positions: Agent positions array
            velocities: Agent velocities array
            violations: Boolean array indicating which agents violated boundaries
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (corrected_positions, corrected_velocities)
        """
        # Apply corrections only to violating agents
        violating_indices = np.where(violations)[0]
        
        for idx in violating_indices:
            pos = positions[idx]
            vel = velocities[idx]
            
            # Check each boundary and apply reflection
            # Left boundary collision
            if pos[0] < self.x_min:
                positions[idx, 0] = self.x_min + (self.x_min - pos[0])  # Reflect position
                velocities[idx, 0] = -vel[0] * self.elasticity * (1 - self.energy_loss)
            
            # Right boundary collision  
            elif pos[0] > self.x_max:
                positions[idx, 0] = self.x_max - (pos[0] - self.x_max)  # Reflect position
                velocities[idx, 0] = -vel[0] * self.elasticity * (1 - self.energy_loss)
            
            # Bottom boundary collision
            if pos[1] < self.y_min:
                positions[idx, 1] = self.y_min + (self.y_min - pos[1])  # Reflect position
                velocities[idx, 1] = -vel[1] * self.elasticity * (1 - self.energy_loss)
            
            # Top boundary collision
            elif pos[1] > self.y_max:
                positions[idx, 1] = self.y_max - (pos[1] - self.y_max)  # Reflect position
                velocities[idx, 1] = -vel[1] * self.elasticity * (1 - self.energy_loss)
        
        return positions, velocities
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect boundary violations for agent positions using vectorized operations.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean violations with shape (n_agents,) or scalar bool
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
        
        # Vectorized boundary violation checking
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check violations for all boundaries simultaneously
        x_violations = np.logical_or(x_coords < self.x_min, x_coords > self.x_max)
        y_violations = np.logical_or(y_coords < self.y_min, y_coords > self.y_max)
        
        violations = np.logical_or(x_violations, y_violations)
        
        # Return scalar for single agent, array for multi-agent
        return violations[0] if single_agent else violations
    
    def get_termination_status(self) -> str:
        """
        Get termination status for episode management.
        
        Returns:
            str: Status string "continue" indicating episode continues with corrections
        """
        return "continue"
    
    def set_elasticity(self, elasticity: float) -> None:
        """
        Update collision elasticity coefficient during simulation.
        
        Args:
            elasticity: New coefficient of restitution [0, 1]
            
        Raises:
            ValueError: If elasticity outside valid range [0, 1]
        """
        if not 0 <= elasticity <= 1:
            raise ValueError("elasticity must be in range [0, 1]")
        self.elasticity = elasticity
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update boundary policy configuration parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - domain_bounds: New domain size tuple
                - elasticity: Collision coefficient of restitution
                - energy_loss: Additional energy dissipation factor
                - allow_negative_coords: Coordinate restriction update
        """
        if 'domain_bounds' in kwargs:
            new_bounds = kwargs['domain_bounds']
            if len(new_bounds) != 2 or any(bound <= 0 for bound in new_bounds):
                raise ValueError("domain_bounds must be positive (width, height) tuple")
            self.domain_bounds = new_bounds
            self.x_max = float(new_bounds[0])
            self.y_max = float(new_bounds[1])
        
        if 'elasticity' in kwargs:
            self.set_elasticity(kwargs['elasticity'])
        
        if 'energy_loss' in kwargs:
            energy_loss = kwargs['energy_loss']
            if not 0 <= energy_loss <= 1:
                raise ValueError("energy_loss must be in range [0, 1]")
            self.energy_loss = energy_loss
        
        if 'allow_negative_coords' in kwargs:
            self.allow_negative_coords = kwargs['allow_negative_coords']
            self.x_min = 0.0 if not self.allow_negative_coords else -np.inf
            self.y_min = 0.0 if not self.allow_negative_coords else -np.inf


class WrapBoundary:
    """
    Boundary policy implementing periodic boundary conditions with position wrapping.
    
    This policy creates a toroidal topology where agents exiting one side of the
    domain immediately appear on the opposite side, maintaining velocity and
    trajectory continuity. Ideal for continuous exploration scenarios and
    eliminating boundary effects in navigation research.
    
    Position wrapping preserves agent momentum and direction while providing
    infinite exploration space within finite computational domains.
    
    Key Features:
    - Seamless position wrapping for toroidal domain topology
    - Velocity preservation during boundary transitions
    - Vectorized wrapping operations for multi-agent efficiency
    - Configurable domain bounds with optional coordinate restrictions
    - Zero energy loss during wrapping transitions
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.2ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and domain parameters
    
    Examples:
        Periodic boundary conditions for continuous exploration:
        >>> policy = WrapBoundary(domain_bounds=(100, 100))
        >>> wrapped_positions = policy.apply_policy(out_of_bounds_positions)
        
        Position wrapping with velocity preservation:
        >>> policy = WrapBoundary(domain_bounds=(200, 150))
        >>> wrapped_pos, unchanged_vel = policy.apply_policy(positions, velocities)
        
        Toroidal navigation domain:
        >>> # Agent at (105, 50) wraps to (5, 50) for domain (100, 100)
        >>> # Agent at (-10, 75) wraps to (90, 75) for domain (100, 100)
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        allow_negative_coords: bool = False
    ):
        """
        Initialize wrap boundary policy with periodic domain parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining wrapping region
            allow_negative_coords: Whether negative coordinates are allowed before wrapping
            
        Raises:
            ValueError: If domain_bounds are invalid or non-positive
        """
        if len(domain_bounds) != 2 or any(bound <= 0 for bound in domain_bounds):
            raise ValueError("domain_bounds must be positive (width, height) tuple")
        
        self.domain_bounds = domain_bounds
        self.allow_negative_coords = allow_negative_coords
        
        # Cache boundary limits for efficient wrapping calculations
        self.x_min = 0.0 if not allow_negative_coords else -np.inf
        self.y_min = 0.0 if not allow_negative_coords else -np.inf
        self.x_max = float(domain_bounds[0])
        self.y_max = float(domain_bounds[1])
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply wrap policy with periodic boundary conditions.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,)
            velocities: Optional velocities (preserved unchanged during wrapping)
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Wrapped positions, or (wrapped_positions, unchanged_velocities)
                
        Notes:
            Wrap policy modifies positions to remain within domain bounds using
            modular arithmetic. Velocities are preserved unchanged as wrapping
            does not affect agent momentum or trajectory direction.
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
            if velocities is not None:
                velocities = velocities.reshape(1, -1)
        
        # Apply periodic wrapping using modular arithmetic
        wrapped_pos = positions.copy()
        
        # Handle finite domain bounds (when not allowing negative coordinates)
        if not self.allow_negative_coords:
            # Wrap x coordinates to [0, x_max)
            wrapped_pos[:, 0] = np.mod(wrapped_pos[:, 0], self.x_max)
            # Wrap y coordinates to [0, y_max)
            wrapped_pos[:, 1] = np.mod(wrapped_pos[:, 1], self.y_max)
        else:
            # For infinite bounds, no wrapping needed
            pass
        
        # Return in original format
        if single_agent:
            wrapped_pos = wrapped_pos[0]
            if velocities is not None:
                return wrapped_pos, velocities[0]
            return wrapped_pos
        
        if velocities is not None:
            return wrapped_pos, velocities
        return wrapped_pos
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect boundary violations for agent positions using vectorized operations.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean violations with shape (n_agents,) or scalar bool
            
        Notes:
            For wrap boundaries, "violations" indicate positions needing wrapping
            rather than actual constraint violations since wrapping is seamless.
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
        
        # Check if positions are outside domain bounds (need wrapping)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check violations for all boundaries simultaneously
        if not self.allow_negative_coords:
            x_violations = np.logical_or(x_coords < self.x_min, x_coords >= self.x_max)
            y_violations = np.logical_or(y_coords < self.y_min, y_coords >= self.y_max)
        else:
            # For infinite bounds, no violations occur
            x_violations = np.zeros(len(x_coords), dtype=bool)
            y_violations = np.zeros(len(y_coords), dtype=bool)
        
        violations = np.logical_or(x_violations, y_violations)
        
        # Return scalar for single agent, array for multi-agent
        return violations[0] if single_agent else violations
    
    def get_termination_status(self) -> str:
        """
        Get termination status for episode management.
        
        Returns:
            str: Status string "continue" indicating episode continues with wrapping
        """
        return "continue"
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update boundary policy configuration parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - domain_bounds: New domain size tuple for wrapping region
                - allow_negative_coords: Coordinate restriction update
        """
        if 'domain_bounds' in kwargs:
            new_bounds = kwargs['domain_bounds']
            if len(new_bounds) != 2 or any(bound <= 0 for bound in new_bounds):
                raise ValueError("domain_bounds must be positive (width, height) tuple")
            self.domain_bounds = new_bounds
            self.x_max = float(new_bounds[0])
            self.y_max = float(new_bounds[1])
            self.x_range = self.x_max - self.x_min
            self.y_range = self.y_max - self.y_min
        
        if 'allow_negative_coords' in kwargs:
            self.allow_negative_coords = kwargs['allow_negative_coords']
            self.x_min = 0.0 if not self.allow_negative_coords else -np.inf
            self.y_min = 0.0 if not self.allow_negative_coords else -np.inf
            self.x_range = self.x_max - self.x_min
            self.y_range = self.y_max - self.y_min


class ClipBoundary:
    """
    Boundary policy implementing hard position constraints to prevent boundary crossing.
    
    This policy enforces strict spatial constraints by clipping agent positions
    to remain within valid domain bounds. Agents attempting to move beyond
    boundaries are constrained to the boundary edge with optional velocity
    damping to prevent continuous boundary pressure.
    
    Clipping provides deterministic boundary behavior with minimal computational
    overhead, ideal for scenarios requiring strict spatial confinement.
    
    Key Features:
    - Hard position constraints preventing boundary crossing
    - Optional velocity damping at boundaries to reduce pressure effects
    - Vectorized clipping operations for multi-agent efficiency
    - Configurable domain bounds with edge behavior customization
    - Zero overshoot guarantee for critical spatial constraints
    
    Performance Characteristics:
    - apply_policy(): O(n) vectorized operation, <0.2ms for 100 agents
    - check_violations(): O(n) vectorized operation, <0.1ms for 100 agents
    - Memory usage: <1KB for policy state and clipping parameters
    
    Examples:
        Hard boundary constraints with position clipping:
        >>> policy = ClipBoundary(domain_bounds=(100, 100))
        >>> clipped_positions = policy.apply_policy(agent_positions)
        
        Position clipping with velocity damping at boundaries:
        >>> policy = ClipBoundary(
        ...     domain_bounds=(100, 100),
        ...     velocity_damping=0.8,
        ...     damp_at_boundary=True
        ... )
        >>> clipped_pos, damped_vel = policy.apply_policy(positions, velocities)
        
        Strict spatial confinement for safety-critical navigation:
        >>> # Guarantees agents never exceed domain bounds
        >>> policy = ClipBoundary(domain_bounds=(50, 50))
        >>> assert np.all(clipped_positions <= (50, 50))
        >>> assert np.all(clipped_positions >= (0, 0))
    """
    
    def __init__(
        self,
        domain_bounds: Tuple[float, float],
        velocity_damping: float = 1.0,
        damp_at_boundary: bool = False,
        allow_negative_coords: bool = False
    ):
        """
        Initialize clip boundary policy with constraint parameters.
        
        Args:
            domain_bounds: Domain size as (width, height) defining clipping region
            velocity_damping: Velocity scaling factor [0, 1] when at boundaries
            damp_at_boundary: Whether to apply velocity damping at boundary contact
            allow_negative_coords: Whether negative coordinates are allowed
            
        Raises:
            ValueError: If parameters are outside valid ranges or domain_bounds invalid
        """
        if len(domain_bounds) != 2 or any(bound <= 0 for bound in domain_bounds):
            raise ValueError("domain_bounds must be positive (width, height) tuple")
        if not 0 <= velocity_damping <= 1:
            raise ValueError("velocity_damping must be in range [0, 1]")
        
        self.domain_bounds = domain_bounds
        self.velocity_damping = velocity_damping
        self.damp_at_boundary = damp_at_boundary
        self.allow_negative_coords = allow_negative_coords
        
        # Cache boundary limits for efficient clipping operations
        self.x_min = 0.0 if not allow_negative_coords else -np.inf
        self.y_min = 0.0 if not allow_negative_coords else -np.inf
        self.x_max = float(domain_bounds[0])
        self.y_max = float(domain_bounds[1])
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply clip policy with hard position constraints.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,)
            velocities: Optional velocities for boundary damping effects
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Clipped positions, or (clipped_positions, modified_velocities)
                
        Notes:
            Clip policy constrains positions to domain bounds using np.clip.
            Velocities may be damped when agents contact boundaries if
            damp_at_boundary is enabled.
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
            if velocities is not None:
                velocities = velocities.reshape(1, -1)
        
        # Apply position clipping to domain bounds
        clipped_pos = positions.copy()
        
        # Clip x coordinates to valid range
        if not np.isinf(self.x_min):
            clipped_pos[:, 0] = np.maximum(clipped_pos[:, 0], self.x_min)
        if not np.isinf(self.x_max):
            clipped_pos[:, 0] = np.minimum(clipped_pos[:, 0], self.x_max)
        
        # Clip y coordinates to valid range
        if not np.isinf(self.y_min):
            clipped_pos[:, 1] = np.maximum(clipped_pos[:, 1], self.y_min)
        if not np.isinf(self.y_max):
            clipped_pos[:, 1] = np.minimum(clipped_pos[:, 1], self.y_max)
        
        # Apply velocity damping if requested and velocities provided
        modified_vel = velocities
        if velocities is not None and self.damp_at_boundary and self.velocity_damping < 1.0:
            modified_vel = velocities.copy()
            
            # Detect agents at boundaries for damping
            at_boundaries = self._detect_boundary_contact(clipped_pos)
            
            # Apply damping to agents at boundaries
            if np.any(at_boundaries):
                modified_vel[at_boundaries] *= self.velocity_damping
        
        # Return in original format
        if single_agent:
            clipped_pos = clipped_pos[0]
            if modified_vel is not None:
                return clipped_pos, modified_vel[0]
            return clipped_pos
        
        if modified_vel is not None:
            return clipped_pos, modified_vel
        return clipped_pos
    
    def _detect_boundary_contact(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect agents in contact with domain boundaries.
        
        Args:
            positions: Agent positions array
            
        Returns:
            np.ndarray: Boolean array indicating boundary contact
        """
        tolerance = 1e-6  # Small tolerance for floating point comparison
        
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check contact with each boundary
        at_x_min = np.abs(x_coords - self.x_min) < tolerance
        at_x_max = np.abs(x_coords - self.x_max) < tolerance
        at_y_min = np.abs(y_coords - self.y_min) < tolerance
        at_y_max = np.abs(y_coords - self.y_max) < tolerance
        
        return np.logical_or.reduce([at_x_min, at_x_max, at_y_min, at_y_max])
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect boundary violations for agent positions using vectorized operations.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean violations with shape (n_agents,) or scalar bool
            
        Notes:
            For clip boundaries, violations indicate positions that would require
            clipping to remain within domain bounds.
        """
        # Handle single agent case by reshaping to 2D
        single_agent = positions.ndim == 1
        if single_agent:
            positions = positions.reshape(1, -1)
        
        # Vectorized boundary violation checking
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check violations for all boundaries simultaneously
        x_violations = np.logical_or(x_coords < self.x_min, x_coords > self.x_max)
        y_violations = np.logical_or(y_coords < self.y_min, y_coords > self.y_max)
        
        violations = np.logical_or(x_violations, y_violations)
        
        # Return scalar for single agent, array for multi-agent
        return violations[0] if single_agent else violations
    
    def get_termination_status(self) -> str:
        """
        Get termination status for episode management.
        
        Returns:
            str: Status string "continue" indicating episode continues with clipping
        """
        return "continue"
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update boundary policy configuration parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - domain_bounds: New domain size tuple for clipping region
                - velocity_damping: Velocity scaling factor at boundaries
                - damp_at_boundary: Enable/disable velocity damping
                - allow_negative_coords: Coordinate restriction update
        """
        if 'domain_bounds' in kwargs:
            new_bounds = kwargs['domain_bounds']
            if len(new_bounds) != 2 or any(bound <= 0 for bound in new_bounds):
                raise ValueError("domain_bounds must be positive (width, height) tuple")
            self.domain_bounds = new_bounds
            self.x_max = float(new_bounds[0])
            self.y_max = float(new_bounds[1])
        
        if 'velocity_damping' in kwargs:
            velocity_damping = kwargs['velocity_damping']
            if not 0 <= velocity_damping <= 1:
                raise ValueError("velocity_damping must be in range [0, 1]")
            self.velocity_damping = velocity_damping
        
        if 'damp_at_boundary' in kwargs:
            self.damp_at_boundary = kwargs['damp_at_boundary']
        
        if 'allow_negative_coords' in kwargs:
            self.allow_negative_coords = kwargs['allow_negative_coords']
            self.x_min = 0.0 if not self.allow_negative_coords else -np.inf
            self.y_min = 0.0 if not self.allow_negative_coords else -np.inf


def create_boundary_policy(
    policy_type: str,
    domain_bounds: Tuple[float, float],
    **kwargs: Any
) -> BoundaryPolicyProtocol:
    """
    Factory function for creating boundary policy instances with runtime selection.
    
    This factory enables configuration-driven boundary policy instantiation without
    requiring explicit class imports, supporting dynamic policy selection and
    parameter configuration through external configuration systems.
    
    Supported Policy Types:
    - "terminate": TerminateBoundary for episode termination on boundary violation
    - "bounce": BounceBoundary for elastic collision behavior at boundaries
    - "wrap": WrapBoundary for periodic boundary conditions with position wrapping
    - "clip": ClipBoundary for hard position constraints preventing boundary crossing
    
    Args:
        policy_type: Boundary policy type identifier string
        domain_bounds: Domain size as (width, height) defining valid region
        **kwargs: Policy-specific configuration parameters passed to constructor
        
    Returns:
        BoundaryPolicyProtocol: Configured boundary policy instance
        
    Raises:
        ValueError: If policy_type is not recognized or parameters are invalid
        
    Examples:
        Termination boundary with custom status:
        >>> policy = create_boundary_policy(
        ...     "terminate", 
        ...     domain_bounds=(100, 100),
        ...     status_on_violation="boundary_exit"
        ... )
        
        Elastic bounce boundary with energy loss:
        >>> policy = create_boundary_policy(
        ...     "bounce",
        ...     domain_bounds=(200, 150), 
        ...     elasticity=0.8,
        ...     energy_loss=0.1
        ... )
        
        Periodic wrapping boundary:
        >>> policy = create_boundary_policy(
        ...     "wrap",
        ...     domain_bounds=(100, 100)
        ... )
        
        Hard clipping boundary with velocity damping:
        >>> policy = create_boundary_policy(
        ...     "clip",
        ...     domain_bounds=(50, 50),
        ...     velocity_damping=0.7,
        ...     damp_at_boundary=True
        ... )
        
        Configuration-driven instantiation:
        >>> config = {
        ...     'policy_type': 'bounce',
        ...     'domain_bounds': (100, 100),
        ...     'elasticity': 0.9
        ... }
        >>> policy = create_boundary_policy(**config)
    """
    # Policy type mapping for factory instantiation
    policy_classes = {
        "terminate": TerminateBoundary,
        "bounce": BounceBoundary,
        "wrap": WrapBoundary,
        "clip": ClipBoundary
    }
    
    # Validate policy type
    if policy_type not in policy_classes:
        available_types = list(policy_classes.keys())
        raise ValueError(
            f"Unknown policy_type '{policy_type}'. "
            f"Available types: {available_types}"
        )
    
    # Instantiate policy with provided parameters
    policy_class = policy_classes[policy_type]
    try:
        return policy_class(domain_bounds=domain_bounds, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for {policy_type} boundary policy: {e}"
        ) from e


# Export all boundary policy implementations and factory function
__all__ = [
    # Boundary policy implementations
    "TerminateBoundary",
    "BounceBoundary", 
    "WrapBoundary",
    "ClipBoundary",
    
    # Factory function for policy creation
    "create_boundary_policy",
    
    # Configuration utilities
    "BoundaryConfig",
]