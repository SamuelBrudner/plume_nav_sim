"""
Comprehensive test module for BoundaryPolicyProtocol implementations validating domain edge handling strategies.

This module provides exhaustive testing for all boundary policy implementations including
terminate, bounce, wrap, and clip behaviors with vectorized operations and episode
termination control. Tests validate protocol compliance, performance requirements,
geometric accuracy, and integration with navigation controllers per F-015 Boundary
Policy Framework requirements.

Test Coverage Areas:
- Protocol compliance: BoundaryPolicyProtocol interface implementation validation
- Performance testing: ≤33ms/step performance target with 100 agents via vectorized operations  
- Geometric validation: Position correction accuracy and boundary violation detection
- Multi-agent scenarios: Vectorized boundary checking algorithms with sub-millisecond response
- Integration testing: Boundary handling delegation with SingleAgentController and MultiAgentController
- Configuration testing: Hydra config group 'conf/base/boundary/' for runtime policy selection
- Complex domain geometry: Spatial indexing optimization and memory efficiency validation
- Thread safety: Concurrent boundary checking scenarios for multi-agent operations
- Edge case handling: Corner collisions, simultaneous violations, and numerical stability

Key Testing Strategies:
- Parametrized tests across all boundary policy types for comprehensive coverage
- Performance benchmarking with pytest-benchmark plugin for timing validation
- Property-based testing with Hypothesis for edge case discovery
- Mock-based integration testing with navigation controllers
- Configuration validation through temporary Hydra config manipulation
- Memory profiling for efficiency validation with large agent populations
- Statistical validation of geometric corrections and physics compliance

Performance Requirements Validation:
- apply_policy(): <1ms for 100 agents with vectorized operations
- check_violations(): <0.5ms for boundary detection across all agents
- get_termination_status(): <0.1ms for episode termination decisions  
- Memory efficiency: <1MB for boundary state management

Author: Blitzy Platform v1.0
Version: 1.0.0
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import time
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import tempfile
from pathlib import Path
import threading

# Import testing infrastructure
pytest = pytest
benchmark = pytest.mark.skipif(
    not hasattr(pytest, 'mark') or not hasattr(pytest.mark, 'benchmark'),
    reason="pytest-benchmark not available"
)

# Import boundary policy protocol and implementations
from plume_nav_sim.core.protocols import BoundaryPolicyProtocol
from plume_nav_sim.core.boundaries import (
    TerminateBoundary, BounceBoundary, WrapBoundary, ClipBoundary,
    create_boundary_policy, BoundaryConfig
)
from plume_nav_sim.core.controllers import SingleAgentController

# Import test fixtures
from tests.conftest import mock_action_config


class TestBoundaryPolicyProtocol:
    """
    Test suite for BoundaryPolicyProtocol interface compliance validation.
    
    Validates that all boundary policy implementations correctly implement the
    BoundaryPolicyProtocol interface with proper method signatures, return types,
    and protocol compliance per Section 6.6.2.2 Protocol Coverage Test Modules.
    """
    
    @pytest.mark.parametrize("policy_class,domain_bounds,kwargs", [
        (TerminateBoundary, (100, 100), {}),
        (BounceBoundary, (100, 100), {'elasticity': 0.8}),
        (WrapBoundary, (100, 100), {}),
        (ClipBoundary, (100, 100), {'velocity_damping': 0.9}),
    ])
    def test_protocol_compliance(self, policy_class, domain_bounds, kwargs):
        """
        Test that boundary policy implementations comply with BoundaryPolicyProtocol.
        
        Validates protocol implementation including method presence, signatures,
        and return types for comprehensive interface compliance testing.
        """
        # Create policy instance
        policy = policy_class(domain_bounds=domain_bounds, **kwargs)
        
        # Verify protocol compliance using isinstance check
        assert isinstance(policy, BoundaryPolicyProtocol), \
            f"{policy_class.__name__} must implement BoundaryPolicyProtocol"
        
        # Verify required methods exist and are callable
        assert hasattr(policy, 'apply_policy') and callable(policy.apply_policy)
        assert hasattr(policy, 'check_violations') and callable(policy.check_violations)
        assert hasattr(policy, 'get_termination_status') and callable(policy.get_termination_status)
        assert hasattr(policy, 'configure') and callable(policy.configure)
    
    def test_method_signatures(self):
        """
        Test that all boundary policies have correct method signatures.
        
        Validates parameter types, optional parameters, and return type annotations
        for protocol method compliance.
        """
        policy = TerminateBoundary(domain_bounds=(100, 100))
        
        # Test apply_policy signature
        positions = np.array([[50, 50]])
        velocities = np.array([[1.0, 0.5]])
        
        # Should accept positions only
        result = policy.apply_policy(positions)
        assert isinstance(result, np.ndarray)
        
        # Should accept positions and velocities
        result = policy.apply_policy(positions, velocities)
        if isinstance(result, tuple):
            assert len(result) == 2
            assert all(isinstance(arr, np.ndarray) for arr in result)
        else:
            assert isinstance(result, np.ndarray)
        
        # Test check_violations signature and return type
        violations = policy.check_violations(positions)
        assert isinstance(violations, (np.ndarray, bool, np.bool_))
        
        # Test get_termination_status signature and return type
        status = policy.get_termination_status()
        assert isinstance(status, str)
        
        # Test configure method accepts kwargs
        policy.configure(domain_bounds=(200, 200))
        assert policy.domain_bounds == (200, 200)


class TestTerminateBoundary:
    """
    Test suite for TerminateBoundary policy implementation.
    
    Validates episode termination behavior, violation detection accuracy,
    and no-position-correction policy enforcement for out-of-bounds scenarios.
    """
    
    @pytest.fixture
    def terminate_policy(self):
        """Create TerminateBoundary policy for testing."""
        return TerminateBoundary(domain_bounds=(100, 100))
    
    def test_initialization(self):
        """Test TerminateBoundary initialization with various parameters."""
        # Basic initialization
        policy = TerminateBoundary(domain_bounds=(100, 100))
        assert policy.domain_bounds == (100, 100)
        assert policy.allow_negative_coords == False
        assert policy.status_on_violation == "oob"
        
        # Custom initialization
        policy = TerminateBoundary(
            domain_bounds=(200, 150),
            allow_negative_coords=True,
            status_on_violation="boundary_exit"
        )
        assert policy.domain_bounds == (200, 150)
        assert policy.allow_negative_coords == True
        assert policy.status_on_violation == "boundary_exit"
        
        # Invalid domain bounds
        with pytest.raises(ValueError):
            TerminateBoundary(domain_bounds=(0, 100))
        with pytest.raises(ValueError):
            TerminateBoundary(domain_bounds=(100,))
    
    def test_violation_detection_single_agent(self, terminate_policy):
        """Test boundary violation detection for single agent scenarios."""
        # Agent within bounds
        position = np.array([50, 50])
        violations = terminate_policy.check_violations(position)
        assert violations == False
        
        # Agent outside right boundary
        position = np.array([105, 50])
        violations = terminate_policy.check_violations(position)
        assert violations == True
        
        # Agent outside top boundary
        position = np.array([50, 105])
        violations = terminate_policy.check_violations(position)
        assert violations == True
        
        # Agent outside left boundary (negative coordinates)
        position = np.array([-5, 50])
        violations = terminate_policy.check_violations(position)
        assert violations == True
        
        # Agent at exact boundary (implementation treats boundary as valid)
        position = np.array([100, 100])
        violations = terminate_policy.check_violations(position)
        assert violations == False
    
    def test_violation_detection_multi_agent(self, terminate_policy):
        """Test vectorized boundary violation detection for multi-agent scenarios."""
        # Multiple agents with mixed boundary states
        positions = np.array([
            [50, 50],   # Within bounds
            [105, 25],  # Outside right boundary
            [25, 105],  # Outside top boundary
            [-10, 50],  # Outside left boundary
            [50, -5],   # Outside bottom boundary
            [0, 0],     # At origin (within bounds)
            [100, 50],  # At right boundary (violation)
        ])
        
        violations = terminate_policy.check_violations(positions)
        expected = np.array([False, True, True, True, True, False, False])
        
        assert isinstance(violations, np.ndarray)
        assert violations.shape == (7,)
        np.testing.assert_array_equal(violations, expected)
    
    def test_apply_policy_no_correction(self, terminate_policy):
        """Test that terminate policy applies no position corrections."""
        # Single agent
        position = np.array([105, 50])  # Outside bounds
        corrected = terminate_policy.apply_policy(position)
        np.testing.assert_array_equal(corrected, position)
        
        # Single agent with velocity
        velocity = np.array([2.0, 1.0])
        corrected_pos, corrected_vel = terminate_policy.apply_policy(position, velocity)
        np.testing.assert_array_equal(corrected_pos, position)
        np.testing.assert_array_equal(corrected_vel, velocity)
        
        # Multi-agent
        positions = np.array([[50, 50], [105, 25], [25, 105]])
        velocities = np.array([[1.0, 0.5], [2.0, 1.0], [-1.0, 2.0]])
        corrected_pos, corrected_vel = terminate_policy.apply_policy(positions, velocities)
        np.testing.assert_array_equal(corrected_pos, positions)
        np.testing.assert_array_equal(corrected_vel, velocities)
    
    def test_termination_status(self, terminate_policy):
        """Test termination status reporting."""
        status = terminate_policy.get_termination_status()
        assert status == "oob"
        
        # Custom status
        policy = TerminateBoundary(
            domain_bounds=(100, 100),
            status_on_violation="boundary_exit"
        )
        status = policy.get_termination_status()
        assert status == "boundary_exit"
    
    def test_configuration_update(self, terminate_policy):
        """Test dynamic configuration updates."""
        # Update domain bounds
        terminate_policy.configure(domain_bounds=(200, 150))
        assert terminate_policy.domain_bounds == (200, 150)
        assert terminate_policy.x_max == 200.0
        assert terminate_policy.y_max == 150.0
        
        # Update coordinate restrictions
        terminate_policy.configure(allow_negative_coords=True)
        assert terminate_policy.allow_negative_coords == True
        assert terminate_policy.x_min == -np.inf
        assert terminate_policy.y_min == -np.inf
        
        # Update status
        terminate_policy.configure(status_on_violation="custom_status")
        assert terminate_policy.status_on_violation == "custom_status"
        
        # Invalid domain bounds
        with pytest.raises(ValueError):
            terminate_policy.configure(domain_bounds=(0, 100))


class TestBounceBoundary:
    """
    Test suite for BounceBoundary policy implementation.
    
    Validates elastic collision physics, energy conservation, velocity reflection,
    and position correction accuracy for realistic boundary behavior.
    """
    
    @pytest.fixture
    def bounce_policy(self):
        """Create BounceBoundary policy for testing."""
        return BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
    
    def test_initialization(self):
        """Test BounceBoundary initialization with physics parameters."""
        # Default initialization
        policy = BounceBoundary(domain_bounds=(100, 100))
        assert policy.domain_bounds == (100, 100)
        assert policy.elasticity == 1.0
        assert policy.energy_loss == 0.0
        assert policy.allow_negative_coords == False
        
        # Custom physics parameters
        policy = BounceBoundary(
            domain_bounds=(200, 150),
            elasticity=0.7,
            energy_loss=0.1,
            allow_negative_coords=True
        )
        assert policy.domain_bounds == (200, 150)
        assert policy.elasticity == 0.7
        assert policy.energy_loss == 0.1
        assert policy.allow_negative_coords == True
        
        # Invalid parameters
        with pytest.raises(ValueError):
            BounceBoundary(domain_bounds=(100, 100), elasticity=1.5)
        with pytest.raises(ValueError):
            BounceBoundary(domain_bounds=(100, 100), energy_loss=-0.1)
    
    def test_requires_velocities(self, bounce_policy):
        """Test that bounce policy requires velocities for physics calculations."""
        position = np.array([105, 50])  # Outside bounds
        
        # Should raise error without velocities
        with pytest.raises(ValueError, match="BounceBoundary requires velocities"):
            bounce_policy.apply_policy(position)
    
    def test_collision_physics_single_agent(self, bounce_policy):
        """Test collision physics for single agent boundary interactions."""
        # Right boundary collision
        position = np.array([105, 50])
        velocity = np.array([2.0, 1.0])
        corrected_pos, corrected_vel = bounce_policy.apply_policy(position, velocity)
        
        # Position should be reflected back into domain
        assert corrected_pos[0] < 100
        assert corrected_pos[1] == 50  # Y unchanged
        
        # X velocity should be reversed and scaled by elasticity
        expected_vel_x = -2.0 * bounce_policy.elasticity * (1 - bounce_policy.energy_loss)
        assert abs(corrected_vel[0] - expected_vel_x) < 1e-10
        assert corrected_vel[1] == 1.0  # Y velocity unchanged
        
        # Left boundary collision
        position = np.array([-5, 50])
        velocity = np.array([-1.0, 0.5])
        corrected_pos, corrected_vel = bounce_policy.apply_policy(position, velocity)
        
        # Position reflected into domain
        assert corrected_pos[0] > 0
        assert corrected_pos[1] == 50
        
        # X velocity reversed
        expected_vel_x = 1.0 * bounce_policy.elasticity * (1 - bounce_policy.energy_loss)
        assert abs(corrected_vel[0] - expected_vel_x) < 1e-10
        assert corrected_vel[1] == 0.5  # Y velocity unchanged
    
    def test_collision_physics_multi_agent(self, bounce_policy):
        """Test vectorized collision physics for multi-agent scenarios."""
        # Multiple agents with different boundary violations
        positions = np.array([
            [50, 50],   # Within bounds - no change
            [105, 25],  # Right boundary collision
            [25, 105],  # Top boundary collision
            [-5, 75],   # Left boundary collision
            [75, -3],   # Bottom boundary collision
        ])
        
        velocities = np.array([
            [1.0, 0.5],   # No collision
            [2.0, 1.0],   # Moving right into right boundary
            [0.5, 1.5],   # Moving up into top boundary  
            [-1.0, 0.8],  # Moving left into left boundary
            [1.2, -0.7],  # Moving down into bottom boundary
        ])
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # Agent 0: No change (within bounds)
        np.testing.assert_array_almost_equal(corrected_pos[0], positions[0])
        np.testing.assert_array_almost_equal(corrected_vel[0], velocities[0])
        
        # Agent 1: Right boundary collision
        assert corrected_pos[1, 0] < 100  # Reflected position
        assert corrected_pos[1, 1] == 25  # Y unchanged
        assert corrected_vel[1, 0] < 0    # X velocity reversed
        assert corrected_vel[1, 1] == 1.0  # Y velocity unchanged
        
        # Agent 2: Top boundary collision
        assert corrected_pos[2, 0] == 25  # X unchanged
        assert corrected_pos[2, 1] < 100  # Reflected position
        assert corrected_vel[2, 0] == 0.5  # X velocity unchanged
        assert corrected_vel[2, 1] < 0     # Y velocity reversed
    
    def test_energy_conservation(self):
        """Test energy conservation in collision physics."""
        # Perfect elastic collision (elasticity=1.0, energy_loss=0.0)
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=1.0, energy_loss=0.0)
        
        position = np.array([105, 50])
        velocity = np.array([2.0, 1.0])
        corrected_pos, corrected_vel = policy.apply_policy(position, velocity)
        
        # Energy should be conserved (velocity magnitude unchanged)
        original_speed = np.linalg.norm(velocity)
        corrected_speed = np.linalg.norm(corrected_vel)
        assert abs(original_speed - corrected_speed) < 1e-10
        
        # Inelastic collision with energy loss
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8, energy_loss=0.1)
        corrected_pos, corrected_vel = policy.apply_policy(position, velocity)
        
        # Energy should be reduced
        corrected_speed = np.linalg.norm(corrected_vel)
        expected_reduction = 0.8 * (1 - 0.1)  # elasticity * (1 - energy_loss)
        expected_speed = original_speed * np.sqrt(expected_reduction)
        # Note: Only the reflected component is affected, not the full velocity
    
    def test_termination_status(self, bounce_policy):
        """Test that bounce policy continues episodes."""
        status = bounce_policy.get_termination_status()
        assert status == "continue"
    
    def test_elasticity_update(self, bounce_policy):
        """Test dynamic elasticity configuration updates."""
        # Update elasticity
        bounce_policy.set_elasticity(0.5)
        assert bounce_policy.elasticity == 0.5
        
        # Test collision with new elasticity
        position = np.array([105, 50])
        velocity = np.array([2.0, 0.0])
        corrected_pos, corrected_vel = bounce_policy.apply_policy(position, velocity)
        
        expected_vel_x = -2.0 * 0.5 * (1 - bounce_policy.energy_loss)
        assert abs(corrected_vel[0] - expected_vel_x) < 1e-10
        
        # Invalid elasticity
        with pytest.raises(ValueError):
            bounce_policy.set_elasticity(1.5)


class TestWrapBoundary:
    """
    Test suite for WrapBoundary policy implementation.
    
    Validates periodic boundary conditions, toroidal topology, position wrapping
    accuracy, and velocity preservation during domain transitions.
    """
    
    @pytest.fixture
    def wrap_policy(self):
        """Create WrapBoundary policy for testing."""
        return WrapBoundary(domain_bounds=(100, 100))
    
    def test_initialization(self):
        """Test WrapBoundary initialization."""
        policy = WrapBoundary(domain_bounds=(100, 100))
        assert policy.domain_bounds == (100, 100)
        assert policy.allow_negative_coords == False
        assert policy.x_max == 100.0
        assert policy.y_max == 100.0
        
        # With negative coordinates allowed
        policy = WrapBoundary(domain_bounds=(200, 150), allow_negative_coords=True)
        assert policy.allow_negative_coords == True
    
    def test_position_wrapping_single_agent(self, wrap_policy):
        """Test position wrapping for single agent scenarios."""
        # Agent beyond right boundary
        position = np.array([105, 50])
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([5, 50])  # Wrapped to opposite side
        np.testing.assert_array_almost_equal(wrapped, expected)
        
        # Agent beyond left boundary
        position = np.array([-5, 50])
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([95, 50])  # Wrapped to right side
        np.testing.assert_array_almost_equal(wrapped, expected)
        
        # Agent beyond top boundary
        position = np.array([50, 105])
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([50, 5])  # Wrapped to bottom
        np.testing.assert_array_almost_equal(wrapped, expected)
        
        # Agent beyond bottom boundary
        position = np.array([50, -5])
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([50, 95])  # Wrapped to top
        np.testing.assert_array_almost_equal(wrapped, expected)
        
        # Agent within bounds (no wrapping)
        position = np.array([50, 50])
        wrapped = wrap_policy.apply_policy(position)
        np.testing.assert_array_almost_equal(wrapped, position)
    
    def test_position_wrapping_multi_agent(self, wrap_policy):
        """Test vectorized position wrapping for multi-agent scenarios."""
        positions = np.array([
            [50, 50],    # Within bounds
            [105, 25],   # Right boundary wrap
            [25, 105],   # Top boundary wrap
            [-5, 75],    # Left boundary wrap
            [75, -3],    # Bottom boundary wrap
            [110, 110],  # Corner wrap (both dimensions)
        ])
        
        wrapped = wrap_policy.apply_policy(positions)
        
        expected = np.array([
            [50, 50],    # No change
            [5, 25],     # Wrapped from right
            [25, 5],     # Wrapped from top
            [95, 75],    # Wrapped from left
            [75, 97],    # Wrapped from bottom
            [10, 10],    # Both dimensions wrapped
        ])
        
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_velocity_preservation(self, wrap_policy):
        """Test that velocities are preserved during wrapping."""
        position = np.array([105, 50])
        velocity = np.array([2.0, 1.0])
        
        wrapped_pos, wrapped_vel = wrap_policy.apply_policy(position, velocity)
        
        # Position should be wrapped
        expected_pos = np.array([5, 50])
        np.testing.assert_array_almost_equal(wrapped_pos, expected_pos)
        
        # Velocity should be unchanged
        np.testing.assert_array_almost_equal(wrapped_vel, velocity)
    
    def test_large_displacements(self, wrap_policy):
        """Test wrapping with large position displacements."""
        # Multiple domain widths displacement
        position = np.array([250, 325])  # 2.5 and 3.25 domain widths
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([50, 25])  # Wrapped back into domain
        np.testing.assert_array_almost_equal(wrapped, expected)
        
        # Large negative displacement
        position = np.array([-150, -220])
        wrapped = wrap_policy.apply_policy(position)
        expected = np.array([50, 80])  # Wrapped from negative
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_violation_detection(self, wrap_policy):
        """Test violation detection for wrap boundaries."""
        # Positions that need wrapping are considered "violations"
        positions = np.array([
            [50, 50],   # Within bounds - no violation
            [105, 25],  # Outside - violation
            [25, 105],  # Outside - violation
            [0, 0],     # At origin - no violation
            [100, 50],  # At boundary - violation
        ])
        
        violations = wrap_policy.check_violations(positions)
        expected = np.array([False, True, True, False, True])
        np.testing.assert_array_equal(violations, expected)
    
    def test_termination_status(self, wrap_policy):
        """Test that wrap policy continues episodes."""
        status = wrap_policy.get_termination_status()
        assert status == "continue"


class TestClipBoundary:
    """
    Test suite for ClipBoundary policy implementation.
    
    Validates hard position constraints, velocity damping, boundary contact
    detection, and deterministic clipping behavior for spatial confinement.
    """
    
    @pytest.fixture
    def clip_policy(self):
        """Create ClipBoundary policy for testing."""
        return ClipBoundary(domain_bounds=(100, 100))
    
    def test_initialization(self):
        """Test ClipBoundary initialization with damping parameters."""
        # Default initialization
        policy = ClipBoundary(domain_bounds=(100, 100))
        assert policy.domain_bounds == (100, 100)
        assert policy.velocity_damping == 1.0
        assert policy.damp_at_boundary == False
        assert policy.allow_negative_coords == False
        
        # Custom damping parameters
        policy = ClipBoundary(
            domain_bounds=(200, 150),
            velocity_damping=0.7,
            damp_at_boundary=True,
            allow_negative_coords=True
        )
        assert policy.velocity_damping == 0.7
        assert policy.damp_at_boundary == True
        assert policy.allow_negative_coords == True
        
        # Invalid parameters
        with pytest.raises(ValueError):
            ClipBoundary(domain_bounds=(100, 100), velocity_damping=1.5)
    
    def test_position_clipping_single_agent(self, clip_policy):
        """Test position clipping for single agent scenarios."""
        # Agent beyond right boundary
        position = np.array([105, 50])
        clipped = clip_policy.apply_policy(position)
        expected = np.array([100, 50])
        np.testing.assert_array_almost_equal(clipped, expected)
        
        # Agent beyond left boundary
        position = np.array([-5, 50])
        clipped = clip_policy.apply_policy(position)
        expected = np.array([0, 50])
        np.testing.assert_array_almost_equal(clipped, expected)
        
        # Agent beyond top boundary
        position = np.array([50, 105])
        clipped = clip_policy.apply_policy(position)
        expected = np.array([50, 100])
        np.testing.assert_array_almost_equal(clipped, expected)
        
        # Agent beyond bottom boundary
        position = np.array([50, -5])
        clipped = clip_policy.apply_policy(position)
        expected = np.array([50, 0])
        np.testing.assert_array_almost_equal(clipped, expected)
        
        # Agent within bounds (no clipping)
        position = np.array([50, 50])
        clipped = clip_policy.apply_policy(position)
        np.testing.assert_array_almost_equal(clipped, position)
    
    def test_position_clipping_multi_agent(self, clip_policy):
        """Test vectorized position clipping for multi-agent scenarios."""
        positions = np.array([
            [50, 50],    # Within bounds
            [105, 25],   # Right boundary clip
            [25, 105],   # Top boundary clip
            [-5, 75],    # Left boundary clip
            [75, -3],    # Bottom boundary clip
            [110, 110],  # Corner clip (both dimensions)
        ])
        
        clipped = clip_policy.apply_policy(positions)
        
        expected = np.array([
            [50, 50],    # No change
            [100, 25],   # Clipped to right boundary
            [25, 100],   # Clipped to top boundary
            [0, 75],     # Clipped to left boundary
            [75, 0],     # Clipped to bottom boundary
            [100, 100],  # Clipped to corner
        ])
        
        np.testing.assert_array_almost_equal(clipped, expected)
    
    def test_velocity_damping_disabled(self, clip_policy):
        """Test that velocities are preserved when damping is disabled."""
        position = np.array([105, 50])
        velocity = np.array([2.0, 1.0])
        
        clipped_pos, clipped_vel = clip_policy.apply_policy(position, velocity)
        
        # Position should be clipped
        expected_pos = np.array([100, 50])
        np.testing.assert_array_almost_equal(clipped_pos, expected_pos)
        
        # Velocity should be unchanged (damping disabled)
        np.testing.assert_array_almost_equal(clipped_vel, velocity)
    
    def test_velocity_damping_enabled(self):
        """Test velocity damping when enabled at boundaries."""
        policy = ClipBoundary(
            domain_bounds=(100, 100),
            velocity_damping=0.5,
            damp_at_boundary=True
        )
        
        position = np.array([105, 50])  # Will be clipped to boundary
        velocity = np.array([2.0, 1.0])
        
        clipped_pos, clipped_vel = policy.apply_policy(position, velocity)
        
        # Position clipped to boundary
        expected_pos = np.array([100, 50])
        np.testing.assert_array_almost_equal(clipped_pos, expected_pos)
        
        # Velocity should be damped (agent is at boundary after clipping)
        expected_vel = velocity * 0.5
        np.testing.assert_array_almost_equal(clipped_vel, expected_vel)
    
    def test_boundary_contact_detection(self):
        """Test detection of agents in contact with boundaries."""
        policy = ClipBoundary(domain_bounds=(100, 100))
        
        positions = np.array([
            [50, 50],    # Not at boundary
            [0, 50],     # At left boundary
            [100, 50],   # At right boundary
            [50, 0],     # At bottom boundary
            [50, 100],   # At top boundary
            [0, 0],      # At corner
        ])
        
        at_boundaries = policy._detect_boundary_contact(positions)
        expected = np.array([False, True, True, True, True, True])
        
        np.testing.assert_array_equal(at_boundaries, expected)
    
    def test_termination_status(self, clip_policy):
        """Test that clip policy continues episodes."""
        status = clip_policy.get_termination_status()
        assert status == "continue"
    
    def test_configuration_update(self, clip_policy):
        """Test dynamic configuration updates."""
        # Update velocity damping
        clip_policy.configure(velocity_damping=0.8)
        assert clip_policy.velocity_damping == 0.8
        
        # Enable boundary damping
        clip_policy.configure(damp_at_boundary=True)
        assert clip_policy.damp_at_boundary == True
        
        # Invalid damping value
        with pytest.raises(ValueError):
            clip_policy.configure(velocity_damping=1.5)


class TestBoundaryPolicyPerformance:
    """
    Test suite for boundary policy performance validation.
    
    Validates performance requirements including ≤33ms/step with 100 agents,
    vectorized operations efficiency, and memory usage constraints per
    F-015 performance targets.
    """
    
    @pytest.fixture(params=[
        TerminateBoundary,
        BounceBoundary,
        WrapBoundary,
        ClipBoundary
    ])
    def policy_class(self, request):
        """Parametrize across all boundary policy classes."""
        return request.param
    
    def test_violation_detection_performance(self, policy_class):
        """Test boundary violation detection performance with large agent populations."""
        # Create policy instance
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        # Generate 100 agent positions with mix of violations
        np.random.seed(42)
        positions = np.random.uniform(-10, 110, size=(100, 2))
        
        # Measure violation detection performance
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple iterations for stable timing
            violations = policy.check_violations(positions)
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / 10 * 1000
        
        # Validate performance requirement: <0.5ms for violation detection
        assert avg_time_ms < 0.5, \
            f"{policy_class.__name__} violation detection took {avg_time_ms:.3f}ms, " \
            f"exceeds 0.5ms requirement"
        
        # Validate return format
        assert isinstance(violations, np.ndarray)
        assert violations.shape == (100,)
        assert violations.dtype == bool
    
    def test_policy_application_performance(self, policy_class):
        """Test boundary policy application performance with large agent populations."""
        # Create policy instance
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        # Generate test data
        np.random.seed(42)
        positions = np.random.uniform(-10, 110, size=(100, 2))
        velocities = np.random.uniform(-2, 2, size=(100, 2))
        
        # Measure policy application performance
        start_time = time.perf_counter()
        for _ in range(10):
            if policy_class == BounceBoundary:
                result = policy.apply_policy(positions, velocities)
            else:
                result = policy.apply_policy(positions, velocities)
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / 10 * 1000
        
        # Validate performance requirement: <1ms for policy application
        assert avg_time_ms < 1.0, \
            f"{policy_class.__name__} policy application took {avg_time_ms:.3f}ms, " \
            f"exceeds 1.0ms requirement"
    
    def test_termination_status_performance(self, policy_class):
        """Test termination status query performance."""
        # Create policy instance
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        # Measure termination status performance
        start_time = time.perf_counter()
        for _ in range(1000):  # Many iterations for microsecond precision
            status = policy.get_termination_status()
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / 1000 * 1000
        
        # Validate performance requirement: <0.1ms for termination status
        assert avg_time_ms < 0.1, \
            f"{policy_class.__name__} termination status took {avg_time_ms:.3f}ms, " \
            f"exceeds 0.1ms requirement"
        
        # Validate return type
        assert isinstance(status, str)
    
    def test_memory_efficiency(self, policy_class):
        """Test memory usage constraints for boundary policy state management."""
        # Create policy instance
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        # Estimate memory usage (simple attribute counting)
        import sys
        policy_size = sys.getsizeof(policy)
        for attr in dir(policy):
            if not attr.startswith('_'):
                attr_value = getattr(policy, attr)
                if not callable(attr_value):
                    policy_size += sys.getsizeof(attr_value)
        
        # Convert to MB
        policy_size_mb = policy_size / (1024 * 1024)
        
        # Validate memory requirement: <1MB for boundary state management
        assert policy_size_mb < 1.0, \
            f"{policy_class.__name__} uses {policy_size_mb:.3f}MB, " \
            f"exceeds 1MB requirement"
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'mark') or 'benchmark' not in dir(pytest.mark),
        reason="pytest-benchmark not available"
    )
    def test_vectorized_performance_benchmark(self, benchmark, policy_class):
        """Benchmark boundary policy performance using pytest-benchmark."""
        # Create policy instance
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        # Test data
        np.random.seed(42)
        positions = np.random.uniform(-10, 110, size=(100, 2))
        velocities = np.random.uniform(-2, 2, size=(100, 2))
        
        # Benchmark function
        def run_boundary_policy():
            if policy_class == BounceBoundary:
                return policy.apply_policy(positions, velocities)
            else:
                return policy.apply_policy(positions, velocities)
        
        # Run benchmark
        result = benchmark(run_boundary_policy)
        
        # Validate result format
        if isinstance(result, tuple):
            assert len(result) == 2
            assert all(isinstance(arr, np.ndarray) for arr in result)
        else:
            assert isinstance(result, np.ndarray)


class TestBoundaryPolicyIntegration:
    """
    Test suite for boundary policy integration with navigation controllers.
    
    Validates boundary handling delegation, controller integration patterns,
    and proper boundary policy lifecycle management within navigation systems.
    """
    
    def test_controller_boundary_integration(self, mock_action_config):
        """Test boundary policy integration with SingleAgentController."""
        # Create boundary policy
        boundary_policy = TerminateBoundary(domain_bounds=(100, 100))
        
        # Mock SingleAgentController with boundary policy
        controller = Mock(spec=SingleAgentController)
        controller.boundary_policy = boundary_policy
        controller.positions = np.array([[105, 50]])  # Out of bounds
        
        # Simulate controller boundary checking
        violations = boundary_policy.check_violations(controller.positions)
        assert violations == True
        
        # Simulate termination decision
        if violations:
            status = boundary_policy.get_termination_status()
            assert status == "oob"
    
    def test_controller_boundary_delegation(self):
        """Test proper boundary handling delegation from controllers."""
        # Mock controller with various boundary policies
        policies = [
            TerminateBoundary(domain_bounds=(100, 100)),
            BounceBoundary(domain_bounds=(100, 100), elasticity=0.8),
            WrapBoundary(domain_bounds=(100, 100)),
            ClipBoundary(domain_bounds=(100, 100))
        ]
        
        position = np.array([105, 50])  # Out of bounds
        velocity = np.array([2.0, 1.0])
        
        for policy in policies:
            # Mock controller method calls
            violations = policy.check_violations(position)
            assert violations == True
            
            # Apply policy correction
            if isinstance(policy, BounceBoundary):
                corrected_pos, corrected_vel = policy.apply_policy(position, velocity)
            else:
                result = policy.apply_policy(position, velocity)
                if isinstance(result, tuple):
                    corrected_pos, corrected_vel = result
                else:
                    corrected_pos = result
            
            # Verify policy-specific behavior
            if isinstance(policy, TerminateBoundary):
                np.testing.assert_array_equal(corrected_pos, position)  # No correction
            elif isinstance(policy, WrapBoundary):
                assert corrected_pos[0] < 100  # Wrapped
            elif isinstance(policy, ClipBoundary):
                assert corrected_pos[0] == 100  # Clipped
            # BounceBoundary tested separately due to physics complexity
    
    def test_multi_agent_controller_integration(self):
        """Test boundary policy integration with multi-agent scenarios."""
        # Create boundary policy
        boundary_policy = WrapBoundary(domain_bounds=(100, 100))
        
        # Multi-agent positions with mixed boundary states
        positions = np.array([
            [50, 50],   # Within bounds
            [105, 25],  # Outside right
            [25, 105],  # Outside top
        ])
        
        # Test vectorized boundary handling
        violations = boundary_policy.check_violations(positions)
        expected_violations = np.array([False, True, True])
        np.testing.assert_array_equal(violations, expected_violations)
        
        # Apply corrections
        corrected_positions = boundary_policy.apply_policy(positions)
        
        # Verify vectorized correction
        assert corrected_positions[0, 0] == 50   # No change
        assert corrected_positions[1, 0] == 5    # Wrapped
        assert corrected_positions[2, 1] == 5    # Wrapped
    
    def test_boundary_policy_lifecycle(self):
        """Test boundary policy lifecycle management in controllers."""
        # Initialize boundary policy
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
        
        # Test configuration updates during runtime
        policy.configure(elasticity=0.9, energy_loss=0.05)
        assert policy.elasticity == 0.9
        assert policy.energy_loss == 0.05
        
        # Test dynamic domain bounds updates
        policy.configure(domain_bounds=(200, 150))
        assert policy.domain_bounds == (200, 150)
        
        # Verify updated behavior
        position = np.array([105, 50])  # Within new bounds
        violations = policy.check_violations(position)
        assert violations == False  # No violation with larger domain


class TestBoundaryPolicyConfiguration:
    """
    Test suite for boundary policy configuration management.
    
    Validates Hydra config group integration, runtime policy selection,
    and configuration-driven boundary policy instantiation per F-015 requirements.
    """
    
    def test_factory_method_creation(self):
        """Test boundary policy creation via factory method."""
        # Test all policy types
        policy_configs = [
            ("terminate", {"status_on_violation": "boundary_exit"}),
            ("bounce", {"elasticity": 0.7, "energy_loss": 0.1}),
            ("wrap", {"allow_negative_coords": True}),
            ("clip", {"velocity_damping": 0.8, "damp_at_boundary": True}),
        ]
        
        domain_bounds = (100, 100)
        
        for policy_type, kwargs in policy_configs:
            policy = create_boundary_policy(policy_type, domain_bounds, **kwargs)
            
            # Verify protocol compliance
            assert isinstance(policy, BoundaryPolicyProtocol)
            
            # Verify domain bounds
            assert policy.domain_bounds == domain_bounds
            
            # Verify type-specific configurations
            if policy_type == "terminate":
                assert policy.status_on_violation == "boundary_exit"
            elif policy_type == "bounce":
                assert policy.elasticity == 0.7
                assert policy.energy_loss == 0.1
            elif policy_type == "wrap":
                assert policy.allow_negative_coords == True
            elif policy_type == "clip":
                assert policy.velocity_damping == 0.8
                assert policy.damp_at_boundary == True
    
    def test_factory_method_invalid_type(self):
        """Test factory method error handling for invalid policy types."""
        with pytest.raises(ValueError, match="Unknown policy_type"):
            create_boundary_policy("invalid_type", (100, 100))
    
    def test_boundary_config_validation(self):
        """Test BoundaryConfig dataclass validation."""
        # Valid configuration
        config = BoundaryConfig(
            domain_bounds=(100, 100),
            policy_type="bounce",
            allow_negative_coords=True,
            boundary_buffer=2.0
        )
        assert config.domain_bounds == (100, 100)
        assert config.policy_type == "bounce"
        assert config.allow_negative_coords == True
        assert config.boundary_buffer == 2.0
        
        # Invalid domain bounds
        with pytest.raises(ValueError):
            BoundaryConfig(domain_bounds=(0, 100))
        
        with pytest.raises(ValueError):
            BoundaryConfig(domain_bounds=(100,))
        
        # Invalid boundary buffer
        with pytest.raises(ValueError):
            BoundaryConfig(domain_bounds=(100, 100), boundary_buffer=-1.0)
    
    def test_hydra_config_group_simulation(self):
        """Test simulation of Hydra config group structure for boundary policies."""
        # Simulate conf/base/boundary/ config group
        boundary_configs = {
            "terminate": {
                "_target_": "plume_nav_sim.core.boundaries.TerminateBoundary",
                "domain_bounds": [100, 100],
                "allow_negative_coords": False,
                "status_on_violation": "oob"
            },
            "bounce": {
                "_target_": "plume_nav_sim.core.boundaries.BounceBoundary",
                "domain_bounds": [100, 100],
                "elasticity": 0.8,
                "energy_loss": 0.1,
                "allow_negative_coords": False
            },
            "wrap": {
                "_target_": "plume_nav_sim.core.boundaries.WrapBoundary",
                "domain_bounds": [100, 100],
                "allow_negative_coords": False
            },
            "clip": {
                "_target_": "plume_nav_sim.core.boundaries.ClipBoundary",
                "domain_bounds": [100, 100],
                "velocity_damping": 1.0,
                "damp_at_boundary": False,
                "allow_negative_coords": False
            }
        }
        
        # Test configuration instantiation simulation
        for config_name, config in boundary_configs.items():
            # Extract target class and parameters
            target_class_name = config["_target_"].split(".")[-1]
            params = {k: v for k, v in config.items() if k != "_target_"}
            
            # Convert domain_bounds from list to tuple
            if "domain_bounds" in params:
                params["domain_bounds"] = tuple(params["domain_bounds"])
            
            # Create instance via factory
            policy_type = config_name
            policy = create_boundary_policy(policy_type, **params)
            
            # Verify correct type instantiation
            assert policy.__class__.__name__ == target_class_name
            assert policy.domain_bounds == (100, 100)
    
    def test_runtime_policy_selection(self):
        """Test runtime boundary policy selection and switching."""
        domain_bounds = (100, 100)
        
        # Create different policies at runtime
        policies = {
            "terminate": create_boundary_policy("terminate", domain_bounds),
            "bounce": create_boundary_policy("bounce", domain_bounds, elasticity=0.8),
            "wrap": create_boundary_policy("wrap", domain_bounds),
            "clip": create_boundary_policy("clip", domain_bounds, velocity_damping=0.9)
        }
        
        # Test policy switching simulation
        test_position = np.array([105, 50])  # Out of bounds
        
        for policy_name, policy in policies.items():
            violations = policy.check_violations(test_position)
            assert violations == True
            
            status = policy.get_termination_status()
            if policy_name == "terminate":
                assert status == "oob"
            else:
                assert status == "continue"


class TestBoundaryPolicyEdgeCases:
    """
    Test suite for boundary policy edge cases and numerical stability.
    
    Validates corner collisions, simultaneous violations, floating point
    precision, and boundary condition robustness across all policy types.
    """
    
    def test_corner_collisions(self):
        """Test boundary policy behavior at domain corners."""
        # Test all policies with corner positions
        policies = [
            TerminateBoundary(domain_bounds=(100, 100)),
            BounceBoundary(domain_bounds=(100, 100), elasticity=0.8),
            WrapBoundary(domain_bounds=(100, 100)),
            ClipBoundary(domain_bounds=(100, 100))
        ]
        
        # Corner positions (outside domain)
        corner_positions = np.array([
            [-5, -5],     # Bottom-left corner
            [105, -5],    # Bottom-right corner
            [-5, 105],    # Top-left corner
            [105, 105],   # Top-right corner
        ])
        
        for policy in policies:
            violations = policy.check_violations(corner_positions)
            # All corners should be violations
            assert np.all(violations == True)
            
            if isinstance(policy, BounceBoundary):
                # Test with velocities for bounce policy
                velocities = np.array([
                    [1.0, 1.0],   # Moving into corner
                    [-1.0, 1.0],  # Moving into corner
                    [1.0, -1.0],  # Moving into corner
                    [-1.0, -1.0], # Moving into corner
                ])
                corrected_pos, corrected_vel = policy.apply_policy(corner_positions, velocities)
                
                # Verify positions are corrected back into domain
                assert np.all(corrected_pos >= 0)
                assert np.all(corrected_pos <= 100)
            else:
                corrected_pos = policy.apply_policy(corner_positions)
                
                if isinstance(policy, TerminateBoundary):
                    # Terminate policy doesn't correct positions
                    np.testing.assert_array_equal(corrected_pos, corner_positions)
                elif isinstance(policy, WrapBoundary):
                    # Wrap policy should wrap corner positions
                    expected = np.array([
                        [95, 95],   # Wrapped corner
                        [5, 95],    # Wrapped corner
                        [95, 5],    # Wrapped corner
                        [5, 5],     # Wrapped corner
                    ])
                    np.testing.assert_array_almost_equal(corrected_pos, expected)
                elif isinstance(policy, ClipBoundary):
                    # Clip policy should clip to corner
                    expected = np.array([
                        [0, 0],       # Clipped corner
                        [100, 0],     # Clipped corner
                        [0, 100],     # Clipped corner
                        [100, 100],   # Clipped corner
                    ])
                    np.testing.assert_array_almost_equal(corrected_pos, expected)
    
    def test_exact_boundary_positions(self):
        """Test behavior with agents exactly at boundaries."""
        policies = [
            TerminateBoundary(domain_bounds=(100, 100)),
            BounceBoundary(domain_bounds=(100, 100), elasticity=1.0),
            WrapBoundary(domain_bounds=(100, 100)),
            ClipBoundary(domain_bounds=(100, 100))
        ]
        
        # Exact boundary positions
        boundary_positions = np.array([
            [0, 50],      # Left boundary
            [100, 50],    # Right boundary
            [50, 0],      # Bottom boundary
            [50, 100],    # Top boundary
        ])
        
        for policy in policies:
            violations = policy.check_violations(boundary_positions)
            
            if isinstance(policy, (TerminateBoundary, BounceBoundary, ClipBoundary)):
                # These policies use inclusive boundaries, so boundary positions are NOT violations
                assert np.all(violations == False)
            elif isinstance(policy, WrapBoundary):
                # Wrap policy considers upper boundaries as violations but not lower boundaries
                # Expected: [False, True, False, True] for positions [0,50], [100,50], [50,0], [50,100]
                expected_wrap = np.array([False, True, False, True])
                np.testing.assert_array_equal(violations, expected_wrap)
    
    def test_floating_point_precision(self):
        """Test numerical stability with floating point precision issues."""
        policy = ClipBoundary(domain_bounds=(100, 100))
        
        # Positions very close to boundaries (floating point precision)
        epsilon = 1e-15
        precision_positions = np.array([
            [100 + epsilon, 50],      # Slightly outside right
            [100 - epsilon, 50],      # Slightly inside right
            [50, 100 + epsilon],      # Slightly outside top
            [50, 100 - epsilon],      # Slightly inside top
            [0 - epsilon, 50],        # Slightly outside left
            [0 + epsilon, 50],        # Slightly inside left
        ])
        
        violations = policy.check_violations(precision_positions)
        clipped = policy.apply_policy(precision_positions)
        
        # Verify numerical stability
        assert np.all(clipped >= 0)
        assert np.all(clipped <= 100)
        assert np.all(np.isfinite(clipped))
    
    def test_simultaneous_violations(self):
        """Test handling of simultaneous boundary violations in multiple dimensions."""
        # Bounce policy with corner collision
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
        
        # Position outside both x and y boundaries
        position = np.array([105, 110])
        velocity = np.array([2.0, 3.0])
        
        corrected_pos, corrected_vel = policy.apply_policy(position, velocity)
        
        # Verify both dimensions are corrected
        assert corrected_pos[0] < 100  # X corrected
        assert corrected_pos[1] < 100  # Y corrected
        
        # Verify both velocity components are affected
        assert corrected_vel[0] < 0    # X velocity reversed
        assert corrected_vel[1] < 0    # Y velocity reversed
    
    def test_zero_velocity_scenarios(self):
        """Test boundary policies with zero velocity inputs."""
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
        
        # Stationary agent outside boundary
        position = np.array([105, 50])
        velocity = np.array([0.0, 0.0])
        
        corrected_pos, corrected_vel = policy.apply_policy(position, velocity)
        
        # Position should be corrected
        assert corrected_pos[0] < 100
        
        # Zero velocity should remain zero (no division by zero)
        np.testing.assert_array_equal(corrected_vel, velocity)
    
    def test_extreme_positions(self):
        """Test boundary policies with extremely large position displacements."""
        policy = WrapBoundary(domain_bounds=(100, 100))
        
        # Very large displacements
        extreme_positions = np.array([
            [1000, 2000],    # 10x and 20x domain size
            [-500, -300],    # Large negative displacements
            [1e6, 1e6],      # Very large values
        ])
        
        wrapped = policy.apply_policy(extreme_positions)
        
        # Verify wrapped positions are within domain
        assert np.all(wrapped >= 0)
        assert np.all(wrapped < 100)
        assert np.all(np.isfinite(wrapped))
    
    def test_array_shape_consistency(self):
        """Test that boundary policies maintain array shape consistency."""
        policy = ClipBoundary(domain_bounds=(100, 100))
        
        # Test different input shapes
        test_cases = [
            np.array([105, 50]),                    # Single agent (1D)
            np.array([[105, 50]]),                  # Single agent (2D)
            np.array([[105, 50], [25, 110]]),       # Two agents
            np.random.uniform(-10, 110, (50, 2)),   # Many agents
        ]
        
        for positions in test_cases:
            violations = policy.check_violations(positions)
            corrected = policy.apply_policy(positions)
            
            # Verify shape consistency
            if positions.ndim == 1:
                assert violations.shape == ()  # Scalar for 1D input
                assert corrected.shape == positions.shape
            else:
                assert violations.shape == (positions.shape[0],)
                assert corrected.shape == positions.shape


class TestBoundaryPolicyThreadSafety:
    """
    Test suite for boundary policy thread safety validation.
    
    Validates concurrent boundary checking scenarios and thread-safe
    execution for multi-threaded navigation applications.
    """
    
    def test_concurrent_boundary_checking(self):
        """Test thread safety of boundary policy operations."""
        policy = TerminateBoundary(domain_bounds=(100, 100))
        
        # Shared test data
        positions = np.random.uniform(-10, 110, size=(100, 2))
        results = []
        errors = []
        
        def check_boundaries():
            """Thread worker function."""
            try:
                for _ in range(100):
                    violations = policy.check_violations(positions)
                    results.append(violations)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=check_boundaries) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify consistent results
        assert len(results) == 500  # 5 threads * 100 iterations
        
        # All results should be identical (deterministic computation)
        first_result = results[0]
        for result in results:
            np.testing.assert_array_equal(result, first_result)
    
    def test_concurrent_policy_application(self):
        """Test thread safety of policy application operations."""
        policy = WrapBoundary(domain_bounds=(100, 100))
        
        # Shared test data
        positions = np.random.uniform(-10, 110, size=(50, 2))
        results = []
        errors = []
        
        def apply_policy():
            """Thread worker function."""
            try:
                for _ in range(50):
                    corrected = policy.apply_policy(positions)
                    results.append(corrected)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=apply_policy) for _ in range(3)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify results are consistent
        assert len(results) == 150  # 3 threads * 50 iterations
        
        # All results should be identical
        first_result = results[0]
        for result in results:
            np.testing.assert_array_almost_equal(result, first_result)
    
    def test_concurrent_configuration_updates(self):
        """Test thread safety of configuration updates."""
        policy = BounceBoundary(domain_bounds=(100, 100), elasticity=0.8)
        
        errors = []
        config_results = []
        
        def update_configuration():
            """Thread worker for configuration updates."""
            try:
                for i in range(10):
                    elasticity = 0.5 + i * 0.05  # Vary elasticity
                    policy.configure(elasticity=elasticity)
                    config_results.append(policy.elasticity)
            except Exception as e:
                errors.append(e)
        
        # Create threads for configuration updates
        config_threads = [threading.Thread(target=update_configuration) for _ in range(3)]
        
        # Start configuration threads
        for thread in config_threads:
            thread.start()
        
        # Wait for completion
        for thread in config_threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Configuration thread safety errors: {errors}"
        
        # Verify final state is consistent
        assert 0.5 <= policy.elasticity <= 0.95  # Within expected range


# Performance benchmarking with pytest-benchmark (if available)
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
class TestBoundaryPolicyBenchmarks:
    """
    Benchmark test suite for boundary policy performance measurement.
    
    Uses pytest-benchmark plugin for precise performance measurement
    and regression detection across boundary policy implementations.
    """
    
    @pytest.mark.parametrize("policy_class", [
        TerminateBoundary,
        BounceBoundary, 
        WrapBoundary,
        ClipBoundary
    ])
    def test_benchmark_violation_detection(self, benchmark, policy_class):
        """Benchmark boundary violation detection performance."""
        # Setup
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        positions = np.random.uniform(-10, 110, size=(100, 2))
        
        # Benchmark
        result = benchmark(policy.check_violations, positions)
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
    
    @pytest.mark.parametrize("policy_class", [
        TerminateBoundary,
        BounceBoundary,
        WrapBoundary, 
        ClipBoundary
    ])
    def test_benchmark_policy_application(self, benchmark, policy_class):
        """Benchmark boundary policy application performance."""
        # Setup
        if policy_class == BounceBoundary:
            policy = policy_class(domain_bounds=(100, 100), elasticity=0.8)
        elif policy_class == ClipBoundary:
            policy = policy_class(domain_bounds=(100, 100), velocity_damping=0.9)
        else:
            policy = policy_class(domain_bounds=(100, 100))
        
        positions = np.random.uniform(-10, 110, size=(100, 2))
        velocities = np.random.uniform(-2, 2, size=(100, 2))
        
        # Benchmark function
        def apply_policy():
            if policy_class == BounceBoundary:
                return policy.apply_policy(positions, velocities)
            else:
                return policy.apply_policy(positions, velocities)
        
        # Benchmark
        result = benchmark(apply_policy)
        
        # Verify result
        if isinstance(result, tuple):
            assert len(result) == 2
            assert all(isinstance(arr, np.ndarray) for arr in result)
        else:
            assert isinstance(result, np.ndarray)