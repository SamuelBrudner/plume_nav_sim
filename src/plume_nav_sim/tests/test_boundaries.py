"""
Comprehensive pytest test module for validating BoundaryPolicyProtocol implementations.

This module provides complete test coverage for boundary policy framework including 
terminate, bounce, wrap, and clip behaviors with vectorized multi-agent boundary
checking and performance compliance validation per Section 6.6.2.4 requirements.

Test Categories:
- BoundaryPolicyProtocol compliance and interface verification
- TerminateBoundary implementation with episode termination behavior
- BounceBoundary implementation with elastic collision physics  
- WrapBoundary implementation with periodic boundary conditions
- ClipBoundary implementation with hard position constraints
- Vectorized boundary checking for multi-agent scenarios up to 100 agents
- Performance validation for ≤33ms step latency and sub-millisecond response
- Factory method integration with runtime policy selection
- Error handling and edge case validation

Performance Requirements:
- Boundary violation detection: <1ms for 100 agents per Section 6.6.6.3
- Policy application: ≤33ms step latency compliance per Section 5.2.6
- Memory efficiency: Linear scaling validation for agent count
- Vectorized operations: NumPy optimization verification

Protocol Coverage:
- 100% coverage for BoundaryPolicyProtocol implementations per Section 6.6.2.4
- All protocol methods: apply_policy, check_violations, get_termination_status, configure
- Interface compliance validation for all boundary policy types
- Method signature and return type verification

Examples:
    Basic boundary policy testing:
    >>> policy = TerminateBoundary(domain_bounds=(100, 100))
    >>> violations = policy.check_violations(positions)
    >>> assert isinstance(violations, np.ndarray)
    
    Performance validation:
    >>> with timer.time_operation("boundary_check") as t:
    ...     violations = policy.check_violations(positions_100_agents)
    >>> assert t.duration < 0.001  # <1ms requirement
    
    Multi-agent scaling:
    >>> positions = np.random.rand(100, 2) * 200  # 100 agents
    >>> violations = policy.check_violations(positions)
    >>> assert len(violations) == 100
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from unittest.mock import MagicMock

# Import boundary policy protocol and implementations
from plume_nav_sim.core.protocols import BoundaryPolicyProtocol
from plume_nav_sim.core.boundaries import (
    TerminateBoundary,
    BounceBoundary, 
    WrapBoundary,
    ClipBoundary,
    create_boundary_policy,
    BoundaryConfig
)


# Test constants for performance validation per Section 6.6.6.3
BOUNDARY_VIOLATION_THRESHOLD_MS = 1.0  # <1ms for 100 agents
STEP_LATENCY_THRESHOLD_MS = 33.0  # ≤33ms step latency  
NUMERICAL_PRECISION_TOLERANCE = 1e-6  # Research accuracy standards


class TestBoundaryPolicyProtocol:
    """Test BoundaryPolicyProtocol interface compliance and type checking."""
    
    def test_protocol_compliance(self):
        """Test that all boundary policy implementations comply with BoundaryPolicyProtocol."""
        # Test TerminateBoundary compliance
        terminate_policy = TerminateBoundary(domain_bounds=(100, 100))
        assert isinstance(terminate_policy, BoundaryPolicyProtocol)
        
        # Test BounceBoundary compliance
        bounce_policy = BounceBoundary(domain_bounds=(100, 100))
        assert isinstance(bounce_policy, BoundaryPolicyProtocol)
        
        # Test WrapBoundary compliance 
        wrap_policy = WrapBoundary(domain_bounds=(100, 100))
        assert isinstance(wrap_policy, BoundaryPolicyProtocol)
        
        # Test ClipBoundary compliance
        clip_policy = ClipBoundary(domain_bounds=(100, 100))
        assert isinstance(clip_policy, BoundaryPolicyProtocol)
    
    def test_interface_implementation(self):
        """Test that all boundary policies implement required protocol methods."""
        policies = [
            TerminateBoundary(domain_bounds=(100, 100)),
            BounceBoundary(domain_bounds=(100, 100)),
            WrapBoundary(domain_bounds=(100, 100)),
            ClipBoundary(domain_bounds=(100, 100))
        ]
        
        required_methods = ['apply_policy', 'check_violations', 'get_termination_status', 'configure']
        
        for policy in policies:
            for method_name in required_methods:
                assert hasattr(policy, method_name), f"{type(policy).__name__} missing {method_name}"
                assert callable(getattr(policy, method_name)), f"{method_name} not callable"
    
    def test_method_signatures(self):
        """Test protocol method signatures and return types."""
        policy = TerminateBoundary(domain_bounds=(100, 100))
        
        # Test apply_policy signature
        positions = np.array([[50.0, 50.0]])
        result = policy.apply_policy(positions)
        assert isinstance(result, np.ndarray) or isinstance(result, tuple)
        
        # Test check_violations signature
        violations = policy.check_violations(positions)
        assert isinstance(violations, (np.ndarray, bool, np.bool_))
        
        # Test get_termination_status signature
        status = policy.get_termination_status()
        assert isinstance(status, str)
        
        # Test configure signature (should not raise)
        policy.configure(domain_bounds=(150, 150))
    
    def test_protocol_inheritance(self):
        """Test that BoundaryPolicyProtocol is properly implemented as runtime checkable."""
        # Verify protocol is runtime checkable
        assert hasattr(BoundaryPolicyProtocol, '__runtime_checkable__')
        
        # Test with mock object that implements protocol
        mock_policy = MagicMock()
        mock_policy.apply_policy = MagicMock(return_value=np.array([[0, 0]]))
        mock_policy.check_violations = MagicMock(return_value=np.array([False]))
        mock_policy.get_termination_status = MagicMock(return_value="continue")
        mock_policy.configure = MagicMock()
        
        # Mock should satisfy protocol (duck typing)
        assert hasattr(mock_policy, 'apply_policy')
        assert hasattr(mock_policy, 'check_violations')
        assert hasattr(mock_policy, 'get_termination_status')
        assert hasattr(mock_policy, 'configure')


class TestTerminateBoundary:
    """Test TerminateBoundary implementation for episode termination behavior."""
    
    @pytest.fixture
    def terminate_policy(self):
        """Create TerminateBoundary instance for testing."""
        return TerminateBoundary(domain_bounds=(100, 100))
    
    @pytest.fixture  
    def terminate_policy_negative_coords(self):
        """Create TerminateBoundary allowing negative coordinates."""
        return TerminateBoundary(domain_bounds=(100, 100), allow_negative_coords=True)
    
    def test_boundary_violation_detection(self, terminate_policy):
        """Test boundary violation detection for various position scenarios."""
        # Test positions within bounds
        valid_positions = np.array([
            [50.0, 50.0],   # Center
            [0.0, 0.0],     # Origin
            [99.9, 99.9],   # Near edge
            [1.0, 1.0]      # Near origin
        ])
        violations = terminate_policy.check_violations(valid_positions)
        assert np.all(violations == False), "Valid positions should not trigger violations"
        
        # Test positions outside bounds
        invalid_positions = np.array([
            [105.0, 50.0],   # Right boundary violation
            [50.0, 105.0],   # Top boundary violation  
            [-5.0, 50.0],    # Left boundary violation
            [50.0, -5.0],    # Bottom boundary violation
            [105.0, 105.0]   # Corner violation
        ])
        violations = terminate_policy.check_violations(invalid_positions)
        assert np.all(violations == True), "Invalid positions should trigger violations"
        
        # Test single agent case
        single_valid = np.array([25.0, 25.0])
        single_violation = terminate_policy.check_violations(single_valid)
        assert single_violation == False, "Single valid position should not violate"
        
        single_invalid = np.array([150.0, 25.0])
        single_violation = terminate_policy.check_violations(single_invalid)
        assert single_violation == True, "Single invalid position should violate"
    
    def test_episode_termination_behavior(self, terminate_policy):
        """Test episode termination status and behavior."""
        # Test termination status
        status = terminate_policy.get_termination_status()
        assert status == "oob", "Default termination status should be 'oob'"
        
        # Test custom termination status
        custom_policy = TerminateBoundary(
            domain_bounds=(100, 100), 
            status_on_violation="boundary_exit"
        )
        custom_status = custom_policy.get_termination_status()
        assert custom_status == "boundary_exit", "Custom status should be preserved"
    
    def test_configurable_status_messages(self, terminate_policy):
        """Test configurable termination status messages."""
        # Update status via configure
        terminate_policy.configure(status_on_violation="custom_termination")
        status = terminate_policy.get_termination_status()
        assert status == "custom_termination", "Status should update via configure"
        
        # Test multiple status updates
        terminate_policy.configure(status_on_violation="final_status")
        status = terminate_policy.get_termination_status()
        assert status == "final_status", "Status should update multiple times"
    
    def test_position_validation(self, terminate_policy):
        """Test position validation and edge cases."""
        # Test exact boundary positions
        boundary_positions = np.array([
            [100.0, 50.0],   # Exact right boundary
            [50.0, 100.0],   # Exact top boundary
            [0.0, 50.0],     # Exact left boundary  
            [50.0, 0.0]      # Exact bottom boundary
        ])
        violations = terminate_policy.check_violations(boundary_positions)
        # Boundary positions should trigger violations (outside domain)
        assert np.all(violations == True), "Exact boundary positions should violate"
        
        # Test negative coordinate handling
        negative_positions = np.array([
            [-1.0, 50.0],
            [50.0, -1.0],
            [-1.0, -1.0]
        ])
        violations = terminate_policy.check_violations(negative_positions)
        assert np.all(violations == True), "Negative positions should violate by default"
    
    def test_multi_agent_termination(self, terminate_policy):
        """Test termination behavior with multiple agents."""
        # Test mixed valid/invalid positions
        mixed_positions = np.array([
            [25.0, 25.0],    # Valid
            [105.0, 25.0],   # Invalid
            [75.0, 75.0],    # Valid
            [25.0, 105.0],   # Invalid
            [50.0, 50.0]     # Valid
        ])
        violations = terminate_policy.check_violations(mixed_positions)
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(violations, expected)
        
        # Test apply_policy returns unchanged positions
        result = terminate_policy.apply_policy(mixed_positions)
        np.testing.assert_array_equal(result, mixed_positions)
        
        # Test with velocities (should return both unchanged)
        velocities = np.array([
            [1.0, 0.0],
            [0.0, 1.0], 
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0]
        ])
        pos_result, vel_result = terminate_policy.apply_policy(mixed_positions, velocities)
        np.testing.assert_array_equal(pos_result, mixed_positions)
        np.testing.assert_array_equal(vel_result, velocities)


class TestBounceBoundary:
    """Test BounceBoundary implementation for elastic collision behavior."""
    
    @pytest.fixture
    def bounce_policy(self):
        """Create BounceBoundary instance for testing."""
        return BounceBoundary(domain_bounds=(100, 100), elasticity=1.0)
    
    @pytest.fixture
    def inelastic_bounce_policy(self):
        """Create inelastic BounceBoundary for energy loss testing."""
        return BounceBoundary(domain_bounds=(100, 100), elasticity=0.8, energy_loss=0.1)
    
    def test_elastic_collision_behavior(self, bounce_policy):
        """Test elastic collision physics with perfect energy conservation."""
        # Test right boundary collision
        positions = np.array([[105.0, 50.0]])  # Outside right boundary
        velocities = np.array([[2.0, 1.0]])    # Moving right
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # Position should be reflected back into domain
        assert corrected_pos[0, 0] <= 100.0, "Position should be within domain"
        
        # X velocity should be reflected, Y velocity unchanged
        assert corrected_vel[0, 0] < 0, "X velocity should be reflected (negative)"
        assert abs(corrected_vel[0, 1] - 1.0) < NUMERICAL_PRECISION_TOLERANCE, "Y velocity should be unchanged"
        
        # Test energy conservation (elastic collision)
        original_speed = np.linalg.norm(velocities[0])
        corrected_speed = np.linalg.norm(corrected_vel[0])
        np.testing.assert_allclose(original_speed, corrected_speed, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_velocity_reflection_physics(self, bounce_policy):
        """Test velocity reflection physics for different boundary collisions."""
        # Test left boundary collision
        positions = np.array([[-5.0, 50.0]])
        velocities = np.array([[-1.0, 0.5]])
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # X velocity should be reflected positive, Y unchanged
        assert corrected_vel[0, 0] > 0, "Left collision should reflect X velocity positive"
        assert abs(corrected_vel[0, 1] - 0.5) < NUMERICAL_PRECISION_TOLERANCE
        
        # Test top boundary collision
        positions = np.array([[50.0, 105.0]])
        velocities = np.array([[0.5, 2.0]])
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # Y velocity should be reflected negative, X unchanged
        assert corrected_vel[0, 1] < 0, "Top collision should reflect Y velocity negative"
        assert abs(corrected_vel[0, 0] - 0.5) < NUMERICAL_PRECISION_TOLERANCE
        
        # Test bottom boundary collision
        positions = np.array([[50.0, -5.0]])
        velocities = np.array([[0.5, -1.0]])
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # Y velocity should be reflected positive, X unchanged
        assert corrected_vel[0, 1] > 0, "Bottom collision should reflect Y velocity positive"
        assert abs(corrected_vel[0, 0] - 0.5) < NUMERICAL_PRECISION_TOLERANCE
    
    def test_configurable_elasticity(self, inelastic_bounce_policy):
        """Test configurable elasticity and energy loss."""
        positions = np.array([[105.0, 50.0]])
        velocities = np.array([[2.0, 0.0]])
        
        corrected_pos, corrected_vel = inelastic_bounce_policy.apply_policy(positions, velocities)
        
        # Velocity should be reduced due to elasticity and energy loss
        expected_speed = 2.0 * 0.8 * (1 - 0.1)  # elasticity * (1 - energy_loss)
        actual_speed = abs(corrected_vel[0, 0])
        np.testing.assert_allclose(actual_speed, expected_speed, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test elasticity configuration
        inelastic_bounce_policy.set_elasticity(0.5)
        corrected_pos, corrected_vel = inelastic_bounce_policy.apply_policy(positions, velocities)
        expected_speed = 2.0 * 0.5 * (1 - 0.1)  # New elasticity
        actual_speed = abs(corrected_vel[0, 0])
        np.testing.assert_allclose(actual_speed, expected_speed, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_energy_conservation(self, bounce_policy):
        """Test energy conservation in elastic collisions."""
        # Test corner collision (both X and Y reflection)
        positions = np.array([[105.0, 105.0]])
        velocities = np.array([[1.5, 1.0]])
        
        original_energy = 0.5 * np.sum(velocities[0]**2)
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        corrected_energy = 0.5 * np.sum(corrected_vel[0]**2)
        
        # Energy should be conserved in elastic collision
        np.testing.assert_allclose(original_energy, corrected_energy, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Both velocity components should be reflected
        assert corrected_vel[0, 0] < 0, "X velocity should be reflected"
        assert corrected_vel[0, 1] < 0, "Y velocity should be reflected"
    
    def test_multi_collision_scenarios(self, bounce_policy):
        """Test multiple agents with simultaneous collisions."""
        positions = np.array([
            [105.0, 50.0],   # Right collision
            [-5.0, 75.0],    # Left collision
            [50.0, 105.0],   # Top collision
            [25.0, -5.0],    # Bottom collision
            [50.0, 50.0]     # No collision
        ])
        
        velocities = np.array([
            [2.0, 1.0],      # Moving right-up
            [-1.0, 0.5],     # Moving left-up
            [1.0, 2.0],      # Moving right-up
            [0.5, -1.0],     # Moving right-down
            [1.0, 1.0]       # Moving right-up (no collision)
        ])
        
        corrected_pos, corrected_vel = bounce_policy.apply_policy(positions, velocities)
        
        # Check collision corrections
        assert corrected_vel[0, 0] < 0, "Right collision X velocity should be negative"
        assert corrected_vel[1, 0] > 0, "Left collision X velocity should be positive"
        assert corrected_vel[2, 1] < 0, "Top collision Y velocity should be negative"
        assert corrected_vel[3, 1] > 0, "Bottom collision Y velocity should be positive"
        
        # No collision case should be unchanged
        np.testing.assert_allclose(corrected_vel[4], velocities[4], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test velocities not provided raises error
        with pytest.raises(ValueError, match="requires velocities"):
            bounce_policy.apply_policy(positions)


class TestWrapBoundary:
    """Test WrapBoundary implementation for periodic boundary conditions."""
    
    @pytest.fixture
    def wrap_policy(self):
        """Create WrapBoundary instance for testing."""
        return WrapBoundary(domain_bounds=(100, 100))
    
    def test_periodic_boundary_conditions(self, wrap_policy):
        """Test periodic wrapping for toroidal domain topology."""
        # Test right boundary wrapping
        positions = np.array([[105.0, 50.0]])  # Beyond right boundary
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected_x = 105.0 % 100.0  # Should wrap to 5.0
        np.testing.assert_allclose(wrapped_pos[0, 0], expected_x, atol=NUMERICAL_PRECISION_TOLERANCE)
        assert wrapped_pos[0, 1] == 50.0, "Y coordinate should be unchanged"
        
        # Test left boundary wrapping (negative values)
        positions = np.array([[-10.0, 50.0]])
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected_x = (-10.0) % 100.0  # Should wrap to 90.0
        np.testing.assert_allclose(wrapped_pos[0, 0], expected_x, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test top boundary wrapping
        positions = np.array([[50.0, 110.0]])
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected_y = 110.0 % 100.0  # Should wrap to 10.0
        np.testing.assert_allclose(wrapped_pos[0, 1], expected_y, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test bottom boundary wrapping
        positions = np.array([[50.0, -15.0]])
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected_y = (-15.0) % 100.0  # Should wrap to 85.0
        np.testing.assert_allclose(wrapped_pos[0, 1], expected_y, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_position_wrapping_behavior(self, wrap_policy):
        """Test position wrapping behavior with different scenarios."""
        # Test corner wrapping (both coordinates)
        positions = np.array([[105.0, 110.0]])
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected = np.array([[5.0, 10.0]])
        np.testing.assert_allclose(wrapped_pos, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test multiple full domain wraps
        positions = np.array([[250.0, 275.0]])  # 2.5 domain widths
        wrapped_pos = wrap_policy.apply_policy(positions)
        expected = np.array([[50.0, 75.0]])  # Should wrap to middle
        np.testing.assert_allclose(wrapped_pos, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test positions already within bounds (no wrapping needed)
        positions = np.array([[25.0, 75.0]])
        wrapped_pos = wrap_policy.apply_policy(positions)
        np.testing.assert_allclose(wrapped_pos, positions, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_continuous_trajectory_validation(self, wrap_policy):
        """Test that wrapping preserves trajectory continuity."""
        # Test velocity preservation during wrapping
        positions = np.array([[105.0, 50.0]])
        velocities = np.array([[2.0, 1.5]])
        
        wrapped_pos, preserved_vel = wrap_policy.apply_policy(positions, velocities)
        
        # Velocities should be completely preserved
        np.testing.assert_allclose(preserved_vel, velocities, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Position should be wrapped
        assert wrapped_pos[0, 0] == 5.0, "Position should be wrapped to 5.0"
        assert wrapped_pos[0, 1] == 50.0, "Y position should be unchanged"
    
    def test_toroidal_domain_management(self, wrap_policy):
        """Test toroidal domain topology management."""
        # Test single agent wrapping
        single_pos = np.array([120.0, 80.0])
        wrapped_single = wrap_policy.apply_policy(single_pos)
        
        # Should work with single agent (1D input)
        assert wrapped_single.shape == (2,), "Single agent should return 1D array"
        np.testing.assert_allclose(wrapped_single, [20.0, 80.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test with velocities for single agent
        single_vel = np.array([1.0, -0.5])
        wrapped_pos, wrapped_vel = wrap_policy.apply_policy(single_pos, single_vel)
        
        assert wrapped_pos.shape == (2,), "Single position should be 1D"
        assert wrapped_vel.shape == (2,), "Single velocity should be 1D"
        np.testing.assert_allclose(wrapped_vel, single_vel, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_wrap_scenarios(self, wrap_policy):
        """Test wrapping with multiple agents simultaneously."""
        positions = np.array([
            [105.0, 50.0],   # Right wrap needed
            [-10.0, 75.0],   # Left wrap needed  
            [50.0, 110.0],   # Top wrap needed
            [25.0, -15.0],   # Bottom wrap needed
            [40.0, 60.0],    # No wrap needed
            [200.0, 250.0]   # Both coordinates wrap needed
        ])
        
        wrapped_pos = wrap_policy.apply_policy(positions)
        
        expected = np.array([
            [5.0, 50.0],     # 105 % 100 = 5
            [90.0, 75.0],    # -10 % 100 = 90
            [50.0, 10.0],    # 110 % 100 = 10
            [25.0, 85.0],    # -15 % 100 = 85
            [40.0, 60.0],    # No change
            [0.0, 50.0]      # 200 % 100 = 0, 250 % 100 = 50
        ])
        
        np.testing.assert_allclose(wrapped_pos, expected, atol=NUMERICAL_PRECISION_TOLERANCE)


class TestClipBoundary:
    """Test ClipBoundary implementation for hard position constraints."""
    
    @pytest.fixture
    def clip_policy(self):
        """Create ClipBoundary instance for testing."""
        return ClipBoundary(domain_bounds=(100, 100))
    
    @pytest.fixture
    def damping_clip_policy(self):
        """Create ClipBoundary with velocity damping enabled."""
        return ClipBoundary(
            domain_bounds=(100, 100),
            velocity_damping=0.7,
            damp_at_boundary=True
        )
    
    def test_hard_boundary_constraints(self, clip_policy):
        """Test hard position constraints preventing boundary crossing."""
        # Test positions outside boundaries get clipped
        positions = np.array([
            [105.0, 50.0],   # Right boundary violation
            [-10.0, 75.0],   # Left boundary violation
            [50.0, 110.0],   # Top boundary violation
            [25.0, -15.0],   # Bottom boundary violation
            [105.0, 110.0]   # Corner violation
        ])
        
        clipped_pos = clip_policy.apply_policy(positions)
        
        # All positions should be within bounds
        assert np.all(clipped_pos[:, 0] >= 0.0), "X coordinates should be >= 0"
        assert np.all(clipped_pos[:, 0] <= 100.0), "X coordinates should be <= 100"
        assert np.all(clipped_pos[:, 1] >= 0.0), "Y coordinates should be >= 0"  
        assert np.all(clipped_pos[:, 1] <= 100.0), "Y coordinates should be <= 100"
        
        # Check specific clipping behavior
        expected = np.array([
            [100.0, 50.0],   # Clipped to right boundary
            [0.0, 75.0],     # Clipped to left boundary
            [50.0, 100.0],   # Clipped to top boundary
            [25.0, 0.0],     # Clipped to bottom boundary
            [100.0, 100.0]   # Clipped to corner
        ])
        
        np.testing.assert_allclose(clipped_pos, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_position_clipping_behavior(self, clip_policy):
        """Test position clipping behavior for edge cases."""
        # Test positions exactly on boundaries
        boundary_positions = np.array([
            [0.0, 50.0],     # Left boundary
            [100.0, 50.0],   # Right boundary
            [50.0, 0.0],     # Bottom boundary
            [50.0, 100.0],   # Top boundary
            [0.0, 0.0],      # Bottom-left corner
            [100.0, 100.0]   # Top-right corner
        ])
        
        clipped_pos = clip_policy.apply_policy(boundary_positions)
        
        # Boundary positions should remain unchanged (already valid)
        np.testing.assert_allclose(clipped_pos, boundary_positions, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test positions slightly outside boundaries
        slightly_outside = np.array([
            [100.1, 50.0],   # Slightly right
            [-0.1, 50.0],    # Slightly left
            [50.0, 100.1],   # Slightly top
            [50.0, -0.1]     # Slightly bottom
        ])
        
        clipped_pos = clip_policy.apply_policy(slightly_outside)
        
        expected = np.array([
            [100.0, 50.0],   # Clipped to boundary
            [0.0, 50.0],     # Clipped to boundary
            [50.0, 100.0],   # Clipped to boundary
            [50.0, 0.0]      # Clipped to boundary
        ])
        
        np.testing.assert_allclose(clipped_pos, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_constraint_enforcement(self, clip_policy):
        """Test constraint enforcement guarantees."""
        # Generate random positions far outside boundaries
        np.random.seed(42)  # Deterministic test
        positions = np.random.rand(20, 2) * 1000 - 500  # Range [-500, 500]
        
        clipped_pos = clip_policy.apply_policy(positions)
        
        # Guarantee: All positions must be within domain bounds
        assert np.all(clipped_pos[:, 0] >= 0.0), "All X coordinates must be >= 0"
        assert np.all(clipped_pos[:, 0] <= 100.0), "All X coordinates must be <= 100"
        assert np.all(clipped_pos[:, 1] >= 0.0), "All Y coordinates must be >= 0"
        assert np.all(clipped_pos[:, 1] <= 100.0), "All Y coordinates must be <= 100"
        
        # Test with very large values
        extreme_positions = np.array([
            [1e6, 1e6],      # Very large positive
            [-1e6, -1e6],    # Very large negative
            [1e6, -1e6],     # Mixed extreme
            [-1e6, 1e6]      # Mixed extreme
        ])
        
        clipped_extreme = clip_policy.apply_policy(extreme_positions)
        
        # Should clip to domain bounds
        expected_extreme = np.array([
            [100.0, 100.0],
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0]
        ])
        
        np.testing.assert_allclose(clipped_extreme, expected_extreme, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_velocity_handling(self, damping_clip_policy):
        """Test velocity handling with damping at boundaries."""
        # Test velocity damping when agents are at boundaries
        positions = np.array([
            [100.0, 50.0],   # At right boundary
            [0.0, 75.0],     # At left boundary
            [50.0, 100.0],   # At top boundary
            [25.0, 0.0],     # At bottom boundary
            [50.0, 50.0]     # Interior (no damping)
        ])
        
        velocities = np.array([
            [2.0, 1.0],      # Moving away from boundary
            [-1.0, 0.5],     # Moving away from boundary
            [1.0, 2.0],      # Moving away from boundary
            [0.5, -1.0],     # Moving away from boundary
            [1.0, 1.0]       # Interior velocity
        ])
        
        clipped_pos, modified_vel = damping_clip_policy.apply_policy(positions, velocities)
        
        # Positions should remain at boundaries
        np.testing.assert_allclose(clipped_pos, positions, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Velocities at boundaries should be damped
        expected_damping = 0.7
        for i in range(4):  # First 4 agents at boundaries
            original_speed = np.linalg.norm(velocities[i])
            modified_speed = np.linalg.norm(modified_vel[i])
            expected_speed = original_speed * expected_damping
            np.testing.assert_allclose(modified_speed, expected_speed, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Interior agent should have unchanged velocity
        np.testing.assert_allclose(modified_vel[4], velocities[4], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_clipping(self, clip_policy):
        """Test clipping behavior with multiple agents."""
        # Test single agent format
        single_pos = np.array([105.0, 50.0])
        clipped_single = clip_policy.apply_policy(single_pos)
        
        assert clipped_single.shape == (2,), "Single agent should return 1D array"
        np.testing.assert_allclose(clipped_single, [100.0, 50.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test with velocities for single agent
        single_vel = np.array([2.0, 1.0])
        clipped_pos, clipped_vel = clip_policy.apply_policy(single_pos, single_vel)
        
        assert clipped_pos.shape == (2,), "Single position should be 1D"
        assert clipped_vel.shape == (2,), "Single velocity should be 1D"
        
        # Test large multi-agent scenario
        num_agents = 50
        np.random.seed(123)
        positions = np.random.rand(num_agents, 2) * 200 - 50  # Range [-50, 150]
        
        clipped_pos = clip_policy.apply_policy(positions)
        
        # All agents should be within bounds
        assert clipped_pos.shape == (num_agents, 2), "Shape should be preserved"
        assert np.all(clipped_pos >= 0.0), "All coordinates should be >= 0"
        assert np.all(clipped_pos <= 100.0), "All coordinates should be <= 100"


class TestVectorizedBoundaryChecking:
    """Test vectorized boundary checking for multi-agent scenarios."""
    
    @pytest.fixture
    def policies(self):
        """Create all boundary policy types for testing."""
        return {
            'terminate': TerminateBoundary(domain_bounds=(100, 100)),
            'bounce': BounceBoundary(domain_bounds=(100, 100)),
            'wrap': WrapBoundary(domain_bounds=(100, 100)),
            'clip': ClipBoundary(domain_bounds=(100, 100))
        }
    
    def test_multi_agent_boundary_validation(self, policies):
        """Test boundary validation with multiple agents."""
        # Create test scenario with 10 agents
        positions = np.array([
            [25.0, 25.0],    # Valid
            [105.0, 50.0],   # Right violation
            [75.0, 75.0],    # Valid
            [-5.0, 25.0],    # Left violation
            [50.0, 105.0],   # Top violation
            [10.0, 10.0],    # Valid
            [50.0, -5.0],    # Bottom violation
            [90.0, 90.0],    # Valid
            [110.0, 110.0],  # Corner violation
            [5.0, 95.0]      # Valid
        ])
        
        expected_violations = np.array([False, True, False, True, True, False, True, False, True, False])
        
        # Test all policy types have consistent violation detection
        for policy_name, policy in policies.items():
            violations = policy.check_violations(positions)
            np.testing.assert_array_equal(
                violations, expected_violations,
                f"{policy_name} policy violation detection mismatch"
            )
    
    def test_vectorized_violation_detection(self, policies):
        """Test vectorized violation detection performance and correctness."""
        # Test with varying agent counts
        agent_counts = [1, 5, 10, 25, 50, 100]
        
        for count in agent_counts:
            # Generate test positions
            np.random.seed(42 + count)  # Deterministic but different per count
            positions = np.random.rand(count, 2) * 150 - 25  # Range [-25, 125]
            
            for policy_name, policy in policies.items():
                violations = policy.check_violations(positions)
                
                # Verify output shape and type
                assert violations.shape == (count,), f"{policy_name}: Wrong violation shape for {count} agents"
                assert violations.dtype == bool, f"{policy_name}: Violations should be boolean"
                
                # Verify vectorized vs individual results match
                individual_violations = []
                for i in range(count):
                    single_violation = policy.check_violations(positions[i])
                    individual_violations.append(single_violation)
                
                np.testing.assert_array_equal(
                    violations, individual_violations,
                    f"{policy_name}: Vectorized results don't match individual results"
                )
    
    def test_batch_position_correction(self, policies):
        """Test batch position correction for boundary policies."""
        # Test positions requiring correction
        positions = np.array([
            [105.0, 50.0],   # Right boundary
            [-10.0, 75.0],   # Left boundary
            [50.0, 110.0],   # Top boundary
            [25.0, -15.0],   # Bottom boundary
            [50.0, 50.0]     # No correction needed
        ])
        
        velocities = np.array([
            [2.0, 1.0],
            [-1.0, 0.5],
            [1.0, 2.0],
            [0.5, -1.0],
            [1.0, 1.0]
        ])
        
        # Test each policy applies corrections correctly
        for policy_name, policy in policies.items():
            if policy_name == 'bounce':
                # Bounce requires velocities
                corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
                assert corrected_pos.shape == positions.shape
                assert corrected_vel.shape == velocities.shape
                
                # Verify all positions are within bounds after bounce correction
                violations_after = policy.check_violations(corrected_pos)
                assert not np.any(violations_after), f"{policy_name}: Positions still violate after correction"
                
            elif policy_name == 'terminate':
                # Terminate doesn't modify positions
                result = policy.apply_policy(positions)
                np.testing.assert_array_equal(result, positions)
                
            else:
                # Wrap and clip modify positions
                corrected_pos = policy.apply_policy(positions)
                assert corrected_pos.shape == positions.shape
                
                # Verify corrections are applied
                violations_after = policy.check_violations(corrected_pos)
                if policy_name == 'wrap':
                    # Wrap should eliminate violations by wrapping to valid range
                    assert not np.any(violations_after), f"{policy_name}: Positions still violate after wrapping"
                elif policy_name == 'clip':
                    # Clip should eliminate violations by clipping to bounds
                    assert not np.any(violations_after), f"{policy_name}: Positions still violate after clipping"
    
    def test_concurrent_policy_application(self, policies):
        """Test concurrent policy application efficiency."""
        # Test all policies can handle same input simultaneously
        num_agents = 20
        np.random.seed(456)
        positions = np.random.rand(num_agents, 2) * 150 - 25
        velocities = np.random.rand(num_agents, 2) * 4 - 2  # Range [-2, 2]
        
        results = {}
        
        # Apply all policies to same input
        for policy_name, policy in policies.items():
            if policy_name == 'bounce':
                results[policy_name] = policy.apply_policy(positions, velocities)
            else:
                results[policy_name] = policy.apply_policy(positions)
        
        # Verify each policy produces valid results
        for policy_name, result in results.items():
            if policy_name == 'bounce':
                corrected_pos, corrected_vel = result
                assert corrected_pos.shape == positions.shape
                assert corrected_vel.shape == velocities.shape
            else:
                if isinstance(result, tuple):
                    corrected_pos, corrected_vel = result
                    assert corrected_pos.shape == positions.shape
                else:
                    corrected_pos = result
                    assert corrected_pos.shape == positions.shape
    
    def test_100_agent_performance(self, policies):
        """Test performance with 100 agents meeting requirements."""
        num_agents = 100
        np.random.seed(789)
        positions = np.random.rand(num_agents, 2) * 150 - 25
        velocities = np.random.rand(num_agents, 2) * 4 - 2
        
        for policy_name, policy in policies.items():
            # Test violation detection performance
            start_time = time.perf_counter()
            violations = policy.check_violations(positions)
            violation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            assert violation_time < BOUNDARY_VIOLATION_THRESHOLD_MS, \
                f"{policy_name}: Violation detection took {violation_time:.2f}ms, should be <{BOUNDARY_VIOLATION_THRESHOLD_MS}ms"
            
            # Test policy application performance
            start_time = time.perf_counter()
            if policy_name == 'bounce':
                result = policy.apply_policy(positions, velocities)
            else:
                result = policy.apply_policy(positions)
            application_time = (time.perf_counter() - start_time) * 1000
            
            assert application_time < STEP_LATENCY_THRESHOLD_MS, \
                f"{policy_name}: Policy application took {application_time:.2f}ms, should be <{STEP_LATENCY_THRESHOLD_MS}ms"
            
            # Verify results are correct
            assert violations.shape == (num_agents,)
            assert violations.dtype == bool


class TestBoundaryPerformance:
    """Test boundary policy performance requirements and scaling."""
    
    @pytest.fixture
    def performance_policies(self):
        """Create boundary policies for performance testing."""
        return [
            TerminateBoundary(domain_bounds=(100, 100)),
            BounceBoundary(domain_bounds=(100, 100)),
            WrapBoundary(domain_bounds=(100, 100)),
            ClipBoundary(domain_bounds=(100, 100))
        ]
    
    def test_sub_millisecond_violation_detection(self, performance_policies):
        """Test sub-millisecond violation detection requirement."""
        # Test with different agent counts
        agent_counts = [10, 25, 50, 100]
        
        for count in agent_counts:
            np.random.seed(100 + count)
            positions = np.random.rand(count, 2) * 150 - 25
            
            for policy in performance_policies:
                policy_name = type(policy).__name__
                
                # Warm up
                policy.check_violations(positions)
                
                # Measure performance
                start_time = time.perf_counter()
                violations = policy.check_violations(positions)
                detection_time = (time.perf_counter() - start_time) * 1000
                
                # Verify sub-millisecond for 100 agents
                if count == 100:
                    assert detection_time < BOUNDARY_VIOLATION_THRESHOLD_MS, \
                        f"{policy_name}: Detection took {detection_time:.3f}ms for {count} agents, should be <{BOUNDARY_VIOLATION_THRESHOLD_MS}ms"
                
                # Verify results are valid
                assert len(violations) == count
                assert isinstance(violations, np.ndarray)
    
    def test_step_latency_compliance(self, performance_policies):
        """Test step latency compliance with ≤33ms requirement."""
        num_agents = 100
        np.random.seed(200)
        positions = np.random.rand(num_agents, 2) * 150 - 25
        velocities = np.random.rand(num_agents, 2) * 4 - 2
        
        for policy in performance_policies:
            policy_name = type(policy).__name__
            
            # Warm up
            if policy_name == 'BounceBoundary':
                policy.apply_policy(positions, velocities)
            else:
                policy.apply_policy(positions)
            
            # Measure step latency (full policy application)
            start_time = time.perf_counter()
            
            if policy_name == 'BounceBoundary':
                result = policy.apply_policy(positions, velocities)
            else:
                result = policy.apply_policy(positions)
            
            step_time = (time.perf_counter() - start_time) * 1000
            
            assert step_time < STEP_LATENCY_THRESHOLD_MS, \
                f"{policy_name}: Step latency {step_time:.2f}ms exceeds {STEP_LATENCY_THRESHOLD_MS}ms limit"
    
    def test_multi_agent_scaling(self, performance_policies):
        """Test performance scaling with increasing agent count."""
        agent_counts = [1, 10, 25, 50, 100]
        scaling_results = {}
        
        for policy in performance_policies:
            policy_name = type(policy).__name__
            scaling_results[policy_name] = []
            
            for count in agent_counts:
                np.random.seed(300 + count)
                positions = np.random.rand(count, 2) * 150 - 25
                velocities = np.random.rand(count, 2) * 4 - 2
                
                # Measure violation detection time
                start_time = time.perf_counter()
                policy.check_violations(positions)
                detection_time = time.perf_counter() - start_time
                
                # Measure policy application time
                start_time = time.perf_counter()
                if policy_name == 'BounceBoundary':
                    policy.apply_policy(positions, velocities)
                else:
                    policy.apply_policy(positions)
                application_time = time.perf_counter() - start_time
                
                scaling_results[policy_name].append({
                    'count': count,
                    'detection_time': detection_time,
                    'application_time': application_time
                })
        
        # Verify linear or sub-linear scaling
        for policy_name, results in scaling_results.items():
            # Check that 100 agents doesn't take more than 10x the time of 10 agents
            time_10 = next(r['detection_time'] for r in results if r['count'] == 10)
            time_100 = next(r['detection_time'] for r in results if r['count'] == 100)
            
            scaling_factor = time_100 / time_10 if time_10 > 0 else 0
            assert scaling_factor <= 15, \
                f"{policy_name}: Poor scaling {scaling_factor:.1f}x from 10 to 100 agents"
    
    def test_memory_efficiency(self, performance_policies):
        """Test memory efficiency for large agent counts."""
        import psutil
        import gc
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with large agent count
        num_agents = 100
        positions = np.random.rand(num_agents, 2) * 150 - 25
        velocities = np.random.rand(num_agents, 2) * 4 - 2
        
        for policy in performance_policies:
            policy_name = type(policy).__name__
            
            # Perform multiple operations
            for _ in range(10):
                violations = policy.check_violations(positions)
                if policy_name == 'BounceBoundary':
                    result = policy.apply_policy(positions, velocities)
                else:
                    result = policy.apply_policy(positions)
            
            # Check memory usage
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - baseline_memory
            
            # Should not use excessive memory for boundary operations
            assert memory_increase < 50, \
                f"{policy_name}: Excessive memory usage {memory_increase:.1f}MB"
    
    def test_performance_regression_detection(self, performance_policies):
        """Test performance regression detection with baseline validation."""
        # Establish performance baselines (adjust based on system capabilities)
        baselines = {
            'TerminateBoundary': {'detection': 0.5, 'application': 0.1},  # ms
            'BounceBoundary': {'detection': 0.5, 'application': 5.0},      # ms
            'WrapBoundary': {'detection': 0.5, 'application': 1.0},        # ms
            'ClipBoundary': {'detection': 0.5, 'application': 1.0}         # ms
        }
        
        num_agents = 50  # Reasonable test size
        np.random.seed(400)
        positions = np.random.rand(num_agents, 2) * 150 - 25
        velocities = np.random.rand(num_agents, 2) * 4 - 2
        
        for policy in performance_policies:
            policy_name = type(policy).__name__
            baseline = baselines[policy_name]
            
            # Warm up
            for _ in range(3):
                policy.check_violations(positions)
                if policy_name == 'BounceBoundary':
                    policy.apply_policy(positions, velocities)
                else:
                    policy.apply_policy(positions)
            
            # Measure detection performance
            detection_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                policy.check_violations(positions)
                detection_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_detection_time = np.mean(detection_times)
            
            # Measure application performance
            application_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                if policy_name == 'BounceBoundary':
                    policy.apply_policy(positions, velocities)
                else:
                    policy.apply_policy(positions)
                application_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_application_time = np.mean(application_times)
            
            # Allow 2x baseline as acceptable performance range
            assert avg_detection_time < baseline['detection'] * 2, \
                f"{policy_name}: Detection time {avg_detection_time:.2f}ms exceeds 2x baseline {baseline['detection'] * 2}ms"
            
            assert avg_application_time < baseline['application'] * 2, \
                f"{policy_name}: Application time {avg_application_time:.2f}ms exceeds 2x baseline {baseline['application'] * 2}ms"


class TestBoundaryPolicyFactory:
    """Test boundary policy factory method and configuration integration."""
    
    def test_factory_instantiation(self):
        """Test factory method instantiation for all policy types."""
        domain_bounds = (100, 100)
        
        # Test terminate policy creation
        terminate_policy = create_boundary_policy("terminate", domain_bounds)
        assert isinstance(terminate_policy, TerminateBoundary)
        assert terminate_policy.domain_bounds == domain_bounds
        
        # Test bounce policy creation
        bounce_policy = create_boundary_policy(
            "bounce", domain_bounds, elasticity=0.8, energy_loss=0.1
        )
        assert isinstance(bounce_policy, BounceBoundary)
        assert bounce_policy.elasticity == 0.8
        assert bounce_policy.energy_loss == 0.1
        
        # Test wrap policy creation
        wrap_policy = create_boundary_policy("wrap", domain_bounds)
        assert isinstance(wrap_policy, WrapBoundary)
        assert wrap_policy.domain_bounds == domain_bounds
        
        # Test clip policy creation  
        clip_policy = create_boundary_policy(
            "clip", domain_bounds, velocity_damping=0.7, damp_at_boundary=True
        )
        assert isinstance(clip_policy, ClipBoundary)
        assert clip_policy.velocity_damping == 0.7
        assert clip_policy.damp_at_boundary == True
    
    def test_configuration_validation(self):
        """Test configuration parameter validation in factory."""
        domain_bounds = (100, 100)
        
        # Test invalid policy type
        with pytest.raises(ValueError, match="Unknown policy_type"):
            create_boundary_policy("invalid_type", domain_bounds)
        
        # Test invalid parameters for specific policies
        with pytest.raises(ValueError):
            create_boundary_policy("bounce", domain_bounds, elasticity=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            create_boundary_policy("clip", domain_bounds, velocity_damping=-0.1)  # < 0.0
        
        # Test invalid domain bounds
        with pytest.raises(ValueError):
            create_boundary_policy("terminate", (-10, 100))  # Negative bounds
        
        with pytest.raises(ValueError):
            create_boundary_policy("wrap", (100,))  # Wrong dimensions
    
    def test_hydra_integration(self):
        """Test Hydra configuration integration patterns."""
        # Test configuration that would work with Hydra instantiate
        configs = [
            {
                'policy_type': 'terminate',
                'domain_bounds': (100, 100),
                'status_on_violation': 'out_of_bounds'
            },
            {
                'policy_type': 'bounce',
                'domain_bounds': (200, 150),
                'elasticity': 0.9,
                'energy_loss': 0.05
            },
            {
                'policy_type': 'wrap',
                'domain_bounds': (50, 50),
                'allow_negative_coords': False
            },
            {
                'policy_type': 'clip',
                'domain_bounds': (75, 75),
                'velocity_damping': 0.8,
                'damp_at_boundary': True
            }
        ]
        
        for config in configs:
            policy_type = config.pop('policy_type')
            policy = create_boundary_policy(policy_type, **config)
            
            # Verify policy was created with correct type
            expected_types = {
                'terminate': TerminateBoundary,
                'bounce': BounceBoundary,
                'wrap': WrapBoundary,
                'clip': ClipBoundary
            }
            assert isinstance(policy, expected_types[policy_type])
    
    def test_runtime_policy_selection(self):
        """Test runtime policy selection and parameter binding."""
        domain_bounds = (100, 100)
        
        # Test policy selection based on runtime parameters
        policy_configs = {
            'experiment_1': {'type': 'terminate', 'params': {}},
            'experiment_2': {'type': 'bounce', 'params': {'elasticity': 0.8}},
            'experiment_3': {'type': 'wrap', 'params': {}},
            'experiment_4': {'type': 'clip', 'params': {'velocity_damping': 0.5}}
        }
        
        created_policies = {}
        for exp_name, config in policy_configs.items():
            policy = create_boundary_policy(
                config['type'], domain_bounds, **config['params']
            )
            created_policies[exp_name] = policy
        
        # Verify each policy has correct configuration
        assert isinstance(created_policies['experiment_1'], TerminateBoundary)
        assert isinstance(created_policies['experiment_2'], BounceBoundary)
        assert created_policies['experiment_2'].elasticity == 0.8
        assert isinstance(created_policies['experiment_3'], WrapBoundary)
        assert isinstance(created_policies['experiment_4'], ClipBoundary)
        assert created_policies['experiment_4'].velocity_damping == 0.5
    
    def test_factory_error_handling(self):
        """Test factory error handling and informative error messages."""
        domain_bounds = (100, 100)
        
        # Test helpful error messages for invalid policy types
        try:
            create_boundary_policy("unknown", domain_bounds)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown policy_type 'unknown'" in str(e)
            assert "Available types:" in str(e)
        
        # Test parameter validation error propagation
        try:
            create_boundary_policy("bounce", domain_bounds, elasticity=2.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid parameters for bounce boundary policy" in str(e)


def test_boundary_policy_backwards_compatibility():
    """Test boundary policy backwards compatibility with legacy interfaces."""
    # Test that all policies work with legacy position formats
    legacy_single_position = np.array([50.0, 75.0])  # 1D array
    modern_single_position = np.array([[50.0, 75.0]])  # 2D array with 1 agent
    
    policies = [
        TerminateBoundary(domain_bounds=(100, 100)),
        BounceBoundary(domain_bounds=(100, 100)),
        WrapBoundary(domain_bounds=(100, 100)),
        ClipBoundary(domain_bounds=(100, 100))
    ]
    
    for policy in policies:
        policy_name = type(policy).__name__
        
        # Test violation detection with both formats
        legacy_violation = policy.check_violations(legacy_single_position)
        modern_violation = policy.check_violations(modern_single_position)
        
        # Results should be equivalent
        assert legacy_violation == modern_violation[0], \
            f"{policy_name}: Legacy and modern violation detection differ"
        
        # Test policy application with both formats
        if policy_name == 'BounceBoundary':
            # Bounce requires velocities
            legacy_vel = np.array([1.0, 0.5])
            modern_vel = np.array([[1.0, 0.5]])
            
            legacy_result = policy.apply_policy(legacy_single_position, legacy_vel)
            modern_result = policy.apply_policy(modern_single_position, modern_vel)
            
            # Results should be compatible
            assert isinstance(legacy_result, tuple), "Legacy should return tuple"
            assert isinstance(modern_result, tuple), "Modern should return tuple"
            
            legacy_pos, legacy_vel_out = legacy_result
            modern_pos, modern_vel_out = modern_result
            
            assert legacy_pos.shape == (2,), "Legacy position should be 1D"
            assert modern_pos.shape == (1, 2), "Modern position should be 2D"
            np.testing.assert_allclose(legacy_pos, modern_pos[0], atol=NUMERICAL_PRECISION_TOLERANCE)
        else:
            legacy_result = policy.apply_policy(legacy_single_position)
            modern_result = policy.apply_policy(modern_single_position)
            
            # Handle different return formats
            if isinstance(legacy_result, tuple):
                legacy_pos = legacy_result[0]
            else:
                legacy_pos = legacy_result
            
            if isinstance(modern_result, tuple):
                modern_pos = modern_result[0]
            else:
                modern_pos = modern_result
            
            assert legacy_pos.shape == (2,), "Legacy position should be 1D"
            assert modern_pos.shape == (1, 2), "Modern position should be 2D"
            np.testing.assert_allclose(legacy_pos, modern_pos[0], atol=NUMERICAL_PRECISION_TOLERANCE)


def test_boundary_error_handling_and_edge_cases():
    """Test boundary policy error handling and edge cases."""
    
    # Test invalid domain bounds
    invalid_bounds_cases = [
        (0, 100),      # Zero width
        (100, 0),      # Zero height
        (-50, 100),    # Negative width
        (100, -50),    # Negative height
        (100,),        # Wrong dimensions
        (100, 100, 50) # Too many dimensions
    ]
    
    for invalid_bounds in invalid_bounds_cases:
        with pytest.raises(ValueError):
            TerminateBoundary(domain_bounds=invalid_bounds)
        
        with pytest.raises(ValueError):
            BounceBoundary(domain_bounds=invalid_bounds)
    
    # Test invalid positions
    policy = TerminateBoundary(domain_bounds=(100, 100))
    
    invalid_positions = [
        np.array([]),                    # Empty array
        np.array([50.0]),               # Wrong dimensions
        np.array([[50.0]]),             # Wrong second dimension
        np.array([[[50.0, 75.0]]]),     # Too many dimensions
        np.array([[np.nan, 50.0]]),     # NaN values
        np.array([[np.inf, 50.0]]),     # Infinite values
    ]
    
    for invalid_pos in invalid_positions:
        with pytest.raises((ValueError, IndexError)):
            policy.check_violations(invalid_pos)
    
    # Test BounceBoundary error cases
    bounce_policy = BounceBoundary(domain_bounds=(100, 100))
    
    # Test missing velocities
    positions = np.array([[50.0, 50.0]])
    with pytest.raises(ValueError, match="requires velocities"):
        bounce_policy.apply_policy(positions)
    
    # Test mismatched position/velocity shapes
    velocities = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2 agents
    positions = np.array([[50.0, 50.0]])             # 1 agent
    
    with pytest.raises((ValueError, IndexError)):
        bounce_policy.apply_policy(positions, velocities)
    
    # Test BoundaryConfig validation
    with pytest.raises(ValueError):
        BoundaryConfig(domain_bounds=(100,))  # Wrong dimensions
    
    with pytest.raises(ValueError):
        BoundaryConfig(domain_bounds=(100, 100), boundary_buffer=-1.0)  # Negative buffer
    
    # Test configure method error handling
    policy = ClipBoundary(domain_bounds=(100, 100))
    
    with pytest.raises(ValueError):
        policy.configure(domain_bounds=(-10, 100))  # Invalid bounds
    
    with pytest.raises(ValueError):
        policy.configure(velocity_damping=1.5)  # Invalid damping value


if __name__ == "__main__":
    # Run tests with appropriate verbosity and coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        f"--cov=src.plume_nav_sim.core.boundaries",
        "--cov-report=term-missing"
    ])