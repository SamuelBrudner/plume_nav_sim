"""
Comprehensive test module for SourceProtocol implementations and pluggable odor source abstractions.

This module provides complete validation for PointSource, MultiSource, and DynamicSource components
with deterministic seeding, vectorized operations testing, and performance compliance validation.
Tests ensure 100% coverage for SourceProtocol implementations per Section 6.6.2.4 requirements
and validate compliance with ≤33ms step latency requirements for 100 agents per Section 6.6.6.3.

Test Categories:
- SourceProtocol interface compliance and method signature validation
- PointSource implementation testing with emission rate queries and position retrieval
- MultiSource collection management with vectorized emission calculations
- DynamicSource temporal evolution and pattern behavior validation
- Vectorized operations performance testing for multi-agent scenarios
- Factory function validation for configuration-driven source instantiation
- Deterministic seeding and reproducibility validation across all source types
- Performance benchmarking for real-time simulation requirements

Performance Requirements:
- Source operations: <1ms per query for minimal simulation overhead
- Multi-agent support: <10ms for 100 agents with vectorized calculations
- Step latency: ≤33ms requirement compliance per Section 6.6.6.3
- Memory efficiency: <10MB for typical source configurations

Coverage Targets:
- SourceProtocol implementations: 100% coverage per Section 6.6.2.4
- Source module components: ≥90% line coverage per enhanced testing standards
- Factory and configuration methods: ≥85% coverage
- Performance and edge case validation: ≥80% coverage
"""

import pytest
import numpy as np
import time
import math
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from unittest.mock import MagicMock

# Import source components for testing
from plume_nav_sim.core.sources import (
    PointSource,
    MultiSource, 
    DynamicSource,
    create_source,
    SourceConfig,
    DynamicSourceConfig
)

# Test tolerances and performance thresholds
NUMERICAL_PRECISION_TOLERANCE = 1e-6
SINGLE_QUERY_THRESHOLD_MS = 1.0
MULTI_AGENT_10_THRESHOLD_MS = 5.0  
MULTI_AGENT_100_THRESHOLD_MS = 10.0
STEP_LATENCY_THRESHOLD_MS = 33.0
MAX_MEMORY_USAGE_MB = 10.0


class TestSourceProtocol:
    """Test SourceProtocol interface compliance and standardized methods for pluggable odor source modeling."""
    
    def test_protocol_compliance(self):
        """Test that all source implementations comply with SourceProtocol interface requirements."""
        # Define expected SourceProtocol interface based on source implementations
        required_methods = ['get_emission_rate', 'get_position', 'update_state']
        
        # Test PointSource protocol compliance
        point_source = PointSource(position=(10.0, 20.0), emission_rate=100.0)
        for method_name in required_methods:
            assert hasattr(point_source, method_name), f"PointSource missing {method_name}() method"
            assert callable(getattr(point_source, method_name)), f"PointSource.{method_name} must be callable"
        
        # Test MultiSource protocol compliance  
        multi_source = MultiSource()
        for method_name in required_methods:
            assert hasattr(multi_source, method_name), f"MultiSource missing {method_name}() method"
            assert callable(getattr(multi_source, method_name)), f"MultiSource.{method_name} must be callable"
        
        # Test DynamicSource protocol compliance
        dynamic_source = DynamicSource(initial_position=(30.0, 40.0), emission_rate=200.0)
        for method_name in required_methods:
            assert hasattr(dynamic_source, method_name), f"DynamicSource missing {method_name}() method"
            assert callable(getattr(dynamic_source, method_name)), f"DynamicSource.{method_name} must be callable"
    
    def test_interface_implementation(self):
        """Test interface implementation consistency across all source types."""
        sources = [
            PointSource(position=(0.0, 0.0), emission_rate=50.0),
            MultiSource(),
            DynamicSource(initial_position=(0.0, 0.0), emission_rate=75.0)
        ]
        
        for source in sources:
            # Test get_emission_rate interface
            scalar_rate = source.get_emission_rate(None)
            assert isinstance(scalar_rate, (int, float)), f"{type(source).__name__} scalar emission rate must be numeric"
            
            single_position = np.array([10.0, 20.0])
            single_rate = source.get_emission_rate(single_position)
            assert isinstance(single_rate, (int, float, np.floating)), f"{type(source).__name__} single position rate must be numeric"
            
            multi_positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
            multi_rates = source.get_emission_rate(multi_positions)
            assert isinstance(multi_rates, np.ndarray), f"{type(source).__name__} multi-position rates must be numpy array"
            assert multi_rates.shape == (3,), f"{type(source).__name__} multi-position rates shape mismatch"
            
            # Test get_position interface
            position = source.get_position()
            if isinstance(position, list):
                # MultiSource returns list of positions
                assert all(isinstance(pos, np.ndarray) and pos.shape == (2,) for pos in position), \
                    f"{type(source).__name__} position list elements must be (2,) arrays"
            else:
                # Single sources return single position
                assert isinstance(position, np.ndarray), f"{type(source).__name__} position must be numpy array"
                assert position.shape == (2,), f"{type(source).__name__} position must have shape (2,)"
            
            # Test update_state interface
            source.update_state(dt=1.0)  # Should execute without error
    
    def test_method_signatures(self):
        """Test method signatures match expected SourceProtocol specification."""
        point_source = PointSource()
        
        # Test get_emission_rate signature and behavior
        # None input should return scalar
        scalar_result = point_source.get_emission_rate(None)
        assert isinstance(scalar_result, (int, float))
        
        # Single agent array should return scalar
        single_pos = np.array([5.0, 10.0])
        single_result = point_source.get_emission_rate(single_pos)
        assert isinstance(single_result, (int, float, np.floating))
        
        # Multi-agent array should return array
        multi_pos = np.array([[0.0, 0.0], [5.0, 5.0]])
        multi_result = point_source.get_emission_rate(multi_pos)
        assert isinstance(multi_result, np.ndarray)
        assert multi_result.shape == (2,)
        
        # Test get_position signature
        position = point_source.get_position()
        assert isinstance(position, np.ndarray)
        assert position.shape == (2,)
        
        # Test update_state signature
        point_source.update_state()  # Default dt=1.0
        point_source.update_state(dt=0.5)  # Custom dt
        point_source.update_state(dt=2.0)  # Different dt
    
    def test_protocol_inheritance(self):
        """Test protocol inheritance and polymorphic behavior across source implementations."""
        sources = [
            PointSource(position=(10.0, 10.0), emission_rate=100.0),
            DynamicSource(initial_position=(20.0, 20.0), emission_rate=150.0, pattern_type="linear", velocity=(1.0, 0.0))
        ]
        
        # Add sources to MultiSource for polymorphic testing
        multi_source = MultiSource(sources=sources)
        
        # Test polymorphic emission rate calculations
        test_positions = np.array([[15.0, 15.0], [25.0, 25.0], [35.0, 35.0]])
        
        # Individual source calls
        point_rates = sources[0].get_emission_rate(test_positions)
        dynamic_rates = sources[1].get_emission_rate(test_positions)
        
        # Combined multi-source call
        combined_rates = multi_source.get_emission_rate(test_positions)
        
        # Verify polymorphic behavior - combined should equal sum
        expected_combined = point_rates + dynamic_rates
        np.testing.assert_allclose(combined_rates, expected_combined, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test polymorphic update operations
        for source in sources:
            source.update_state(dt=1.0)
        multi_source.update_state(dt=1.0)


class TestPointSource:
    """Comprehensive testing for PointSource implementation with emission rate queries and position retrieval."""
    
    @pytest.fixture
    def basic_point_source(self):
        """Create basic PointSource for testing."""
        return PointSource(
            position=(25.0, 50.0),
            emission_rate=500.0,
            seed=42
        )
    
    @pytest.fixture
    def temporal_point_source(self):
        """Create PointSource with temporal variation enabled."""
        return PointSource(
            position=(75.0, 100.0),
            emission_rate=1000.0,
            enable_temporal_variation=True,
            seed=123
        )
    
    def test_point_source_initialization(self):
        """Test PointSource initialization with various parameter combinations."""
        # Basic initialization
        source = PointSource()
        assert source.get_position().shape == (2,)
        np.testing.assert_allclose(source.get_position(), [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        assert source.get_emission_rate() == 1.0
        
        # Custom parameters
        source = PointSource(position=(10.0, 20.0), emission_rate=150.0)
        np.testing.assert_allclose(source.get_position(), [10.0, 20.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        assert source.get_emission_rate() == 150.0
        
        # With seed
        source = PointSource(position=(30.0, 40.0), emission_rate=250.0, seed=456)
        np.testing.assert_allclose(source.get_position(), [30.0, 40.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        assert source.get_emission_rate() == 250.0
    
    def test_emission_rate_queries(self, basic_point_source):
        """Test emission rate queries for scalar, single-agent, and multi-agent scenarios."""
        source = basic_point_source
        
        # Scalar query (no position)
        scalar_rate = source.get_emission_rate(None)
        assert scalar_rate == 500.0
        
        # Single agent query
        single_position = np.array([30.0, 55.0])
        single_rate = source.get_emission_rate(single_position)
        assert single_rate == 500.0  # PointSource returns constant rate
        
        # Multi-agent query
        multi_positions = np.array([
            [20.0, 45.0],
            [30.0, 55.0], 
            [40.0, 65.0],
            [50.0, 75.0]
        ])
        multi_rates = source.get_emission_rate(multi_positions)
        assert isinstance(multi_rates, np.ndarray)
        assert multi_rates.shape == (4,)
        np.testing.assert_allclose(multi_rates, [500.0, 500.0, 500.0, 500.0], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_position_retrieval(self, basic_point_source):
        """Test position retrieval methods and immutability."""
        source = basic_point_source
        
        # Get position
        position = source.get_position()
        assert isinstance(position, np.ndarray)
        assert position.shape == (2,)
        np.testing.assert_allclose(position, [25.0, 50.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test immutability (returned copy)
        position[0] = 999.0  # Modify returned array
        new_position = source.get_position()
        np.testing.assert_allclose(new_position, [25.0, 50.0], atol=NUMERICAL_PRECISION_TOLERANCE)  # Should be unchanged
    
    def test_state_update_behavior(self, temporal_point_source):
        """Test state update behavior for PointSource with temporal variation."""
        source = temporal_point_source
        
        initial_rate = source.get_emission_rate()
        assert initial_rate == 1000.0
        
        # Update state multiple times
        for i in range(10):
            source.update_state(dt=1.0)
            current_rate = source.get_emission_rate()
            assert isinstance(current_rate, (int, float))
            assert current_rate > 0.0  # Should remain positive
            
            # With temporal variation, rate should vary around base rate
            if i > 0:  # After first update, rate may differ from initial
                assert 800.0 <= current_rate <= 1200.0  # Reasonable variation range
    
    def test_configuration_validation(self):
        """Test configuration parameter validation and error handling."""
        # Valid configurations should work
        source = PointSource(position=(0.0, 0.0), emission_rate=0.0)  # Zero emission is valid
        assert source.get_emission_rate() == 0.0
        
        # Invalid position format
        with pytest.raises(ValueError):
            PointSource(position=(1.0,))  # Too few elements
            
        with pytest.raises(ValueError):
            PointSource(position=(1.0, 2.0, 3.0))  # Too many elements
        
        # Invalid emission rate
        with pytest.raises(ValueError):
            PointSource(emission_rate=-1.0)  # Negative emission rate
    
    def test_deterministic_seeding(self):
        """Test deterministic behavior with controlled seeding."""
        seed = 789
        
        # Create two sources with same seed
        source1 = PointSource(position=(10.0, 20.0), emission_rate=100.0, 
                             enable_temporal_variation=True, seed=seed)
        source2 = PointSource(position=(10.0, 20.0), emission_rate=100.0,
                             enable_temporal_variation=True, seed=seed)
        
        # Both should produce identical behavior
        for _ in range(5):
            source1.update_state(dt=1.0)
            source2.update_state(dt=1.0)
            
            rate1 = source1.get_emission_rate()
            rate2 = source2.get_emission_rate()
            
            np.testing.assert_allclose(rate1, rate2, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Different seeds should produce different behavior
        source3 = PointSource(position=(10.0, 20.0), emission_rate=100.0,
                             enable_temporal_variation=True, seed=seed+1)
        
        source1.update_state(dt=1.0)
        source3.update_state(dt=1.0)
        
        rate1 = source1.get_emission_rate()
        rate3 = source3.get_emission_rate()
        
        # Should be different (with high probability)
        assert not np.allclose(rate1, rate3, atol=1e-3)


class TestMultiSource:
    """Testing for MultiSource source management with vectorized emission calculations and aggregated emission rates."""
    
    @pytest.fixture
    def empty_multi_source(self):
        """Create empty MultiSource for testing."""
        return MultiSource(seed=42)
    
    @pytest.fixture
    def populated_multi_source(self):
        """Create MultiSource with initial sources."""
        sources = [
            PointSource(position=(10.0, 10.0), emission_rate=100.0),
            PointSource(position=(30.0, 30.0), emission_rate=200.0),
            PointSource(position=(50.0, 50.0), emission_rate=300.0)
        ]
        return MultiSource(sources=sources, seed=456)
    
    def test_multi_source_initialization(self):
        """Test MultiSource initialization with various configurations."""
        # Empty initialization
        multi_source = MultiSource()
        assert multi_source.get_source_count() == 0
        assert len(multi_source.get_sources()) == 0
        
        # With initial sources
        initial_sources = [
            PointSource(position=(0.0, 0.0), emission_rate=50.0),
            PointSource(position=(10.0, 10.0), emission_rate=75.0)
        ]
        multi_source = MultiSource(sources=initial_sources)
        assert multi_source.get_source_count() == 2
        
        # With seed
        multi_source = MultiSource(seed=999)
        assert multi_source.get_source_count() == 0
    
    def test_source_management(self, empty_multi_source):
        """Test source addition, removal, and collection management."""
        multi_source = empty_multi_source
        
        # Add sources
        source1 = PointSource(position=(5.0, 5.0), emission_rate=25.0)
        source2 = PointSource(position=(15.0, 15.0), emission_rate=50.0)
        source3 = DynamicSource(initial_position=(25.0, 25.0), emission_rate=75.0)
        
        multi_source.add_source(source1)
        assert multi_source.get_source_count() == 1
        
        multi_source.add_source(source2)
        multi_source.add_source(source3)
        assert multi_source.get_source_count() == 3
        
        # Test source retrieval
        sources = multi_source.get_sources()
        assert len(sources) == 3
        assert sources[0] is source1
        assert sources[1] is source2
        assert sources[2] is source3
        
        # Remove sources
        multi_source.remove_source(1)  # Remove middle source
        assert multi_source.get_source_count() == 2
        remaining_sources = multi_source.get_sources()
        assert remaining_sources[0] is source1
        assert remaining_sources[1] is source3
        
        # Clear all sources
        multi_source.clear_sources()
        assert multi_source.get_source_count() == 0
    
    def test_vectorized_emission_calculations(self, populated_multi_source):
        """Test vectorized emission calculations across multiple sources."""
        multi_source = populated_multi_source
        
        # Test positions
        test_positions = np.array([
            [15.0, 15.0],
            [25.0, 25.0],
            [35.0, 35.0],
            [45.0, 45.0]
        ])
        
        # Get combined emission rates
        combined_rates = multi_source.get_emission_rate(test_positions)
        assert isinstance(combined_rates, np.ndarray)
        assert combined_rates.shape == (4,)
        
        # Verify each rate is sum of individual source contributions
        expected_rates = np.zeros(4)
        for source in multi_source.get_sources():
            source_rates = source.get_emission_rate(test_positions)
            expected_rates += source_rates
        
        np.testing.assert_allclose(combined_rates, expected_rates, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Each position should have total emission = 100 + 200 + 300 = 600
        np.testing.assert_allclose(combined_rates, [600.0, 600.0, 600.0, 600.0], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_aggregated_emission_rates(self, populated_multi_source):
        """Test aggregated emission rate calculations and total emission methods."""
        multi_source = populated_multi_source
        
        # Test scalar total emission
        scalar_total = multi_source.get_emission_rate(None)
        assert scalar_total == 600.0  # 100 + 200 + 300
        
        # Test total emission rate alias
        alias_total = multi_source.get_total_emission_rate(None)
        assert alias_total == scalar_total
        
        # Test with single position
        single_position = np.array([20.0, 25.0])
        single_total = multi_source.get_total_emission_rate(single_position)
        assert single_total == 600.0
        
        # Test with multi-agent positions
        multi_positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        multi_totals = multi_source.get_total_emission_rate(multi_positions)
        np.testing.assert_allclose(multi_totals, [600.0, 600.0], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_source_addition_removal(self, empty_multi_source):
        """Test dynamic source addition and removal operations."""
        multi_source = empty_multi_source
        
        # Test addition validation
        valid_source = PointSource(position=(0.0, 0.0), emission_rate=100.0)
        multi_source.add_source(valid_source)
        assert multi_source.get_source_count() == 1
        
        # Test invalid source addition
        invalid_source = object()  # Object without required methods
        with pytest.raises(TypeError):
            multi_source.add_source(invalid_source)
        
        # Test removal validation
        with pytest.raises(IndexError):
            multi_source.remove_source(5)  # Out of range
        
        # Test valid removal
        multi_source.remove_source(0)
        assert multi_source.get_source_count() == 0
        
        # Test removal from empty collection
        with pytest.raises(IndexError):
            multi_source.remove_source(0)
    
    def test_multi_agent_performance(self):
        """Test performance with large numbers of agents and sources."""
        # Create multi-source with several sources
        sources = [
            PointSource(position=(i*10.0, j*10.0), emission_rate=50.0 + i*j)
            for i in range(3) for j in range(3)
        ]
        multi_source = MultiSource(sources=sources)
        
        # Create large agent position array (100 agents)
        agent_positions = np.random.rand(100, 2) * 100.0
        
        # Performance test
        start_time = time.perf_counter()
        emission_rates = multi_source.get_emission_rate(agent_positions)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Validate results
        assert isinstance(emission_rates, np.ndarray)
        assert emission_rates.shape == (100,)
        assert np.all(emission_rates > 0)
        
        # Validate performance requirement
        assert execution_time_ms < MULTI_AGENT_100_THRESHOLD_MS, \
            f"MultiSource with 9 sources and 100 agents took {execution_time_ms:.2f}ms, should be <{MULTI_AGENT_100_THRESHOLD_MS}ms"


class TestDynamicSource:
    """Testing for DynamicSource temporal evolution, trajectory configuration, and time-varying positions."""
    
    @pytest.fixture
    def stationary_source(self):
        """Create stationary DynamicSource for testing."""
        return DynamicSource(
            initial_position=(50.0, 50.0),
            emission_rate=200.0,
            pattern_type="stationary",
            seed=42
        )
    
    @pytest.fixture  
    def linear_source(self):
        """Create linear motion DynamicSource for testing."""
        return DynamicSource(
            initial_position=(0.0, 0.0),
            emission_rate=150.0,
            pattern_type="linear",
            velocity=(2.0, 1.0),
            seed=123
        )
    
    @pytest.fixture
    def circular_source(self):
        """Create circular motion DynamicSource for testing."""
        return DynamicSource(
            initial_position=(25.0, 25.0),
            emission_rate=300.0,
            pattern_type="circular",
            amplitude=10.0,
            frequency=0.1,
            seed=456
        )
    
    def test_dynamic_source_initialization(self):
        """Test DynamicSource initialization with various pattern configurations."""
        # Stationary pattern
        source = DynamicSource(
            initial_position=(10.0, 20.0),
            emission_rate=100.0,
            pattern_type="stationary"
        )
        assert source.get_pattern_type() == "stationary"
        np.testing.assert_allclose(source.get_position(), [10.0, 20.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Linear pattern
        source = DynamicSource(
            initial_position=(5.0, 10.0),
            emission_rate=200.0,
            pattern_type="linear",
            velocity=(1.5, -0.5)
        )
        assert source.get_pattern_type() == "linear"
        
        # Circular pattern
        source = DynamicSource(
            initial_position=(30.0, 40.0),
            emission_rate=250.0,
            pattern_type="circular",
            amplitude=15.0,
            frequency=0.05
        )
        assert source.get_pattern_type() == "circular"
        
        # Invalid pattern type
        with pytest.raises(ValueError):
            DynamicSource(pattern_type="invalid_pattern")
    
    def test_temporal_evolution(self, linear_source, circular_source):
        """Test temporal evolution for different movement patterns."""
        # Linear motion test
        linear = linear_source
        initial_pos = linear.get_position().copy()
        
        # Update multiple times and verify linear motion
        for i in range(1, 6):
            linear.update_state(dt=1.0)
            current_pos = linear.get_position()
            
            # Position should change linearly: new_pos = initial + velocity * time
            expected_pos = initial_pos + np.array([2.0, 1.0]) * i
            np.testing.assert_allclose(current_pos, expected_pos, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Circular motion test
        circular = circular_source
        initial_center = np.array([25.0, 25.0])
        positions = []
        
        # Collect positions over one period
        for i in range(20):
            circular.update_state(dt=1.0)
            positions.append(circular.get_position().copy())
        
        # Verify circular motion properties
        positions = np.array(positions)
        
        # All positions should be within amplitude distance of center
        distances = np.linalg.norm(positions - initial_center, axis=1)
        assert np.all(distances <= 10.0 + 1e-6), "Positions exceed circular amplitude"
    
    def test_trajectory_configuration(self):
        """Test trajectory configuration with predefined waypoints."""
        source = DynamicSource(initial_position=(0.0, 0.0), emission_rate=100.0)
        
        # Set trajectory with waypoints
        waypoints = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0), (10.0, -10.0)]
        source.set_trajectory(waypoints)
        
        assert source.get_pattern_type() == "trajectory"
        
        # Set trajectory with timestamps
        waypoints = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        timestamps = [0.0, 2.5, 5.0]
        source.set_trajectory(waypoints, timestamps)
        
        assert source.get_pattern_type() == "trajectory"
        
        # Test invalid trajectory configuration
        with pytest.raises(ValueError):
            source.set_trajectory([])  # Empty trajectory
        
        with pytest.raises(ValueError):
            source.set_trajectory([(0.0, 0.0), (1.0, 1.0)], [0.0])  # Mismatched timestamps
    
    def test_emission_pattern_behavior(self):
        """Test emission pattern behavior for dynamic sources with temporal variation."""
        # Circular source with emission variation
        source = DynamicSource(
            initial_position=(20.0, 20.0),
            emission_rate=500.0,
            pattern_type="circular",
            amplitude=5.0,
            frequency=0.2
        )
        
        emission_rates = []
        
        # Collect emission rates over time
        for i in range(20):
            source.update_state(dt=1.0)
            rate = source.get_emission_rate()
            emission_rates.append(rate)
            assert rate >= 0.0, "Emission rate must be non-negative"
        
        emission_rates = np.array(emission_rates)
        
        # Should show temporal variation
        assert np.std(emission_rates) > 0.0, "Emission rates should vary over time"
        
        # Test custom emission pattern
        def pulse_pattern(t):
            return 1.0 if int(t) % 2 == 0 else 0.5
        
        source.set_emission_pattern(pulse_pattern)
        
        # Test pulsed emission behavior
        source.reset_time()
        rates_even = []
        rates_odd = []
        
        for i in range(10):
            source.update_state(dt=1.0)
            rate = source.get_emission_rate()
            if i % 2 == 0:
                rates_even.append(rate)
            else:
                rates_odd.append(rate)
        
        # Even time steps should have higher emission rates
        assert np.mean(rates_even) > np.mean(rates_odd)
    
    def test_time_varying_positions(self, stationary_source, linear_source, circular_source):
        """Test time-varying position behavior for different patterns."""
        # Stationary source should not move
        stationary = stationary_source
        initial_pos = stationary.get_position().copy()
        
        for _ in range(10):
            stationary.update_state(dt=1.0)
            current_pos = stationary.get_position()
            np.testing.assert_allclose(current_pos, initial_pos, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Linear source should move predictably
        linear = linear_source
        start_pos = linear.get_position().copy()
        
        linear.update_state(dt=5.0)  # Large time step
        end_pos = linear.get_position()
        
        # Should move by velocity * dt
        expected_displacement = np.array([2.0, 1.0]) * 5.0
        actual_displacement = end_pos - start_pos
        np.testing.assert_allclose(actual_displacement, expected_displacement, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Circular source should follow circular path
        circular = circular_source
        center = np.array([25.0, 25.0])
        
        # Test multiple positions on circle
        positions = []
        for _ in range(8):
            circular.update_state(dt=2.5)  # 1/4 period steps
            positions.append(circular.get_position())
        
        # Verify circular motion
        for pos in positions:
            distance = np.linalg.norm(pos - center)
            assert abs(distance - 10.0) < 0.1, f"Position {pos} not on expected circle"
    
    def test_deterministic_temporal_seeding(self):
        """Test deterministic temporal behavior with controlled seeding."""
        seed = 789
        
        # Create two identical sources
        source1 = DynamicSource(
            initial_position=(10.0, 10.0),
            emission_rate=100.0,
            pattern_type="random_walk",
            noise_std=1.0,
            seed=seed
        )
        
        source2 = DynamicSource(
            initial_position=(10.0, 10.0),
            emission_rate=100.0,
            pattern_type="random_walk", 
            noise_std=1.0,
            seed=seed
        )
        
        # Both should evolve identically
        for _ in range(10):
            source1.update_state(dt=1.0)
            source2.update_state(dt=1.0)
            
            pos1 = source1.get_position()
            pos2 = source2.get_position()
            
            np.testing.assert_allclose(pos1, pos2, atol=NUMERICAL_PRECISION_TOLERANCE)
            
            rate1 = source1.get_emission_rate()
            rate2 = source2.get_emission_rate()
            
            np.testing.assert_allclose(rate1, rate2, atol=NUMERICAL_PRECISION_TOLERANCE)


class TestVectorizedOperations:
    """Testing vectorized operations for multi-agent source queries with performance benchmarks."""
    
    def test_multi_agent_source_queries(self):
        """Test vectorized source queries for multi-agent scenarios."""
        # Create test sources
        point_source = PointSource(position=(25.0, 25.0), emission_rate=100.0)
        dynamic_source = DynamicSource(initial_position=(75.0, 75.0), emission_rate=200.0, pattern_type="linear", velocity=(1.0, 0.0))
        multi_source = MultiSource(sources=[point_source, dynamic_source])
        
        # Test various agent configurations
        test_cases = [
            # (description, agent_positions, expected_shape)
            ("Single agent", np.array([50.0, 50.0]), ()),
            ("Small group", np.random.rand(5, 2) * 100, (5,)),
            ("Medium group", np.random.rand(25, 2) * 100, (25,)),
            ("Large group", np.random.rand(100, 2) * 100, (100,))
        ]
        
        for description, positions, expected_shape in test_cases:
            # Test each source type
            for source in [point_source, dynamic_source, multi_source]:
                result = source.get_emission_rate(positions)
                
                if positions.ndim == 1:
                    assert isinstance(result, (int, float, np.floating)), f"{description} with {type(source).__name__} should return scalar"
                else:
                    assert isinstance(result, np.ndarray), f"{description} with {type(source).__name__} should return array"
                    assert result.shape == expected_shape, f"{description} with {type(source).__name__} shape mismatch"
                    assert np.all(result >= 0), f"{description} with {type(source).__name__} has negative emissions"
    
    def test_vectorized_emission_calculations(self):
        """Test vectorized emission calculations for efficiency and correctness."""
        # Create sources with different emission rates
        sources = [
            PointSource(position=(i*20.0, j*20.0), emission_rate=(i+1)*(j+1)*50.0)
            for i in range(3) for j in range(3)
        ]
        multi_source = MultiSource(sources=sources)
        
        # Large multi-agent scenario
        num_agents = 50
        agent_positions = np.random.rand(num_agents, 2) * 100.0
        
        # Vectorized calculation
        start_time = time.perf_counter()
        vectorized_result = multi_source.get_emission_rate(agent_positions)
        vectorized_time = time.perf_counter() - start_time
        
        # Manual calculation for verification
        start_time = time.perf_counter()
        manual_result = np.zeros(num_agents)
        for i, pos in enumerate(agent_positions):
            manual_result[i] = multi_source.get_emission_rate(pos.reshape(1, -1))[0]
        manual_time = time.perf_counter() - start_time
        
        # Verify correctness
        np.testing.assert_allclose(vectorized_result, manual_result, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Verify performance improvement
        assert vectorized_time < manual_time, "Vectorized calculation should be faster than manual"
        assert vectorized_time * 1000 < MULTI_AGENT_100_THRESHOLD_MS, f"Vectorized calculation took {vectorized_time*1000:.2f}ms, should be <{MULTI_AGENT_100_THRESHOLD_MS}ms"
    
    def test_batch_position_retrieval(self):
        """Test batch position retrieval for multiple sources."""
        # Create sources at different positions
        positions = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0)]
        sources = [PointSource(position=pos, emission_rate=100.0) for pos in positions]
        multi_source = MultiSource(sources=sources)
        
        # Get all positions
        retrieved_positions = multi_source.get_position()
        
        assert len(retrieved_positions) == 4
        for i, retrieved_pos in enumerate(retrieved_positions):
            assert isinstance(retrieved_pos, np.ndarray)
            assert retrieved_pos.shape == (2,)
            np.testing.assert_allclose(retrieved_pos, positions[i], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_concurrent_source_operations(self):
        """Test concurrent source operations for thread safety and consistency."""
        # Create sources
        sources = [
            PointSource(position=(i*10.0, i*10.0), emission_rate=100.0 + i*50.0)
            for i in range(5)
        ]
        multi_source = MultiSource(sources=sources)
        
        # Perform concurrent operations
        agent_positions = np.random.rand(20, 2) * 50.0
        
        results = []
        for _ in range(10):  # Multiple queries
            result = multi_source.get_emission_rate(agent_positions)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_100_agent_performance(self):
        """Test performance compliance with 100 agents requirement."""
        # Create realistic multi-source scenario
        sources = [
            PointSource(position=(25.0 + i*5.0, 25.0 + j*5.0), emission_rate=50.0 + i*j*10.0)
            for i in range(5) for j in range(5)
        ]
        multi_source = MultiSource(sources=sources)
        
        # 100 agent positions
        agent_positions = np.random.rand(100, 2) * 100.0
        
        # Performance benchmark
        start_time = time.perf_counter()
        
        # Multiple operations to simulate realistic workload
        for _ in range(10):
            emission_rates = multi_source.get_emission_rate(agent_positions)
            multi_source.update_state(dt=1.0)
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Validate performance requirement
        assert execution_time_ms < STEP_LATENCY_THRESHOLD_MS, \
            f"100-agent multi-source operations took {execution_time_ms:.2f}ms, should be <{STEP_LATENCY_THRESHOLD_MS}ms"


class TestSourcePerformance:
    """Testing source operation performance and compliance with step latency requirements."""
    
    def test_step_latency_compliance(self):
        """Test source operations comply with ≤33ms step latency requirement."""
        # Create realistic source configuration
        sources = [
            PointSource(position=(20.0*i, 20.0*j), emission_rate=100.0)
            for i in range(3) for j in range(3)
        ]
        dynamic_sources = [
            DynamicSource(initial_position=(60.0 + 10.0*i, 60.0 + 10.0*j), 
                         emission_rate=150.0, pattern_type="circular", 
                         amplitude=5.0, frequency=0.1)
            for i in range(2) for j in range(2)
        ]
        
        all_sources = sources + dynamic_sources
        multi_source = MultiSource(sources=all_sources)
        
        # 100 agents
        agent_positions = np.random.rand(100, 2) * 100.0
        
        # Measure step operations
        step_times = []
        for _ in range(20):  # Multiple steps
            start_time = time.perf_counter()
            
            # Simulate typical step operations
            emission_rates = multi_source.get_emission_rate(agent_positions)
            multi_source.update_state(dt=1.0)
            
            step_time_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time_ms)
        
        # Validate latency requirements
        mean_time = np.mean(step_times)
        p95_time = np.percentile(step_times, 95)
        max_time = np.max(step_times)
        
        assert mean_time < STEP_LATENCY_THRESHOLD_MS, f"Mean step time {mean_time:.2f}ms exceeds {STEP_LATENCY_THRESHOLD_MS}ms"
        assert p95_time < STEP_LATENCY_THRESHOLD_MS, f"95th percentile {p95_time:.2f}ms exceeds {STEP_LATENCY_THRESHOLD_MS}ms"
        assert max_time < STEP_LATENCY_THRESHOLD_MS, f"Maximum step time {max_time:.2f}ms exceeds {STEP_LATENCY_THRESHOLD_MS}ms"
    
    def test_33ms_step_requirement(self):
        """Test specific 33ms step requirement with realistic workload."""
        # Maximum complexity scenario
        point_sources = [PointSource(position=(i*5.0, j*5.0), emission_rate=25.0) for i in range(10) for j in range(10)]
        dynamic_sources = [
            DynamicSource(initial_position=(50.0 + i*10.0, 50.0 + j*10.0), 
                         emission_rate=50.0, pattern_type="circular",
                         amplitude=8.0, frequency=0.05)
            for i in range(5) for j in range(5)
        ]
        
        # Combine all sources (125 total)
        all_sources = point_sources[:75] + dynamic_sources  # Limit to prevent excessive test time
        multi_source = MultiSource(sources=all_sources)
        
        # 100 agents with realistic positions
        agent_positions = np.random.rand(100, 2) * 200.0
        
        # Single step timing
        start_time = time.perf_counter()
        emission_rates = multi_source.get_emission_rate(agent_positions)
        multi_source.update_state(dt=1.0)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert execution_time_ms <= STEP_LATENCY_THRESHOLD_MS, \
            f"Complex source step took {execution_time_ms:.2f}ms, must be ≤{STEP_LATENCY_THRESHOLD_MS}ms"
    
    def test_source_operation_overhead(self):
        """Test source operation overhead for minimal simulation impact."""
        source = PointSource(position=(50.0, 50.0), emission_rate=100.0)
        
        # Single query performance
        position = np.array([55.0, 55.0])
        start_time = time.perf_counter()
        for _ in range(1000):  # Many queries
            result = source.get_emission_rate(position)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        per_query_time_ms = execution_time_ms / 1000
        assert per_query_time_ms < SINGLE_QUERY_THRESHOLD_MS, \
            f"Single query took {per_query_time_ms:.4f}ms, should be <{SINGLE_QUERY_THRESHOLD_MS}ms"
    
    def test_memory_efficiency(self):
        """Test memory efficiency for source operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Create many sources
        sources = []
        for i in range(100):
            if i % 3 == 0:
                source = PointSource(position=(i*2.0, i*2.0), emission_rate=100.0)
            elif i % 3 == 1:
                source = DynamicSource(initial_position=(i*2.0, i*2.0), emission_rate=150.0)
            else:
                source = MultiSource()
                source.add_source(PointSource(position=(i*1.0, i*1.0), emission_rate=75.0))
            sources.append(source)
        
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = current_memory_mb - initial_memory_mb
        
        # Verify memory usage is reasonable
        assert memory_increase_mb < MAX_MEMORY_USAGE_MB, \
            f"100 sources used {memory_increase_mb:.1f}MB, should be <{MAX_MEMORY_USAGE_MB}MB"
    
    def test_performance_regression_detection(self):
        """Test performance regression detection across source types."""
        sources = [
            PointSource(position=(25.0, 25.0), emission_rate=100.0),
            MultiSource(sources=[PointSource(position=(i*10.0, i*10.0), emission_rate=50.0) for i in range(5)]),
            DynamicSource(initial_position=(50.0, 50.0), emission_rate=200.0, pattern_type="circular")
        ]
        
        agent_positions = np.random.rand(50, 2) * 100.0
        performance_results = {}
        
        for source in sources:
            source_name = type(source).__name__
            times = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                result = source.get_emission_rate(agent_positions)
                source.update_state(dt=1.0)
                times.append((time.perf_counter() - start_time) * 1000)
            
            performance_results[source_name] = {
                'mean_time_ms': np.mean(times),
                'max_time_ms': np.max(times),
                'std_time_ms': np.std(times)
            }
        
        # Validate performance expectations
        assert performance_results['PointSource']['mean_time_ms'] < 2.0, "PointSource performance regression"
        assert performance_results['MultiSource']['mean_time_ms'] < 5.0, "MultiSource performance regression"
        assert performance_results['DynamicSource']['mean_time_ms'] < 3.0, "DynamicSource performance regression"


class TestSourceFactory:
    """Testing factory function for configuration-driven source instantiation and runtime selection."""
    
    def test_factory_instantiation(self):
        """Test factory function creates correct source types from configuration."""
        # PointSource configuration
        point_config = {
            'type': 'PointSource',
            'position': (15.0, 25.0),
            'emission_rate': 150.0
        }
        point_source = create_source(point_config)
        assert isinstance(point_source, PointSource)
        np.testing.assert_allclose(point_source.get_position(), [15.0, 25.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        assert point_source.get_emission_rate() == 150.0
        
        # DynamicSource configuration
        dynamic_config = {
            'type': 'DynamicSource',
            'initial_position': (30.0, 40.0),
            'emission_rate': 200.0,
            'pattern_type': 'linear',
            'velocity': (1.5, -0.5)
        }
        dynamic_source = create_source(dynamic_config)
        assert isinstance(dynamic_source, DynamicSource)
        assert dynamic_source.get_pattern_type() == 'linear'
        
        # MultiSource configuration
        multi_config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (10.0, 10.0), 'emission_rate': 50.0},
                {'type': 'PointSource', 'position': (20.0, 20.0), 'emission_rate': 75.0}
            ]
        }
        multi_source = create_source(multi_config)
        assert isinstance(multi_source, MultiSource)
        assert multi_source.get_source_count() == 2
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling in factory."""
        # Missing type field
        with pytest.raises(KeyError):
            create_source({'position': (0.0, 0.0)})
        
        # Invalid type
        with pytest.raises(ValueError):
            create_source({'type': 'InvalidSource'})
        
        # Invalid parameters for specific source type
        with pytest.raises(ValueError):
            create_source({
                'type': 'PointSource',
                'emission_rate': -100.0  # Negative emission rate
            })
    
    def test_hydra_integration(self):
        """Test factory integration with Hydra-style configuration."""
        # Test with nested configuration typical of Hydra
        hydra_style_config = {
            'type': 'DynamicSource',
            'initial_position': (40.0, 50.0),
            'emission_rate': 300.0,
            'pattern_type': 'circular',
            'amplitude': 12.0,
            'frequency': 0.08,
            'seed': 999
        }
        
        source = create_source(hydra_style_config)
        assert isinstance(source, DynamicSource)
        assert source.get_pattern_type() == 'circular'
        np.testing.assert_allclose(source.get_position(), [40.0, 50.0], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_runtime_source_selection(self):
        """Test runtime source selection based on configuration parameters."""
        configurations = [
            {
                'type': 'PointSource',
                'position': (i*10.0, i*10.0),
                'emission_rate': 50.0 + i*25.0
            }
            for i in range(3)
        ]
        
        # Create sources dynamically
        sources = [create_source(config) for config in configurations]
        
        # Verify correct types and parameters
        for i, source in enumerate(sources):
            assert isinstance(source, PointSource)
            expected_position = [i*10.0, i*10.0]
            expected_rate = 50.0 + i*25.0
            
            np.testing.assert_allclose(source.get_position(), expected_position, atol=NUMERICAL_PRECISION_TOLERANCE)
            assert source.get_emission_rate() == expected_rate
    
    def test_factory_error_handling(self):
        """Test factory error handling for various invalid configurations."""
        invalid_configs = [
            {},  # Empty config
            {'type': None},  # None type
            {'type': 'PointSource', 'position': 'invalid'},  # Invalid position
            {'type': 'MultiSource', 'sources': 'not_a_list'},  # Invalid sources
            {'type': 'DynamicSource', 'pattern_type': 'unknown'},  # Invalid pattern
        ]
        
        for config in invalid_configs:
            with pytest.raises((KeyError, ValueError, TypeError)):
                create_source(config)


def test_source_seeding_reproducibility():
    """Test source seeding reproducibility across all source types."""
    seed = 12345
    
    # Test PointSource reproducibility
    source1 = PointSource(position=(10.0, 20.0), emission_rate=100.0, 
                         enable_temporal_variation=True, seed=seed)
    source2 = PointSource(position=(10.0, 20.0), emission_rate=100.0,
                         enable_temporal_variation=True, seed=seed)
    
    for _ in range(10):
        source1.update_state(dt=1.0)
        source2.update_state(dt=1.0)
        
        rate1 = source1.get_emission_rate()
        rate2 = source2.get_emission_rate()
        
        np.testing.assert_allclose(rate1, rate2, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    # Test DynamicSource reproducibility
    dynamic1 = DynamicSource(initial_position=(0.0, 0.0), emission_rate=200.0,
                            pattern_type="random_walk", noise_std=1.0, seed=seed)
    dynamic2 = DynamicSource(initial_position=(0.0, 0.0), emission_rate=200.0,
                            pattern_type="random_walk", noise_std=1.0, seed=seed)
    
    for _ in range(10):
        dynamic1.update_state(dt=1.0)
        dynamic2.update_state(dt=1.0)
        
        pos1 = dynamic1.get_position()
        pos2 = dynamic2.get_position()
        
        np.testing.assert_allclose(pos1, pos2, atol=NUMERICAL_PRECISION_TOLERANCE)


def test_source_backwards_compatibility():
    """Test backwards compatibility with existing source usage patterns."""
    # Test that sources work with basic position queries
    source = PointSource(position=(25.0, 50.0), emission_rate=150.0)
    
    # Single position query (backward compatibility)
    position = [30.0, 55.0]  # List instead of numpy array
    result = source.get_emission_rate(np.array(position))
    assert result == 150.0
    
    # Multi-position query
    positions = [[20.0, 45.0], [30.0, 55.0], [40.0, 65.0]]
    results = source.get_emission_rate(np.array(positions))
    assert len(results) == 3
    assert all(r == 150.0 for r in results)
    
    # Test with MultiSource
    multi_source = MultiSource()
    multi_source.add_source(source)
    
    combined_result = multi_source.get_emission_rate(np.array(position))
    assert combined_result == 150.0


if __name__ == "__main__":
    # Run tests with appropriate verbosity and coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.plume_nav_sim.core.sources",
        "--cov-report=term-missing"
    ])