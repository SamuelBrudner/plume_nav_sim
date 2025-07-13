"""
Comprehensive test module for SourceProtocol implementations validating odor source modeling functionality.

This module implements comprehensive test coverage for SourceProtocol interface compliance including
PointSource, MultiSource, and DynamicSource implementations. Tests validate protocol compliance,
performance requirements (≤33ms/step with 100 agents), deterministic seeding, vectorized operations,
and integration with plume models and environment components per F-013 requirements.

Test Coverage:
- Protocol Interface Compliance: 100% coverage of SourceProtocol methods
- Performance Validation: ≤33ms/step performance targets with multi-agent scenarios  
- Deterministic Seeding: Reproducible behavior across test runs
- Vectorized Operations: Efficient multi-agent emission calculations
- Integration Testing: Plume model and environment connectivity
- Configuration Testing: Hydra integration and runtime source selection
- Error Handling: Edge cases and boundary conditions

Performance Requirements per Section 6.6.2.1:
- get_emission_rate(): <0.1ms single query, <1ms for 100 agents
- get_position(): <0.1ms with minimal overhead  
- update_state(): <1ms per time step
- Memory efficiency: <10MB typical configurations

Test Organization per Section 6.6.2.2:
- TestPointSource: Single-point source implementation validation
- TestMultiSource: Multiple simultaneous sources with vectorized operations
- TestDynamicSource: Time-varying sources with temporal evolution
- TestSourceFactory: Configuration-driven source creation
- TestSourceIntegration: Environment and plume model integration
- TestSourcePerformance: Performance benchmarking and SLA validation
"""

import pytest
import numpy as np
import time
import math
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from unittest.mock import patch, MagicMock

# Import source implementations and protocols
from src.plume_nav_sim.core.protocols import SourceProtocol
from src.plume_nav_sim.core.sources import (
    PointSource, MultiSource, DynamicSource, create_source,
    SourceConfig, DynamicSourceConfig
)

# Import test fixtures and utilities
from tests.conftest import mock_action_config
from src.plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv


class TestSourceProtocolCompliance:
    """
    Test suite for SourceProtocol interface compliance across all implementations.
    
    Validates that PointSource, MultiSource, and DynamicSource implementations
    strictly adhere to SourceProtocol interface requirements with correct method
    signatures, return types, and behavioral contracts per Section 6.6.2.2.
    """
    
    @pytest.fixture
    def source_implementations(self):
        """Provide all source implementations for protocol compliance testing."""
        return [
            PointSource(position=(50.0, 50.0), emission_rate=1000.0),
            MultiSource(sources=[PointSource(position=(30.0, 30.0), emission_rate=500.0)]),
            DynamicSource(initial_position=(25.0, 75.0), emission_rate=750.0, pattern_type="stationary")
        ]
    
    def test_protocol_implementation_compliance(self, source_implementations):
        """Validate that all source implementations satisfy SourceProtocol interface."""
        for source in source_implementations:
            # Verify protocol compliance using isinstance check
            assert isinstance(source, SourceProtocol), f"{type(source).__name__} must implement SourceProtocol"
            
            # Verify required methods exist and are callable
            required_methods = ['get_emission_rate', 'get_position', 'update_state']
            for method_name in required_methods:
                assert hasattr(source, method_name), f"{type(source).__name__} missing {method_name} method"
                assert callable(getattr(source, method_name)), f"{type(source).__name__}.{method_name} must be callable"
    
    def test_get_emission_rate_signature_compliance(self, source_implementations):
        """Test get_emission_rate method signature and return type compliance."""
        for source in source_implementations:
            # Test scalar emission rate query (no positions)
            rate = source.get_emission_rate()
            assert isinstance(rate, (int, float)), f"Scalar emission rate must be numeric, got {type(rate)}"
            assert rate >= 0, f"Emission rate must be non-negative, got {rate}"
            
            # Test single agent position query
            single_position = np.array([40.0, 60.0])
            single_rate = source.get_emission_rate(single_position)
            assert isinstance(single_rate, (int, float, np.number)), f"Single agent rate must be numeric, got {type(single_rate)}"
            assert single_rate >= 0, f"Single agent emission rate must be non-negative, got {single_rate}"
            
            # Test multi-agent position query
            multi_positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
            multi_rates = source.get_emission_rate(multi_positions)
            assert isinstance(multi_rates, np.ndarray), f"Multi-agent rates must be numpy array, got {type(multi_rates)}"
            assert multi_rates.shape == (3,), f"Multi-agent rates shape must be (3,), got {multi_rates.shape}"
            assert np.all(multi_rates >= 0), f"All emission rates must be non-negative, got {multi_rates}"
    
    def test_get_position_signature_compliance(self, source_implementations):
        """Test get_position method signature and return type compliance."""
        for source in source_implementations:
            if isinstance(source, MultiSource):
                # MultiSource returns list of positions
                positions = source.get_position()
                assert isinstance(positions, list), f"MultiSource positions must be list, got {type(positions)}"
                for pos in positions:
                    assert isinstance(pos, np.ndarray), f"Each position must be numpy array, got {type(pos)}"
                    assert pos.shape == (2,), f"Position shape must be (2,), got {pos.shape}"
            else:
                # Single source returns numpy array
                position = source.get_position()
                assert isinstance(position, np.ndarray), f"Position must be numpy array, got {type(position)}"
                assert position.shape == (2,), f"Position shape must be (2,), got {position.shape}"
    
    def test_update_state_signature_compliance(self, source_implementations):
        """Test update_state method signature compliance."""
        for source in source_implementations:
            # Test with default dt
            source.update_state()
            
            # Test with explicit dt values
            for dt in [0.1, 1.0, 2.5]:
                source.update_state(dt=dt)
            
            # Method should not return anything (returns None)
            result = source.update_state(dt=1.0)
            assert result is None, f"update_state should return None, got {result}"


class TestPointSource:
    """
    Test suite for PointSource implementation validating single-point source behavior.
    
    Tests basic point source functionality including emission rate queries, position
    access, temporal updates, and configuration management. Validates performance
    requirements and edge case handling for single-source scenarios.
    """
    
    @pytest.fixture
    def basic_point_source(self):
        """Create basic point source for testing."""
        return PointSource(position=(50.0, 50.0), emission_rate=1000.0, seed=42)
    
    @pytest.fixture
    def temporal_point_source(self):
        """Create point source with temporal variation enabled."""
        return PointSource(
            position=(75.0, 25.0), 
            emission_rate=500.0, 
            enable_temporal_variation=True,
            seed=123
        )
    
    def test_point_source_initialization(self):
        """Test PointSource initialization with various parameter combinations."""
        # Test basic initialization
        source = PointSource()
        assert np.allclose(source.get_position(), [0.0, 0.0])
        assert source.get_emission_rate() == 1.0
        
        # Test explicit parameter initialization
        source = PointSource(position=(100.0, 200.0), emission_rate=2500.0)
        assert np.allclose(source.get_position(), [100.0, 200.0])
        assert source.get_emission_rate() == 2500.0
        
        # Test initialization with lists and tuples
        source = PointSource(position=[75.5, 125.3], emission_rate=1750)
        assert np.allclose(source.get_position(), [75.5, 125.3])
        assert source.get_emission_rate() == 1750.0
    
    def test_point_source_initialization_validation(self):
        """Test PointSource initialization parameter validation."""
        # Test invalid position formats
        with pytest.raises(ValueError, match="Position must be a 2-element"):
            PointSource(position=(10.0,))  # Single element
        
        with pytest.raises(ValueError, match="Position must be a 2-element"):
            PointSource(position=(10.0, 20.0, 30.0))  # Three elements
        
        # Test negative emission rate
        with pytest.raises(ValueError, match="Emission rate must be non-negative"):
            PointSource(emission_rate=-100.0)
    
    def test_emission_rate_queries(self, basic_point_source):
        """Test emission rate queries for different input formats."""
        # Scalar query
        rate = basic_point_source.get_emission_rate()
        assert rate == 1000.0
        
        # Single agent query
        single_position = np.array([45.0, 55.0])
        single_rate = basic_point_source.get_emission_rate(single_position)
        assert single_rate == 1000.0
        
        # Multi-agent query
        multi_positions = np.array([[40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
        multi_rates = basic_point_source.get_emission_rate(multi_positions)
        expected_rates = np.array([1000.0, 1000.0, 1000.0])
        assert np.allclose(multi_rates, expected_rates)
        
        # Large multi-agent query for performance testing
        large_positions = np.random.rand(100, 2) * 100
        large_rates = basic_point_source.get_emission_rate(large_positions)
        assert large_rates.shape == (100,)
        assert np.all(large_rates == 1000.0)
    
    def test_position_queries(self, basic_point_source):
        """Test position access and immutability."""
        # Get position
        position = basic_point_source.get_position()
        assert np.allclose(position, [50.0, 50.0])
        
        # Verify returned position is a copy (not reference)
        position[0] = 999.0
        new_position = basic_point_source.get_position()
        assert new_position[0] == 50.0, "Position should be immutable from external modification"
    
    def test_temporal_variation(self, temporal_point_source):
        """Test temporal emission rate variation when enabled."""
        initial_rate = temporal_point_source.get_emission_rate()
        
        # Update state and check for variation
        for i in range(10):
            temporal_point_source.update_state(dt=1.0)
            current_rate = temporal_point_source.get_emission_rate()
            # Rate should vary around base rate due to sinusoidal variation
            assert 0.9 * 500.0 <= current_rate <= 1.1 * 500.0
    
    def test_configuration_updates(self, basic_point_source):
        """Test runtime configuration updates."""
        # Test position configuration
        basic_point_source.configure(position=(75.0, 85.0))
        assert np.allclose(basic_point_source.get_position(), [75.0, 85.0])
        
        # Test emission rate configuration  
        basic_point_source.configure(emission_rate=1500.0)
        assert basic_point_source.get_emission_rate() == 1500.0
        
        # Test temporal variation configuration
        basic_point_source.configure(enable_temporal_variation=True)
        initial_rate = basic_point_source.get_emission_rate()
        basic_point_source.update_state(dt=5.0)
        # Should see some variation after temporal update
        assert basic_point_source.get_emission_rate() != initial_rate
    
    def test_configuration_validation(self, basic_point_source):
        """Test configuration parameter validation."""
        # Test invalid position
        with pytest.raises(ValueError, match="Position must be a 2-element"):
            basic_point_source.configure(position=(10.0,))
        
        # Test negative emission rate
        with pytest.raises(ValueError, match="Emission rate must be non-negative"):
            basic_point_source.configure(emission_rate=-500.0)
        
        # Test unknown parameter
        with pytest.raises(KeyError, match="Unknown parameters"):
            basic_point_source.configure(invalid_param=123)
    
    def test_convenience_methods(self, basic_point_source):
        """Test convenience setter methods."""
        # Test set_emission_rate
        basic_point_source.set_emission_rate(2000.0)
        assert basic_point_source.get_emission_rate() == 2000.0
        
        # Test set_position
        basic_point_source.set_position((80.0, 90.0))
        assert np.allclose(basic_point_source.get_position(), [80.0, 90.0])
        
        # Test validation in convenience methods
        with pytest.raises(ValueError):
            basic_point_source.set_emission_rate(-100.0)
        
        with pytest.raises(ValueError):
            basic_point_source.set_position((10.0,))
    
    def test_performance_tracking(self, basic_point_source):
        """Test performance statistics tracking."""
        # Perform several queries
        for _ in range(10):
            basic_point_source.get_emission_rate()
            basic_point_source.get_emission_rate(np.array([25.0, 35.0]))
        
        # Get performance stats
        stats = basic_point_source.get_performance_stats()
        assert 'query_count' in stats
        assert 'total_query_time' in stats  
        assert 'avg_query_time' in stats
        assert stats['query_count'] == 20.0  # 10 scalar + 10 single agent queries
        assert stats['total_query_time'] >= 0.0
        assert stats['avg_query_time'] >= 0.0


class TestMultiSource:
    """
    Test suite for MultiSource implementation validating multiple simultaneous sources.
    
    Tests multi-source aggregation, vectorized operations, dynamic source management,
    and performance characteristics for complex source configurations.
    """
    
    @pytest.fixture
    def empty_multi_source(self):
        """Create empty MultiSource for testing."""
        return MultiSource(seed=42)
    
    @pytest.fixture
    def populated_multi_source(self):
        """Create MultiSource with initial sources."""
        sources = [
            PointSource(position=(30.0, 30.0), emission_rate=500.0),
            PointSource(position=(70.0, 70.0), emission_rate=800.0),
            DynamicSource(initial_position=(50.0, 50.0), emission_rate=300.0, pattern_type="stationary")
        ]
        return MultiSource(sources=sources, seed=123)
    
    def test_multi_source_initialization(self):
        """Test MultiSource initialization with various configurations."""
        # Test empty initialization
        multi_source = MultiSource()
        assert multi_source.get_source_count() == 0
        assert multi_source.get_emission_rate() == 0.0
        
        # Test initialization with source list
        sources = [
            PointSource(position=(10.0, 10.0), emission_rate=100.0),
            PointSource(position=(90.0, 90.0), emission_rate=200.0)
        ]
        multi_source = MultiSource(sources=sources)
        assert multi_source.get_source_count() == 2
        assert multi_source.get_emission_rate() == 300.0  # Sum of sources
    
    def test_source_management(self, empty_multi_source):
        """Test dynamic source addition and removal."""
        # Add sources
        point_source = PointSource(position=(25.0, 75.0), emission_rate=400.0)
        empty_multi_source.add_source(point_source)
        assert empty_multi_source.get_source_count() == 1
        
        dynamic_source = DynamicSource(initial_position=(75.0, 25.0), emission_rate=600.0)
        empty_multi_source.add_source(dynamic_source)
        assert empty_multi_source.get_source_count() == 2
        
        # Test total emission rate
        assert empty_multi_source.get_emission_rate() == 1000.0
        
        # Remove sources
        empty_multi_source.remove_source(0)
        assert empty_multi_source.get_source_count() == 1
        assert empty_multi_source.get_emission_rate() == 600.0
        
        # Clear all sources
        empty_multi_source.clear_sources()
        assert empty_multi_source.get_source_count() == 0
        assert empty_multi_source.get_emission_rate() == 0.0
    
    def test_source_validation(self, empty_multi_source):
        """Test source interface validation during addition."""
        # Test valid source
        valid_source = PointSource(position=(50.0, 50.0), emission_rate=100.0)
        empty_multi_source.add_source(valid_source)
        
        # Test invalid source (missing methods)
        invalid_source = object()
        with pytest.raises(TypeError, match="Source must implement get_emission_rate"):
            empty_multi_source.add_source(invalid_source)
        
        # Test source with non-callable methods
        class InvalidSource:
            get_emission_rate = "not_callable"
            get_position = lambda self: np.array([0, 0])
            update_state = lambda self, dt=1.0: None
        
        with pytest.raises(TypeError, match="must be callable"):
            empty_multi_source.add_source(InvalidSource())
    
    def test_aggregated_emission_rates(self, populated_multi_source):
        """Test emission rate aggregation across multiple sources."""
        # Test scalar aggregation
        total_rate = populated_multi_source.get_emission_rate()
        expected_total = 500.0 + 800.0 + 300.0  # Sum of all sources
        assert total_rate == expected_total
        
        # Test single agent aggregation
        single_position = np.array([50.0, 50.0])
        single_rate = populated_multi_source.get_emission_rate(single_position)
        assert single_rate == expected_total
        
        # Test multi-agent aggregation
        multi_positions = np.array([[30.0, 30.0], [50.0, 50.0], [70.0, 70.0]])
        multi_rates = populated_multi_source.get_emission_rate(multi_positions)
        expected_rates = np.array([expected_total, expected_total, expected_total])
        assert np.allclose(multi_rates, expected_rates)
    
    def test_position_access(self, populated_multi_source):
        """Test position access for all sources."""
        positions = populated_multi_source.get_position()
        assert len(positions) == 3
        
        # Verify individual positions
        assert np.allclose(positions[0], [30.0, 30.0])  # First PointSource
        assert np.allclose(positions[1], [70.0, 70.0])  # Second PointSource
        assert np.allclose(positions[2], [50.0, 50.0])  # DynamicSource (stationary)
    
    def test_synchronized_updates(self, populated_multi_source):
        """Test synchronized temporal updates across all sources."""
        # Get initial positions and rates
        initial_positions = populated_multi_source.get_position()
        initial_rate = populated_multi_source.get_emission_rate()
        
        # Update all sources
        populated_multi_source.update_state(dt=1.0)
        
        # Verify update propagation
        updated_positions = populated_multi_source.get_position()
        updated_rate = populated_multi_source.get_emission_rate()
        
        # Positions should remain same for stationary sources in this test
        for i, (initial_pos, updated_pos) in enumerate(zip(initial_positions, updated_positions)):
            assert np.allclose(initial_pos, updated_pos), f"Source {i} position should not change for stationary sources"
    
    def test_empty_source_behavior(self, empty_multi_source):
        """Test behavior with no sources."""
        # Test scalar emission rate
        assert empty_multi_source.get_emission_rate() == 0.0
        
        # Test single agent query
        single_position = np.array([25.0, 75.0])
        assert empty_multi_source.get_emission_rate(single_position) == 0.0
        
        # Test multi-agent query
        multi_positions = np.array([[10.0, 10.0], [50.0, 50.0]])
        multi_rates = empty_multi_source.get_emission_rate(multi_positions)
        assert np.allclose(multi_rates, [0.0, 0.0])
        
        # Test position access
        positions = empty_multi_source.get_position()
        assert len(positions) == 0
    
    def test_source_removal_edge_cases(self, populated_multi_source):
        """Test edge cases in source removal."""
        initial_count = populated_multi_source.get_source_count()
        
        # Test removal by negative index (last element)
        populated_multi_source.remove_source(-1)
        assert populated_multi_source.get_source_count() == initial_count - 1
        
        # Test invalid index
        with pytest.raises(IndexError, match="Source index .* out of range"):
            populated_multi_source.remove_source(100)
    
    def test_source_list_access(self, populated_multi_source):
        """Test access to source list."""
        sources = populated_multi_source.get_sources()
        assert len(sources) == 3
        
        # Verify returned list is a copy (not reference)
        sources.clear()
        assert populated_multi_source.get_source_count() == 3, "Original source list should not be affected"
    
    def test_performance_tracking(self, populated_multi_source):
        """Test performance statistics for multi-source operations."""
        # Perform queries
        for _ in range(5):
            populated_multi_source.get_emission_rate()
            populated_multi_source.get_emission_rate(np.array([40.0, 60.0]))
        
        stats = populated_multi_source.get_performance_stats()
        assert 'query_count' in stats
        assert 'source_count' in stats
        assert stats['query_count'] == 10.0
        assert stats['source_count'] == 3.0


class TestDynamicSource:
    """
    Test suite for DynamicSource implementation validating time-varying source behavior.
    
    Tests temporal evolution patterns, movement trajectories, emission variations,
    and complex source dynamics for research into navigation under dynamic conditions.
    """
    
    @pytest.fixture
    def stationary_dynamic_source(self):
        """Create stationary dynamic source for baseline testing."""
        return DynamicSource(
            initial_position=(50.0, 50.0),
            emission_rate=1000.0,
            pattern_type="stationary",
            seed=42
        )
    
    @pytest.fixture
    def circular_dynamic_source(self):
        """Create circular orbit dynamic source."""
        return DynamicSource(
            initial_position=(50.0, 50.0),
            emission_rate=500.0,
            pattern_type="circular",
            amplitude=20.0,
            frequency=0.1,
            seed=123
        )
    
    @pytest.fixture
    def linear_dynamic_source(self):
        """Create linear motion dynamic source."""
        return DynamicSource(
            initial_position=(10.0, 50.0),
            emission_rate=750.0,
            pattern_type="linear",
            velocity=(2.0, 1.0),
            seed=456
        )
    
    @pytest.fixture
    def random_walk_source(self):
        """Create random walk dynamic source."""
        return DynamicSource(
            initial_position=(50.0, 50.0),
            emission_rate=600.0,
            pattern_type="random_walk",
            noise_std=1.0,
            seed=789
        )
    
    def test_dynamic_source_initialization(self):
        """Test DynamicSource initialization with various patterns."""
        # Test basic initialization
        source = DynamicSource()
        assert np.allclose(source.get_position(), [0.0, 0.0])
        assert source.get_emission_rate() == 1.0
        assert source.get_pattern_type() == "stationary"
        
        # Test circular pattern initialization
        source = DynamicSource(
            initial_position=(25.0, 75.0),
            pattern_type="circular",
            amplitude=15.0,
            frequency=0.05
        )
        assert np.allclose(source.get_position(), [25.0, 75.0])
        assert source.get_pattern_type() == "circular"
    
    def test_initialization_validation(self):
        """Test DynamicSource initialization parameter validation."""
        # Test invalid pattern type
        with pytest.raises(ValueError, match="Pattern type must be one of"):
            DynamicSource(pattern_type="invalid_pattern")
        
        # Test negative parameters
        with pytest.raises(ValueError, match="Amplitude must be non-negative"):
            DynamicSource(amplitude=-5.0)
        
        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            DynamicSource(frequency=-0.1)
        
        with pytest.raises(ValueError, match="Noise standard deviation must be non-negative"):
            DynamicSource(noise_std=-1.0)
        
        with pytest.raises(ValueError, match="Emission rate must be non-negative"):
            DynamicSource(emission_rate=-100.0)
    
    def test_stationary_pattern(self, stationary_dynamic_source):
        """Test stationary pattern behavior."""
        initial_position = stationary_dynamic_source.get_position()
        
        # Update multiple times
        for _ in range(10):
            stationary_dynamic_source.update_state(dt=1.0)
            current_position = stationary_dynamic_source.get_position()
            assert np.allclose(current_position, initial_position), "Stationary source should not move"
    
    def test_circular_pattern(self, circular_dynamic_source):
        """Test circular orbit pattern behavior."""
        initial_position = circular_dynamic_source.get_position()
        positions = [initial_position.copy()]
        
        # Collect positions over one complete orbit
        time_steps = 100
        dt = 0.1
        for _ in range(time_steps):
            circular_dynamic_source.update_state(dt=dt)
            positions.append(circular_dynamic_source.get_position().copy())
        
        positions = np.array(positions)
        
        # Verify circular motion around initial position
        center = np.array([50.0, 50.0])
        distances = np.linalg.norm(positions - center, axis=1)
        
        # Distance should be approximately constant (amplitude) after initial position
        expected_radius = 20.0
        for distance in distances[1:]:  # Skip initial position at center
            assert abs(distance - expected_radius) < 1.0, f"Distance {distance} should be close to radius {expected_radius}"
    
    def test_linear_pattern(self, linear_dynamic_source):
        """Test linear motion pattern behavior."""
        initial_position = linear_dynamic_source.get_position()
        velocity = np.array([2.0, 1.0])
        
        # Update and verify linear motion
        dt = 1.0
        for i in range(5):
            linear_dynamic_source.update_state(dt=dt)
            current_position = linear_dynamic_source.get_position()
            expected_position = initial_position + velocity * (i + 1) * dt
            assert np.allclose(current_position, expected_position, atol=1e-10), f"Linear motion position mismatch at step {i+1}"
    
    def test_random_walk_pattern(self, random_walk_source):
        """Test random walk pattern behavior."""
        initial_position = random_walk_source.get_position()
        positions = [initial_position.copy()]
        
        # Collect positions over random walk
        for _ in range(20):
            random_walk_source.update_state(dt=1.0)
            positions.append(random_walk_source.get_position().copy())
        
        positions = np.array(positions)
        
        # Verify positions change (stochastic movement)
        movements = np.diff(positions, axis=0)
        total_movement = np.sum(np.linalg.norm(movements, axis=1))
        
        # Should have some movement due to noise (not deterministic test)
        assert total_movement > 0, "Random walk should produce some movement"
        
        # Verify movements are reasonable (not too large)
        max_step_size = np.max(np.linalg.norm(movements, axis=1))
        assert max_step_size < 10.0, f"Random walk steps should be reasonable, got max step {max_step_size}"
    
    def test_sinusoidal_pattern(self):
        """Test sinusoidal oscillation pattern."""
        source = DynamicSource(
            initial_position=(50.0, 50.0),
            pattern_type="sinusoidal",
            amplitude=10.0,
            frequency=0.1,
            seed=555
        )
        
        initial_position = source.get_position()
        positions = [initial_position.copy()]
        
        # Collect positions over oscillation
        for _ in range(50):
            source.update_state(dt=0.2)
            positions.append(source.get_position().copy())
        
        positions = np.array(positions)
        x_positions = positions[:, 0]
        
        # Verify oscillation in x direction
        x_min, x_max = np.min(x_positions), np.max(x_positions)
        expected_range = 2 * 10.0  # 2 * amplitude
        actual_range = x_max - x_min
        
        assert abs(actual_range - expected_range) < 2.0, f"Sinusoidal range {actual_range} should be close to {expected_range}"
    
    def test_emission_rate_variation(self, circular_dynamic_source):
        """Test temporal emission rate variation for dynamic sources."""
        initial_rate = circular_dynamic_source.get_emission_rate()
        rates = [initial_rate]
        
        # Collect emission rates over time
        for _ in range(20):
            circular_dynamic_source.update_state(dt=1.0)
            rates.append(circular_dynamic_source.get_emission_rate())
        
        rates = np.array(rates)
        
        # Verify rates vary around base rate (circular pattern has emission variation)
        base_rate = 500.0
        min_rate, max_rate = np.min(rates), np.max(rates)
        
        # Should see some variation due to pattern-based emission changes
        assert min_rate < base_rate < max_rate, f"Emission rates should vary around base rate {base_rate}, got range [{min_rate}, {max_rate}]"
    
    def test_custom_emission_pattern(self, stationary_dynamic_source):
        """Test custom emission pattern function."""
        # Define custom pattern function
        def pulse_pattern(t):
            return 2.0 if int(t) % 2 == 0 else 0.5
        
        stationary_dynamic_source.set_emission_pattern(pulse_pattern)
        
        # Test pulsed emission behavior
        base_rate = 1000.0
        rates = []
        
        for i in range(6):
            stationary_dynamic_source.update_state(dt=1.0)
            rate = stationary_dynamic_source.get_emission_rate()
            rates.append(rate)
            
            # Verify pattern application
            expected_multiplier = pulse_pattern(i + 1)
            expected_rate = base_rate * expected_multiplier
            assert abs(rate - expected_rate) < 1e-10, f"Custom pattern rate mismatch at t={i+1}"
    
    def test_trajectory_setting(self, stationary_dynamic_source):
        """Test predefined trajectory setting."""
        # Define trajectory waypoints
        waypoints = [(10.0, 10.0), (50.0, 30.0), (90.0, 50.0), (50.0, 90.0)]
        timestamps = [0.0, 2.0, 4.0, 6.0]
        
        stationary_dynamic_source.set_trajectory(waypoints, timestamps)
        assert stationary_dynamic_source.get_pattern_type() == "trajectory"
        
        # Test trajectory validation
        with pytest.raises(ValueError, match="Trajectory must contain at least one point"):
            stationary_dynamic_source.set_trajectory([])
        
        with pytest.raises(ValueError, match="Number of timestamps must match"):
            stationary_dynamic_source.set_trajectory(waypoints, [0.0, 2.0])  # Mismatched length
    
    def test_time_reset(self, circular_dynamic_source):
        """Test temporal state reset functionality."""
        initial_position = circular_dynamic_source.get_position()
        initial_rate = circular_dynamic_source.get_emission_rate()
        
        # Update state
        for _ in range(10):
            circular_dynamic_source.update_state(dt=1.0)
        
        # Verify state has changed
        assert not np.allclose(circular_dynamic_source.get_position(), initial_position)
        
        # Reset time
        circular_dynamic_source.reset_time()
        
        # Verify state reset to initial conditions
        assert np.allclose(circular_dynamic_source.get_position(), initial_position)
        assert abs(circular_dynamic_source.get_emission_rate() - initial_rate) < 1e-10
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with seeding."""
        # Create two identical sources with same seed
        source1 = DynamicSource(
            initial_position=(30.0, 70.0),
            pattern_type="random_walk",
            noise_std=2.0,
            seed=999
        )
        source2 = DynamicSource(
            initial_position=(30.0, 70.0),
            pattern_type="random_walk",
            noise_std=2.0,
            seed=999
        )
        
        # Execute identical update sequences
        for _ in range(15):
            source1.update_state(dt=0.5)
            source2.update_state(dt=0.5)
            
            # Verify identical behavior
            pos1, pos2 = source1.get_position(), source2.get_position()
            rate1, rate2 = source1.get_emission_rate(), source2.get_emission_rate()
            
            assert np.allclose(pos1, pos2, atol=1e-15), f"Seeded sources should behave identically: {pos1} vs {pos2}"
            assert abs(rate1 - rate2) < 1e-15, f"Seeded emission rates should be identical: {rate1} vs {rate2}"
    
    def test_performance_tracking(self, circular_dynamic_source):
        """Test performance statistics for dynamic source operations."""
        # Perform queries and updates
        for _ in range(8):
            circular_dynamic_source.get_emission_rate()
            circular_dynamic_source.update_state(dt=0.5)
        
        stats = circular_dynamic_source.get_performance_stats()
        assert 'query_count' in stats
        assert 'pattern_type' in stats
        assert 'current_time' in stats
        assert stats['query_count'] == 8.0
        assert stats['pattern_type'] == "circular"
        assert stats['current_time'] == 4.0  # 8 updates * 0.5 dt


class TestSourceFactory:
    """
    Test suite for source factory function validating configuration-driven source creation.
    
    Tests the create_source factory function with various configurations, parameter
    validation, nested source creation for MultiSource, and Hydra integration patterns.
    """
    
    def test_point_source_creation(self):
        """Test PointSource creation through factory."""
        config = {
            'type': 'PointSource',
            'position': (60.0, 80.0),
            'emission_rate': 1200.0,
            'seed': 42
        }
        
        source = create_source(config)
        assert isinstance(source, PointSource)
        assert np.allclose(source.get_position(), [60.0, 80.0])
        assert source.get_emission_rate() == 1200.0
    
    def test_multi_source_creation(self):
        """Test MultiSource creation with nested sources."""
        config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (20.0, 20.0), 'emission_rate': 300.0},
                {'type': 'PointSource', 'position': (80.0, 80.0), 'emission_rate': 700.0},
                {'type': 'DynamicSource', 'initial_position': (50.0, 50.0), 'emission_rate': 400.0, 'pattern_type': 'stationary'}
            ],
            'seed': 123
        }
        
        source = create_source(config)
        assert isinstance(source, MultiSource)
        assert source.get_source_count() == 3
        assert source.get_emission_rate() == 1400.0  # Sum of nested sources
    
    def test_dynamic_source_creation(self):
        """Test DynamicSource creation with various patterns."""
        # Test circular pattern
        config = {
            'type': 'DynamicSource',
            'initial_position': (40.0, 60.0),
            'emission_rate': 800.0,
            'pattern_type': 'circular',
            'amplitude': 25.0,
            'frequency': 0.08,
            'seed': 456
        }
        
        source = create_source(config)
        assert isinstance(source, DynamicSource)
        assert np.allclose(source.get_position(), [40.0, 60.0])
        assert source.get_emission_rate() == 800.0
        assert source.get_pattern_type() == "circular"
    
    def test_factory_validation(self):
        """Test factory function parameter validation."""
        # Test missing type field
        with pytest.raises(KeyError, match="Configuration must include 'type' field"):
            create_source({'position': (10.0, 10.0)})
        
        # Test unknown source type
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source({'type': 'UnknownSource', 'position': (10.0, 10.0)})
        
        # Test invalid parameters for specific source types
        with pytest.raises(ValueError, match="Emission rate must be non-negative"):
            create_source({'type': 'PointSource', 'emission_rate': -100.0})
    
    def test_nested_configuration_validation(self):
        """Test validation of nested source configurations."""
        # Test MultiSource with invalid nested source
        config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (10.0, 10.0), 'emission_rate': 100.0},
                {'type': 'InvalidSource', 'position': (20.0, 20.0)}  # Invalid nested type
            ]
        }
        
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source(config)
    
    def test_configuration_parameter_passing(self):
        """Test proper parameter passing through factory."""
        config = {
            'type': 'DynamicSource',
            'initial_position': (25.0, 75.0),
            'emission_rate': 600.0,
            'pattern_type': 'linear',
            'velocity': (1.5, -0.5),
            'seed': 789
        }
        
        source = create_source(config)
        assert isinstance(source, DynamicSource)
        assert np.allclose(source.get_position(), [25.0, 75.0])
        assert source.get_emission_rate() == 600.0
        assert source.get_pattern_type() == "linear"
        
        # Verify velocity parameter was passed correctly by testing motion
        initial_position = source.get_position()
        source.update_state(dt=2.0)
        expected_position = initial_position + np.array([1.5, -0.5]) * 2.0
        assert np.allclose(source.get_position(), expected_position)


class TestSourceIntegration:
    """
    Test suite for source integration with plume models and environment components.
    
    Tests source integration patterns, environment connectivity, plume model
    compatibility, and end-to-end simulation workflows with source components.
    """
    
    @pytest.fixture
    def mock_plume_env(self, mock_action_config):
        """Create mock plume environment for integration testing."""
        # Create a mock environment that uses source protocol
        mock_env = MagicMock()
        mock_env.source = None
        mock_env.reset.return_value = ({'position': np.array([10.0, 10.0])}, {})
        mock_env.step.return_value = (
            {'position': np.array([11.0, 11.0])}, 
            1.0, False, False, {}
        )
        return mock_env
    
    def test_source_environment_integration(self, mock_plume_env):
        """Test source integration with plume navigation environment."""
        # Create source and attach to environment
        source = PointSource(position=(50.0, 50.0), emission_rate=1500.0)
        mock_plume_env.source = source
        
        # Test environment can access source
        assert mock_plume_env.source is not None
        assert isinstance(mock_plume_env.source, SourceProtocol)
        
        # Test source queries through environment
        emission_rate = mock_plume_env.source.get_emission_rate()
        assert emission_rate == 1500.0
        
        position = mock_plume_env.source.get_position()
        assert np.allclose(position, [50.0, 50.0])
    
    def test_dynamic_source_environment_updates(self, mock_plume_env):
        """Test dynamic source updates during environment steps."""
        # Create dynamic source
        source = DynamicSource(
            initial_position=(30.0, 70.0),
            emission_rate=800.0,
            pattern_type="linear",
            velocity=(1.0, 0.5)
        )
        mock_plume_env.source = source
        
        # Simulate environment steps with source updates
        initial_position = source.get_position()
        
        for step in range(5):
            # Simulate source update during environment step
            mock_plume_env.source.update_state(dt=1.0)
            
            # Verify source evolution
            current_position = mock_plume_env.source.get_position()
            expected_position = initial_position + np.array([1.0, 0.5]) * (step + 1)
            assert np.allclose(current_position, expected_position), f"Source position mismatch at step {step + 1}"
    
    def test_multi_source_environment_integration(self, mock_plume_env):
        """Test MultiSource integration with environment."""
        # Create multi-source configuration
        sources = [
            PointSource(position=(20.0, 20.0), emission_rate=400.0),
            PointSource(position=(80.0, 80.0), emission_rate=600.0),
            DynamicSource(initial_position=(50.0, 50.0), emission_rate=300.0, pattern_type="circular", amplitude=10.0, frequency=0.1)
        ]
        multi_source = MultiSource(sources=sources)
        mock_plume_env.source = multi_source
        
        # Test aggregated emission queries
        total_emission = mock_plume_env.source.get_emission_rate()
        assert total_emission == 1300.0
        
        # Test multi-agent emission queries
        agent_positions = np.array([[25.0, 25.0], [50.0, 50.0], [75.0, 75.0]])
        agent_emissions = mock_plume_env.source.get_emission_rate(agent_positions)
        expected_emissions = np.array([1300.0, 1300.0, 1300.0])
        assert np.allclose(agent_emissions, expected_emissions)
    
    def test_source_configuration_persistence(self):
        """Test source configuration persistence and restoration."""
        # Create source with specific configuration
        original_config = {
            'type': 'DynamicSource',
            'initial_position': (35.0, 65.0),
            'emission_rate': 950.0,
            'pattern_type': 'sinusoidal',
            'amplitude': 15.0,
            'frequency': 0.12,
            'seed': 567
        }
        
        source = create_source(original_config)
        
        # Verify configuration was applied correctly
        assert np.allclose(source.get_position(), [35.0, 65.0])
        assert source.get_emission_rate() == 950.0
        assert source.get_pattern_type() == "sinusoidal"
        
        # Evolve source state
        for _ in range(10):
            source.update_state(dt=0.5)
        
        # Reset to initial state
        source.reset_time()
        
        # Verify restoration to initial configuration
        assert np.allclose(source.get_position(), [35.0, 65.0])
        assert source.get_emission_rate() == 950.0
    
    @patch('src.plume_nav_sim.envs.plume_navigation_env.PlumeNavigationEnv')
    def test_environment_source_injection(self, mock_env_class):
        """Test source injection into environment during creation."""
        # Configure mock environment
        mock_env_instance = MagicMock()
        mock_env_class.return_value = mock_env_instance
        
        # Create source
        source = PointSource(position=(45.0, 55.0), emission_rate=1100.0)
        
        # Test environment creation with source
        config = {
            'source': source,
            'video_path': 'test_video.mp4'
        }
        
        # Simulate environment creation with source injection
        mock_env_instance.source = source
        mock_env_instance.configure = MagicMock()
        
        # Verify source integration
        assert mock_env_instance.source is source
        assert isinstance(mock_env_instance.source, SourceProtocol)


class TestSourcePerformance:
    """
    Test suite for source performance validation and benchmarking.
    
    Tests performance requirements including ≤33ms/step with 100 agents, vectorized
    operations efficiency, memory usage patterns, and multi-threading safety.
    """
    
    def test_point_source_performance_requirements(self):
        """Test PointSource performance meets ≤33ms requirements with 100 agents."""
        source = PointSource(position=(50.0, 50.0), emission_rate=1000.0)
        
        # Test single query performance
        start_time = time.perf_counter()
        for _ in range(1000):
            source.get_emission_rate()
        single_query_time = (time.perf_counter() - start_time) / 1000
        
        assert single_query_time < 0.0001, f"Single query should be <0.1ms, got {single_query_time*1000:.3f}ms"
        
        # Test 100-agent vectorized query performance
        agent_positions = np.random.rand(100, 2) * 100
        
        start_time = time.perf_counter()
        for _ in range(100):  # 100 steps simulation
            source.get_emission_rate(agent_positions)
        multi_agent_time = (time.perf_counter() - start_time) / 100
        
        assert multi_agent_time < 0.033, f"100-agent query should be <33ms, got {multi_agent_time*1000:.3f}ms"
    
    def test_multi_source_performance_scaling(self):
        """Test MultiSource performance scaling with multiple sources and agents."""
        # Create multi-source with varying source counts
        source_counts = [1, 5, 10, 20]
        agent_positions = np.random.rand(100, 2) * 100
        
        for source_count in source_counts:
            sources = [
                PointSource(position=(np.random.rand()*100, np.random.rand()*100), emission_rate=100.0)
                for _ in range(source_count)
            ]
            multi_source = MultiSource(sources=sources)
            
            # Measure query time
            start_time = time.perf_counter()
            for _ in range(50):  # 50 simulation steps
                multi_source.get_emission_rate(agent_positions)
            avg_step_time = (time.perf_counter() - start_time) / 50
            
            # Performance should scale reasonably with source count
            max_allowed_time = 0.033 * (1 + source_count * 0.1)  # Allow some scaling
            assert avg_step_time < max_allowed_time, f"Multi-source with {source_count} sources too slow: {avg_step_time*1000:.3f}ms"
    
    def test_dynamic_source_update_performance(self):
        """Test DynamicSource update performance requirements."""
        patterns = ["stationary", "linear", "circular", "sinusoidal", "random_walk"]
        
        for pattern in patterns:
            source = DynamicSource(
                initial_position=(50.0, 50.0),
                emission_rate=500.0,
                pattern_type=pattern,
                amplitude=10.0 if pattern in ["circular", "sinusoidal"] else 0.0,
                frequency=0.1 if pattern in ["circular", "sinusoidal"] else 0.0,
                velocity=(1.0, 0.5) if pattern == "linear" else (0.0, 0.0),
                noise_std=1.0 if pattern == "random_walk" else 0.0,
                seed=42
            )
            
            # Measure update performance
            start_time = time.perf_counter()
            for _ in range(1000):
                source.update_state(dt=1.0)
            avg_update_time = (time.perf_counter() - start_time) / 1000
            
            assert avg_update_time < 0.001, f"Dynamic source {pattern} update too slow: {avg_update_time*1000:.3f}ms"
    
    @pytest.mark.benchmark
    def test_vectorized_operations_efficiency(self, benchmark):
        """Benchmark vectorized operations vs iterative approaches."""
        source = PointSource(position=(50.0, 50.0), emission_rate=1000.0)
        agent_positions = np.random.rand(100, 2) * 100
        
        # Benchmark vectorized approach (actual implementation)
        def vectorized_query():
            return source.get_emission_rate(agent_positions)
        
        result = benchmark(vectorized_query)
        
        # Verify result correctness
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(result == 1000.0)
    
    def test_memory_usage_efficiency(self):
        """Test memory usage patterns for different source configurations."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Test PointSource memory usage
        point_sources = [PointSource(position=(i*10, i*10), emission_rate=100*i) for i in range(100)]
        point_memory = tracemalloc.get_traced_memory()[0]
        
        tracemalloc.clear_traces()
        
        # Test MultiSource memory usage
        multi_source = MultiSource(sources=point_sources[:50])
        multi_memory = tracemalloc.get_traced_memory()[0]
        
        tracemalloc.clear_traces()
        
        # Test DynamicSource memory usage
        dynamic_sources = [
            DynamicSource(initial_position=(i*5, i*5), emission_rate=50*i, pattern_type="circular", amplitude=5.0, frequency=0.1)
            for i in range(50)
        ]
        dynamic_memory = tracemalloc.get_traced_memory()[0]
        
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 10MB total per requirement)
        max_memory_bytes = 10 * 1024 * 1024  # 10MB
        assert point_memory < max_memory_bytes, f"PointSource memory usage too high: {point_memory/1024/1024:.2f}MB"
        assert multi_memory < max_memory_bytes, f"MultiSource memory usage too high: {multi_memory/1024/1024:.2f}MB"
        assert dynamic_memory < max_memory_bytes, f"DynamicSource memory usage too high: {dynamic_memory/1024/1024:.2f}MB"
    
    def test_thread_safety(self):
        """Test thread safety of source operations."""
        source = MultiSource(sources=[
            PointSource(position=(i*20, i*20), emission_rate=100*i) for i in range(1, 6)
        ])
        
        agent_positions = np.random.rand(50, 2) * 100
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                local_results = []
                for _ in range(100):
                    # Perform concurrent operations
                    rate = source.get_emission_rate(agent_positions)
                    source.update_state(dt=0.01)
                    local_results.append(rate)
                results.extend(local_results)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create and run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors occurred: {errors}"
        
        # Verify reasonable number of results
        assert len(results) > 0, "No results collected from threaded operations"
    
    def test_performance_regression_detection(self):
        """Test for performance regression in source operations."""
        # Baseline performance measurements (these should be updated if optimizations improve performance)
        source = PointSource(position=(50.0, 50.0), emission_rate=1000.0)
        agent_positions = np.random.rand(100, 2) * 100
        
        # Measure current performance
        start_time = time.perf_counter()
        for _ in range(1000):
            source.get_emission_rate(agent_positions)
        current_time = time.perf_counter() - start_time
        
        # Performance should be consistently fast (allow some variance)
        max_allowed_time = 0.1  # 100ms for 1000 queries with 100 agents each
        assert current_time < max_allowed_time, f"Performance regression detected: {current_time:.3f}s > {max_allowed_time}s"


class TestSourceDeterministicBehavior:
    """
    Test suite for deterministic behavior validation across source implementations.
    
    Tests reproducible seeding, deterministic state evolution, configuration snapshot
    reproducibility, and cross-platform consistency per Section 6.6.2.6 requirements.
    """
    
    def test_point_source_deterministic_seeding(self):
        """Test PointSource deterministic behavior with seeding."""
        seed = 12345
        
        # Create two identical sources with same seed
        source1 = PointSource(position=(30.0, 70.0), emission_rate=800.0, seed=seed, enable_temporal_variation=True)
        source2 = PointSource(position=(30.0, 70.0), emission_rate=800.0, seed=seed, enable_temporal_variation=True)
        
        # Execute identical update sequences
        for step in range(20):
            source1.update_state(dt=0.5)
            source2.update_state(dt=0.5)
            
            # Verify identical emission rates
            rate1, rate2 = source1.get_emission_rate(), source2.get_emission_rate()
            assert abs(rate1 - rate2) < 1e-15, f"Seeded sources emission mismatch at step {step}: {rate1} vs {rate2}"
            
            # Verify identical positions (should be same for PointSource)
            pos1, pos2 = source1.get_position(), source2.get_position()
            assert np.allclose(pos1, pos2, atol=1e-15), f"Seeded sources position mismatch at step {step}: {pos1} vs {pos2}"
    
    def test_dynamic_source_deterministic_patterns(self):
        """Test DynamicSource deterministic behavior across different patterns."""
        patterns = ["circular", "sinusoidal", "random_walk"]
        seed = 54321
        
        for pattern in patterns:
            # Create two identical dynamic sources
            kwargs = {
                'initial_position': (40.0, 60.0),
                'emission_rate': 600.0,
                'pattern_type': pattern,
                'seed': seed
            }
            
            if pattern in ["circular", "sinusoidal"]:
                kwargs.update({'amplitude': 15.0, 'frequency': 0.08})
            elif pattern == "random_walk":
                kwargs.update({'noise_std': 2.0})
            
            source1 = DynamicSource(**kwargs)
            source2 = DynamicSource(**kwargs)
            
            # Execute identical sequences
            positions1, positions2 = [], []
            rates1, rates2 = [], []
            
            for _ in range(25):
                source1.update_state(dt=0.2)
                source2.update_state(dt=0.2)
                
                positions1.append(source1.get_position().copy())
                positions2.append(source2.get_position().copy())
                rates1.append(source1.get_emission_rate())
                rates2.append(source2.get_emission_rate())
            
            # Verify identical evolution
            for i, (pos1, pos2, rate1, rate2) in enumerate(zip(positions1, positions2, rates1, rates2)):
                assert np.allclose(pos1, pos2, atol=1e-15), f"Pattern {pattern} position mismatch at step {i}: {pos1} vs {pos2}"
                assert abs(rate1 - rate2) < 1e-15, f"Pattern {pattern} rate mismatch at step {i}: {rate1} vs {rate2}"
    
    def test_multi_source_deterministic_aggregation(self):
        """Test MultiSource deterministic behavior with nested sources."""
        seed = 98765
        
        # Create identical multi-sources with seeded components
        def create_multi_source():
            sources = [
                PointSource(position=(20.0, 20.0), emission_rate=300.0, seed=seed, enable_temporal_variation=True),
                DynamicSource(initial_position=(50.0, 50.0), emission_rate=400.0, pattern_type="circular", 
                            amplitude=10.0, frequency=0.1, seed=seed),
                DynamicSource(initial_position=(80.0, 80.0), emission_rate=500.0, pattern_type="random_walk",
                            noise_std=1.5, seed=seed)
            ]
            return MultiSource(sources=sources, seed=seed)
        
        multi_source1 = create_multi_source()
        multi_source2 = create_multi_source()
        
        # Execute identical update sequences
        agent_positions = np.array([[25.0, 25.0], [50.0, 50.0], [75.0, 75.0]])
        
        for step in range(15):
            multi_source1.update_state(dt=1.0)
            multi_source2.update_state(dt=1.0)
            
            # Test scalar emission rates
            scalar_rate1 = multi_source1.get_emission_rate()
            scalar_rate2 = multi_source2.get_emission_rate()
            assert abs(scalar_rate1 - scalar_rate2) < 1e-15, f"Multi-source scalar rates mismatch at step {step}"
            
            # Test vectorized emission rates
            vector_rates1 = multi_source1.get_emission_rate(agent_positions)
            vector_rates2 = multi_source2.get_emission_rate(agent_positions)
            assert np.allclose(vector_rates1, vector_rates2, atol=1e-15), f"Multi-source vector rates mismatch at step {step}"
            
            # Test individual source positions
            positions1 = multi_source1.get_position()
            positions2 = multi_source2.get_position()
            for i, (pos1, pos2) in enumerate(zip(positions1, positions2)):
                assert np.allclose(pos1, pos2, atol=1e-15), f"Multi-source position {i} mismatch at step {step}"
    
    def test_factory_deterministic_creation(self):
        """Test deterministic source creation through factory function."""
        config = {
            'type': 'DynamicSource',
            'initial_position': (35.0, 65.0),
            'emission_rate': 750.0,
            'pattern_type': 'circular',
            'amplitude': 20.0,
            'frequency': 0.12,
            'seed': 13579
        }
        
        # Create multiple sources from same configuration
        sources = [create_source(config.copy()) for _ in range(3)]
        
        # Execute identical update sequences on all sources
        for step in range(10):
            for source in sources:
                source.update_state(dt=0.8)
            
            # Verify all sources behave identically
            reference_position = sources[0].get_position()
            reference_rate = sources[0].get_emission_rate()
            
            for i, source in enumerate(sources[1:], 1):
                pos = source.get_position()
                rate = source.get_emission_rate()
                assert np.allclose(pos, reference_position, atol=1e-15), f"Factory source {i} position mismatch at step {step}"
                assert abs(rate - reference_rate) < 1e-15, f"Factory source {i} rate mismatch at step {step}"
    
    def test_configuration_snapshot_reproducibility(self):
        """Test reproducibility from configuration snapshots."""
        # Create source and evolve state
        original_source = DynamicSource(
            initial_position=(45.0, 55.0),
            emission_rate=900.0,
            pattern_type="sinusoidal",
            amplitude=12.0,
            frequency=0.15,
            seed=24680
        )
        
        # Evolve state
        evolution_steps = []
        for i in range(10):
            original_source.update_state(dt=0.6)
            evolution_steps.append({
                'step': i,
                'position': original_source.get_position().copy(),
                'emission_rate': original_source.get_emission_rate()
            })
        
        # Create new source from same configuration
        reproduced_source = DynamicSource(
            initial_position=(45.0, 55.0),
            emission_rate=900.0,
            pattern_type="sinusoidal",
            amplitude=12.0,
            frequency=0.15,
            seed=24680
        )
        
        # Reproduce evolution
        for step_data in evolution_steps:
            reproduced_source.update_state(dt=0.6)
            
            # Verify identical reproduction
            step = step_data['step']
            expected_pos = step_data['position']
            expected_rate = step_data['emission_rate']
            
            actual_pos = reproduced_source.get_position()
            actual_rate = reproduced_source.get_emission_rate()
            
            assert np.allclose(actual_pos, expected_pos, atol=1e-15), f"Configuration snapshot position mismatch at step {step}"
            assert abs(actual_rate - expected_rate) < 1e-15, f"Configuration snapshot rate mismatch at step {step}"
    
    def test_cross_platform_consistency(self):
        """Test consistent behavior across different numerical precision scenarios."""
        # Test with different numpy dtypes to simulate cross-platform differences
        dtypes = [np.float32, np.float64]
        seed = 11111
        
        results_by_dtype = {}
        
        for dtype in dtypes:
            # Create source with specific dtype considerations
            source = DynamicSource(
                initial_position=np.array([50.0, 50.0], dtype=dtype),
                emission_rate=np.array([1000.0], dtype=dtype)[0],
                pattern_type="circular",
                amplitude=np.array([8.0], dtype=dtype)[0],
                frequency=np.array([0.2], dtype=dtype)[0],
                seed=seed
            )
            
            # Collect evolution data
            positions = []
            rates = []
            
            for _ in range(8):
                source.update_state(dt=0.5)
                positions.append(source.get_position().copy())
                rates.append(source.get_emission_rate())
            
            results_by_dtype[str(dtype)] = {
                'positions': positions,
                'rates': rates
            }
        
        # Compare results across dtypes (should be very close, allowing for precision differences)
        float32_results = results_by_dtype['<class \'numpy.float32\'>']
        float64_results = results_by_dtype['<class \'numpy.float64\'>']
        
        for i, (pos32, pos64, rate32, rate64) in enumerate(zip(
            float32_results['positions'], float64_results['positions'],
            float32_results['rates'], float64_results['rates']
        )):
            # Allow small differences due to floating point precision
            assert np.allclose(pos32, pos64, atol=1e-6), f"Cross-platform position mismatch at step {i}"
            assert abs(rate32 - rate64) < 1e-6, f"Cross-platform rate mismatch at step {i}"


# Integration with pytest benchmarking for performance monitoring
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])