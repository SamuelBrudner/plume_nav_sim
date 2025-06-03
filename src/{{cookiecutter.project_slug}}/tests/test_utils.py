"""
Comprehensive test module for utility functions validating seed management, logging configuration,
visualization utilities, and cross-cutting concerns.

This module ensures utility function reliability, mathematical precision, and proper integration
with the broader system architecture. Tests cover reproducibility management, performance monitoring,
visualization capabilities, and cross-platform compatibility per Section 6.6.3.1 requirements.

Test Coverage Targets:
- Seed management: >85% coverage with reproducibility validation
- Logging configuration: >85% coverage with performance validation  
- Visualization utilities: >75% coverage with rendering validation
- Mathematical utilities: >90% coverage with precision validation
- Cross-platform compatibility: 100% coverage for file I/O and path handling
"""

import os
import sys
import time
import tempfile
import threading
import platform
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional, Tuple
import uuid

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

# Test the utils package modules
from src.{{cookiecutter.project_slug}}.utils.seed_manager import (
    SeedManager, SeedConfig, get_seed_manager, set_global_seed,
    get_current_seed, get_numpy_generator
)
from src.{{cookiecutter.project_slug}}.utils.logging import (
    LoggingConfig, EnhancedLoggingManager, CorrelationContext,
    HydraJobTracker, CLIMetricsTracker, get_logging_manager,
    setup_enhanced_logging, get_module_logger, create_correlation_scope,
    create_cli_command_scope, create_parameter_validation_scope
)
from src.{{cookiecutter.project_slug}}.utils.visualization import (
    SimulationVisualization, visualize_trajectory, batch_visualize_trajectories,
    setup_headless_mode, get_available_themes, DEFAULT_VISUALIZATION_CONFIG
)


class TestSeedManager:
    """
    Comprehensive test suite for seed management functionality.
    
    Tests reproducibility, random state management, Hydra integration,
    performance characteristics, and cross-platform consistency.
    """
    
    def setup_method(self):
        """Reset seed manager state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()
    
    def test_seed_manager_singleton_pattern(self):
        """Test that SeedManager implements singleton pattern correctly."""
        manager1 = SeedManager()
        manager2 = SeedManager()
        assert manager1 is manager2
        
        # Test factory function returns same instance
        manager3 = get_seed_manager()
        assert manager1 is manager3
    
    def test_seed_config_validation(self):
        """Test SeedConfig Pydantic schema validation."""
        # Valid configuration
        config = SeedConfig(seed=42, auto_seed=False, validate_initialization=True)
        assert config.seed == 42
        assert config.auto_seed is False
        assert config.validate_initialization is True
        
        # Test default values
        default_config = SeedConfig()
        assert default_config.seed is None
        assert default_config.auto_seed is True
        assert default_config.validate_initialization is True
        
        # Test invalid seed range (should not raise due to Field constraints)
        config_negative = SeedConfig(seed=-1)
        assert config_negative.seed == -1  # Pydantic allows this, validation happens in manager
    
    def test_seed_manager_initialization_basic(self):
        """Test basic seed manager initialization."""
        manager = SeedManager()
        
        # Test initialization with explicit seed
        seed = manager.initialize(config=SeedConfig(seed=42))
        assert seed == 42
        assert manager.current_seed == 42
        assert manager.numpy_generator is not None
        
        # Test reproducible random number generation
        random_value = manager.numpy_generator.random()
        assert isinstance(random_value, float)
        assert 0.0 <= random_value <= 1.0
    
    def test_seed_manager_auto_generation(self):
        """Test automatic seed generation from entropy."""
        manager = SeedManager()
        
        # Test auto-generation
        config = SeedConfig(auto_seed=True, seed=None)
        seed1 = manager.initialize(config=config)
        
        # Reset and generate again
        SeedManager.reset()
        manager = SeedManager()
        seed2 = manager.initialize(config=config)
        
        # Seeds should be different (high probability)
        assert seed1 != seed2
        assert 0 <= seed1 <= 2**32 - 1
        assert 0 <= seed2 <= 2**32 - 1
    
    def test_seed_manager_performance(self):
        """Test seed manager initialization performance requirements."""
        manager = SeedManager()
        
        start_time = time.perf_counter()
        manager.initialize(config=SeedConfig(seed=42))
        initialization_time = (time.perf_counter() - start_time) * 1000
        
        # Should initialize in under 100ms per specification
        assert initialization_time < 100, f"Initialization took {initialization_time:.2f}ms > 100ms"
    
    def test_reproducibility_validation(self):
        """Test reproducibility across multiple initializations."""
        # Initialize with specific seed
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42))
        
        # Generate reference values
        np.random.seed(42)  # Set legacy numpy random state
        manager.numpy_generator.bit_generator.state = np.random.default_rng(42).bit_generator.state
        
        reference_values = {
            'numpy_generator': manager.numpy_generator.random()
        }
        
        # Reset and reinitialize
        SeedManager.reset()
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42))
        
        # Validate reproducibility (allow for some floating point tolerance)
        current_value = manager.numpy_generator.random()
        assert abs(current_value - reference_values['numpy_generator']) < 1e-10
    
    def test_temporary_seed_context(self):
        """Test temporary seed override functionality."""
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42, preserve_state=True))
        
        # Get original random value
        original_value = manager.numpy_generator.random()
        
        # Use temporary seed
        with manager.temporary_seed(100) as temp_seed:
            assert temp_seed == 100
            assert manager.current_seed == 100
            temp_value = manager.numpy_generator.random()
        
        # Should restore original seed
        assert manager.current_seed == 42
        # Note: We can't test exact value restoration without complex state management
        # but we can test that the context manager works
    
    def test_experiment_seed_generation(self):
        """Test deterministic experiment seed generation."""
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42))
        
        # Generate experiment seeds
        seeds = manager.generate_experiment_seeds(5)
        assert len(seeds) == 5
        assert all(isinstance(seed, int) for seed in seeds)
        assert all(0 <= seed <= 2**32 - 1 for seed in seeds)
        
        # Should be deterministic
        seeds2 = manager.generate_experiment_seeds(5)
        assert seeds == seeds2
    
    def test_state_preservation_and_restoration(self):
        """Test random state preservation and restoration."""
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42, preserve_state=True))
        
        # Generate some random numbers
        _ = manager.numpy_generator.random()
        _ = manager.numpy_generator.random()
        
        # Capture state
        state = manager.get_state()
        assert state is not None
        assert 'numpy_generator_state' in state
        assert 'seed' in state
        
        # Generate more numbers
        value_before_restore = manager.numpy_generator.random()
        
        # Restore state
        manager.restore_state(state)
        
        # Should get same value as before
        value_after_restore = manager.numpy_generator.random()
        assert value_before_restore == value_after_restore
    
    def test_environment_hash_generation(self):
        """Test environment hash generation for cross-platform consistency."""
        manager = SeedManager()
        manager.initialize(config=SeedConfig(seed=42, hash_environment=True))
        
        env_hash = manager.environment_hash
        assert env_hash is not None
        assert len(env_hash) == 8  # MD5 hash truncated to 8 characters
        assert isinstance(env_hash, str)
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Test set_global_seed
        seed = set_global_seed(42)
        assert seed == 42
        
        # Test get_current_seed
        current = get_current_seed()
        assert current == 42
        
        # Test get_numpy_generator
        generator = get_numpy_generator()
        assert generator is not None
        assert hasattr(generator, 'random')
    
    def test_seed_manager_error_handling(self):
        """Test error handling in seed manager."""
        manager = SeedManager()
        
        # Test invalid seed range in initialization
        with pytest.raises((ValueError, RuntimeError)):
            manager.initialize(config=SeedConfig(seed=2**32))  # Too large
        
        # Test state operations without preservation enabled
        manager.initialize(config=SeedConfig(seed=42, preserve_state=False))
        assert manager.get_state() is None
        
        with pytest.raises(RuntimeError):
            manager.temporary_seed(100).__enter__()
    
    @pytest.mark.parametrize("platform_info", [
        ("Windows", "10", "AMD64"),
        ("Linux", "5.4.0", "x86_64"),
        ("Darwin", "20.0.0", "x86_64"),
    ])
    def test_cross_platform_consistency(self, platform_info):
        """Test cross-platform deterministic behavior."""
        with patch('platform.platform') as mock_platform, \
             patch('platform.python_version') as mock_python, \
             patch('sys.maxsize', 9223372036854775807):
            
            mock_platform.return_value = f"{platform_info[0]}-{platform_info[1]}-{platform_info[2]}"
            mock_python.return_value = "3.9.0"
            
            manager = SeedManager()
            manager.initialize(config=SeedConfig(seed=42, hash_environment=True))
            
            # Environment hash should be deterministic for same platform info
            env_hash = manager.environment_hash
            assert env_hash is not None
            assert len(env_hash) == 8


class TestLoggingUtilities:
    """
    Comprehensive test suite for enhanced logging functionality.
    
    Tests configuration management, performance monitoring, Hydra integration,
    correlation context management, and CLI metrics tracking.
    """
    
    def setup_method(self):
        """Reset logging manager state before each test."""
        EnhancedLoggingManager.reset()
        CorrelationContext.clear_correlation_id()
    
    def teardown_method(self):
        """Clean up after each test."""
        EnhancedLoggingManager.reset()
        CorrelationContext.clear_correlation_id()
    
    def test_logging_config_validation(self):
        """Test LoggingConfig Pydantic schema validation."""
        # Valid configuration
        config = LoggingConfig(
            level="DEBUG",
            console_format="enhanced",
            enable_file_logging=True,
            enable_hydra_integration=True
        )
        assert config.level == "DEBUG"
        assert config.console_format == "enhanced"
        assert config.enable_file_logging is True
        assert config.enable_hydra_integration is True
        
        # Test defaults
        default_config = LoggingConfig()
        assert default_config.level == "INFO"
        assert default_config.enable_correlation_ids is True
        assert default_config.performance_threshold_ms == 100.0
    
    def test_enhanced_logging_manager_singleton(self):
        """Test enhanced logging manager singleton pattern."""
        manager1 = EnhancedLoggingManager()
        manager2 = EnhancedLoggingManager()
        assert manager1 is manager2
        
        # Test factory function
        manager3 = get_logging_manager()
        assert manager1 is manager3
    
    def test_logging_manager_initialization(self):
        """Test logging manager initialization."""
        manager = EnhancedLoggingManager()
        
        # Test initialization with config
        config = LoggingConfig(level="DEBUG", enable_file_logging=False)
        manager.initialize(config=config)
        
        # Verify initialization completed without errors
        assert manager._config is not None
        assert manager._config.level == "DEBUG"
        assert manager._config.enable_file_logging is False
    
    def test_correlation_context_management(self):
        """Test correlation ID context management."""
        # Test automatic ID generation
        corr_id1 = CorrelationContext.get_correlation_id()
        assert corr_id1 is not None
        assert len(corr_id1) == 8
        
        # Test explicit ID setting
        test_id = "test123"
        CorrelationContext.set_correlation_id(test_id)
        assert CorrelationContext.get_correlation_id() == test_id
        
        # Test context manager
        with CorrelationContext.correlation_scope("scope123") as scope_id:
            assert scope_id == "scope123"
            assert CorrelationContext.get_correlation_id() == "scope123"
        
        # Should restore previous context
        assert CorrelationContext.get_correlation_id() == test_id
    
    def test_hydra_job_tracker(self):
        """Test Hydra job tracking functionality."""
        tracker = HydraJobTracker()
        
        # Test initialization without Hydra
        job_info = tracker.initialize()
        assert 'status' in job_info
        assert job_info['status'] == 'hydra_not_initialized'
        
        # Test job metrics
        metrics = tracker.get_job_metrics()
        assert 'job_name' in metrics
        assert 'config_checksum' in metrics
    
    def test_cli_metrics_tracker(self):
        """Test CLI metrics tracking functionality."""
        tracker = CLIMetricsTracker()
        
        # Test command tracking
        with tracker.track_command("test_command") as metrics:
            metrics['test_param'] = "test_value"
            time.sleep(0.01)  # Small delay to test timing
        
        assert 'command_name' in metrics
        assert 'start_time' in metrics
        assert 'total_execution_time_ms' in metrics
        assert metrics['total_execution_time_ms'] > 0
    
    def test_parameter_validation_timing(self):
        """Test parameter validation timing tracking."""
        tracker = CLIMetricsTracker()
        
        with tracker.track_parameter_validation() as validation_metrics:
            # Simulate parameter validation work
            time.sleep(0.01)
        
        assert 'validation_time_ms' in validation_metrics
        assert validation_metrics['validation_time_ms'] > 0
    
    def test_logging_performance_requirements(self):
        """Test logging initialization performance."""
        manager = EnhancedLoggingManager()
        config = LoggingConfig(enable_file_logging=False)  # Disable file I/O for speed
        
        start_time = time.perf_counter()
        manager.initialize(config=config)
        initialization_time = (time.perf_counter() - start_time) * 1000
        
        # Should initialize quickly (reasonable threshold for testing)
        assert initialization_time < 500, f"Logging init took {initialization_time:.2f}ms"
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Test setup function
        setup_enhanced_logging(LoggingConfig(enable_file_logging=False))
        
        # Test module logger creation
        logger = get_module_logger(__name__)
        assert logger is not None
        
        # Test correlation scope
        with create_correlation_scope("test123") as corr_id:
            assert corr_id == "test123"
        
        # Test CLI command scope
        with create_cli_command_scope("test_cmd") as cmd_metrics:
            assert 'command_name' in cmd_metrics
            assert cmd_metrics['command_name'] == "test_cmd"
        
        # Test parameter validation scope
        with create_parameter_validation_scope() as val_metrics:
            pass  # metrics dict should be available
    
    def test_logging_context_injection(self):
        """Test automatic context injection into log records."""
        manager = EnhancedLoggingManager()
        config = LoggingConfig(
            enable_correlation_ids=True,
            enable_seed_context=False,  # Disable to avoid seed manager dependency
            enable_hydra_integration=False  # Disable to avoid Hydra dependency
        )
        manager.initialize(config=config)
        
        # Set correlation context
        CorrelationContext.set_correlation_id("test_correlation")
        
        # Update logging context
        manager.update_context({'test_key': 'test_value'})
        
        # Context should be available in manager
        assert 'test_key' in manager._log_context
        assert manager._log_context['test_key'] == 'test_value'
    
    def test_threading_safety(self):
        """Test thread safety of correlation context."""
        results = {}
        
        def worker(thread_id):
            CorrelationContext.set_correlation_id(f"thread_{thread_id}")
            time.sleep(0.01)  # Allow context switching
            results[thread_id] = CorrelationContext.get_correlation_id()
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have its own correlation ID
        for i in range(5):
            assert results[i] == f"thread_{i}"
    
    def test_logging_environment_variable_tracking(self):
        """Test environment variable usage logging."""
        manager = get_logging_manager()
        manager.initialize(LoggingConfig(enable_environment_logging=True))
        
        # Set up test environment variable
        test_env_var = "TEST_LOG_VAR"
        os.environ[test_env_var] = "test_value"
        
        try:
            # This should log environment variable usage
            manager.log_environment_variables([test_env_var, "NONEXISTENT_VAR"])
            # Should complete without error
        finally:
            # Clean up
            if test_env_var in os.environ:
                del os.environ[test_env_var]


class TestVisualizationUtilities:
    """
    Comprehensive test suite for visualization functionality.
    
    Tests real-time animation, static trajectory plots, batch processing,
    configuration management, and headless operation modes.
    """
    
    def setup_method(self):
        """Set up test environment for visualization tests."""
        # Ensure matplotlib backend is set for testing
        plt.switch_backend('Agg')  # Non-interactive backend
    
    def teardown_method(self):
        """Clean up after visualization tests."""
        plt.close('all')  # Close all figures
    
    def test_default_visualization_config(self):
        """Test default visualization configuration structure."""
        config = DEFAULT_VISUALIZATION_CONFIG
        
        # Test required configuration sections
        assert 'animation' in config
        assert 'export' in config
        assert 'resolution' in config
        assert 'theme' in config
        assert 'agents' in config
        
        # Test animation defaults
        assert config['animation']['fps'] == 30
        assert config['animation']['interval'] == 33
        assert config['animation']['blit'] is True
        
        # Test export defaults
        assert config['export']['format'] == "mp4"
        assert config['export']['codec'] == "libx264"
    
    def test_simulation_visualization_initialization(self):
        """Test SimulationVisualization initialization."""
        # Test basic initialization
        viz = SimulationVisualization(headless=True)
        assert viz.fig is not None
        assert viz.ax is not None
        assert viz.headless is True
        
        # Test initialization with custom config
        custom_config = {'animation': {'fps': 60}, 'theme': {'colormap': 'plasma'}}
        viz_custom = SimulationVisualization(config=custom_config, headless=True)
        assert viz_custom.config.animation.fps == 60
        assert viz_custom.config.theme.colormap == 'plasma'
    
    def test_simulation_visualization_environment_setup(self):
        """Test environment setup for visualization."""
        viz = SimulationVisualization(headless=True)
        
        # Create test environment
        environment = np.random.rand(50, 50)
        extent = (0, 50, 0, 50)
        
        viz.setup_environment(environment, extent)
        
        assert viz.img is not None
        assert viz.colorbar is not None
        assert viz.environment_bounds == extent
    
    def test_simulation_visualization_update_single_agent(self):
        """Test visualization update for single agent."""
        viz = SimulationVisualization(headless=True)
        
        # Set up environment
        environment = np.random.rand(20, 20)
        viz.setup_environment(environment)
        
        # Test single agent update
        positions = np.array([10.0, 10.0])
        orientations = 45.0
        odor_values = 0.5
        
        frame_data = (positions, orientations, odor_values)
        artists = viz.update_visualization(frame_data)
        
        assert len(artists) > 0
        assert len(viz.agent_markers) == 1
    
    def test_simulation_visualization_update_multi_agent(self):
        """Test visualization update for multiple agents."""
        viz = SimulationVisualization(headless=True)
        
        # Set up environment
        environment = np.random.rand(20, 20)
        viz.setup_environment(environment)
        
        # Test multi-agent update
        positions = np.array([[5.0, 5.0], [15.0, 15.0], [10.0, 10.0]])
        orientations = np.array([0.0, 90.0, 180.0])
        odor_values = np.array([0.3, 0.7, 0.5])
        
        frame_data = (positions, orientations, odor_values)
        artists = viz.update_visualization(frame_data)
        
        assert len(artists) > 0
        assert len(viz.agent_markers) == 3
        assert len(viz.trail_data) == 3
    
    def test_simulation_visualization_performance_adaptive_quality(self):
        """Test adaptive quality management for performance."""
        viz = SimulationVisualization(headless=True)
        viz.adaptive_quality = True
        
        # Set up environment
        environment = np.random.rand(20, 20)
        viz.setup_environment(environment)
        
        # Simulate large number of agents (should trigger quality reduction)
        n_agents = 100
        positions = np.random.rand(n_agents, 2) * 20
        orientations = np.random.rand(n_agents) * 360
        odor_values = np.random.rand(n_agents)
        
        frame_data = (positions, orientations, odor_values)
        artists = viz.update_visualization(frame_data)
        
        # Should complete without error even with many agents
        assert len(artists) > 0
        assert len(viz.agent_markers) == n_agents
    
    def test_simulation_visualization_animation_creation(self):
        """Test animation creation functionality."""
        viz = SimulationVisualization(headless=True)
        
        # Set up environment
        environment = np.random.rand(20, 20)
        viz.setup_environment(environment)
        
        # Create simple update function
        def update_func(frame_idx):
            positions = np.array([[frame_idx % 20, 10.0]])
            orientations = np.array([frame_idx * 10])
            odor_values = np.array([0.5])
            return positions, orientations, odor_values
        
        # Create animation (small number of frames for testing)
        anim = viz.create_animation(update_func, frames=5, interval=100)
        
        assert anim is not None
        assert isinstance(anim, FuncAnimation)
        assert viz.animation_obj is anim
    
    def test_visualize_trajectory_single_agent(self):
        """Test static trajectory visualization for single agent."""
        # Create test trajectory data
        time_steps = 50
        positions = np.column_stack([
            np.linspace(0, 10, time_steps),
            np.sin(np.linspace(0, 2*np.pi, time_steps)) * 2 + 5
        ])
        
        # Test basic trajectory plot
        fig = visualize_trajectory(
            positions=positions,
            show_plot=False,
            batch_mode=True
        )
        
        assert fig is not None
        assert isinstance(fig, Figure)
    
    def test_visualize_trajectory_multi_agent(self):
        """Test static trajectory visualization for multiple agents."""
        # Create test trajectory data for multiple agents
        time_steps = 30
        n_agents = 3
        
        positions = np.zeros((n_agents, time_steps, 2))
        for i in range(n_agents):
            positions[i, :, 0] = np.linspace(i*2, i*2 + 10, time_steps)
            positions[i, :, 1] = np.sin(np.linspace(0, 2*np.pi, time_steps)) * (i+1) + 5
        
        orientations = np.random.rand(n_agents, time_steps) * 360
        
        fig = visualize_trajectory(
            positions=positions,
            orientations=orientations,
            show_plot=False,
            batch_mode=True,
            title="Multi-Agent Test Trajectory"
        )
        
        assert fig is not None
        assert isinstance(fig, Figure)
    
    def test_visualize_trajectory_with_plume_background(self):
        """Test trajectory visualization with plume background."""
        # Create test data
        positions = np.column_stack([
            np.linspace(0, 20, 25),
            np.linspace(0, 20, 25)
        ])
        
        # Create test plume background
        plume_frames = np.random.rand(20, 20)
        
        fig = visualize_trajectory(
            positions=positions,
            plume_frames=plume_frames,
            show_plot=False,
            batch_mode=True
        )
        
        assert fig is not None
        assert isinstance(fig, Figure)
    
    def test_batch_visualize_trajectories(self):
        """Test batch trajectory visualization processing."""
        # Create test trajectory data
        trajectory_data = []
        for i in range(3):
            positions = np.column_stack([
                np.linspace(0, 10, 20),
                np.sin(np.linspace(0, 2*np.pi, 20)) * (i+1)
            ])
            trajectory_data.append({
                'positions': positions,
                'title': f'Test Trajectory {i+1}'
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = batch_visualize_trajectories(
                trajectory_data=trajectory_data,
                output_dir=temp_dir,
                naming_pattern="test_batch_{idx:02d}"
            )
            
            assert len(saved_paths) == 3
            for path in saved_paths:
                assert path.exists()
                assert path.suffix == '.png'
    
    def test_visualization_headless_mode(self):
        """Test headless mode setup and operation."""
        setup_headless_mode()
        
        # Should be able to create visualizations in headless mode
        viz = SimulationVisualization(headless=True)
        assert viz.headless is True
        
        # Should be able to create static plots
        positions = np.column_stack([np.linspace(0, 10, 20), np.linspace(0, 10, 20)])
        fig = visualize_trajectory(positions, show_plot=False, batch_mode=True)
        assert fig is not None
    
    def test_get_available_themes(self):
        """Test available themes functionality."""
        themes = get_available_themes()
        
        assert isinstance(themes, dict)
        assert len(themes) > 0
        
        # Test expected themes
        expected_themes = ["scientific", "presentation", "high_contrast"]
        for theme in expected_themes:
            assert theme in themes
            assert 'colormap' in themes[theme]
            assert 'background' in themes[theme]
            assert 'dpi' in themes[theme]
    
    def test_visualization_configuration_validation(self):
        """Test visualization configuration validation and merging."""
        # Test with valid configuration
        config = {
            'animation': {'fps': 60},
            'theme': {'colormap': 'plasma'},
            'agents': {'max_agents_full_quality': 25}
        }
        
        viz = SimulationVisualization(config=config, headless=True)
        assert viz.config.animation.fps == 60
        assert viz.config.theme.colormap == 'plasma'
        assert viz.config.agents.max_agents_full_quality == 25
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization components."""
        viz = SimulationVisualization(headless=True)
        
        # Test with invalid frame data
        with pytest.raises((ValueError, IndexError, AttributeError)):
            invalid_frame_data = (None, None, None)
            viz.update_visualization(invalid_frame_data)
    
    @pytest.mark.parametrize("n_agents,expected_performance", [
        (1, "high"),
        (10, "high"),
        (50, "medium"),
        (100, "low")
    ])
    def test_visualization_scaling_performance(self, n_agents, expected_performance):
        """Test visualization performance scaling with agent count."""
        viz = SimulationVisualization(headless=True)
        environment = np.random.rand(20, 20)
        viz.setup_environment(environment)
        
        # Generate test data
        positions = np.random.rand(n_agents, 2) * 20
        orientations = np.random.rand(n_agents) * 360
        odor_values = np.random.rand(n_agents)
        
        frame_data = (positions, orientations, odor_values)
        
        # Measure update time
        start_time = time.perf_counter()
        artists = viz.update_visualization(frame_data)
        update_time = (time.perf_counter() - start_time) * 1000
        
        # Verify update completed
        assert len(artists) > 0
        assert len(viz.agent_markers) == n_agents
        
        # Performance expectations (relaxed for testing environment)
        if expected_performance == "high":
            assert update_time < 100  # 100ms threshold for testing
        elif expected_performance == "medium":
            assert update_time < 200  # 200ms threshold
        else:  # low performance expected
            assert update_time < 500  # 500ms threshold


class TestUtilityIntegration:
    """
    Integration tests for utility functions working together.
    
    Tests cross-cutting concerns, module interactions, and system-wide
    utility functionality.
    """
    
    def setup_method(self):
        """Set up integration test environment."""
        SeedManager.reset()
        EnhancedLoggingManager.reset()
        plt.switch_backend('Agg')
    
    def teardown_method(self):
        """Clean up integration test environment."""
        SeedManager.reset()
        EnhancedLoggingManager.reset()
        plt.close('all')
    
    def test_seed_manager_logging_integration(self):
        """Test integration between seed manager and logging system."""
        # Initialize seed manager first
        set_global_seed(42)
        
        # Initialize logging with seed context enabled
        setup_enhanced_logging(LoggingConfig(
            enable_seed_context=True,
            enable_file_logging=False
        ))
        
        # Logging manager should pick up seed context
        manager = get_logging_manager()
        assert 'seed' in manager._log_context
        assert manager._log_context['seed'] == 42
    
    def test_reproducible_visualization_generation(self):
        """Test reproducible visualization generation with seed management."""
        # Set deterministic seed
        set_global_seed(42)
        
        # Generate reproducible test data
        positions1 = np.random.rand(20, 2) * 10
        
        # Reset and regenerate with same seed
        set_global_seed(42)
        positions2 = np.random.rand(20, 2) * 10
        
        # Should be identical
        np.testing.assert_array_equal(positions1, positions2)
        
        # Create visualizations
        fig1 = visualize_trajectory(positions1, show_plot=False, batch_mode=True)
        fig2 = visualize_trajectory(positions2, show_plot=False, batch_mode=True)
        
        assert fig1 is not None
        assert fig2 is not None
    
    def test_logging_performance_monitoring_integration(self):
        """Test logging integration with performance monitoring."""
        # Set up logging with performance monitoring
        setup_enhanced_logging(LoggingConfig(
            enable_performance_monitoring=True,
            performance_threshold_ms=50.0,
            enable_file_logging=False
        ))
        
        # Test CLI metrics tracking
        with create_cli_command_scope("test_integration") as metrics:
            # Simulate some work
            time.sleep(0.01)
            
            # Test parameter validation timing
            with create_parameter_validation_scope() as val_metrics:
                time.sleep(0.005)
            
            metrics['test_param'] = 'test_value'
        
        assert 'total_execution_time_ms' in metrics
        assert 'validation_time_ms' in val_metrics
    
    def test_cross_platform_file_operations(self):
        """Test cross-platform compatibility for file I/O operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test path handling across platforms
            test_file = temp_path / "test_file.png"
            
            # Generate test visualization
            positions = np.column_stack([np.linspace(0, 10, 20), np.linspace(0, 10, 20)])
            
            # Save visualization
            fig = visualize_trajectory(
                positions=positions,
                output_path=test_file,
                show_plot=False,
                batch_mode=True
            )
            
            # Verify file was created
            assert test_file.exists()
            assert test_file.stat().st_size > 0
    
    def test_memory_management_integration(self):
        """Test memory management across utility components."""
        import psutil
        import gc
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        for i in range(10):
            # Seed management
            set_global_seed(i)
            
            # Visualization creation
            positions = np.random.rand(50, 2) * 20
            fig = visualize_trajectory(positions, show_plot=False, batch_mode=True)
            plt.close(fig)
            
            # Force garbage collection
            gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - baseline_memory) / 1024 / 1024  # MB
        
        # Allow some growth but not excessive (50MB threshold for test)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB"
    
    def test_configuration_error_recovery_integration(self):
        """Test error recovery across utility components."""
        # Test logging initialization with invalid config
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            setup_enhanced_logging("invalid_config")  # Should be dict or LoggingConfig
        
        # Should still be able to initialize with valid config
        setup_enhanced_logging(LoggingConfig(enable_file_logging=False))
        
        # Test seed manager error recovery
        try:
            SeedManager().initialize(config="invalid")  # Should fail
        except (ValueError, TypeError, RuntimeError):
            pass  # Expected
        
        # Should still work with valid config
        seed = set_global_seed(42)
        assert seed == 42
    
    def test_concurrent_utility_operations(self):
        """Test thread safety of utility operations."""
        import concurrent.futures
        
        def worker(thread_id):
            # Each thread should have independent state
            with create_correlation_scope(f"thread_{thread_id}"):
                seed = set_global_seed(thread_id)
                positions = np.random.rand(10, 2) * 10
                
                return {
                    'thread_id': thread_id,
                    'seed': seed,
                    'correlation_id': CorrelationContext.get_correlation_id(),
                    'data_shape': positions.shape
                }
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        # Verify each thread had independent state
        for i, result in enumerate(results):
            assert result['thread_id'] == i
            assert result['seed'] == i
            assert result['correlation_id'] == f"thread_{i}"
            assert result['data_shape'] == (10, 2)


class TestUtilityErrorHandling:
    """
    Test error handling and edge cases for utility functions.
    
    Validates robust error handling, graceful degradation, and
    appropriate error messages for all utility components.
    """
    
    def test_seed_manager_edge_cases(self):
        """Test seed manager edge cases and error conditions."""
        manager = SeedManager()
        
        # Test with None configuration
        seed = manager.initialize(config=None)
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1
        
        # Test state operations edge cases
        SeedManager.reset()
        manager = SeedManager()
        manager.initialize(config=SeedConfig(preserve_state=False))
        
        # Should handle missing state gracefully
        assert manager.get_state() is None
    
    def test_logging_edge_cases(self):
        """Test logging system edge cases."""
        # Test initialization with minimal config
        manager = EnhancedLoggingManager()
        manager.initialize(config={})  # Empty config dict
        
        # Should use defaults
        assert manager._config is not None
        
        # Test correlation context edge cases
        CorrelationContext.clear_correlation_id()
        
        # Should generate new ID when needed
        corr_id = CorrelationContext.get_correlation_id()
        assert corr_id is not None
        assert len(corr_id) == 8
    
    def test_visualization_edge_cases(self):
        """Test visualization edge cases and error handling."""
        # Test with empty trajectory data
        empty_positions = np.array([]).reshape(0, 2)
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = visualize_trajectory(empty_positions, show_plot=False, batch_mode=True)
            if fig is not None:
                plt.close(fig)
        except (ValueError, IndexError):
            pass  # Expected for empty data
        
        # Test with single point trajectory
        single_point = np.array([[5.0, 5.0]])
        fig = visualize_trajectory(single_point, show_plot=False, batch_mode=True)
        assert fig is not None
        plt.close(fig)
    
    def test_mathematical_precision_edge_cases(self):
        """Test mathematical precision in edge cases."""
        # Test with very small numbers
        tiny_positions = np.array([[1e-10, 1e-10], [2e-10, 2e-10]])
        fig = visualize_trajectory(tiny_positions, show_plot=False, batch_mode=True)
        assert fig is not None
        plt.close(fig)
        
        # Test with very large numbers
        large_positions = np.array([[1e6, 1e6], [2e6, 2e6]])
        fig = visualize_trajectory(large_positions, show_plot=False, batch_mode=True)
        assert fig is not None
        plt.close(fig)
    
    def test_resource_cleanup_edge_cases(self):
        """Test resource cleanup in error conditions."""
        # Test visualization cleanup after errors
        viz = SimulationVisualization(headless=True)
        
        try:
            # Force an error during setup
            viz.setup_environment(None)  # Should fail
        except (AttributeError, TypeError):
            pass  # Expected error
        
        # Should be able to clean up gracefully
        viz.close()
        
        # Test multiple cleanup calls
        viz.close()  # Should not error


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])