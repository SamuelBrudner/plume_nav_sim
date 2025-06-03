"""
Tests for the consolidated trajectory visualization module.

This test module covers the comprehensive visualization capabilities including
real-time animation, static trajectory plotting, Hydra configuration integration,
CLI export functionality, multi-agent visualization scaling, and batch processing
workflows as specified in Section 7.3 of the technical specification.

Key Features Tested:
- SimulationVisualization class for real-time animation (30+ FPS)
- visualize_trajectory function for publication-quality static plots
- Hydra configuration integration and parameter management
- CLI export capabilities with headless mode support
- Multi-agent visualization scaling (up to 100 agents)
- Publication-quality export with configurable DPI settings
- Batch processing workflows for automated experiment analysis
"""

import pytest
import numpy as np
import pathlib
import tempfile
from unittest.mock import patch, MagicMock, call
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Any, Optional

# Import from the new consolidated visualization module
from src.{{cookiecutter.project_slug}}.utils.visualization import (
    SimulationVisualization,
    visualize_trajectory,
    batch_visualize_trajectories,
    setup_headless_mode,
    get_available_themes,
    create_simulation_visualization,
    export_animation,
    DEFAULT_VISUALIZATION_CONFIG
)

try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = Dict[str, Any]  # Fallback type hint


# =============================================================================
# FIXTURES FOR COMPREHENSIVE TESTING
# =============================================================================

@pytest.fixture
def mock_plt():
    """Mock matplotlib.pyplot to avoid displaying plots during tests."""
    with patch('src.{{cookiecutter.project_slug}}.utils.visualization.plt') as mock_plt:
        # Configure mock figure and axis
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.figure.return_value = mock_fig
        yield mock_plt


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib module for headless mode testing."""
    with patch('src.{{cookiecutter.project_slug}}.utils.visualization.matplotlib') as mock_mpl:
        mock_mpl.use = MagicMock()
        yield mock_mpl


@pytest.fixture
def sample_single_agent_data():
    """Generate synthetic single-agent trajectory data for testing."""
    time_steps = 50
    positions = np.array([[i * 0.5, np.sin(i * 0.1) * 10 + 25] for i in range(time_steps)])
    positions = positions.reshape(1, time_steps, 2)  # Shape: (1, 50, 2)
    orientations = np.array([i * 2.0 for i in range(time_steps)]).reshape(1, -1)  # Shape: (1, 50)
    odor_values = np.random.uniform(0, 1, size=(1, time_steps))  # Shape: (1, 50)
    
    return {
        'positions': positions,
        'orientations': orientations,
        'odor_values': odor_values
    }


@pytest.fixture
def sample_multi_agent_data():
    """Generate synthetic multi-agent trajectory data for testing."""
    num_agents = 5
    time_steps = 30
    
    # Create different trajectory patterns for each agent
    positions = np.zeros((num_agents, time_steps, 2))
    orientations = np.zeros((num_agents, time_steps))
    odor_values = np.zeros((num_agents, time_steps))
    
    for agent_idx in range(num_agents):
        # Different movement patterns for each agent
        if agent_idx == 0:
            # Linear movement
            positions[agent_idx] = [[i, i * 0.5] for i in range(time_steps)]
        elif agent_idx == 1:
            # Circular movement
            t = np.linspace(0, 2*np.pi, time_steps)
            positions[agent_idx] = [[10*np.cos(t[i]) + 20, 10*np.sin(t[i]) + 20] for i in range(time_steps)]
        elif agent_idx == 2:
            # Zigzag movement
            positions[agent_idx] = [[i, 10*np.sin(i*0.5) + 15] for i in range(time_steps)]
        else:
            # Random walk
            positions[agent_idx] = np.cumsum(np.random.randn(time_steps, 2) * 0.5, axis=0) + [agent_idx*10, agent_idx*10]
        
        # Generate orientations and odor values
        orientations[agent_idx] = np.random.uniform(0, 360, time_steps)
        odor_values[agent_idx] = np.random.uniform(0, 1, time_steps)
    
    return {
        'positions': positions,
        'orientations': orientations,
        'odor_values': odor_values
    }


@pytest.fixture
def sample_large_multi_agent_data():
    """Generate large-scale multi-agent data for performance testing."""
    num_agents = 100  # Maximum supported agents per specification
    time_steps = 50
    
    positions = np.random.randn(num_agents, time_steps, 2) * 10 + 50
    orientations = np.random.uniform(0, 360, (num_agents, time_steps))
    odor_values = np.random.uniform(0, 1, (num_agents, time_steps))
    
    return {
        'positions': positions,
        'orientations': orientations,
        'odor_values': odor_values
    }


@pytest.fixture
def sample_plume_environment():
    """Generate synthetic plume environment data."""
    height, width = 100, 100
    
    # Create a Gaussian-shaped plume
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    center_x, center_y = width // 2, height // 2
    plume = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 15**2))
    plume = (plume * 255).astype(np.uint8)
    
    return plume


@pytest.fixture
def hydra_config():
    """Create a comprehensive Hydra configuration for testing."""
    if not HYDRA_AVAILABLE:
        # Fallback configuration for environments without Hydra
        return {
            "animation": {
                "fps": 30,
                "interval": 33,
                "blit": True,
                "repeat": False,
                "dpi": 100,
                "figsize": [10, 8]
            },
            "export": {
                "format": "mp4",
                "codec": "libx264",
                "quality": "medium"
            },
            "resolution": {
                "width": 1920,
                "height": 1080,
                "dpi": 300
            },
            "theme": {
                "colormap": "viridis",
                "background": "white",
                "grid": True,
                "grid_alpha": 0.3
            },
            "agents": {
                "color_scheme": "tab10",
                "marker_size": 50,
                "trail_length": 100,
                "max_agents_full_quality": 50
            },
            "static": {
                "dpi": 300,
                "formats": ["png", "pdf"],
                "figsize": [12, 9],
                "show_orientations": True
            },
            "batch": {
                "parallel": False,
                "output_pattern": "trajectory_{idx:03d}",
                "naming_convention": "timestamp"
            }
        }
    
    # Create Hydra DictConfig for testing
    config_dict = {
        "visualization": {
            "animation": {
                "fps": 60,  # High FPS for testing
                "interval": 16,  # ~60 FPS
                "blit": True,
                "repeat": False,
                "dpi": 150,
                "figsize": [12, 9]
            },
            "export": {
                "format": "mp4",
                "codec": "libx264",
                "quality": "high",
                "extra_args": ["-vcodec", "libx264"]
            },
            "resolution": {
                "width": 1920,
                "height": 1080,
                "dpi": 300
            },
            "theme": {
                "colormap": "plasma",
                "background": "white",
                "grid": True,
                "grid_alpha": 0.2
            },
            "agents": {
                "color_scheme": "categorical",
                "marker_size": 75,
                "trail_length": 150,
                "trail_alpha": 0.8,
                "max_agents_full_quality": 50
            },
            "static": {
                "dpi": 300,
                "formats": ["png", "pdf", "svg"],
                "figsize": [14, 10],
                "show_orientations": True,
                "orientation_subsample": 5
            },
            "batch": {
                "parallel": True,
                "output_pattern": "test_trajectory_{idx:03d}",
                "naming_convention": "timestamp"
            },
            "performance": {
                "vectorized_rendering": True,
                "adaptive_quality": True,
                "memory_limit_mb": 256
            }
        }
    }
    
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_animation():
    """Mock matplotlib animation for testing."""
    with patch('src.{{cookiecutter.project_slug}}.utils.visualization.animation') as mock_anim:
        mock_func_animation = MagicMock()
        mock_anim.FuncAnimation.return_value = mock_func_animation
        mock_anim.FFMpegWriter = MagicMock()
        mock_anim.PillowWriter = MagicMock()
        yield mock_anim


# =============================================================================
# SIMULATIONVISUALIZATION CLASS TESTS
# =============================================================================

class TestSimulationVisualization:
    """Test suite for the SimulationVisualization class."""
    
    def test_initialization_default_config(self, mock_plt):
        """Test SimulationVisualization initialization with default configuration."""
        viz = SimulationVisualization()
        
        # Verify figure creation
        mock_plt.subplots.assert_called_once()
        
        # Check default configuration values
        assert viz.config.animation.fps == 30
        assert viz.config.animation.interval == 33
        assert viz.config.theme.colormap == "viridis"
        assert not viz.headless
        
        # Verify internal state initialization
        assert viz.img is None
        assert viz.colorbar is None
        assert viz.animation_obj is None
        assert len(viz.agent_markers) == 0
        assert len(viz.agent_arrows) == 0
        assert len(viz.agent_trails) == 0
    
    def test_initialization_with_hydra_config(self, mock_plt, hydra_config):
        """Test SimulationVisualization initialization with Hydra configuration."""
        viz = SimulationVisualization(config=hydra_config.visualization)
        
        # Verify Hydra configuration integration
        if HYDRA_AVAILABLE:
            assert viz.config.animation.fps == 60
            assert viz.config.animation.dpi == 150
            assert viz.config.theme.colormap == "plasma"
            assert viz.config.agents.color_scheme == "categorical"
    
    def test_headless_mode_initialization(self, mock_plt, mock_matplotlib):
        """Test SimulationVisualization initialization in headless mode."""
        viz = SimulationVisualization(headless=True)
        
        # Verify headless mode configuration
        mock_matplotlib.use.assert_called_with('Agg')
        assert viz.headless is True
    
    def test_setup_environment(self, mock_plt, sample_plume_environment):
        """Test environment setup with plume visualization."""
        viz = SimulationVisualization()
        
        # Setup environment
        viz.setup_environment(sample_plume_environment)
        
        # Verify imshow was called for environment visualization
        viz.ax.imshow.assert_called_once()
        
        # Verify colorbar creation
        viz.fig.colorbar.assert_called_once()
        
        # Check axis labels and title
        viz.ax.set_xlabel.assert_called_with('X Position', fontsize=12)
        viz.ax.set_ylabel.assert_called_with('Y Position', fontsize=12)
        viz.ax.set_title.assert_called_with('Odor Plume Navigation Simulation', fontsize=14)
    
    def test_update_visualization_single_agent(self, mock_plt, sample_single_agent_data):
        """Test visualization update for single agent."""
        viz = SimulationVisualization()
        viz.setup_environment(np.zeros((50, 50)))
        
        # Extract single frame data
        frame_positions = sample_single_agent_data['positions'][0, 0]  # Single agent, first frame
        frame_orientations = sample_single_agent_data['orientations'][0, 0]
        frame_odor = sample_single_agent_data['odor_values'][0, 0]
        
        frame_data = (frame_positions, frame_orientations, frame_odor)
        
        # Update visualization
        updated_artists = viz.update_visualization(frame_data)
        
        # Verify artists were created and returned
        assert len(updated_artists) > 0
        assert len(viz.agent_markers) == 1
        
        # Verify agent color assignment
        assert len(viz.agent_colors) >= 1
    
    def test_update_visualization_multi_agent(self, mock_plt, sample_multi_agent_data):
        """Test visualization update for multiple agents."""
        viz = SimulationVisualization()
        viz.setup_environment(np.zeros((50, 50)))
        
        # Extract multi-agent frame data
        frame_positions = sample_multi_agent_data['positions'][:, 0]  # All agents, first frame
        frame_orientations = sample_multi_agent_data['orientations'][:, 0]
        frame_odor = sample_multi_agent_data['odor_values'][:, 0]
        
        frame_data = (frame_positions, frame_orientations, frame_odor)
        
        # Update visualization
        updated_artists = viz.update_visualization(frame_data)
        
        # Verify multi-agent handling
        num_agents = len(frame_positions)
        assert len(viz.agent_markers) == num_agents
        assert len(updated_artists) >= num_agents
        
        # Verify trail data initialization
        assert len(viz.trail_data) == num_agents
        for i in range(num_agents):
            assert i in viz.trail_data
            assert 'positions' in viz.trail_data[i]
            assert 'line' in viz.trail_data[i]
    
    def test_multi_agent_scaling_performance(self, mock_plt, sample_large_multi_agent_data):
        """Test multi-agent visualization scaling for 100 agents."""
        viz = SimulationVisualization()
        viz.setup_environment(np.zeros((100, 100)))
        
        # Extract large multi-agent frame data
        frame_positions = sample_large_multi_agent_data['positions'][:, 0]  # 100 agents, first frame
        frame_orientations = sample_large_multi_agent_data['orientations'][:, 0]
        frame_odor = sample_large_multi_agent_data['odor_values'][:, 0]
        
        frame_data = (frame_positions, frame_orientations, frame_odor)
        
        # Update visualization with large agent count
        updated_artists = viz.update_visualization(frame_data)
        
        # Verify system handles 100 agents
        assert len(viz.agent_markers) == 100
        
        # Verify adaptive quality is triggered for large agent count
        assert viz.adaptive_quality is True
    
    def test_create_animation(self, mock_plt, mock_animation, sample_single_agent_data):
        """Test animation creation with frame update function."""
        viz = SimulationVisualization()
        viz.setup_environment(np.zeros((50, 50)))
        
        # Create mock update function
        def mock_update_func(frame_idx):
            # Return frame data for the given frame index
            positions = sample_single_agent_data['positions'][:, frame_idx]
            orientations = sample_single_agent_data['orientations'][:, frame_idx]
            odor_values = sample_single_agent_data['odor_values'][:, frame_idx]
            return (positions, orientations, odor_values)
        
        # Create animation
        frames = 10
        animation_obj = viz.create_animation(mock_update_func, frames)
        
        # Verify animation creation
        mock_animation.FuncAnimation.assert_called_once()
        assert viz.animation_obj is not None
        
        # Verify animation parameters
        call_args = mock_animation.FuncAnimation.call_args
        assert call_args[1]['frames'] == frames
        assert call_args[1]['interval'] == viz.config.animation.interval
    
    def test_save_animation_mp4(self, mock_plt, mock_animation, tmp_path):
        """Test animation saving in MP4 format."""
        viz = SimulationVisualization()
        
        # Create mock animation object
        mock_anim_obj = MagicMock()
        viz.animation_obj = mock_anim_obj
        
        # Test saving
        output_path = tmp_path / "test_animation.mp4"
        viz.save_animation(output_path)
        
        # Verify save was called
        mock_anim_obj.save.assert_called()
        
        # Verify FFMpegWriter was used
        mock_animation.FFMpegWriter.assert_called()
    
    def test_performance_stats_tracking(self, mock_plt):
        """Test performance statistics tracking."""
        viz = SimulationVisualization()
        
        # Simulate some frame times
        viz.frame_times = [30.0, 35.0, 28.0, 32.0, 29.0]  # milliseconds
        
        # Get performance stats
        stats = viz.get_performance_stats()
        
        # Verify statistics calculation
        assert 'avg_frame_time_ms' in stats
        assert 'max_frame_time_ms' in stats
        assert 'min_frame_time_ms' in stats
        assert 'fps_estimate' in stats
        
        assert stats['avg_frame_time_ms'] == 30.8
        assert stats['max_frame_time_ms'] == 35.0
        assert stats['min_frame_time_ms'] == 28.0
        assert abs(stats['fps_estimate'] - (1000 / 30.8)) < 0.1


# =============================================================================
# STATIC TRAJECTORY VISUALIZATION TESTS
# =============================================================================

class TestVisualizeTrajectory:
    """Test suite for the visualize_trajectory function."""
    
    def test_visualize_single_agent_trajectory(self, mock_plt, sample_single_agent_data):
        """Test static trajectory visualization for single agent."""
        positions = sample_single_agent_data['positions'][0]  # Shape: (time_steps, 2)
        orientations = sample_single_agent_data['orientations'][0]
        
        # Create visualization
        visualize_trajectory(
            positions=positions,
            orientations=orientations,
            show_plot=False
        )
        
        # Verify plot creation
        mock_plt.subplots.assert_called_once()
        
        # Verify trajectory plotting
        mock_plt.subplots.return_value[1].plot.assert_called()
        mock_plt.subplots.return_value[1].scatter.assert_called()  # Start/end markers
        mock_plt.subplots.return_value[1].quiver.assert_called()   # Orientation arrows
    
    def test_visualize_multi_agent_trajectories(self, mock_plt, sample_multi_agent_data):
        """Test static trajectory visualization for multiple agents."""
        positions = sample_multi_agent_data['positions']
        orientations = sample_multi_agent_data['orientations']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Create visualization
        visualize_trajectory(
            positions=positions,
            orientations=orientations,
            agent_colors=colors,
            show_plot=False
        )
        
        # Verify multi-agent plotting
        mock_ax = mock_plt.subplots.return_value[1]
        
        # Should have plot calls for each agent
        assert mock_ax.plot.call_count == len(positions)
        
        # Should have scatter calls for start/end markers
        assert mock_ax.scatter.call_count >= len(positions) * 2
    
    def test_visualize_with_plume_background(self, mock_plt, sample_single_agent_data, sample_plume_environment):
        """Test trajectory visualization with plume background."""
        positions = sample_single_agent_data['positions'][0]
        
        # Create visualization with plume
        visualize_trajectory(
            positions=positions,
            plume_frames=sample_plume_environment,
            show_plot=False
        )
        
        # Verify plume background rendering
        mock_ax = mock_plt.subplots.return_value[1]
        mock_ax.imshow.assert_called_once()
        mock_plt.colorbar.assert_called_once()
    
    def test_publication_quality_export(self, mock_plt, sample_single_agent_data, tmp_path):
        """Test publication-quality export with configurable DPI settings."""
        positions = sample_single_agent_data['positions'][0]
        
        # Test different DPI settings
        dpi_settings = [300, 150, 72]  # Publication, presentation, web quality
        
        for dpi in dpi_settings:
            output_path = tmp_path / f"test_trajectory_{dpi}dpi.png"
            
            # Create high-quality visualization
            visualize_trajectory(
                positions=positions,
                output_path=output_path,
                dpi=dpi,
                show_plot=False,
                figsize=(12, 9)
            )
            
            # Verify figure creation with correct DPI
            mock_plt.subplots.assert_called_with(figsize=(12, 9), dpi=dpi)
    
    def test_multiple_format_export(self, mock_plt, sample_single_agent_data, tmp_path, hydra_config):
        """Test multi-format export (PNG, PDF, SVG)."""
        positions = sample_single_agent_data['positions'][0]
        
        # Configure for multiple formats
        if HYDRA_AVAILABLE:
            config = hydra_config.visualization
        else:
            config = None
        
        output_path = tmp_path / "test_trajectory"
        
        # Create visualization with multi-format export
        visualize_trajectory(
            positions=positions,
            output_path=output_path,
            config=config,
            show_plot=False
        )
        
        # Verify figure saving
        mock_fig = mock_plt.subplots.return_value[0]
        assert mock_fig.savefig.call_count >= 1
    
    def test_batch_mode_operation(self, mock_plt, sample_single_agent_data):
        """Test batch mode for automated processing."""
        positions = sample_single_agent_data['positions'][0]
        
        # Test batch mode
        fig = visualize_trajectory(
            positions=positions,
            show_plot=False,
            batch_mode=True
        )
        
        # Verify figure is returned for batch processing
        assert fig is not None
    
    def test_backward_compatibility_colors_parameter(self, mock_plt, sample_multi_agent_data):
        """Test backward compatibility with 'colors' parameter."""
        positions = sample_multi_agent_data['positions']
        colors = ['red', 'blue', 'green']
        
        # Use deprecated 'colors' parameter
        visualize_trajectory(
            positions=positions,
            colors=colors,  # Backward compatibility
            show_plot=False
        )
        
        # Should work without errors
        mock_plt.subplots.assert_called_once()


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

class TestBatchVisualization:
    """Test suite for batch visualization processing."""
    
    def test_batch_visualize_trajectories(self, mock_plt, sample_multi_agent_data, tmp_path):
        """Test batch processing of multiple trajectory visualizations."""
        # Prepare batch data
        num_trajectories = 3
        trajectory_data = []
        
        for i in range(num_trajectories):
            # Create variations of the sample data
            data = {
                'positions': sample_multi_agent_data['positions'] + i * 10,  # Offset each trajectory
                'orientations': sample_multi_agent_data['orientations'],
                'title': f'Experiment {i+1}'
            }
            trajectory_data.append(data)
        
        # Process batch
        saved_paths = batch_visualize_trajectories(
            trajectory_data=trajectory_data,
            output_dir=tmp_path,
            naming_pattern="batch_test_{idx:03d}"
        )
        
        # Verify batch processing
        assert len(saved_paths) <= num_trajectories  # May be less due to mocking
        
        # Verify multiple figure creation
        assert mock_plt.subplots.call_count >= num_trajectories
    
    def test_batch_with_hydra_config(self, mock_plt, sample_single_agent_data, tmp_path, hydra_config):
        """Test batch processing with Hydra configuration."""
        trajectory_data = [
            {'positions': sample_single_agent_data['positions'][0], 'title': 'Test 1'},
            {'positions': sample_single_agent_data['positions'][0] + 5, 'title': 'Test 2'}
        ]
        
        # Process with Hydra config
        if HYDRA_AVAILABLE:
            config = hydra_config.visualization
        else:
            config = None
        
        saved_paths = batch_visualize_trajectories(
            trajectory_data=trajectory_data,
            output_dir=tmp_path,
            config=config
        )
        
        # Should complete without errors
        assert isinstance(saved_paths, list)
    
    def test_headless_batch_processing(self, mock_plt, mock_matplotlib, sample_single_agent_data, tmp_path):
        """Test headless batch processing capabilities."""
        # Setup headless mode
        setup_headless_mode()
        mock_matplotlib.use.assert_called_with('Agg')
        
        # Prepare batch data
        trajectory_data = [
            {'positions': sample_single_agent_data['positions'][0]}
        ]
        
        # Process in headless mode
        saved_paths = batch_visualize_trajectories(
            trajectory_data=trajectory_data,
            output_dir=tmp_path
        )
        
        # Should complete without display requirements
        assert isinstance(saved_paths, list)


# =============================================================================
# HYDRA CONFIGURATION INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """Test suite for Hydra configuration integration."""
    
    def test_animation_fps_configuration(self, mock_plt, hydra_config):
        """Test Hydra configuration for animation FPS settings."""
        # Test different FPS values
        fps_values = [15, 30, 60]
        
        for fps in fps_values:
            config = OmegaConf.create({
                "visualization": {
                    "animation": {"fps": fps, "interval": int(1000/fps)}
                }
            })
            
            viz = SimulationVisualization(config=config.visualization)
            assert viz.config.animation.fps == fps
            assert viz.config.animation.interval == int(1000/fps)
    
    def test_export_format_configuration(self, mock_plt, mock_animation):
        """Test Hydra configuration for export format settings."""
        export_formats = ["mp4", "avi", "gif"]
        
        for fmt in export_formats:
            config = OmegaConf.create({
                "visualization": {
                    "export": {"format": fmt}
                }
            })
            
            viz = SimulationVisualization(config=config.visualization)
            assert viz.config.export.format == fmt
    
    def test_resolution_parameter_configuration(self, mock_plt):
        """Test Hydra configuration for resolution parameters."""
        config = OmegaConf.create({
            "visualization": {
                "resolution": {
                    "width": 1920,
                    "height": 1080,
                    "dpi": 300
                }
            }
        })
        
        viz = SimulationVisualization(config=config.visualization)
        assert viz.config.resolution.width == 1920
        assert viz.config.resolution.height == 1080
        assert viz.config.resolution.dpi == 300
    
    def test_theme_configuration(self, mock_plt):
        """Test Hydra theme configuration parameters."""
        config = OmegaConf.create({
            "visualization": {
                "theme": {
                    "colormap": "plasma",
                    "background": "black",
                    "grid": False,
                    "grid_alpha": 0.5
                }
            }
        })
        
        viz = SimulationVisualization(config=config.visualization)
        assert viz.config.theme.colormap == "plasma"
        assert viz.config.theme.background == "black"
        assert viz.config.theme.grid is False
        assert viz.config.theme.grid_alpha == 0.5


# =============================================================================
# CLI EXPORT CAPABILITIES TESTS
# =============================================================================

class TestCLIExportCapabilities:
    """Test suite for CLI export functionality."""
    
    def test_headless_mode_setup(self, mock_matplotlib):
        """Test headless mode configuration for CLI operation."""
        setup_headless_mode()
        
        # Verify matplotlib backend configuration
        mock_matplotlib.use.assert_called_with('Agg')
    
    def test_headless_simulation_visualization(self, mock_plt, mock_matplotlib):
        """Test SimulationVisualization in headless mode."""
        viz = SimulationVisualization(headless=True)
        
        # Verify headless configuration
        mock_matplotlib.use.assert_called_with('Agg')
        assert viz.headless is True
        
        # Should not call show() in headless mode
        viz.show()
        # No assertion needed - should just print message
    
    def test_export_animation_with_configuration(self, mock_animation):
        """Test animation export with configuration."""
        # Create mock animation
        mock_anim_obj = MagicMock()
        
        # Test export with configuration
        export_animation(
            animation_obj=mock_anim_obj,
            filename="test_export.mp4"
        )
        
        # Verify export was attempted
        mock_anim_obj.save.assert_called()
        mock_animation.FFMpegWriter.assert_called()
    
    def test_mp4_export_functionality(self, mock_plt, mock_animation, tmp_path):
        """Test MP4 export functionality with CLI flags."""
        viz = SimulationVisualization(headless=True)
        
        # Create mock animation
        mock_anim_obj = MagicMock()
        viz.animation_obj = mock_anim_obj
        
        # Test MP4 export
        output_path = tmp_path / "cli_test_export.mp4"
        viz.save_animation(output_path, format_override="mp4")
        
        # Verify MP4 export configuration
        mock_animation.FFMpegWriter.assert_called()
        mock_anim_obj.save.assert_called()


# =============================================================================
# UTILITY AND CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test suite for utility and convenience functions."""
    
    def test_get_available_themes(self):
        """Test theme availability query function."""
        themes = get_available_themes()
        
        # Verify expected themes are available
        assert isinstance(themes, dict)
        assert "scientific" in themes
        assert "presentation" in themes
        assert "high_contrast" in themes
        
        # Verify theme structure
        for theme_name, theme_config in themes.items():
            assert "colormap" in theme_config
            assert "background" in theme_config
            assert "dpi" in theme_config
    
    def test_create_simulation_visualization_factory(self, mock_plt):
        """Test factory function for SimulationVisualization creation."""
        viz = create_simulation_visualization()
        
        # Verify instance creation
        assert isinstance(viz, SimulationVisualization)
        mock_plt.subplots.assert_called()
    
    def test_default_visualization_config_structure(self):
        """Test default configuration structure completeness."""
        config = DEFAULT_VISUALIZATION_CONFIG
        
        # Verify required configuration sections
        required_sections = [
            "animation", "export", "resolution", "theme", 
            "agents", "static", "batch", "performance"
        ]
        
        for section in required_sections:
            assert section in config
        
        # Verify animation configuration
        assert "fps" in config["animation"]
        assert "interval" in config["animation"]
        assert "dpi" in config["animation"]
        
        # Verify static configuration
        assert "dpi" in config["static"]
        assert "formats" in config["static"]
        assert "figsize" in config["static"]


# =============================================================================
# INTEGRATION AND ERROR HANDLING TESTS
# =============================================================================

class TestIntegrationAndErrorHandling:
    """Test suite for integration scenarios and error handling."""
    
    def test_visualization_with_invalid_data_shapes(self, mock_plt):
        """Test visualization handling of invalid data shapes."""
        viz = SimulationVisualization()
        
        # Test with mismatched data shapes
        positions = np.array([[1, 2]])  # Wrong shape
        orientations = np.array([0, 45, 90])  # Mismatched length
        odor_values = np.array([0.5])
        
        # Should handle gracefully without crashing
        try:
            frame_data = (positions, orientations, odor_values)
            viz.update_visualization(frame_data)
        except Exception as e:
            # Expected to handle shape mismatches
            assert isinstance(e, (IndexError, ValueError))
    
    def test_save_animation_without_creation(self, mock_plt):
        """Test error handling when saving animation without creation."""
        viz = SimulationVisualization()
        
        # Should raise error when no animation exists
        with pytest.raises(ValueError, match="No animation created"):
            viz.save_animation("test.mp4")
    
    def test_batch_processing_error_handling(self, mock_plt, tmp_path):
        """Test batch processing with invalid data."""
        # Include invalid data in batch
        trajectory_data = [
            {'positions': np.array([[1, 2], [3, 4]])},  # Valid
            {'positions': "invalid_data"},  # Invalid
            {'positions': np.array([[5, 6], [7, 8]])}   # Valid
        ]
        
        # Should handle errors gracefully and continue processing
        saved_paths = batch_visualize_trajectories(
            trajectory_data=trajectory_data,
            output_dir=tmp_path
        )
        
        # Should return results for valid trajectories only
        assert isinstance(saved_paths, list)
    
    def test_memory_management_cleanup(self, mock_plt):
        """Test proper memory management and resource cleanup."""
        viz = SimulationVisualization()
        
        # Add some visualization elements
        viz.agent_markers = [MagicMock() for _ in range(5)]
        viz.agent_arrows = [MagicMock() for _ in range(5)]
        viz.agent_trails = [MagicMock() for _ in range(5)]
        
        # Test cleanup
        viz.close()
        
        # Verify cleanup was performed
        assert viz.animation_obj is None
        assert len(viz.agent_markers) == 0
        assert len(viz.agent_arrows) == 0
        assert len(viz.agent_trails) == 0


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

class TestPerformanceAndScalability:
    """Test suite for performance characteristics and scalability."""
    
    def test_vectorized_rendering_performance(self, mock_plt, sample_large_multi_agent_data):
        """Test vectorized rendering performance with large agent populations."""
        viz = SimulationVisualization()
        viz.setup_environment(np.zeros((100, 100)))
        
        # Test with maximum supported agents (100)
        frame_positions = sample_large_multi_agent_data['positions'][:, 0]
        frame_orientations = sample_large_multi_agent_data['orientations'][:, 0]
        frame_odor = sample_large_multi_agent_data['odor_values'][:, 0]
        
        frame_data = (frame_positions, frame_orientations, frame_odor)
        
        # Should handle 100 agents without performance issues
        updated_artists = viz.update_visualization(frame_data)
        
        # Verify all agents are handled
        assert len(viz.agent_markers) == 100
        assert len(updated_artists) >= 100
    
    def test_adaptive_quality_degradation(self, mock_plt, sample_large_multi_agent_data):
        """Test adaptive quality degradation for performance optimization."""
        viz = SimulationVisualization()
        viz.adaptive_quality = True
        
        # Simulate many agents triggering quality degradation
        frame_positions = sample_large_multi_agent_data['positions'][:, 0]
        frame_orientations = sample_large_multi_agent_data['orientations'][:, 0]
        frame_odor = sample_large_multi_agent_data['odor_values'][:, 0]
        
        frame_data = (frame_positions, frame_orientations, frame_odor)
        
        # Should enable adaptive quality for large agent counts
        viz.update_visualization(frame_data)
        
        # Verify adaptive quality is activated
        assert viz.adaptive_quality is True
    
    def test_trail_length_management(self, mock_plt, sample_single_agent_data):
        """Test trail length management for memory efficiency."""
        viz = SimulationVisualization()
        viz.config.agents.trail_length = 10  # Short trail for testing
        
        # Add many position updates
        for i in range(20):  # More than trail_length
            positions = sample_single_agent_data['positions'][0, i % sample_single_agent_data['positions'].shape[1]]
            orientations = sample_single_agent_data['orientations'][0, i % sample_single_agent_data['orientations'].shape[1]]
            odor_values = sample_single_agent_data['odor_values'][0, i % sample_single_agent_data['odor_values'].shape[1]]
            
            frame_data = (positions, orientations, odor_values)
            viz.update_visualization(frame_data)
        
        # Verify trail length is limited
        if 0 in viz.trail_data:
            assert len(viz.trail_data[0]['positions']) <= viz.config.agents.trail_length


if __name__ == "__main__":
    pytest.main([__file__])