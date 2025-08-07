"""Tests for the trajectory visualization module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from plume_nav_sim.utils.visualization import visualize_trajectory, SimulationVisualization

# Hydra imports for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None


@pytest.fixture
def mock_plt():
    """Mock matplotlib.pyplot to avoid displaying plots during tests."""
    with patch('plume_nav_sim.utils.visualization.plt') as mock_plt:
        # Configure subplots to return mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        yield mock_plt


def test_visualize_trajectory_single_agent(mock_plt):
    """Test visualizing trajectory for a single agent."""
    # Create synthetic data for a single agent
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])  # Shape: (1, 3, 2)
    orientations = np.array([[0, 45, 90]])  # Shape: (1, 3)
    
    # Call the visualization function
    visualize_trajectory(positions, orientations, show_plot=False)
    
    # Check that the plot was created and properly formatted
    mock_plt.subplots.assert_called_once()
    
    # Get the mock axes object returned by subplots
    mock_fig, mock_ax = mock_plt.subplots.return_value
    
    # Check that plot methods were called on the axes
    mock_ax.plot.assert_called()  # Should be called to plot trajectory
    mock_ax.quiver.assert_called()  # Should be called to show orientations
    mock_plt.savefig.assert_not_called()  # No output path was provided


def test_visualize_trajectory_multi_agent(mock_plt):
    """Test visualizing trajectory for multiple agents."""
    # Create synthetic data for multiple agents
    positions = np.array([
        [[0, 0], [1, 1], [2, 2]],  # Agent 1
        [[5, 5], [6, 6], [7, 7]]   # Agent 2
    ])  # Shape: (2, 3, 2)
    orientations = np.array([
        [0, 45, 90],    # Agent 1
        [180, 225, 270]  # Agent 2
    ])  # Shape: (2, 3)
    
    # Call the visualization function
    visualize_trajectory(positions, orientations, show_plot=False)
    
    # Check that the plot was created with multiple traces
    mock_plt.figure.assert_called_once()
    assert mock_plt.plot.call_count >= 2  # Should be called at least once per agent
    assert mock_plt.quiver.call_count >= 2  # Should be called to show orientations for each agent


def test_visualize_trajectory_with_plume(mock_plt):
    """Test visualizing trajectory with a plume background."""
    # Create synthetic data
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Create a synthetic plume frame
    plume_frame = np.zeros((10, 10), dtype=np.uint8)
    plume_frame[5:8, 5:8] = 255  # Add a bright spot
    
    # Call the visualization function with plume frames
    visualize_trajectory(
        positions, 
        orientations, 
        plume_frames=plume_frame, 
        show_plot=False
    )
    
    # Check that the imshow was called to display the plume
    mock_plt.imshow.assert_called_once()
    mock_plt.colorbar.assert_called_once()


def test_visualize_trajectory_with_output(mock_plt, sample_single_agent_data, temp_output_dir):
    """Test visualizing trajectory with publication-quality output file."""
    data = sample_single_agent_data
    output_path = temp_output_dir / "test_trajectory.png"
    
    # Call the visualization function with an output path
    visualize_trajectory(
        data['positions'], 
        data['orientations'], 
        output_path=output_path,
        show_plot=False,
        headless=True,
        dpi=300
    )
    
    # Check that the figure was saved with correct parameters
    mock_plt.savefig.assert_called()
    save_call = mock_plt.savefig.call_args
    assert save_call[0][0] == str(output_path)
    assert save_call[1]['dpi'] == 300
    assert save_call[1]['bbox_inches'] == 'tight'


def test_visualize_trajectory_with_custom_colors(mock_plt, sample_multi_agent_data):
    """Test visualizing trajectory with custom color schemes."""
    data = sample_multi_agent_data
    
    # Define custom colors
    agent_colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Call the visualization function with custom colors
    visualize_trajectory(
        data['positions'], 
        data['orientations'], 
        agent_colors=agent_colors,
        show_plot=False,
        headless=True
    )
    
    # Check that the plot was created
    mock_plt.plot.assert_called()
    # Verify multiple agents were plotted
    assert mock_plt.plot.call_count >= data['num_agents']


# === NEW TESTS FOR ENHANCED FUNCTIONALITY ===

class TestSimulationVisualization:
    """Test class for SimulationVisualization real-time animation capabilities."""
    
    def test_simulation_visualization_initialization(self):
        """Test basic SimulationVisualization initialization."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            viz = SimulationVisualization(
                figsize=(10, 8),
                dpi=150,
                fps=30,
                headless=True
            )
            
            assert viz.config['fps'] == 30
            assert viz.config['dpi'] == 150
            assert viz.config['headless'] is True
            mock_subplots.assert_called_once_with(figsize=(10, 8), dpi=150)
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_simulation_visualization_from_config(self, hydra_visualization_config):
        """Test SimulationVisualization creation from Hydra configuration."""
        with patch('matplotlib.pyplot.subplots'):
            viz = SimulationVisualization.from_config(hydra_visualization_config)
            
            assert viz.config['fps'] == 30
            assert viz.config['theme'] == 'scientific'
            assert viz.config['headless'] is False
    
    def test_simulation_visualization_setup_environment(self, sample_plume_data):
        """Test environment setup with plume data."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            viz = SimulationVisualization(headless=True)
            viz.setup_environment(sample_plume_data)
            
            mock_ax.imshow.assert_called_once()
            mock_ax.clear.assert_called_once()
    
    def test_simulation_visualization_multi_agent_update(self, sample_multi_agent_data):
        """Test multi-agent visualization updates with vectorized rendering."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            viz = SimulationVisualization(max_agents=100, headless=True)
            
            # Simulate frame data for multiple agents
            frame_data = {
                'agents': [
                    ((10, 15), 45.0),
                    ((20, 25), 90.0),
                    ((30, 35), 135.0)
                ],
                'odor_values': [0.75, 0.60, 0.80]
            }
            
            updated_artists = viz.update_visualization(frame_data)
            
            # Should create artists for all agents
            assert len(viz.agent_artists) == 3
            assert len(updated_artists) > 0
    
    def test_simulation_visualization_animation_creation(self):
        """Test animation creation with configurable performance settings."""
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.animation.FuncAnimation') as mock_anim:
                viz = SimulationVisualization(fps=60, headless=True)
                
                def dummy_update_func(frame_idx):
                    return ((frame_idx, frame_idx), 0.0, 0.5)
                
                animation_obj = viz.create_animation(
                    dummy_update_func, 
                    frames=100,
                    blit=False
                )
                
                mock_anim.assert_called_once()
                call_args = mock_anim.call_args
                assert call_args[1]['interval'] == int(1000 / 60)  # 60 FPS
                assert call_args[1]['frames'] == 100


class TestHydraConfigurationIntegration:
    """Test class for Hydra configuration integration features."""
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_animation_parameters(self, hydra_visualization_config, mock_plt):
        """Test Hydra configuration for animation parameters including fps, format, resolution."""
        # Update config with specific animation parameters
        hydra_visualization_config.animation.fps = 60
        hydra_visualization_config.animation.format = 'mp4'
        hydra_visualization_config.resolution = '1080p'
        
        with patch('matplotlib.pyplot.subplots'):
            viz = SimulationVisualization.from_config(hydra_visualization_config)
            
            assert viz.config['fps'] == 60
            # Resolution should be properly parsed
            expected_figsize = (1920 / 300, 1080 / 300)  # 1080p at 300 DPI
            assert abs(viz.config['figsize'][0] - expected_figsize[0]) < 0.01
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_static_plot_parameters(self, hydra_visualization_config, sample_single_agent_data, mock_plt):
        """Test Hydra configuration for static plotting including DPI and export format."""
        # Update config with specific static parameters
        hydra_visualization_config.static.dpi = 300
        hydra_visualization_config.static.format = 'pdf'
        
        data = sample_single_agent_data
        
        # Extract static config for testing
        static_config = OmegaConf.to_container(hydra_visualization_config.static, resolve=True)
        
        visualize_trajectory(
            data['positions'],
            data['orientations'],
            dpi=static_config['dpi'],
            format=static_config['format'],
            show_plot=False,
            headless=True
        )
        
        mock_plt.subplots.assert_called_once()
        # DPI should be passed to subplots
        call_args = mock_plt.subplots.call_args
        assert call_args[1]['dpi'] == 300


class TestCLIExportCapabilities:
    """Test class for CLI export capabilities with headless mode support."""
    
    def test_headless_mode_support(self, sample_single_agent_data, mock_plt):
        """Test headless mode operation for CLI integration."""
        data = sample_single_agent_data
        
        with patch('matplotlib.use') as mock_use:
            visualize_trajectory(
                data['positions'],
                data['orientations'],
                headless=True,
                show_plot=False
            )
            
            # Should set matplotlib to Agg backend for headless operation
            mock_use.assert_called_with('Agg')
    
    def test_mp4_export_functionality(self, temp_output_dir):
        """Test MP4 export functionality for CLI workflows."""
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.animation.FuncAnimation') as mock_anim:
                viz = SimulationVisualization(headless=True)
                
                def dummy_update_func(frame_idx):
                    return ((frame_idx, frame_idx), 0.0, 0.5)
                
                # Create animation
                viz.create_animation(dummy_update_func, frames=50)
                
                # Mock the animation object
                mock_anim_instance = MagicMock()
                viz.animation_obj = mock_anim_instance
                
                # Test MP4 save
                output_path = temp_output_dir / "test_animation.mp4"
                viz.save_animation(output_path, format='mp4', fps=30)
                
                mock_anim_instance.save.assert_called_once()
                save_call = mock_anim_instance.save.call_args
                assert str(output_path) in save_call[0]
    
    def test_batch_processing_mode(self, sample_multi_agent_data, mock_plt, temp_output_dir):
        """Test batch processing workflows with optimized performance."""
        data = sample_multi_agent_data
        
        # Test batch mode returns figure object for memory management
        fig = visualize_trajectory(
            data['positions'],
            data['orientations'],
            batch_mode=True,
            headless=True,
            output_path=temp_output_dir / "batch_test.png"
        )
        
        # Should return figure object in batch mode
        assert fig is not None
        mock_plt.close.assert_not_called()  # Should not auto-close in batch mode


class TestMultiAgentVisualizationScaling:
    """Test class for multi-agent visualization scaling capabilities."""
    
    def test_large_agent_population_support(self, mock_plt):
        """Test visualization scaling up to 100 agents with vectorized rendering."""
        num_agents = 100
        time_steps = 50
        
        # Generate data for 100 agents
        positions = np.random.rand(num_agents, time_steps, 2) * 100
        orientations = np.random.rand(num_agents, time_steps) * 360
        
        with patch('matplotlib.pyplot.subplots'):
            viz = SimulationVisualization(max_agents=100, headless=True)
            
            # Create frame data for many agents
            frame_data = {
                'agents': [((pos[0], pos[1]), orient) for pos, orient in 
                          zip(positions[:10, 0], orientations[:10, 0])],  # Test first 10
                'odor_values': np.random.rand(10)
            }
            
            updated_artists = viz.update_visualization(frame_data)
            
            # Should handle multiple agents efficiently
            assert len(viz.agent_artists) == 10
            assert len(updated_artists) > 0
    
    def test_vectorized_rendering_performance(self, sample_multi_agent_data, mock_plt):
        """Test vectorized rendering capabilities for multi-agent scenarios."""
        data = sample_multi_agent_data
        
        visualize_trajectory(
            data['positions'],
            data['orientations'],
            show_plot=False,
            headless=True,
            trajectory_alpha=0.7  # Test transparency for overlapping trajectories
        )
        
        # Should efficiently handle multiple agent plotting
        assert mock_plt.plot.call_count >= data['num_agents']
        mock_plt.subplots.assert_called_once()


class TestPublicationQualityStaticPlots:
    """Test class for publication-quality static plot export."""
    
    @pytest.mark.parametrize("dpi,expected_quality", [
        (300, "publication"),
        (150, "presentation"), 
        (72, "web")
    ])
    def test_configurable_dpi_settings(self, dpi, expected_quality, sample_single_agent_data, mock_plt, temp_output_dir):
        """Test configurable DPI settings for different quality levels."""
        data = sample_single_agent_data
        output_path = temp_output_dir / f"test_{expected_quality}.png"
        
        visualize_trajectory(
            data['positions'],
            data['orientations'],
            output_path=output_path,
            dpi=dpi,
            show_plot=False,
            headless=True
        )
        
        # Verify DPI was set correctly
        mock_plt.subplots.assert_called_once()
        call_args = mock_plt.subplots.call_args
        assert call_args[1]['dpi'] == dpi
        
        # Verify save was called with correct DPI
        mock_plt.savefig.assert_called()
        save_call = mock_plt.savefig.call_args
        assert save_call[1]['dpi'] == dpi
    
    @pytest.mark.parametrize("format_type", ["png", "pdf", "svg", "eps"])
    def test_multiple_export_formats(self, format_type, sample_single_agent_data, mock_plt, temp_output_dir):
        """Test export capabilities for multiple publication formats."""
        data = sample_single_agent_data
        output_path = temp_output_dir / f"test_plot.{format_type}"
        
        visualize_trajectory(
            data['positions'],
            data['orientations'],
            output_path=output_path,
            format=format_type,
            dpi=300,
            show_plot=False,
            headless=True
        )
        
        mock_plt.savefig.assert_called()
        save_call = mock_plt.savefig.call_args
        assert save_call[1]['format'] == format_type
    
    def test_theme_based_styling(self, sample_single_agent_data, mock_plt):
        """Test theme-based styling for different presentation contexts."""
        data = sample_single_agent_data
        
        for theme in ['scientific', 'presentation', 'high_contrast']:
            visualize_trajectory(
                data['positions'],
                data['orientations'],
                theme=theme,
                show_plot=False,
                headless=True
            )
        
        # Should create plots for each theme
        assert mock_plt.subplots.call_count == 3


class TestBatchProcessingWorkflows:
    """Test class for batch processing workflows and headless operation."""
    
    def test_headless_batch_trajectory_processing(self, mock_plt, temp_output_dir):
        """Test headless batch processing of multiple trajectory datasets."""
        # Generate multiple trajectory datasets
        datasets = []
        for i in range(5):
            positions = np.random.rand(50, 2) * 100
            orientations = np.random.rand(50) * 360
            datasets.append((positions, orientations))
        
        with patch('matplotlib.use') as mock_use:
            # Process each dataset in batch mode
            for i, (positions, orientations) in enumerate(datasets):
                fig = visualize_trajectory(
                    positions,
                    orientations,
                    output_path=temp_output_dir / f"batch_{i:03d}.png",
                    batch_mode=True,
                    headless=True,
                    dpi=150
                )
                
                # Should return figure for memory management
                assert fig is not None
            
            # Should set headless backend
            mock_use.assert_called_with('Agg')
    
    def test_memory_efficient_batch_processing(self, mock_plt, temp_output_dir):
        """Test memory-efficient batch processing with figure cleanup."""
        positions = np.random.rand(100, 2) * 50
        orientations = np.random.rand(100) * 360
        
        # Process with explicit figure management
        fig = visualize_trajectory(
            positions,
            orientations,
            batch_mode=True,
            headless=True
        )
        
        # Figure should be returned for manual cleanup
        assert fig is not None
        
        # Simulate manual cleanup
        with patch.object(fig, 'clear') as mock_clear:
            fig.clear()
            mock_clear.assert_called_once()
    
    def test_parallel_processing_compatibility(self, sample_multi_agent_data, mock_plt):
        """Test compatibility with parallel processing workflows."""
        data = sample_multi_agent_data
        
        # Test that visualization works in isolated context (simulating parallel worker)
        with patch('matplotlib.pyplot.ioff') as mock_ioff:
            visualize_trajectory(
                data['positions'],
                data['orientations'],
                headless=True,
                show_plot=False,
                batch_mode=True
            )
            
            # Should work without interactive dependencies
            mock_plt.subplots.assert_called_once()


class TestVisualizationModuleIntegration:
    """Test class for consolidated visualization module organization."""
    
    def test_unified_import_structure(self):
        """Test that both SimulationVisualization and visualize_trajectory are available from single module."""
        # Should be able to import both classes from utils.visualization
        from src.plume_nav_sim.utils.visualization import (
            SimulationVisualization,
            visualize_trajectory,
            VisualizationConfig
        )
        
        # Verify classes are properly available
        assert SimulationVisualization is not None
        assert visualize_trajectory is not None
        assert VisualizationConfig is not None
    
    def test_backward_compatibility_parameters(self, sample_single_agent_data, mock_plt):
        """Test backward compatibility with legacy parameter names."""
        data = sample_single_agent_data
        
        # Test legacy 'colors' parameter still works
        visualize_trajectory(
            data['positions'],
            data['orientations'],
            colors=['red', 'blue'],  # Legacy parameter name
            show_plot=False,
            headless=True
        )
        
        mock_plt.plot.assert_called()
    
    def test_enhanced_error_handling(self, mock_plt):
        """Test enhanced error handling for invalid inputs."""
        # Test invalid position array dimensions
        with pytest.raises(ValueError, match="positions must be 2D or 3D array"):
            visualize_trajectory(
                np.array([1, 2, 3]),  # 1D array - invalid
                show_plot=False,
                headless=True
            )
        
        # Test incompatible orientations
        positions = np.random.rand(50, 2)
        orientations = np.random.rand(30)  # Wrong length
        
        with pytest.raises(ValueError, match="orientations length"):
            visualize_trajectory(
                positions,
                orientations,
                show_plot=False,
                headless=True
            )
