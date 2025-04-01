"""Tests for the integrated simulation system that combines video plume and navigator."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import the simulation module once to avoid reimporting in each test
from odor_plume_nav.simulation import Simulation


class TestSimulation:
    """Tests for the Simulation class that integrates plume and navigator."""
    
    def test_simulation_initialization(self, mock_video_plume, mock_navigator):
        """Test that a Simulation can be initialized with plume and navigator."""
        with patch('odor_plume_nav.simulation.VideoPlume', return_value=mock_video_plume), \
             patch('odor_plume_nav.simulation.SimpleNavigator', return_value=mock_navigator):
            
            # Create a simulation with mocked components
            sim = Simulation(
                video_path="test_video.mp4",
                dt=0.1
            )
            
            # Check that the simulation was initialized with the correct components
            assert sim.plume is not None
            assert sim.navigator is not None
            assert sim.dt == 0.1
            assert sim.time == 0.0
            assert sim.frame_index == 0
            
            # Verify that get_frame was called once during initialization
            mock_video_plume.get_frame.assert_called_once()
    
    def test_simulation_step(self, mock_video_plume, mock_navigator):
        """Test that a simulation step advances the plume and moves the navigator."""
        with patch('odor_plume_nav.simulation.VideoPlume', return_value=mock_video_plume), \
             patch('odor_plume_nav.simulation.SimpleNavigator', return_value=mock_navigator):
            
            # Create a simulation with mocked components
            sim = Simulation(
                video_path="test_video.mp4",
                dt=0.1
            )
            
            # Reset the mock to isolate calls during the step
            mock_video_plume.get_frame.reset_mock()
            mock_navigator.update.reset_mock()
            
            # Perform a simulation step
            sim.step()
            
            # Check that the plume was advanced
            mock_video_plume.get_frame.assert_called_once()
            
            # Check that the navigator was updated
            mock_navigator.update.assert_called_once_with(dt=0.1)
            
            # Check that simulation state was updated
            assert sim.time == 0.1
            assert sim.frame_index == 1
    
    def test_multiple_simulation_steps(self, mock_video_plume, mock_navigator):
        """Test that multiple simulation steps work correctly."""
        with patch("odor_plume_nav.simulation.VideoPlume", return_value=mock_video_plume):
            with patch("odor_plume_nav.simulation.SimpleNavigator", return_value=mock_navigator):
                # Create a simulation with the mocked objects
                sim = Simulation(
                    video_path="test_video.mp4",
                    dt=0.1
                )
                
                # Reset the mock to isolate calls during the steps
                mock_video_plume.get_frame.reset_mock()
                mock_navigator.update.reset_mock()
                
                # Perform 5 simulation steps directly
                sim.step()
                sim.step()
                sim.step()
                sim.step()
                sim.step()
                
                # Check that the plume was advanced multiple times
                assert mock_video_plume.get_frame.call_count == 5
                
                # Check that the navigator was updated multiple times
                assert mock_navigator.update.call_count == 5
                
                # Check that simulation state reflects multiple steps
                assert sim.time == 0.5
                assert sim.frame_index == 5
    
    def test_simulation_with_config(self, mock_video_plume, mock_navigator, config_files):
        """Test that a simulation can be created with configuration settings."""
        with patch('odor_plume_nav.simulation.create_video_plume_from_config', return_value=mock_video_plume), \
             patch('odor_plume_nav.simulation.create_navigator_from_config', return_value=mock_navigator), \
             patch('odor_plume_nav.simulation.load_config', return_value=config_files["default_config"]):
            
            # Create a simulation with configuration
            sim = Simulation(
                video_path="test_video.mp4",
                config_path="test_config.yaml"
            )
            
            # Check that factory functions were called with the right parameters
            from odor_plume_nav.simulation import create_video_plume_from_config, create_navigator_from_config
            
            create_video_plume_from_config.assert_called_once_with(
                "test_video.mp4", 
                config_path="test_config.yaml"
            )
            
            create_navigator_from_config.assert_called_once_with(
                config_path="test_config.yaml"
            )
    
    def test_get_agent_position(self, mock_video_plume, mock_navigator):
        """Test that we can get the agent's position from the simulation."""
        with patch('odor_plume_nav.simulation.VideoPlume', return_value=mock_video_plume), \
             patch('odor_plume_nav.simulation.SimpleNavigator', return_value=mock_navigator):
            
            # Create a simulation with mocked components
            sim = Simulation(
                video_path="test_video.mp4",
                dt=0.1
            )
            
            # Set up the mock to return a specific position
            mock_navigator.get_position.return_value = (2.5, 3.5)
            
            # Get the agent's position
            position = sim.get_agent_position()
            
            # Check that we got the right position
            assert position == (2.5, 3.5)
    
    def test_get_current_frame(self, mock_video_plume, mock_navigator):
        """Test that we can get the current frame from the simulation."""
        # Set up the mock to return a test frame
        test_frame = np.ones((10, 10), dtype=np.uint8)
        mock_video_plume.get_frame.return_value = test_frame
        
        with patch('odor_plume_nav.simulation.VideoPlume', return_value=mock_video_plume), \
             patch('odor_plume_nav.simulation.SimpleNavigator', return_value=mock_navigator):
            
            # Create a simulation with mocked components
            sim = Simulation(
                video_path="test_video.mp4",
                dt=0.1
            )
            
            # Since we already set up the mock before creating the simulation,
            # the constructor should have gotten the test frame
            assert np.array_equal(sim.current_frame, test_frame)
