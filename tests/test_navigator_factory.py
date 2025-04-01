"""Tests for the navigator factory module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from odor_plume_nav.navigator_factory import create_navigator_from_config


def test_create_navigator_with_default_config(config_files):
    """Test creating a navigator with default configuration."""
    with patch('odor_plume_nav.navigator_factory.load_config', 
               return_value=config_files["default_config"]):
        
        # Create a navigator with default config
        navigator = create_navigator_from_config()
        
        # Check that the navigator was created with default settings
        assert navigator.orientation == 0.0
        assert navigator.speed == 0.0
        assert navigator.max_speed == 1.0


def test_create_navigator_with_user_config(config_files):
    """Test creating a navigator with user configuration."""
    with patch('odor_plume_nav.navigator_factory.load_config', 
               return_value=config_files["user_config"]):
        
        # Create a navigator with user config
        navigator = create_navigator_from_config()
        
        # Check that the navigator was created with user settings
        assert navigator.orientation == 45.0
        assert navigator.speed == 0.5
        assert navigator.max_speed == 2.0


def test_create_navigator_with_merged_config(config_files):
    """Test creating a navigator with merged configuration."""
    # Create a merged config by combining default and parts of user config
    merged_config = config_files["default_config"].copy()
    merged_config["navigator"]["orientation"] = 90.0  # Override just the orientation parameter
    
    with patch('odor_plume_nav.navigator_factory.load_config', 
               return_value=merged_config):
        
        # Create a navigator with merged config
        navigator = create_navigator_from_config()
        
        # Check that the navigator was created with merged settings
        assert navigator.orientation == 90.0  # Overridden
        assert navigator.speed == 0.0        # Default
        assert navigator.max_speed == 1.0    # Default


def test_create_navigator_with_additional_params(config_files):
    """Test creating a navigator with additional parameters."""
    with patch('odor_plume_nav.navigator_factory.load_config', 
               return_value=config_files["default_config"]):
        
        # Create a navigator with default config but override some parameters
        navigator = create_navigator_from_config(orientation=180.0, speed=0.75)
        
        # Check that the navigator was created with overridden settings
        assert navigator.orientation == 180.0  # Explicitly provided
        assert navigator.speed == 0.75         # Explicitly provided
        assert navigator.max_speed == 1.0      # From default config
