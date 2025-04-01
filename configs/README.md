# Odor Plume Navigation Configuration System

This document explains how the configuration system works in the Odor Plume Navigation package.

## Overview

The configuration system follows a layered approach that allows customization at different levels:

1. **Default configuration** - Base settings for all components
2. **User configuration** - Custom overrides for specific use cases
3. **Parameter overrides** - Direct parameter values passed to factory functions

## Configuration Files

### default.yaml

The `default.yaml` file contains baseline settings for all components in the system. It defines:

- Default values for all configurable parameters
- Structure of the configuration (sections and subsections)
- Documentation of each parameter (via comments)

You should **never modify** this file directly. Instead, create your own configuration file that overrides specific values.

### example_user_config.yaml

This is a template showing how to create your own configuration file. It demonstrates:

- How to override specific settings while inheriting defaults for others
- The hierarchical structure of the configuration
- Proper formatting and commenting practices

To use it, copy this file and customize as needed.

## Current Configuration Sections

### video_plume

Settings for the `VideoPlume` class that handles video processing:

```yaml
video_plume:
  flip: false       # Whether to flip video frames
  kernel_size: 0    # Gaussian kernel size for smoothing (0 = no smoothing)
  kernel_sigma: 1.0 # Gaussian kernel sigma for smoothing
```

### navigator

Settings for the `SimpleNavigator` class that controls movement:

```yaml
navigator:
  orientation: 0.0  # Initial orientation in degrees (0 = right, 90 = up)
  speed: 0.0        # Initial speed in units per time
  max_speed: 1.0    # Maximum allowed speed
```

## How Configuration is Loaded

1. The `load_config()` function first loads the default configuration
2. If a user config path is provided, it loads and deep merges those settings
3. The environment variable `ODOR_PLUME_NAV_CONFIG_DIR` can specify a custom config directory

## Using the Configuration in Code

### Factory Functions

The package provides factory functions that create objects using configuration settings:

```python
# Create a VideoPlume with config settings
from odor_plume_nav.video_plume_factory import create_video_plume_from_config

plume = create_video_plume_from_config(
    video_path="path/to/video.mp4",       # Required
    config_path="path/to/user_config.yaml" # Optional
)

# Create a Navigator with config settings
from odor_plume_nav.navigator_factory import create_navigator_from_config

navigator = create_navigator_from_config(
    config_path="path/to/user_config.yaml", # Optional
    orientation=45.0  # Direct override (highest priority)
)
```

### Priority of Settings

When multiple sources define the same setting, this is the priority order (highest to lowest):

1. Explicit parameters passed to factory functions
2. Values in user configuration files
3. Values in the default configuration

## Extending the Configuration

When adding new configurable components:

1. Add default settings to `default.yaml`
2. Create a factory function that uses `load_config()`
3. Extract relevant settings from the configuration
4. Apply any explicit overrides
5. Update this documentation

## Best Practices

- Keep configuration files minimal and focused
- Only override settings that differ from defaults
- Use meaningful comments to explain non-obvious settings
- Follow the established naming and structure patterns
