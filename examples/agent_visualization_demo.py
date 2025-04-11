"""
Demo script for agent visualization in odor plume environment.

This script demonstrates how to visualize an agent navigating in an odor environment
using matplotlib's animation framework.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src"
sys.path.append(str(src_dir))

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.visualization import SimulationVisualization


def create_gaussian_odor_field(width: int = 50, height: int = 50, centers=None):
    """
    Create a Gaussian odor field with multiple sources.
    
    Args:
        width: Width of the environment
        height: Height of the environment
        centers: List of (x, y, intensity, sigma) for odor sources, or None to use defaults
    
    Returns:
        2D numpy array with odor values
    """
    if centers is None:
        # Default: two odor sources
        centers = [
            (width * 0.7, height * 0.6, 1.0, 5.0),   # (x, y, intensity, sigma)
            (width * 0.3, height * 0.3, 0.7, 7.0),
        ]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    odor_field = np.zeros((height, width), dtype=np.float32)
    
    # Add each Gaussian odor source
    for cx, cy, intensity, sigma in centers:
        gaussian = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
        odor_field += intensity * gaussian
    
    # Normalize if the maximum is > 1
    max_val = np.max(odor_field)
    if max_val > 1:
        odor_field /= max_val
        
    return odor_field


def demo_simple_movement():
    """
    Demonstrate simple movement of an agent in an odor field.
    """
    # Create an odor field
    width, height = 50, 50
    odor_field = create_gaussian_odor_field(width, height)
    
    # Create a navigator starting at the bottom left corner
    navigator = Navigator(
        position=(5, 5),
        orientation=45,  # 45 degrees = diagonally up and right
        speed=0.5,
        max_speed=1.0
    )
    
    # Create visualization
    viz = SimulationVisualization(figsize=(10, 8))
    viz.setup_environment(odor_field)
    
    # Maximum frames for animation
    num_frames = 40
    
    # Time step for simulation
    dt = 0.5
    
    # Define the update function for each frame
    def update_frame(frame_num):
        """Return data for the current animation frame."""
        print(f"Generating frame {frame_num}/{num_frames}")
        
        # Update agent position
        navigator.update(dt)
        current_pos = navigator.get_position()
        
        # Read odor at current position
        odor_value = navigator.read_single_antenna_odor(odor_field)
        
        # Print info to console
        print(f"Step {frame_num}: Position={current_pos}, "
              f"Orientation={navigator.orientation:.1f}°, Odor={odor_value:.3f}")
        
        # Return data for this frame: (position, orientation, odor_value)
        return (current_pos, navigator.orientation, odor_value)
    
    # Create and display the animation
    anim = viz.create_animation(
        update_func=update_frame,
        frames=num_frames,
        interval=200,  # milliseconds between frames
        blit=True,     # optimize performance
        repeat=True    # repeat the animation
    )
    
    # Save the animation to a file
    output_dir = Path(script_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "simple_movement.mp4"
    
    print(f"Saving animation to {output_file}...")
    viz.save_animation(str(output_file), fps=10)
    
    # Display the animation
    print("Displaying animation. Close the window when done.")
    viz.show()


def demo_odor_following():
    """
    Demonstrate a simple algorithm for an agent following an odor gradient.
    """
    # Create an odor field
    width, height = 50, 50
    odor_field = create_gaussian_odor_field(width, height)
    
    # Create a navigator starting at the bottom left corner
    navigator = Navigator(
        position=(5, 5),
        orientation=0,
        speed=0.0,  # Start with zero speed
        max_speed=1.0
    )
    
    # Create visualization
    viz = SimulationVisualization(figsize=(10, 8))
    viz.setup_environment(odor_field)
    
    # Maximum frames for animation
    num_frames = 40
    
    # Time step for simulation
    dt = 0.5
    
    # Storage for movement data
    stop_frame = None
    
    # Define the update function for each frame
    def update_frame(frame_num):
        """Return data for the current animation frame."""
        nonlocal stop_frame
        
        # If we already reached the target, repeat the last frame
        if stop_frame is not None and frame_num >= stop_frame:
            current_pos = navigator.get_position()
            odor_value = navigator.read_single_antenna_odor(odor_field)
            return (current_pos, navigator.orientation, odor_value)
        
        print(f"Generating frame {frame_num}/{num_frames}")
        
        # Get current state
        current_pos = navigator.get_position()
        odor_value = navigator.read_single_antenna_odor(odor_field)
        
        # Simple gradient-following algorithm:
        # 1. Look in different directions
        # 2. Move in the direction with the highest odor value
        best_direction = 0
        best_odor = odor_value
        
        # Test different directions
        for test_direction in [0, 45, 90, 135, 180, 225, 270, 315]:
            # Calculate position if we moved in this direction
            test_radians = np.radians(test_direction)
            test_x = current_pos[0] + np.cos(test_radians) * 2
            test_y = current_pos[1] + np.sin(test_radians) * 2
            
            # Create a temporary navigator to read the odor at this position
            test_navigator = Navigator(position=(test_x, test_y))
            test_odor = test_navigator.read_single_antenna_odor(odor_field)
            
            # Update best direction if this is better
            if test_odor > best_odor:
                best_odor = test_odor
                best_direction = test_direction
        
        # Set the navigator's orientation and speed based on the best direction
        navigator.set_orientation(best_direction)
        
        # Set speed proportional to the odor gradient
        odor_improvement = max(0, best_odor - odor_value)
        navigator.set_speed(0.3 + odor_improvement * 2)
        
        # Update agent position
        navigator.update(dt)
        
        # Print info to console
        print(f"Step {frame_num}: Position={current_pos}, Direction={navigator.orientation:.1f}°, "
              f"Speed={navigator.speed:.2f}, Odor={odor_value:.3f}, Best odor={best_odor:.3f}")
        
        # If we've reached a very strong odor source, remember this frame
        if odor_value > 0.9 and stop_frame is None:
            print("Reached strong odor source!")
            stop_frame = frame_num + 1
        
        # Return data for this frame: (position, orientation, odor_value)
        return (current_pos, navigator.orientation, odor_value)
    
    # Create and display the animation
    anim = viz.create_animation(
        update_func=update_frame,
        frames=num_frames,
        interval=200,  # milliseconds between frames
        blit=True,     # optimize performance
        repeat=True    # repeat the animation
    )
    
    # Save the animation to a file
    output_dir = Path(script_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "odor_following.mp4"
    
    print(f"Saving animation to {output_file}...")
    viz.save_animation(str(output_file), fps=10)
    
    # Display the animation
    print("Displaying animation. Close the window when done.")
    viz.show()


if __name__ == "__main__":
    print("1. Simple Movement Demo")
    print("2. Odor Following Demo")
    choice = input("Select a demo (1 or 2): ")
    
    # Create output directory
    output_dir = Path(script_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    if choice == "2":
        demo_odor_following()
    else:
        demo_simple_movement()
