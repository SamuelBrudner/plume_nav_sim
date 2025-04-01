"""
Visualization module for odor plume simulation.

This module provides visualization tools for the odor plume environment and navigating agents.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.animation as animation
from typing import Tuple, Optional, Union, Any, List, Callable


class SimulationVisualization:
    """Visualization class for odor plume simulation."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        """
        Initialize the visualization.
        
        Args:
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch (resolution)
        """
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.img = None
        self.agent_marker = None
        self.agent_direction = None
        self.colorbar = None
        self.ax.set_title("Odor Plume Simulation")
        self.animation = None
        self.traces = []
        self.annotations = []
    
    def setup_environment(self, environment: np.ndarray):
        """
        Set up the environment visualization.
        
        Args:
            environment: 2D numpy array representing the odor environment
        """
        # Clear previous visualization
        self.ax.clear()
        self.traces = []
        self.annotations = []
        
        # Plot the environment as a heatmap
        height, width = environment.shape
        extent = (0, width, 0, height)  # (left, right, bottom, top)
        self.img = self.ax.imshow(
            environment,
            origin='lower',  # To match standard cartesian coordinates
            extent=extent,
            cmap='viridis',  # Colormap for odor intensity
            vmin=0,
            vmax=np.max(environment) if np.max(environment) > 0 else 1
        )
        
        # Add colorbar
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(self.img, ax=self.ax)
        self.colorbar.set_label('Odor Concentration')
        
        # Set axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Create empty agent marker and direction arrow for animation
        self.agent_marker = plt.Circle((0, 0), 0.5, color='red', fill=True)
        self.ax.add_patch(self.agent_marker)
        
        self.agent_direction = self.ax.arrow(
            0, 0, 0, 0, 
            head_width=0.3, head_length=0.5, fc='red', ec='red'
        )
    
    def update_visualization(self, frame_data):
        """
        Update the visualization for animation.
        
        Args:
            frame_data: (position, orientation, odor_value) for current frame
        
        Returns:
            List of artists that were updated
        """
        position, orientation, odor_value = frame_data
        
        # Update agent marker position
        self.agent_marker.center = position
        
        # Update orientation arrow
        import math
        arrow_length = 1.0
        dx = arrow_length * math.cos(math.radians(orientation))
        dy = arrow_length * math.sin(math.radians(orientation))
        
        # Remove old arrow and create new one
        self.agent_direction.remove()
        self.agent_direction = self.ax.arrow(
            position[0], position[1], dx, dy, 
            head_width=0.3, head_length=0.5, fc='red', ec='red'
        )
        
        # Update traces
        if len(self.traces) > 1:
            x_vals = [pos[0] for pos in self.traces]
            y_vals = [pos[1] for pos in self.traces]
            
            # Update or create trace line
            trace_line = self.ax.plot(x_vals, y_vals, '-', color='blue', alpha=0.5)[0]
        else:
            trace_line = None
        
        # Add odor annotation
        for ann in self.annotations:
            ann.remove()
        self.annotations = []
        
        annotation = self.ax.annotate(
            f"Odor: {odor_value:.2f}",
            xy=position,
            xytext=(position[0], position[1]-2),
            fontsize=8,
            color='white',
            bbox=dict(facecolor='black', alpha=0.5)
        )
        self.annotations.append(annotation)
        
        # Return updated artists
        artists = [self.agent_marker, self.agent_direction]
        if trace_line:
            artists.append(trace_line)
        artists.extend(self.annotations)
        
        return artists
    
    def create_animation(self, update_func: Callable, frames: int, interval: int = 200, 
                        blit: bool = True, repeat: bool = False):
        """
        Create an animation using the provided update function.
        
        Args:
            update_func: Function that returns the data for each frame
            frames: Number of frames in the animation
            interval: Delay between frames in milliseconds
            blit: Whether to use blitting for improved performance
            repeat: Whether to repeat the animation
            
        Returns:
            The animation object
        """
        def animate(i):
            frame_data = update_func(i)
            self.traces.append(frame_data[0])  # Add position to traces
            return self.update_visualization(frame_data)
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=frames, interval=interval, 
            blit=blit, repeat=repeat
        )
        
        return self.animation
    
    def save_animation(self, filename: str, fps: int = 10, extra_args=None):
        """
        Save the animation to a file.
        
        Args:
            filename: Output file path
            fps: Frames per second
            extra_args: Additional arguments passed to the writer
        """
        if self.animation is None:
            raise ValueError("No animation has been created yet. Call create_animation first.")
        
        # Default to ffmpeg writer if available
        if extra_args is None:
            extra_args = ['-vcodec', 'libx264']
            
        # Save the animation
        self.animation.save(filename, fps=fps, extra_args=extra_args)
        print(f"Animation saved to {filename}")
    
    def show(self):
        """Show the animation."""
        plt.show()
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)
