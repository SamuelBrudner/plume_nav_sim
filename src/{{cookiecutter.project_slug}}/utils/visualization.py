"""
Comprehensive visualization module for odor plume navigation simulation.

This module consolidates all visualization capabilities including:
- Real-time animation (SimulationVisualization) with 30+ FPS performance
- Static trajectory plotting (visualize_trajectory) with publication-quality output  
- Hydra configuration integration for all visualization parameters
- CLI integration with headless mode support for automation
- Multi-agent visualization supporting up to 100 agents with vectorized rendering
- Batch processing capabilities for multiple simulation results

The module provides unified visualization interfaces that support both interactive
and automated visualization workflows for odor plume navigation experiments.

Examples:
    Real-time visualization with Hydra configuration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../../conf"):
        ...     cfg = compose(config_name="config")
        >>> viz = SimulationVisualization.from_config(cfg.visualization)
        >>> viz.setup_environment(plume_data)
        >>> animation = viz.create_animation(frame_callback, frames=1000)
        >>> viz.save_animation("output.mp4", headless=True)
        
    Static trajectory plotting with batch processing:
        >>> plot_config = cfg.visualization.static
        >>> visualize_trajectory(
        ...     positions, orientations, plume_frames,
        ...     output_path="trajectory.png", 
        ...     **plot_config
        ... )
        
    CLI headless export:
        $ python -m project_slug.cli.main visualize --headless \\
            experiment=multi_agent \\
            visualization.animation.fps=60 \\
            visualization.export.format=mp4 \\
            visualization.resolution=1080p
"""

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Protocol

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# Conditional imports for Hydra configuration support
try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Fallback types for environments without Hydra
    DictConfig = dict
    OmegaConf = None

# Import configuration schemas if available
try:
    from ..config.schemas import NavigatorConfig
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    CONFIG_SCHEMAS_AVAILABLE = False
    NavigatorConfig = None


class VisualizationConfig(Protocol):
    """Protocol defining visualization configuration interface."""
    
    fps: Optional[int] = 30
    format: Optional[str] = "mp4"
    resolution: Optional[str] = "720p"
    dpi: Optional[int] = 150
    headless: Optional[bool] = False
    theme: Optional[str] = "scientific"


class SimulationVisualization:
    """
    Real-time visualization class for odor plume simulation with Hydra configuration support.
    
    Provides interactive and headless visualization capabilities with 30+ FPS performance,
    multi-agent support, and comprehensive export options. Integrates seamlessly with
    Hydra configuration management and CLI workflows.
    
    Features:
    - Real-time animation with FuncAnimation framework
    - Multi-agent visualization up to 100 agents with vectorized rendering
    - Configurable frame rates (15-60 FPS), output formats, and quality settings
    - Headless operation for batch processing and CI/CD pipelines
    - Publication-quality export with MP4, AVI, and GIF support
    - Interactive controls with zoom, pan, and parameter adjustment
    - Memory-efficient trajectory management for extended simulations
    
    Attributes:
        fig (Figure): Matplotlib figure for visualization
        ax (Axes): Primary axes for plotting
        config (Dict): Visualization configuration parameters
        agent_artists (Dict): Collection of agent visual elements
        trajectory_lines (Dict): Agent trajectory line collections
        annotations (List): Text annotations and overlays
        animation_obj (FuncAnimation): Animation object for playback control
    """
    
    # Pre-defined color schemes for agent differentiation
    COLOR_SCHEMES = {
        'scientific': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'presentation': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'],
        'high_contrast': ['#000000', '#e60049', '#0bb4ff', '#50e991', '#e6d800',
                         '#9b19f5', '#ffa300', '#dc0ab4', '#b3d4ff', '#00bfa0']
    }
    
    # Resolution presets for different quality levels
    RESOLUTION_PRESETS = {
        '480p': (854, 480, 72),
        '720p': (1280, 720, 150), 
        '1080p': (1920, 1080, 300),
        'presentation': (1024, 768, 150),
        'publication': (1200, 900, 300)
    }
    
    def __init__(self, 
                 figsize: Tuple[float, float] = (12, 8),
                 dpi: int = 150,
                 fps: int = 30,
                 max_agents: int = 100,
                 theme: str = 'scientific',
                 headless: bool = False,
                 **kwargs):
        """
        Initialize the simulation visualization system.
        
        Args:
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for rendering quality
            fps: Target frame rate for animations (15-60 FPS)
            max_agents: Maximum number of agents to support
            theme: Color scheme theme ('scientific', 'presentation', 'high_contrast')
            headless: Enable headless mode for automated processing
            **kwargs: Additional configuration parameters
        """
        # Configure matplotlib backend for headless operation
        if headless:
            matplotlib.use('Agg')
        
        # Initialize figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Store configuration parameters
        self.config = {
            'figsize': figsize,
            'dpi': dpi,
            'fps': max(15, min(60, fps)),  # Clamp FPS to valid range
            'max_agents': max_agents,
            'theme': theme,
            'headless': headless,
            **kwargs
        }
        
        # Initialize visualization state
        self.img = None
        self.colorbar = None
        self.animation_obj = None
        
        # Agent visualization components
        self.agent_artists = {}
        self.trajectory_lines = {}
        self.trajectory_data = {}
        self.annotations = []
        
        # Performance optimization settings
        self.artist_cache = {}
        self.update_counter = 0
        self.memory_limit = 10000  # Maximum trajectory points per agent
        
        # Color scheme and styling
        self.colors = self.COLOR_SCHEMES.get(theme, self.COLOR_SCHEMES['scientific'])
        self._setup_figure_style()
        
    @classmethod
    def from_config(cls, config: Union[DictConfig, Dict[str, Any]]) -> 'SimulationVisualization':
        """
        Create SimulationVisualization instance from Hydra configuration.
        
        Args:
            config: Hydra configuration object or dictionary
            
        Returns:
            Configured SimulationVisualization instance
            
        Examples:
            >>> with initialize(config_path="conf"):
            ...     cfg = compose(config_name="config")
            >>> viz = SimulationVisualization.from_config(cfg.visualization)
        """
        if HYDRA_AVAILABLE and isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = dict(config) if config else {}
        
        # Extract resolution settings
        resolution = config_dict.get('resolution', '720p')
        if resolution in cls.RESOLUTION_PRESETS:
            width, height, dpi = cls.RESOLUTION_PRESETS[resolution]
            figsize = (width / dpi, height / dpi)
        else:
            figsize = config_dict.get('figsize', (12, 8))
            dpi = config_dict.get('dpi', 150)
        
        # Extract animation settings
        animation_config = config_dict.get('animation', {})
        fps = animation_config.get('fps', 30)
        
        # Extract agent settings
        agent_config = config_dict.get('agents', {})
        max_agents = agent_config.get('max_agents', 100)
        theme = agent_config.get('color_scheme', 'scientific')
        
        # Extract general settings
        headless = config_dict.get('headless', False)
        
        return cls(
            figsize=figsize,
            dpi=dpi,
            fps=fps,
            max_agents=max_agents,
            theme=theme,
            headless=headless,
            **config_dict
        )
    
    def _setup_figure_style(self) -> None:
        """Configure figure styling based on theme."""
        self.ax.set_title("Odor Plume Navigation Simulation", fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Apply theme-specific styling
        if self.config['theme'] == 'presentation':
            self.ax.set_facecolor('#f8f9fa')
            self.fig.patch.set_facecolor('white')
        elif self.config['theme'] == 'high_contrast':
            self.ax.set_facecolor('white')
            self.fig.patch.set_facecolor('white')
    
    def setup_environment(self, environment: np.ndarray, extent: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Set up the environment visualization with optimized rendering.
        
        Args:
            environment: 2D numpy array representing the odor environment
            extent: Optional extent as (left, right, bottom, top) for coordinate mapping
            
        Raises:
            ValueError: If environment is not a 2D array
        """
        if environment.ndim != 2:
            raise ValueError(f"Environment must be 2D array, got {environment.ndim}D")
        
        # Clear previous visualization components
        self.ax.clear()
        self._setup_figure_style()
        
        # Reset state
        self.agent_artists.clear()
        self.trajectory_lines.clear()
        self.trajectory_data.clear()
        self.annotations.clear()
        
        # Set up coordinate system
        height, width = environment.shape
        if extent is None:
            extent = (0, width, 0, height)
        
        # Plot environment as heatmap with optimized colormap
        vmax = np.max(environment) if np.max(environment) > 0 else 1
        self.img = self.ax.imshow(
            environment,
            origin='lower',
            extent=extent,
            cmap='viridis',
            vmin=0,
            vmax=vmax,
            interpolation='bilinear',
            aspect='auto'
        )
        
        # Add colorbar with improved formatting
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(self.img, ax=self.ax, shrink=0.8)
        self.colorbar.set_label('Odor Concentration', fontsize=10)
        
        # Configure axis limits and aspect ratio
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        
    def _get_agent_color(self, agent_id: int) -> str:
        """Get color for specified agent with cycling support."""
        return self.colors[agent_id % len(self.colors)]
    
    def _create_agent_artists(self, agent_id: int, position: Tuple[float, float], orientation: float) -> Dict[str, Any]:
        """
        Create visual artists for a new agent with vectorized rendering.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position as (x, y)
            orientation: Initial orientation in degrees
            
        Returns:
            Dictionary containing agent visual elements
        """
        color = self._get_agent_color(agent_id)
        
        # Create agent marker (circle)
        marker = plt.Circle(position, 0.5, color=color, fill=True, alpha=0.8, zorder=10)
        self.ax.add_patch(marker)
        
        # Create orientation arrow
        arrow_length = 1.0
        dx = arrow_length * math.cos(math.radians(orientation))
        dy = arrow_length * math.sin(math.radians(orientation))
        
        arrow = self.ax.arrow(
            position[0], position[1], dx, dy,
            head_width=0.3, head_length=0.4,
            fc=color, ec=color, alpha=0.9,
            zorder=11
        )
        
        # Initialize trajectory line collection for efficient updates
        self.trajectory_data[agent_id] = []
        line_collection = LineCollection([], colors=[color], alpha=0.6, linewidths=1.5, zorder=5)
        self.ax.add_collection(line_collection)
        self.trajectory_lines[agent_id] = line_collection
        
        return {
            'marker': marker,
            'arrow': arrow,
            'color': color,
            'trajectory_collection': line_collection
        }
    
    def update_visualization(self, frame_data: Union[Tuple, Dict]) -> List[Any]:
        """
        Update visualization for animation with vectorized multi-agent support.
        
        Args:
            frame_data: Frame data containing agent states. Can be:
                - Tuple: (position, orientation, odor_value) for single agent
                - Dict: {'agents': [agent_states], 'odor_values': [values]} for multi-agent
                
        Returns:
            List of updated matplotlib artists
            
        Examples:
            Single agent:
                >>> frame_data = ((10, 15), 45.0, 0.75)
                >>> artists = viz.update_visualization(frame_data)
                
            Multi-agent:
                >>> frame_data = {
                ...     'agents': [((10, 15), 45.0), ((20, 25), 90.0)],
                ...     'odor_values': [0.75, 0.60]
                ... }
                >>> artists = viz.update_visualization(frame_data)
        """
        updated_artists = []
        self.update_counter += 1
        
        # Handle different frame data formats
        if isinstance(frame_data, tuple) and len(frame_data) == 3:
            # Single agent format: (position, orientation, odor_value)
            agents_data = [(frame_data[0], frame_data[1])]
            odor_values = [frame_data[2]]
        elif isinstance(frame_data, dict):
            # Multi-agent format
            agents_data = frame_data.get('agents', [])
            odor_values = frame_data.get('odor_values', [0.0] * len(agents_data))
        else:
            raise ValueError(f"Unsupported frame_data format: {type(frame_data)}")
        
        # Clear previous annotations
        for annotation in self.annotations:
            annotation.remove()
        self.annotations.clear()
        
        # Update each agent
        for agent_id, (position, orientation) in enumerate(agents_data):
            # Create agent artists if they don't exist
            if agent_id not in self.agent_artists:
                if agent_id >= self.config['max_agents']:
                    warnings.warn(f"Agent {agent_id} exceeds max_agents limit ({self.config['max_agents']})")
                    continue
                self.agent_artists[agent_id] = self._create_agent_artists(agent_id, position, orientation)
            
            # Update agent position
            artists = self.agent_artists[agent_id]
            artists['marker'].center = position
            updated_artists.append(artists['marker'])
            
            # Update orientation arrow
            arrow_length = 1.0
            dx = arrow_length * math.cos(math.radians(orientation))
            dy = arrow_length * math.sin(math.radians(orientation))
            
            # Remove old arrow and create new one (matplotlib limitation)
            artists['arrow'].remove()
            artists['arrow'] = self.ax.arrow(
                position[0], position[1], dx, dy,
                head_width=0.3, head_length=0.4,
                fc=artists['color'], ec=artists['color'], alpha=0.9,
                zorder=11
            )
            updated_artists.append(artists['arrow'])
            
            # Update trajectory with memory management
            self.trajectory_data[agent_id].append(position)
            
            # Limit trajectory length for memory efficiency
            if len(self.trajectory_data[agent_id]) > self.memory_limit:
                self.trajectory_data[agent_id] = self.trajectory_data[agent_id][-self.memory_limit:]
            
            # Update trajectory line collection
            if len(self.trajectory_data[agent_id]) > 1:
                # Create line segments for efficient rendering
                points = np.array(self.trajectory_data[agent_id])
                segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
                self.trajectory_lines[agent_id].set_segments(segments)
                updated_artists.append(self.trajectory_lines[agent_id])
            
            # Add odor concentration annotation
            if agent_id < len(odor_values):
                annotation = self.ax.annotate(
                    f"Agent {agent_id + 1}: {odor_values[agent_id]:.3f}",
                    xy=position,
                    xytext=(position[0] + 1, position[1] - 1.5),
                    fontsize=8,
                    color='white',
                    bbox=dict(facecolor=artists['color'], alpha=0.7, boxstyle='round,pad=0.3'),
                    zorder=15
                )
                self.annotations.append(annotation)
                updated_artists.append(annotation)
        
        return updated_artists
    
    def create_animation(self, 
                        update_func: Callable[[int], Union[Tuple, Dict]], 
                        frames: int,
                        interval: Optional[int] = None,
                        blit: bool = True,
                        repeat: bool = False,
                        save_count: Optional[int] = None) -> animation.FuncAnimation:
        """
        Create optimized animation with configurable performance settings.
        
        Args:
            update_func: Function returning frame data for each animation step
            frames: Number of frames in the animation
            interval: Delay between frames in milliseconds (derived from fps if None)
            blit: Use blitting for improved performance (disable for complex scenes)
            repeat: Whether to loop the animation
            save_count: Number of frames to cache for saving (defaults to frames)
            
        Returns:
            FuncAnimation object for playback control
            
        Examples:
            >>> def frame_callback(frame_idx):
            ...     # Return single agent data
            ...     return (positions[frame_idx], orientations[frame_idx], odor_values[frame_idx])
            >>> animation = viz.create_animation(frame_callback, frames=1000)
            >>> viz.show()  # Start interactive playback
        """
        # Calculate interval from configured FPS
        if interval is None:
            interval = int(1000 / self.config['fps'])
        
        # Set default save_count
        if save_count is None:
            save_count = min(frames, 1000)  # Limit memory usage
        
        def animate_wrapper(frame_idx: int) -> List[Any]:
            """Wrapper function for animation updates."""
            try:
                frame_data = update_func(frame_idx)
                return self.update_visualization(frame_data)
            except Exception as e:
                print(f"Animation error at frame {frame_idx}: {e}")
                return []
        
        # Create animation with optimized settings
        self.animation_obj = animation.FuncAnimation(
            self.fig,
            animate_wrapper,
            frames=frames,
            interval=interval,
            blit=blit and not self.config['headless'],  # Disable blitting in headless mode
            repeat=repeat,
            save_count=save_count
        )
        
        return self.animation_obj
    
    def save_animation(self, 
                      filename: Union[str, Path],
                      fps: Optional[int] = None,
                      format: Optional[str] = None,
                      quality: Optional[str] = None,
                      extra_args: Optional[List[str]] = None,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """
        Save animation with configurable export settings and progress monitoring.
        
        Args:
            filename: Output file path with extension determining format
            fps: Frames per second for export (uses configuration default if None)
            format: Export format override ('mp4', 'avi', 'gif')
            quality: Quality preset ('low', 'medium', 'high', 'publication')
            extra_args: Additional ffmpeg arguments for fine-tuning
            progress_callback: Optional callback for progress updates
            
        Raises:
            ValueError: If no animation has been created
            RuntimeError: If export fails
            
        Examples:
            >>> viz.save_animation("output.mp4", fps=30, quality="high")
            >>> viz.save_animation("animation.gif", fps=10, quality="medium")
        """
        if self.animation_obj is None:
            raise ValueError("No animation created. Call create_animation() first.")
        
        # Use configured FPS if not specified
        export_fps = fps or self.config['fps']
        
        # Configure export settings based on quality preset
        quality_presets = {
            'low': {'bitrate': '500k', 'crf': '28'},
            'medium': {'bitrate': '1000k', 'crf': '23'},
            'high': {'bitrate': '2000k', 'crf': '18'},
            'publication': {'bitrate': '5000k', 'crf': '15'}
        }
        
        # Determine output format
        filepath = Path(filename)
        export_format = format or filepath.suffix[1:].lower()
        
        # Configure writer settings
        writer_args = {}
        if export_format in ['mp4', 'avi']:
            writer = 'ffmpeg'
            if extra_args is None:
                extra_args = ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                
                # Add quality settings
                if quality and quality in quality_presets:
                    preset = quality_presets[quality]
                    extra_args.extend(['-crf', preset['crf']])
                    
            writer_args['extra_args'] = extra_args
            
        elif export_format == 'gif':
            writer = 'pillow'
            writer_args['fps'] = min(export_fps, 20)  # GIF fps limitation
            
        else:
            writer = 'ffmpeg'  # Fallback to ffmpeg
        
        # Create progress callback wrapper
        if progress_callback:
            def progress_wrapper(frame_num, total_frames):
                progress_callback(frame_num, total_frames)
                
            writer_args['progress_callback'] = progress_wrapper
        
        try:
            # Save animation with error handling
            print(f"Saving animation to {filepath} (format: {export_format}, fps: {export_fps})")
            self.animation_obj.save(
                str(filepath),
                writer=writer,
                fps=export_fps,
                **writer_args
            )
            print(f"Animation successfully saved to {filepath}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save animation: {e}") from e
    
    def show(self) -> None:
        """
        Display interactive visualization window.
        
        Note: Does nothing in headless mode to prevent display errors.
        """
        if not self.config['headless']:
            plt.show()
        else:
            print("Headless mode: Use save_animation() to export results")
    
    def close(self) -> None:
        """Clean up visualization resources and close figure."""
        if self.animation_obj:
            self.animation_obj.event_source.stop()
        plt.close(self.fig)
        
        # Clear state
        self.agent_artists.clear()
        self.trajectory_lines.clear()
        self.trajectory_data.clear()
        self.annotations.clear()


def visualize_trajectory(
    positions: np.ndarray,
    orientations: Optional[np.ndarray] = None,
    plume_frames: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    agent_colors: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,  # Backward compatibility
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 300,
    format: str = 'png',
    theme: str = 'scientific',
    headless: bool = None,
    start_markers: bool = True,
    end_markers: bool = True,
    orientation_arrows: bool = True,
    trajectory_alpha: float = 0.7,
    grid: bool = True,
    legend: bool = True,
    colorbar: bool = True,
    batch_mode: bool = False,
    **kwargs
) -> Optional[Figure]:
    """
    Generate publication-quality static trajectory plots with comprehensive customization.
    
    Supports both single and multi-agent trajectory visualization with optional plume
    background overlay, orientation arrows, and extensive formatting options. Optimized
    for batch processing and headless operation in automated workflows.
    
    Args:
        positions: Agent positions as array of shape:
            - (time_steps, 2) for single agent
            - (num_agents, time_steps, 2) for multi-agent
            - (time_steps, num_agents, 2) alternative format (auto-detected)
        orientations: Optional orientation angles in degrees with compatible shape
        plume_frames: Optional plume concentration data for background overlay
        output_path: Path for saving the plot (format determined by extension)
        show_plot: Display plot interactively (ignored in headless mode)
        agent_colors: Custom color list for agents (overrides theme)
        colors: Legacy parameter name for agent_colors (backward compatibility)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height) in inches
        dpi: Resolution for saved figures (publication: 300, web: 150, preview: 72)
        format: Output format override ('png', 'pdf', 'svg', 'eps')
        theme: Color scheme ('scientific', 'presentation', 'high_contrast')
        headless: Force headless mode (auto-detected if None)
        start_markers: Show start position markers
        end_markers: Show end position markers  
        orientation_arrows: Show orientation arrow indicators
        trajectory_alpha: Transparency for trajectory lines (0-1)
        grid: Show coordinate grid
        legend: Show agent legend
        colorbar: Show plume concentration colorbar
        batch_mode: Optimize for batch processing (disables interactive features)
        **kwargs: Additional matplotlib configuration parameters
        
    Returns:
        Figure object if batch_mode=True, None otherwise
        
    Raises:
        ValueError: If positions array has invalid shape or incompatible orientations
        TypeError: If plume_frames has incorrect format
        
    Examples:
        Basic single agent plot:
            >>> positions = np.random.rand(100, 2) * 50
            >>> visualize_trajectory(positions, output_path="trajectory.png")
            
        Multi-agent with orientations and plume overlay:
            >>> positions = np.random.rand(3, 100, 2) * 50  # 3 agents, 100 steps
            >>> orientations = np.random.rand(3, 100) * 360
            >>> plume = np.random.rand(50, 50)
            >>> visualize_trajectory(
            ...     positions, orientations, plume,
            ...     output_path="multi_agent.pdf",
            ...     dpi=300, theme="presentation"
            ... )
            
        Batch processing configuration:
            >>> for i, (pos, orient) in enumerate(trajectory_data):
            ...     fig = visualize_trajectory(
            ...         pos, orient, 
            ...         output_path=f"batch_{i:03d}.png",
            ...         batch_mode=True, headless=True
            ...     )
            ...     fig.clear()  # Memory management
    """
    # Auto-detect headless mode
    if headless is None:
        # Check if running in headless environment
        headless = (
            batch_mode or 
            output_path is not None and not show_plot or
            matplotlib.get_backend() == 'Agg'
        )
    
    # Configure matplotlib backend for headless operation
    if headless:
        matplotlib.use('Agg')
    
    # Validate and reshape positions array
    positions = np.asarray(positions)
    if positions.ndim < 2 or positions.ndim > 3:
        raise ValueError(f"positions must be 2D or 3D array, got {positions.ndim}D")
    
    # Reshape to standard format: (num_agents, time_steps, 2)
    if positions.ndim == 2:
        # Single agent: (time_steps, 2) -> (1, time_steps, 2)
        if positions.shape[1] != 2:
            raise ValueError(f"For 2D positions, last dimension must be 2, got {positions.shape[1]}")
        positions = np.expand_dims(positions, axis=0)
    elif positions.ndim == 3:
        if positions.shape[2] != 2:
            # Check if in (time_steps, num_agents, 2) format
            if positions.shape[0] > positions.shape[1] and positions.shape[0] > 10:
                # Likely (time_steps, num_agents, 2) -> transpose to (num_agents, time_steps, 2)
                positions = np.transpose(positions, (1, 0, 2))
            else:
                raise ValueError(f"positions array must have shape [..., 2], got {positions.shape}")
    
    num_agents, time_steps, _ = positions.shape
    
    # Validate and reshape orientations if provided
    if orientations is not None:
        orientations = np.asarray(orientations)
        if orientations.ndim == 1:
            # Single agent case
            if len(orientations) != time_steps:
                raise ValueError(f"orientations length ({len(orientations)}) must match time_steps ({time_steps})")
            orientations = np.expand_dims(orientations, axis=0)
        elif orientations.ndim == 2:
            if orientations.shape[1] != time_steps:
                # Check if transposition is needed
                if orientations.shape[0] == time_steps:
                    orientations = orientations.T
                else:
                    raise ValueError(f"orientations shape {orientations.shape} incompatible with positions")
            if orientations.shape[0] != num_agents:
                raise ValueError(f"orientations agents ({orientations.shape[0]}) must match positions agents ({num_agents})")
    
    # Set up color scheme
    if agent_colors is None and colors is not None:
        agent_colors = colors  # Backward compatibility
    
    if agent_colors is None:
        color_schemes = SimulationVisualization.COLOR_SCHEMES
        agent_colors = color_schemes.get(theme, color_schemes['scientific'])
    
    # Ensure sufficient colors for all agents
    while len(agent_colors) < num_agents:
        agent_colors.extend(agent_colors)  # Repeat color scheme
    
    # Create figure with specified settings
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot plume background if provided
    if plume_frames is not None:
        plume_array = np.asarray(plume_frames)
        
        # Validate plume data format
        if plume_array.ndim < 2:
            raise TypeError(f"plume_frames must be at least 2D, got {plume_array.ndim}D")
        
        # Use first frame if multiple frames provided
        if plume_array.ndim > 2:
            plume_array = plume_array[0] if plume_array.shape[0] > 1 else plume_array.squeeze()
        
        # Display plume with appropriate colormap
        im = ax.imshow(
            plume_array, 
            cmap='viridis', 
            alpha=0.6, 
            origin='lower',
            extent=[0, plume_array.shape[1], 0, plume_array.shape[0]],
            aspect='auto'
        )
        
        # Add colorbar if requested
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Odor Concentration', fontsize=10)
    
    # Plot trajectories for each agent
    for agent_id in range(num_agents):
        agent_positions = positions[agent_id]
        color = agent_colors[agent_id % len(agent_colors)]
        
        # Plot trajectory line
        ax.plot(
            agent_positions[:, 0], agent_positions[:, 1],
            color=color, alpha=trajectory_alpha, linewidth=2,
            label=f'Agent {agent_id + 1}' if legend else None,
            zorder=5
        )
        
        # Mark start position
        if start_markers:
            ax.scatter(
                agent_positions[0, 0], agent_positions[0, 1],
                color=color, marker='o', s=100, edgecolors='white',
                linewidth=2, zorder=10, alpha=0.9
            )
        
        # Mark end position
        if end_markers:
            ax.scatter(
                agent_positions[-1, 0], agent_positions[-1, 1],
                color=color, marker='X', s=150, edgecolors='white',
                linewidth=2, zorder=10, alpha=0.9
            )
        
        # Plot orientation arrows if provided and requested
        if orientation_arrows and orientations is not None:
            agent_orientations = orientations[agent_id]
            
            # Subsample positions for arrow plotting (avoid overcrowding)
            arrow_step = max(1, time_steps // 20)  # Show ~20 arrows per trajectory
            arrow_indices = np.arange(0, time_steps, arrow_step)
            
            # Calculate arrow components
            orient_rad = np.deg2rad(agent_orientations[arrow_indices])
            arrow_scale = min(figsize) * 0.5  # Scale arrows relative to figure size
            
            u = np.cos(orient_rad) * arrow_scale
            v = np.sin(orient_rad) * arrow_scale
            
            # Plot orientation arrows
            ax.quiver(
                agent_positions[arrow_indices, 0], agent_positions[arrow_indices, 1],
                u, v, color=color, alpha=0.7, scale=50, width=0.003,
                zorder=8
            )
    
    # Configure plot appearance
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif num_agents == 1:
        ax.set_title('Agent Trajectory', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Multi-Agent Trajectories ({num_agents} agents)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if legend and num_agents > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Apply theme-specific styling
    if theme == 'presentation':
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
    elif theme == 'high_contrast':
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    
    # Set equal aspect ratio for accurate spatial representation
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot if output path specified
    if output_path:
        output_path = Path(output_path)
        
        # Determine format from extension or parameter
        save_format = format
        if output_path.suffix:
            save_format = output_path.suffix[1:].lower()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure save parameters based on format
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': fig.get_facecolor(),
            'edgecolor': 'none'
        }
        
        # Format-specific optimizations
        if save_format in ['png', 'jpg', 'jpeg']:
            save_kwargs['optimize'] = True
        elif save_format == 'pdf':
            save_kwargs['metadata'] = {'Creator': 'Odor Plume Navigation Visualization'}
        elif save_format == 'svg':
            save_kwargs['metadata'] = {'Creator': 'Odor Plume Navigation'}
        
        try:
            fig.savefig(str(output_path), format=save_format, **save_kwargs)
            if not batch_mode:
                print(f"Trajectory plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
            raise
    
    # Show plot if requested and not in headless mode
    if show_plot and not headless:
        plt.show()
    
    # Return figure for batch mode, close otherwise
    if batch_mode:
        return fig
    else:
        if not show_plot:
            plt.close(fig)
        return None


# Hydra ConfigStore registration for visualization configurations
if HYDRA_AVAILABLE:
    cs = ConfigStore.instance()
    
    # Register visualization configuration schemas
    try:
        from dataclasses import dataclass
        from typing import Optional
        
        @dataclass
        class AnimationConfig:
            fps: int = 30
            format: str = "mp4"
            quality: str = "medium"
            
        @dataclass  
        class StaticConfig:
            dpi: int = 300
            format: str = "png"
            figsize: Tuple[float, float] = (12, 8)
            
        @dataclass
        class AgentConfig:
            max_agents: int = 100
            color_scheme: str = "scientific"
            trail_length: int = 1000
            
        @dataclass
        class VisualizationConfig:
            animation: AnimationConfig = AnimationConfig()
            static: StaticConfig = StaticConfig()
            agents: AgentConfig = AgentConfig()
            headless: bool = False
            resolution: str = "720p"
            theme: str = "scientific"
        
        # Register configuration groups
        cs.store(group="visualization", name="animation", node=AnimationConfig)
        cs.store(group="visualization", name="static", node=StaticConfig)
        cs.store(group="visualization", name="agents", node=AgentConfig)
        cs.store(group="visualization", name="base", node=VisualizationConfig)
        
    except ImportError:
        # Fallback if dataclasses not available
        pass


# Public API exports
__all__ = [
    "SimulationVisualization",
    "visualize_trajectory",
    "VisualizationConfig"
]