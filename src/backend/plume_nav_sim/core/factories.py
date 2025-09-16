"""
Core factory functions for creating default data structures.
"""

from .geometry import GridSize
from .state import AgentState

def create_default_agent_state(grid_size: GridSize) -> AgentState:
    """Factory function to create a default agent state at the grid center."""
    center_position = grid_size.center()
    return AgentState(position=center_position)
