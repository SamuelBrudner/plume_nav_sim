"""
Configuration schema base classes and utilities for the odor plume navigation system.

This module provides base classes and common schema components used across
the configuration system, including re-exports of Pydantic base classes
for consistent schema definition and validation.
"""

from typing import Any, Dict, Optional, Type, Union, List, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Re-export Pydantic base classes for consistent use across config modules
__all__ = [
    "BaseModel",
    "Field", 
    "ConfigDict",
    "field_validator",
    "model_validator",
]

# Additional schema utilities can be added here as needed