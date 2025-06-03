"""Database session management module for optional persistence capabilities.

This module provides clean import interfaces for SQLAlchemy session management 
components while maintaining the optional nature of database infrastructure.
The database infrastructure remains completely inactive by default, ensuring
zero performance impact on file-based operations while providing ready-to-extend
infrastructure for advanced persistence requirements.

Key Features:
- Optional database activation without affecting default file-based operation
- Support for both active and inactive database modes
- Seamless integration with existing scientific Python ecosystem workflows
- Modular database infrastructure that integrates with existing file-based persistence
- Enterprise-grade SQLAlchemy session management when activated

Database Backends Supported:
- SQLite for development and testing
- PostgreSQL for production environments  
- In-memory databases for unit testing
- Future cloud database integrations

Usage Patterns:
    
    # For file-based workflows (default behavior)
    from {{cookiecutter.project_slug}}.db import database_available
    if not database_available:
        # Continue with file-based persistence
        save_trajectory_to_file(data)
    
    # For database-enabled workflows (when configured)
    from {{cookiecutter.project_slug}}.db import get_session, database_available
    if database_available:
        with get_session() as session:
            session.add(trajectory_record)
            session.commit()

The module follows cookiecutter template conventions with proper module organization
and maintains compatibility with existing research workflows while providing
extensibility for collaborative research environments.
"""

import logging
from typing import Optional, Any, Union, ContextManager

# Configure module logger
logger = logging.getLogger(__name__)

# Track database availability and initialization state
_database_available: bool = False
_session_factory: Optional[Any] = None
_database_initialized: bool = False

try:
    # Attempt to import database session management components
    from .session import (
        SessionManager,
        get_session_factory,
        create_session,
        DatabaseConfig,
        initialize_database,
        close_database_connections,
    )
    
    _database_available = True
    logger.debug("Database session management components successfully imported")
    
except ImportError as e:
    logger.debug(f"Database components not available: {e}")
    logger.debug("System will operate in file-based mode (default behavior)")
    
    # Define stub implementations for graceful degradation
    class SessionManager:
        """Stub session manager for non-database mode."""
        
        def __init__(self, *args, **kwargs):
            logger.warning("Database not configured - SessionManager operating in stub mode")
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def add(self, *args, **kwargs):
            pass
            
        def commit(self):
            pass
            
        def rollback(self):
            pass
            
        @property
        def is_active(self) -> bool:
            return False
    
    def get_session_factory(*args, **kwargs) -> None:
        """Stub session factory for non-database mode."""
        logger.debug("Database not available - returning None for session factory")
        return None
    
    def create_session(*args, **kwargs) -> SessionManager:
        """Stub session creator for non-database mode."""
        logger.debug("Database not available - returning stub session")
        return SessionManager()
    
    class DatabaseConfig:
        """Stub database configuration for non-database mode."""
        pass
    
    def initialize_database(*args, **kwargs) -> bool:
        """Stub database initializer for non-database mode."""
        logger.debug("Database not available - initialization skipped")
        return False
    
    def close_database_connections():
        """Stub database cleanup for non-database mode."""
        logger.debug("Database not available - connection cleanup skipped")
        pass

except Exception as e:
    logger.error(f"Unexpected error importing database components: {e}")
    logger.info("System will continue in file-based mode")
    _database_available = False


def is_database_available() -> bool:
    """Check if database infrastructure is available and properly configured.
    
    Returns:
        bool: True if database components are available and can be used,
              False if system should operate in file-based mode.
    
    Note:
        This function only checks for component availability, not actual
        database connectivity. Use initialize_database() to test connectivity.
    """
    return _database_available


def get_session() -> Union[ContextManager[Any], SessionManager]:
    """Get a database session context manager if database is available.
    
    Returns:
        Context manager for database session when database is configured,
        or stub session manager when operating in file-based mode.
    
    Example:
        >>> with get_session() as session:
        ...     if session.is_active:
        ...         session.add(trajectory_record)
        ...         session.commit()
        ...     else:
        ...         save_trajectory_to_file(trajectory_data)
    """
    if _database_available and _session_factory is not None:
        try:
            return create_session()
        except Exception as e:
            logger.warning(f"Failed to create database session: {e}")
            logger.info("Falling back to stub session manager")
            return SessionManager()
    else:
        logger.debug("Database not configured - returning stub session")
        return SessionManager()


def setup_database(config: Optional[Any] = None) -> bool:
    """Initialize database infrastructure if available and configured.
    
    Args:
        config: Hydra configuration object containing database parameters.
                If None, will attempt to load from environment variables.
    
    Returns:
        bool: True if database was successfully initialized,
              False if initialization failed or database is not available.
    
    Note:
        This function is safe to call even when database infrastructure
        is not available. It will gracefully return False without raising
        exceptions, allowing the system to continue in file-based mode.
    """
    global _session_factory, _database_initialized
    
    if not _database_available:
        logger.debug("Database infrastructure not available - skipping initialization")
        return False
    
    if _database_initialized:
        logger.debug("Database already initialized")
        return True
    
    try:
        success = initialize_database(config)
        if success:
            _session_factory = get_session_factory()
            _database_initialized = True
            logger.info("Database infrastructure successfully initialized")
        else:
            logger.warning("Database initialization failed - continuing in file-based mode")
        return success
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        logger.info("Continuing in file-based mode")
        return False


def cleanup_database():
    """Clean up database connections and resources.
    
    This function is safe to call even when database infrastructure
    is not available or not initialized. It ensures proper cleanup
    without raising exceptions.
    """
    global _session_factory, _database_initialized
    
    if not _database_available:
        logger.debug("Database infrastructure not available - skipping cleanup")
        return
    
    try:
        close_database_connections()
        _session_factory = None
        _database_initialized = False
        logger.debug("Database connections cleaned up successfully")
        
    except Exception as e:
        logger.warning(f"Error during database cleanup: {e}")
        # Still reset state even if cleanup failed
        _session_factory = None
        _database_initialized = False


# Provide convenient aliases for backward compatibility and clean imports
database_available = is_database_available()
DatabaseSession = get_session

# Export public API
__all__ = [
    # Core session management
    "SessionManager",
    "get_session",
    "DatabaseSession",  # Alias for convenience
    
    # Database lifecycle management
    "setup_database", 
    "cleanup_database",
    "initialize_database",
    "close_database_connections",
    
    # Configuration and factory functions
    "DatabaseConfig",
    "get_session_factory",
    "create_session",
    
    # Utility functions
    "is_database_available",
    "database_available",  # Convenience constant
]

# Module-level initialization logging
if _database_available:
    logger.info("Database session management module loaded - ready for optional activation")
else:
    logger.info("Database session management module loaded in file-based mode")