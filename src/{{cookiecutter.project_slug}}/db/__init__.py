"""
Database Session Management Package for Odor Plume Navigation System

This package provides comprehensive database session management infrastructure using 
SQLAlchemy ≥2.0 patterns with optional activation capabilities. The database system 
remains completely inactive by default, ensuring zero performance impact on file-based 
operations while providing ready-to-activate persistence capabilities for trajectory 
storage, experiment metadata persistence, and collaborative research data sharing.

Key Features:
- Optional database activation without affecting default file-based operation
- Clean import interfaces for SQLAlchemy session management components
- Zero performance impact when database features are not configured
- Seamless integration with existing scientific Python ecosystem workflows
- Modular database infrastructure following cookiecutter template conventions
- Support for both synchronous and asynchronous database operations

Architecture Integration:
- Follows src/{{cookiecutter.project_slug}} layout with proper module organization
- Integrates with Hydra configuration hierarchy for database parameters
- Supports environment variable interpolation through python-dotenv
- Maintains compatibility with existing file-based persistence patterns
- Provides pluggable persistence adapter patterns for future extensibility

Optional Persistence Capabilities:
- Trajectory recording and experiment metadata storage through configuration toggles
- Multi-database backend support (SQLite, PostgreSQL, in-memory)
- Connection pooling with configurable parameters for performance optimization
- Async/await compatibility for high-performance concurrent operations
- Automatic transaction handling and rollback on errors
- Secure credential management via environment variable interpolation

Usage Examples:

    # Check if database features are available and enabled
    from {{cookiecutter.project_slug}}.db import is_database_enabled
    
    if is_database_enabled():
        # Database operations only when configured
        from {{cookiecutter.project_slug}}.db import get_session
        
        with get_session() as session:
            if session:
                # Perform database operations
                session.add(trajectory_record)
                # Automatic commit/rollback handled by context manager
    
    # Factory pattern for session configuration
    from {{cookiecutter.project_slug}}.db import SessionManager
    
    session_manager = SessionManager.from_config(config.database)
    with session_manager.session() as session:
        if session:
            # Database operations with configured backend
            pass
    
    # Async session support for high-performance operations
    from {{cookiecutter.project_slug}}.db import get_async_session
    
    async with get_async_session() as session:
        if session:
            # Async database operations
            await session.execute(query)

Integration Patterns:
- Import interfaces designed for both active and inactive database modes
- Graceful degradation when SQLAlchemy or related dependencies unavailable
- Compatibility with existing research workflows and file-based persistence
- Support for future cloud-based research infrastructure and collaboration

Authors: Cookiecutter Template Generator
License: MIT
Version: 2.0.0
"""

import warnings
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

# Type checking imports for development tooling support
if TYPE_CHECKING:
    from sqlalchemy.orm import Session as SQLASession
    from sqlalchemy.ext.asyncio import AsyncSession
    from .session import (
        SessionManager,
        DatabaseConfig,
        DatabaseConnectionInfo,
        DatabaseBackend,
        PersistenceHooks,
    )

# Global availability flags for dependency tracking
_SESSION_MODULE_AVAILABLE = False
_IMPORT_ERROR_MESSAGE = None


def _attempt_session_import():
    """
    Attempt to import session management components with graceful degradation.
    
    This function handles the optional nature of database functionality by attempting
    to import the session management infrastructure and capturing any import failures.
    When imports fail, the system maintains compatibility by providing None fallbacks
    while preserving error information for debugging purposes.
    
    Returns:
        tuple: (success_flag, error_message)
    """
    global _SESSION_MODULE_AVAILABLE, _IMPORT_ERROR_MESSAGE
    
    if _SESSION_MODULE_AVAILABLE:
        return True, None
    
    try:
        # Attempt to import session management infrastructure
        from .session import (
            SessionManager,
            DatabaseConfig,
            DatabaseConnectionInfo,
            DatabaseBackend,
            get_session_manager,
            get_session,
            get_async_session,
            is_database_enabled,
            test_database_connection,
            test_async_database_connection,
            cleanup_database,
            PersistenceHooks,
            SQLALCHEMY_AVAILABLE,
            HYDRA_AVAILABLE,
            DOTENV_AVAILABLE,
        )
        
        # Cache successful import for future use
        _SESSION_MODULE_AVAILABLE = True
        _IMPORT_ERROR_MESSAGE = None
        
        # Store references for module-level exports
        globals().update({
            'SessionManager': SessionManager,
            'DatabaseConfig': DatabaseConfig,
            'DatabaseConnectionInfo': DatabaseConnectionInfo,
            'DatabaseBackend': DatabaseBackend,
            'get_session_manager': get_session_manager,
            'get_session': get_session,
            'get_async_session': get_async_session,
            'is_database_enabled': is_database_enabled,
            'test_database_connection': test_database_connection,
            'test_async_database_connection': test_async_database_connection,
            'cleanup_database': cleanup_database,
            'PersistenceHooks': PersistenceHooks,
            'SQLALCHEMY_AVAILABLE': SQLALCHEMY_AVAILABLE,
            'HYDRA_AVAILABLE': HYDRA_AVAILABLE,
            'DOTENV_AVAILABLE': DOTENV_AVAILABLE,
        })
        
        return True, None
        
    except ImportError as e:
        # Capture import error for debugging while maintaining system functionality
        _IMPORT_ERROR_MESSAGE = str(e)
        warnings.warn(
            f"Database session management not available: {e}. "
            "System will operate in file-only mode. "
            "Install SQLAlchemy ≥2.0.41 to enable database features.",
            UserWarning,
            stacklevel=3
        )
        return False, str(e)
    
    except Exception as e:
        # Handle any other initialization errors
        _IMPORT_ERROR_MESSAGE = str(e)
        warnings.warn(
            f"Database initialization error: {e}. "
            "Database features will be unavailable.",
            UserWarning,
            stacklevel=3
        )
        return False, str(e)


def _create_fallback_functions():
    """
    Create fallback functions that maintain API compatibility when database features unavailable.
    
    These fallback implementations ensure that code can safely import and call database
    functions without conditional imports, with graceful degradation to None returns
    or no-op behavior when database infrastructure is not available.
    """
    
    def _fallback_is_database_enabled() -> bool:
        """Fallback function returning False when database infrastructure unavailable."""
        return False
    
    def _fallback_get_session(*args, **kwargs):
        """Fallback session context manager returning None when database unavailable."""
        from contextlib import nullcontext
        return nullcontext(None)
    
    def _fallback_get_async_session(*args, **kwargs):
        """Fallback async session context manager returning None when database unavailable."""
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def _null_async_context():
            yield None
        
        return _null_async_context()
    
    def _fallback_test_connection(*args, **kwargs) -> bool:
        """Fallback connection test returning False when database unavailable."""
        return False
    
    def _fallback_cleanup(*args, **kwargs) -> None:
        """Fallback cleanup function with no-op behavior when database unavailable."""
        pass
    
    def _fallback_get_session_manager(*args, **kwargs):
        """Fallback session manager returning None when database unavailable."""
        return None
    
    # Set fallback functions in module globals
    globals().update({
        'is_database_enabled': _fallback_is_database_enabled,
        'get_session': _fallback_get_session,
        'get_async_session': _fallback_get_async_session,
        'test_database_connection': _fallback_test_connection,
        'test_async_database_connection': _fallback_test_connection,
        'cleanup_database': _fallback_cleanup,
        'get_session_manager': _fallback_get_session_manager,
        'SessionManager': None,
        'DatabaseConfig': None,
        'DatabaseConnectionInfo': None,
        'DatabaseBackend': None,
        'PersistenceHooks': None,
        'SQLALCHEMY_AVAILABLE': False,
        'HYDRA_AVAILABLE': False,
        'DOTENV_AVAILABLE': False,
    })


# Initialize database session management with graceful degradation
_session_available, _session_error = _attempt_session_import()

if not _session_available:
    # Create fallback functions for API compatibility
    _create_fallback_functions()


def get_database_status() -> Dict[str, Any]:
    """
    Get comprehensive database infrastructure status information.
    
    Provides detailed information about database module availability, dependency
    status, and configuration readiness for debugging and system monitoring.
    
    Returns:
        Dict containing status information including:
        - session_module_available: Whether session.py module loaded successfully
        - database_enabled: Whether database features are currently active
        - dependencies: Status of required dependencies (SQLAlchemy, Hydra, dotenv)
        - error_message: Any import or initialization errors encountered
        - configuration_ready: Whether database configuration is available
    """
    status = {
        'session_module_available': _SESSION_MODULE_AVAILABLE,
        'database_enabled': False,
        'dependencies': {
            'sqlalchemy_available': globals().get('SQLALCHEMY_AVAILABLE', False),
            'hydra_available': globals().get('HYDRA_AVAILABLE', False),
            'dotenv_available': globals().get('DOTENV_AVAILABLE', False),
        },
        'error_message': _IMPORT_ERROR_MESSAGE,
        'configuration_ready': False,
    }
    
    # Check if database is actually enabled (not just available)
    if _SESSION_MODULE_AVAILABLE:
        try:
            is_enabled_func = globals().get('is_database_enabled')
            if is_enabled_func:
                status['database_enabled'] = is_enabled_func()
                
            # Test configuration readiness
            get_manager_func = globals().get('get_session_manager')
            if get_manager_func:
                manager = get_manager_func()
                status['configuration_ready'] = manager is not None and manager.enabled
                
        except Exception as e:
            status['error_message'] = f"Status check error: {e}"
    
    return status


def ensure_database_compatibility(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Ensure database system compatibility and attempt initialization if configured.
    
    This function provides a programmatic way to verify database readiness and
    attempt initialization when configuration is available. It's designed for
    use in application startup sequences where database features might be
    conditionally enabled based on runtime configuration.
    
    Args:
        config: Optional database configuration dictionary
        
    Returns:
        True if database features are available and ready, False otherwise
    """
    if not _SESSION_MODULE_AVAILABLE:
        return False
    
    try:
        # Check if database is enabled
        is_enabled_func = globals().get('is_database_enabled')
        if not is_enabled_func or not is_enabled_func():
            return False
        
        # Test connection if database is enabled
        test_func = globals().get('test_database_connection')
        if test_func:
            return test_func()
        
        return False
        
    except Exception as e:
        warnings.warn(
            f"Database compatibility check failed: {e}",
            UserWarning,
            stacklevel=2
        )
        return False


def configure_database_from_config(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Configure database session management from Hydra or dictionary configuration.
    
    Provides a convenience function for initializing database features from
    configuration objects, supporting both Hydra DictConfig and standard
    dictionary configurations with automatic type handling.
    
    Args:
        config: Database configuration (DictConfig, dict, or Pydantic model)
        
    Returns:
        True if database was successfully configured, False otherwise
    """
    if not _SESSION_MODULE_AVAILABLE:
        return False
    
    try:
        # Get session manager factory
        manager_factory = globals().get('get_session_manager')
        if not manager_factory:
            return False
        
        # Create session manager with provided configuration
        manager = manager_factory(config=config)
        
        # Test the configuration
        return manager is not None and manager.enabled
        
    except Exception as e:
        warnings.warn(
            f"Database configuration failed: {e}",
            UserWarning,
            stacklevel=2
        )
        return False


# Public API exports with comprehensive fallback handling
__all__ = [
    # Core session management
    'SessionManager',
    'get_session_manager',
    'get_session',
    'get_async_session',
    
    # Configuration and status
    'DatabaseConfig',
    'DatabaseConnectionInfo',
    'DatabaseBackend',
    'is_database_enabled',
    'get_database_status',
    
    # Connection testing and lifecycle
    'test_database_connection',
    'test_async_database_connection',
    'cleanup_database',
    
    # Persistence capabilities
    'PersistenceHooks',
    
    # Utility functions
    'ensure_database_compatibility',
    'configure_database_from_config',
    
    # Dependency availability flags
    'SQLALCHEMY_AVAILABLE',
    'HYDRA_AVAILABLE',
    'DOTENV_AVAILABLE',
]


# Module-level initialization and compatibility verification
def _initialize_database_module():
    """
    Initialize database module with comprehensive compatibility checking.
    
    Performs module-level initialization including dependency verification,
    configuration validation, and status logging for debugging and monitoring.
    This function is called once during module import to establish the
    database infrastructure state.
    """
    try:
        # Log initialization status for debugging
        import logging
        logger = logging.getLogger(__name__)
        
        if _SESSION_MODULE_AVAILABLE:
            logger.debug("Database session management module loaded successfully")
            
            # Check if database is actually configured and enabled
            is_enabled_func = globals().get('is_database_enabled')
            if is_enabled_func and is_enabled_func():
                logger.info("Database features are enabled and ready")
            else:
                logger.debug("Database infrastructure available but not configured")
        else:
            logger.debug(
                f"Database session management not available: {_IMPORT_ERROR_MESSAGE}. "
                "Operating in file-only mode."
            )
        
        # Validate public API completeness
        missing_exports = []
        for export_name in __all__:
            if export_name not in globals() or globals()[export_name] is None:
                if export_name not in ['SessionManager', 'DatabaseConfig', 
                                     'DatabaseConnectionInfo', 'DatabaseBackend', 
                                     'PersistenceHooks']:
                    missing_exports.append(export_name)
        
        if missing_exports:
            logger.warning(f"Missing database API exports: {missing_exports}")
        
    except Exception as e:
        # Initialization errors should not prevent module loading
        warnings.warn(
            f"Database module initialization warning: {e}",
            UserWarning,
            stacklevel=2
        )


# Perform module initialization
_initialize_database_module()


# Example usage and integration patterns for development reference
if __name__ == "__main__":
    """
    Example usage patterns for database session management.
    
    These examples demonstrate the optional nature of database features
    and proper integration patterns for research workflows.
    """
    
    print("Database Session Management Module Status")
    print("=" * 50)
    
    # Display comprehensive status information
    status = get_database_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("\nTesting database functionality...")
    
    # Test basic database availability
    if is_database_enabled():
        print("✓ Database features are enabled and ready")
        
        # Test database connection
        if test_database_connection():
            print("✓ Database connection successful")
        else:
            print("⚠ Database connection failed")
        
        # Example session usage
        with get_session() as session:
            if session:
                print("✓ Database session created successfully")
            else:
                print("⚠ Database session creation failed")
    
    else:
        print("ℹ Database features not enabled - file-based mode active")
        print("  This is expected for default configuration")
    
    # Test compatibility function
    if ensure_database_compatibility():
        print("✓ Database system fully compatible and ready")
    else:
        print("ℹ Database system not configured or compatible")
    
    print("\nModule initialization complete.")