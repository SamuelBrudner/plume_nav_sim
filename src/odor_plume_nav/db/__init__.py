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
- Modular database infrastructure following unified package conventions
- Support for both synchronous and asynchronous database operations

Architecture Integration:
- Follows src/odor_plume_nav layout with proper module organization
- Integrates with Hydra configuration hierarchy for database parameters
- Supports environment variable interpolation through python-dotenv
- Maintains compatibility with existing file-based persistence patterns
- Provides pluggable persistence adapter patterns for future extensibility

Optional Persistence Capabilities:
- Trajectory recording and experiment metadata storage through configuration toggles
- Multi-database backend support (SQLite, PostgreSQL, in-memory)
- Connection pooling with configurable parameters for performance optimization
- Context-managed sessions with automatic transaction handling and connection cleanup
- Secure credential management via environment variable interpolation
- Enterprise-grade session management with comprehensive error handling

Usage Examples:

    # Check if database features are available and enabled
    from odor_plume_nav.db import is_database_enabled
    
    if is_database_enabled():
        # Database operations only when configured
        from odor_plume_nav.db import get_session_manager
        
        manager = get_session_manager()
        with manager.get_session() as session:
            if session:
                # Perform database operations
                session.add(trajectory_record)
                # Automatic commit/rollback handled by context manager
    
    # Factory pattern for session configuration
    from odor_plume_nav.db import DatabaseSessionManager, DatabaseConfig
    
    config = DatabaseConfig(url="sqlite:///experiments.db")
    session_manager = DatabaseSessionManager(config)
    with session_manager.get_session() as session:
        if session:
            # Database operations with configured backend
            pass
    
    # Environment-based configuration
    from odor_plume_nav.db import get_session_manager
    
    # Automatically detects DATABASE_URL environment variable
    session_manager = get_session_manager()
    
    with session_manager.get_session() as session:
        if session:
            # Database operations with environment configuration
            pass

Integration Patterns:
- Import interfaces designed for both active and inactive database modes
- Graceful degradation when SQLAlchemy or related dependencies unavailable
- Compatibility with existing research workflows and file-based persistence
- Support for future cloud-based research infrastructure and collaboration

Authors: Blitzy Platform Engineering Team
License: MIT
Version: 2.0.0
"""

import logging
import warnings
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# Type checking imports for development tooling support
if TYPE_CHECKING:
    from sqlalchemy.orm import Session as SQLASession
    from .session import (
        DatabaseSessionManager,
        DatabaseConfig,
    )

# Global availability flags for dependency tracking
_SESSION_MODULE_AVAILABLE = False
_SQLALCHEMY_AVAILABLE = False
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
    global _SESSION_MODULE_AVAILABLE, _SQLALCHEMY_AVAILABLE, _IMPORT_ERROR_MESSAGE
    
    if _SESSION_MODULE_AVAILABLE:
        return True, None
    
    try:
        # Attempt to import session management infrastructure
        from .session import (
            DatabaseSessionManager,
            DatabaseConfig,
            DatabaseConnectionError,
            DatabaseTransactionError,
            DatabaseConfigurationError,
            get_session_manager,
            get_session,
            configure_database,
            get_default_session,
            SQLALCHEMY_AVAILABLE,
        )
        
        # Cache successful import for future use
        _SESSION_MODULE_AVAILABLE = True
        _SQLALCHEMY_AVAILABLE = SQLALCHEMY_AVAILABLE
        _IMPORT_ERROR_MESSAGE = None
        
        # Store references for module-level exports
        globals().update({
            'DatabaseSessionManager': DatabaseSessionManager,
            'DatabaseConfig': DatabaseConfig,
            'DatabaseConnectionError': DatabaseConnectionError,
            'DatabaseTransactionError': DatabaseTransactionError,
            'DatabaseConfigurationError': DatabaseConfigurationError,
            'get_session_manager': get_session_manager,
            'get_session': get_session,
            'configure_database': configure_database,
            'get_default_session': get_default_session,
            'SQLALCHEMY_AVAILABLE': SQLALCHEMY_AVAILABLE,
        })
        
        logger.debug("Database session management module loaded successfully")
        return True, None
        
    except ImportError as e:
        # Capture import error for debugging while maintaining system functionality
        _IMPORT_ERROR_MESSAGE = str(e)
        logger.debug(
            f"Database session management not available: {e}. "
            "System will operate in file-only mode."
        )
        return False, str(e)
    
    except Exception as e:
        # Handle any other initialization errors
        _IMPORT_ERROR_MESSAGE = str(e)
        logger.warning(f"Database initialization error: {e}")
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
    
    @contextmanager
    def _fallback_get_session(*args, **kwargs):
        """Fallback session context manager returning None when database unavailable."""
        yield None
    
    @contextmanager
    def _fallback_get_default_session(*args, **kwargs):
        """Fallback default session context manager returning None when database unavailable."""
        yield None
    
    def _fallback_test_connection(*args, **kwargs) -> bool:
        """Fallback connection test returning False when database unavailable."""
        return False
    
    def _fallback_cleanup(*args, **kwargs) -> None:
        """Fallback cleanup function with no-op behavior when database unavailable."""
        pass
    
    def _fallback_get_session_manager(*args, **kwargs):
        """Fallback session manager returning None when database unavailable."""
        return None
    
    def _fallback_configure_database(*args, **kwargs) -> None:
        """Fallback database configuration with no-op behavior when database unavailable."""
        pass
    
    # Set fallback functions in module globals
    globals().update({
        'is_database_enabled': _fallback_is_database_enabled,
        'get_session_manager': _fallback_get_session_manager,
        'get_session': _fallback_get_session,
        'get_default_session': _fallback_get_default_session,
        'configure_database': _fallback_configure_database,
        'test_database_connection': _fallback_test_connection,
        'cleanup_database': _fallback_cleanup,
        'DatabaseSessionManager': None,
        'DatabaseConfig': None,
        'DatabaseConnectionError': None,
        'DatabaseTransactionError': None,
        'DatabaseConfigurationError': None,
        'SQLALCHEMY_AVAILABLE': False,
    })


# Initialize database session management with graceful degradation
_session_available, _session_error = _attempt_session_import()

if not _session_available:
    # Create fallback functions for API compatibility
    _create_fallback_functions()


def is_database_enabled() -> bool:
    """
    Check if database session management is enabled and operational.
    
    This function provides a comprehensive check for database availability,
    including dependency availability, configuration presence, and actual
    connectivity. It serves as the primary gate for database-dependent operations.
    
    Returns:
        True if database features are enabled and ready, False otherwise
        
    Examples:
        Basic usage pattern:
            ```python
            from odor_plume_nav.db import is_database_enabled
            
            if is_database_enabled():
                # Safe to use database operations
                from odor_plume_nav.db import get_session_manager
                manager = get_session_manager()
                # Proceed with database operations
            else:
                # Use file-based persistence instead
                pass
            ```
        
        Integration with configuration:
            ```python
            import hydra
            from omegaconf import DictConfig
            from odor_plume_nav.db import is_database_enabled, configure_database_from_config
            
            @hydra.main(version_base=None, config_path="conf", config_name="config")
            def main(cfg: DictConfig) -> None:
                # Attempt to configure database from Hydra config
                configure_database_from_config(cfg)
                
                if is_database_enabled():
                    print("Database persistence enabled")
                else:
                    print("Using file-based persistence")
            ```
    """
    if not _SESSION_MODULE_AVAILABLE:
        return False
    
    try:
        # Get session manager to test actual availability
        manager_func = globals().get('get_session_manager')
        if not manager_func:
            return False
        
        manager = manager_func()
        return manager is not None and manager.is_active
        
    except Exception as e:
        logger.debug(f"Database enablement check failed: {e}")
        return False


def get_database_status() -> Dict[str, Any]:
    """
    Get comprehensive database infrastructure status information.
    
    Provides detailed information about database module availability, dependency
    status, and configuration readiness for debugging and system monitoring.
    
    Returns:
        Dict containing status information including:
        - session_module_available: Whether session.py module loaded successfully
        - database_enabled: Whether database features are currently active
        - sqlalchemy_available: Whether SQLAlchemy dependency is installed
        - error_message: Any import or initialization errors encountered
        - configuration_ready: Whether database configuration is available
        - session_manager_active: Whether session manager is operational
        
    Examples:
        System diagnostics:
            ```python
            from odor_plume_nav.db import get_database_status
            
            status = get_database_status()
            print(f"Database available: {status['database_enabled']}")
            print(f"SQLAlchemy installed: {status['sqlalchemy_available']}")
            
            if not status['database_enabled'] and status['error_message']:
                print(f"Database unavailable: {status['error_message']}")
            ```
        
        Configuration validation:
            ```python
            from odor_plume_nav.db import get_database_status
            
            status = get_database_status()
            if status['session_module_available'] and not status['configuration_ready']:
                print("Database infrastructure available but not configured")
                print("Set DATABASE_URL environment variable to enable database features")
            ```
    """
    status = {
        'session_module_available': _SESSION_MODULE_AVAILABLE,
        'database_enabled': False,
        'sqlalchemy_available': _SQLALCHEMY_AVAILABLE,
        'error_message': _IMPORT_ERROR_MESSAGE,
        'configuration_ready': False,
        'session_manager_active': False,
    }
    
    # Check if database is actually enabled (not just available)
    if _SESSION_MODULE_AVAILABLE:
        try:
            status['database_enabled'] = is_database_enabled()
            
            # Test session manager status
            get_manager_func = globals().get('get_session_manager')
            if get_manager_func:
                manager = get_manager_func()
                status['session_manager_active'] = manager is not None and manager.is_active
                status['configuration_ready'] = manager is not None
                
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
        
    Examples:
        Application startup:
            ```python
            from odor_plume_nav.db import ensure_database_compatibility
            
            # Test compatibility without configuration
            if ensure_database_compatibility():
                print("Database system ready")
            else:
                print("Database system not available - using file persistence")
            ```
        
        With configuration:
            ```python
            from odor_plume_nav.db import ensure_database_compatibility
            
            db_config = {
                'url': 'sqlite:///experiments.db',
                'pool_size': 5
            }
            
            if ensure_database_compatibility(db_config):
                print("Database configured and ready")
            else:
                print("Database configuration failed")
            ```
    """
    if not _SESSION_MODULE_AVAILABLE:
        return False
    
    try:
        # Check if database is enabled
        if not is_database_enabled():
            return False
        
        # Test basic connectivity if database is enabled
        manager_func = globals().get('get_session_manager')
        if not manager_func:
            return False
        
        manager = manager_func()
        if not manager or not manager.is_active:
            return False
        
        # Attempt to create a session to verify connectivity
        with manager.get_session() as session:
            return session is not None
        
    except Exception as e:
        logger.debug(f"Database compatibility check failed: {e}")
        return False


def configure_database_from_config(config: Union[Dict[str, Any], Any]) -> bool:
    """
    Configure database session management from Hydra or dictionary configuration.
    
    Provides a convenience function for initializing database features from
    configuration objects, supporting both Hydra DictConfig and standard
    dictionary configurations with automatic type handling.
    
    Args:
        config: Database configuration (DictConfig, dict, or DatabaseConfig)
        
    Returns:
        True if database was successfully configured, False otherwise
        
    Examples:
        With Hydra configuration:
            ```python
            import hydra
            from omegaconf import DictConfig
            from odor_plume_nav.db import configure_database_from_config
            
            @hydra.main(version_base=None, config_path="conf", config_name="config")
            def main(cfg: DictConfig) -> None:
                if hasattr(cfg, 'database') and cfg.database:
                    success = configure_database_from_config(cfg.database)
                    if success:
                        print("Database configured successfully")
                    else:
                        print("Database configuration failed")
            ```
        
        With dictionary configuration:
            ```python
            from odor_plume_nav.db import configure_database_from_config
            
            config = {
                'url': 'postgresql://user:pass@localhost:5432/experiments',
                'pool_size': 10,
                'echo': False
            }
            
            if configure_database_from_config(config):
                print("Database configured successfully")
            ```
    """
    if not _SESSION_MODULE_AVAILABLE:
        return False
    
    try:
        # Get configuration function
        configure_func = globals().get('configure_database')
        if not configure_func:
            return False
        
        # Handle different configuration types
        if hasattr(config, 'database') and config.database:
            # Hydra DictConfig with database section
            db_config = config.database
        elif isinstance(config, dict) and 'database' in config:
            # Dictionary with database section
            db_config = config['database']
        else:
            # Direct database configuration
            db_config = config
        
        # Create DatabaseConfig if needed
        DatabaseConfig = globals().get('DatabaseConfig')
        if DatabaseConfig and isinstance(db_config, dict):
            db_config = DatabaseConfig(**db_config)
        
        # Configure database
        configure_func(db_config)
        
        # Verify configuration worked
        return is_database_enabled()
        
    except Exception as e:
        logger.debug(f"Database configuration failed: {e}")
        return False


def test_database_connection() -> bool:
    """
    Test database connectivity and return connection status.
    
    Performs a simple connectivity test to verify that the database is accessible
    and properly configured. This function is safe to call regardless of database
    availability and will return False if database features are not enabled.
    
    Returns:
        True if database connection is successful, False otherwise
        
    Examples:
        Connection verification:
            ```python
            from odor_plume_nav.db import test_database_connection, is_database_enabled
            
            if is_database_enabled():
                if test_database_connection():
                    print("Database connection successful")
                else:
                    print("Database connection failed")
            else:
                print("Database not enabled")
            ```
        
        Pre-operation check:
            ```python
            from odor_plume_nav.db import test_database_connection, get_session_manager
            
            if test_database_connection():
                # Safe to proceed with database operations
                manager = get_session_manager()
                with manager.get_session() as session:
                    # Perform database operations
                    pass
            else:
                # Fall back to file-based operations
                pass
            ```
    """
    if not _SESSION_MODULE_AVAILABLE or not is_database_enabled():
        return False
    
    try:
        # Get session manager and test connection
        manager_func = globals().get('get_session_manager')
        if not manager_func:
            return False
        
        manager = manager_func()
        if not manager or not manager.is_active:
            return False
        
        # Attempt to create and use a session
        with manager.get_session() as session:
            return session is not None
        
    except Exception as e:
        logger.debug(f"Database connection test failed: {e}")
        return False


def cleanup_database() -> None:
    """
    Clean up database connections and resources.
    
    Properly shuts down database connections and cleans up resources. Should be
    called during application shutdown for clean resource management. This function
    is safe to call regardless of database availability.
    
    Examples:
        Application shutdown:
            ```python
            from odor_plume_nav.db import cleanup_database
            
            def shutdown_application():
                # Clean up database resources
                cleanup_database()
                print("Database resources cleaned up")
            ```
        
        Context manager pattern:
            ```python
            from contextlib import contextmanager
            from odor_plume_nav.db import get_session_manager, cleanup_database
            
            @contextmanager
            def database_context():
                try:
                    manager = get_session_manager()
                    yield manager
                finally:
                    cleanup_database()
            ```
    """
    if not _SESSION_MODULE_AVAILABLE:
        return
    
    try:
        # Get session manager and clean up
        manager_func = globals().get('get_session_manager')
        if manager_func:
            manager = manager_func()
            if manager and hasattr(manager, 'close'):
                manager.close()
                logger.debug("Database connections cleaned up")
        
    except Exception as e:
        logger.debug(f"Database cleanup error: {e}")


# Public API exports with comprehensive fallback handling
__all__ = [
    # Core session management
    'DatabaseSessionManager',
    'get_session_manager',
    'get_session',
    'get_default_session',
    
    # Configuration and management
    'DatabaseConfig',
    'configure_database',
    'configure_database_from_config',
    
    # Status and compatibility
    'is_database_enabled',
    'get_database_status',
    'ensure_database_compatibility',
    
    # Connection testing and lifecycle
    'test_database_connection',
    'cleanup_database',
    
    # Exception classes
    'DatabaseConnectionError',
    'DatabaseTransactionError',
    'DatabaseConfigurationError',
    
    # Dependency availability flags
    'SQLALCHEMY_AVAILABLE',
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
        if _SESSION_MODULE_AVAILABLE:
            logger.debug("Database session management infrastructure loaded successfully")
            
            # Check if database is actually configured and enabled
            if is_database_enabled():
                logger.info("Database features are enabled and ready for trajectory storage")
            else:
                logger.debug(
                    "Database infrastructure available but not configured. "
                    "Set DATABASE_URL environment variable to enable database features."
                )
        else:
            logger.debug(
                f"Database session management not available: {_IMPORT_ERROR_MESSAGE}. "
                "System will operate in file-only mode with zero performance impact."
            )
        
        # Validate public API completeness
        missing_exports = []
        for export_name in __all__:
            if export_name not in globals() or globals()[export_name] is None:
                # Skip optional class exports that may not be available
                if export_name not in ['DatabaseSessionManager', 'DatabaseConfig', 
                                     'DatabaseConnectionError', 'DatabaseTransactionError', 
                                     'DatabaseConfigurationError']:
                    missing_exports.append(export_name)
        
        if missing_exports:
            logger.warning(f"Missing database API exports: {missing_exports}")
        
    except Exception as e:
        # Initialization errors should not prevent module loading
        logger.debug(f"Database module initialization completed with warnings: {e}")


# Perform module initialization
_initialize_database_module()


# Example usage and integration patterns for development reference
if __name__ == "__main__":
    """
    Example usage patterns for database session management.
    
    These examples demonstrate the optional nature of database features
    and proper integration patterns for research workflows using the
    unified odor_plume_nav package structure.
    """
    
    print("Odor Plume Navigation Database Module Status")
    print("=" * 55)
    
    # Display comprehensive status information
    status = get_database_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\nTesting database functionality...")
    
    # Test basic database availability
    if is_database_enabled():
        print("✓ Database features are enabled and ready")
        
        # Test database connection
        if test_database_connection():
            print("✓ Database connection successful")
            
            # Example session usage with the unified package
            from odor_plume_nav.db import get_session_manager
            
            manager = get_session_manager()
            with manager.get_session() as session:
                if session:
                    print("✓ Database session created successfully")
                    print("  Ready for trajectory recording and experiment metadata storage")
                else:
                    print("⚠ Database session creation failed")
        else:
            print("⚠ Database connection failed")
    
    else:
        print("ℹ Database features not enabled - file-based mode active")
        print("  This is expected for default configuration and ensures zero performance impact")
        print("  Set DATABASE_URL environment variable to enable optional database features")
    
    # Test compatibility function
    if ensure_database_compatibility():
        print("✓ Database system fully compatible and ready for experiment persistence")
    else:
        print("ℹ Database system not configured - continuing with optimized file-based storage")
    
    # Demonstrate configuration from environment
    import os
    if 'DATABASE_URL' in os.environ:
        print(f"\n✓ DATABASE_URL detected: {os.environ['DATABASE_URL'][:20]}...")
    else:
        print(f"\nℹ No DATABASE_URL environment variable found")
        print("  Example: export DATABASE_URL='sqlite:///experiments.db'")
    
    print(f"\nModule initialization complete - odor_plume_nav.db ready for use")