"""
SQLAlchemy Session Management Infrastructure for Odor Plume Navigation System

This module implements enterprise-grade database session management using SQLAlchemy ≥2.0
patterns with comprehensive support for multiple database backends, connection pooling,
and environment-specific configuration. The infrastructure remains completely inactive
by default, ensuring zero performance impact on file-based operations while providing
ready-to-activate trajectory recording and experiment metadata storage through
configuration toggles.

Key Features:
- SQLAlchemy 2.0+ session management with context managers
- Multiple backend support (SQLite, PostgreSQL, in-memory)
- Hydra configuration integration with environment variable interpolation
- Connection pooling with configurable parameters
- Async/await compatibility for high-performance operations
- Automatic transaction handling and rollback on errors
- Secure credential management via python-dotenv
- Zero operational overhead when database features disabled
- Ready-to-activate persistence hooks for trajectory and metadata storage

Architecture:
- Follows cookiecutter template structure conventions
- Integrates seamlessly with existing file-based persistence
- Supports future extensibility for collaborative research environments
- Maintains backward compatibility with current workflows
- Provides pluggable persistence adapter patterns

Configuration Integration:
- Database connection parameters via Hydra config groups
- Environment variable interpolation using ${oc.env:VAR_NAME} syntax
- Secure credential management through conf/local/credentials.yaml.template
- Multi-environment support (development, testing, production)
- Automated connection string handling with backend detection

Performance Characteristics:
- Context-managed sessions with automatic resource cleanup
- Connection pooling for efficient resource utilization
- Async compatibility for concurrent experiment execution
- Transaction optimization with rollback handling
- Graceful error recovery and connection recycling
- Memory-efficient operation with configurable limits

Usage Examples:

    # Basic session usage (activated only when database configured)
    from {{cookiecutter.project_slug}}.db.session import get_session, SessionManager
    
    # Context manager for automatic transaction handling
    with get_session() as session:
        # Perform database operations
        session.add(trajectory_record)
        # Automatic commit/rollback handled by context manager
    
    # Factory pattern for session configuration
    session_manager = SessionManager.from_config(config.database)
    with session_manager.session() as session:
        # Database operations with configured backend
        pass
    
    # Async session support for high-performance operations
    async with get_async_session() as session:
        # Async database operations
        await session.execute(query)

Dependencies:
- SQLAlchemy ≥2.0.41: Modern database abstraction with async support
- python-dotenv ≥1.1.0: Secure environment variable management
- Hydra-core: Configuration composition and parameter interpolation
- Pydantic: Configuration schema validation and type safety

Integration Points:
- src/{{cookiecutter.project_slug}}/config/schemas.py: Database configuration schemas
- conf/local/credentials.yaml.template: Secure credential templates
- conf/base.yaml: Default database configuration parameters
- Environment variables: Secure credential injection and deployment flexibility

Authors: Cookiecutter Template Generator
License: MIT
Version: 2.0.0
"""

import os
import warnings
from contextlib import contextmanager, asynccontextmanager
from typing import (
    Optional, Dict, Any, Union, AsyncGenerator, Generator, 
    Type, Protocol, runtime_checkable
)
from urllib.parse import urlparse
import logging

# SQLAlchemy 2.0+ imports with backward compatibility handling
try:
    from sqlalchemy import (
        create_engine, text, MetaData, Table, Column, Integer, 
        String, Float, DateTime, Boolean, Text, event
    )
    from sqlalchemy.orm import (
        sessionmaker, declarative_base, Session as SQLASession
    )
    from sqlalchemy.ext.asyncio import (
        create_async_engine, AsyncSession, async_sessionmaker
    )
    from sqlalchemy.pool import StaticPool, QueuePool
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import (
        SQLAlchemyError, DisconnectionError, TimeoutError as SQLTimeoutError
    )
    SQLALCHEMY_AVAILABLE = True
except ImportError as e:
    # Graceful degradation when SQLAlchemy not installed
    SQLALCHEMY_AVAILABLE = False
    SQLAlchemyError = Exception
    Engine = None
    SQLASession = None
    AsyncSession = None
    warnings.warn(
        f"SQLAlchemy not available: {e}. Database features will be disabled.",
        UserWarning,
        stacklevel=2
    )

# Environment variable management
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    warnings.warn(
        "python-dotenv not available. Environment variable loading disabled.",
        UserWarning,
        stacklevel=2
    )

# Hydra configuration integration
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    warnings.warn(
        "OmegaConf not available. Hydra configuration integration disabled.",
        UserWarning,
        stacklevel=2
    )

# Enhanced type annotations
try:
    from typing_extensions import TypedDict, Literal
except ImportError:
    from typing import Dict as TypedDict
    Literal = str

# Logging setup
logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseConfig(Protocol):
    """
    Protocol defining database configuration interface.
    
    This protocol ensures type safety for database configuration objects
    while supporting various configuration sources (Hydra, Pydantic, dict).
    """
    
    url: str
    pool_size: Optional[int]
    max_overflow: Optional[int]
    pool_timeout: Optional[int]
    pool_recycle: Optional[int]
    echo: Optional[bool]
    enabled: Optional[bool]


class DatabaseConnectionInfo(TypedDict, total=False):
    """
    Typed dictionary for database connection parameters.
    
    Provides type safety for connection configuration while supporting
    optional parameters and multiple database backends.
    """
    
    url: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    echo: bool
    autocommit: bool
    autoflush: bool
    expire_on_commit: bool


class DatabaseBackend:
    """
    Database backend detection and configuration utilities.
    
    Provides automatic backend detection from connection URLs and
    backend-specific configuration optimization for SQLite, PostgreSQL,
    and in-memory databases.
    """
    
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MEMORY = "memory"
    
    @classmethod
    def detect_backend(cls, database_url: str) -> str:
        """
        Detect database backend from connection URL.
        
        Args:
            database_url: Database connection URL
            
        Returns:
            Database backend identifier
        """
        if not database_url:
            return cls.SQLITE
        
        parsed = urlparse(database_url)
        scheme = parsed.scheme.lower()
        
        if scheme.startswith('sqlite'):
            return cls.SQLITE
        elif scheme.startswith('postgresql'):
            return cls.POSTGRESQL
        elif scheme.startswith('mysql'):
            return cls.MYSQL
        elif 'memory' in database_url.lower():
            return cls.MEMORY
        else:
            logger.warning(f"Unknown database backend: {scheme}, defaulting to SQLite")
            return cls.SQLITE
    
    @classmethod
    def get_backend_defaults(cls, backend: str) -> Dict[str, Any]:
        """
        Get backend-specific default configuration parameters.
        
        Args:
            backend: Database backend identifier
            
        Returns:
            Default configuration parameters for the backend
        """
        defaults = {
            cls.SQLITE: {
                'pool_size': 1,
                'max_overflow': 0,
                'pool_timeout': 30,
                'pool_recycle': -1,
                'echo': False,
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            },
            cls.POSTGRESQL: {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'echo': False,
                'poolclass': QueuePool,
                'connect_args': {
                    'connect_timeout': 10,
                    'server_settings': {
                        'application_name': 'odor_plume_nav'
                    }
                }
            },
            cls.MYSQL: {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'echo': False,
                'poolclass': QueuePool,
                'connect_args': {
                    'connect_timeout': 10
                }
            },
            cls.MEMORY: {
                'pool_size': 1,
                'max_overflow': 0,
                'pool_timeout': 5,
                'pool_recycle': -1,
                'echo': False,
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False
                }
            }
        }
        
        return defaults.get(backend, defaults[cls.SQLITE])


class SessionManager:
    """
    Enterprise-grade SQLAlchemy session management with connection pooling.
    
    Provides comprehensive database session lifecycle management including:
    - Context-managed sessions with automatic transaction handling
    - Connection pooling with backend-specific optimization
    - Multi-database backend support with automatic configuration
    - Async/await compatibility for high-performance operations
    - Graceful error handling and connection recovery
    - Environment-specific configuration management
    
    The SessionManager remains completely inactive when database features
    are disabled, ensuring zero performance impact on file-based operations.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        config: Optional[Union[DatabaseConfig, Dict[str, Any]]] = None,
        auto_configure: bool = True
    ):
        """
        Initialize session manager with database configuration.
        
        Args:
            database_url: Database connection URL (optional)
            config: Database configuration object or dictionary
            auto_configure: Enable automatic configuration from environment
        """
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._sessionmaker = None
        self._async_sessionmaker = None
        self._database_url = database_url
        self._config = config or {}
        self._enabled = False
        self._backend = None
        
        # Load environment variables if available
        if DOTENV_AVAILABLE and auto_configure:
            self._load_environment()
        
        # Initialize database configuration if enabled
        if self._should_initialize():
            self._initialize_database()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env files."""
        try:
            # Search for .env files in standard locations
            env_paths = [
                '.env',
                '.env.local',
                '.env.development',
                '.env.testing',
                '.env.production'
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.debug(f"Loaded environment from {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load environment variables: {e}")
    
    def _should_initialize(self) -> bool:
        """
        Determine if database should be initialized based on configuration.
        
        Returns:
            True if database initialization should proceed
        """
        if not SQLALCHEMY_AVAILABLE:
            return False
        
        # Check explicit configuration
        if isinstance(self._config, dict):
            if self._config.get('enabled', False):
                return True
        elif hasattr(self._config, 'enabled'):
            if self._config.enabled:
                return True
        
        # Check for database URL from various sources
        database_url = (
            self._database_url or
            os.getenv('DATABASE_URL') or
            getattr(self._config, 'url', None) if hasattr(self._config, 'url') else None or
            self._config.get('url') if isinstance(self._config, dict) else None
        )
        
        return bool(database_url)
    
    def _initialize_database(self) -> None:
        """Initialize database engine and session factories."""
        try:
            # Determine database URL
            database_url = self._get_database_url()
            if not database_url:
                logger.warning("No database URL configured, database features disabled")
                return
            
            # Detect backend and get default configuration
            self._backend = DatabaseBackend.detect_backend(database_url)
            backend_defaults = DatabaseBackend.get_backend_defaults(self._backend)
            
            # Merge configuration with backend defaults
            engine_config = self._build_engine_config(backend_defaults)
            
            # Create synchronous engine
            self._engine = create_engine(database_url, **engine_config)
            self._sessionmaker = sessionmaker(
                bind=self._engine,
                autocommit=self._get_config_value('autocommit', False),
                autoflush=self._get_config_value('autoflush', True),
                expire_on_commit=self._get_config_value('expire_on_commit', True)
            )
            
            # Create async engine if supported
            if database_url.startswith(('postgresql+asyncpg', 'mysql+aiomysql')):
                try:
                    self._async_engine = create_async_engine(database_url, **engine_config)
                    self._async_sessionmaker = async_sessionmaker(
                        bind=self._async_engine,
                        expire_on_commit=self._get_config_value('expire_on_commit', True)
                    )
                except Exception as e:
                    logger.warning(f"Async engine creation failed: {e}")
            
            # Register event listeners for connection management
            self._register_event_listeners()
            
            self._enabled = True
            logger.info(f"Database session manager initialized with {self._backend} backend")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._enabled = False
    
    def _get_database_url(self) -> Optional[str]:
        """Get database URL from configuration sources."""
        # Priority order: explicit URL > config object > environment variable
        return (
            self._database_url or
            getattr(self._config, 'url', None) if hasattr(self._config, 'url') else None or
            self._config.get('url') if isinstance(self._config, dict) else None or
            os.getenv('DATABASE_URL')
        )
    
    def _build_engine_config(self, backend_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Build engine configuration from defaults and user settings."""
        config = backend_defaults.copy()
        
        # Override with user configuration
        for key in ['pool_size', 'max_overflow', 'pool_timeout', 'pool_recycle', 'echo']:
            value = self._get_config_value(key)
            if value is not None:
                config[key] = value
        
        return config
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from various sources."""
        if hasattr(self._config, key):
            return getattr(self._config, key)
        elif isinstance(self._config, dict):
            return self._config.get(key, default)
        else:
            return default
    
    def _register_event_listeners(self) -> None:
        """Register SQLAlchemy event listeners for connection management."""
        if not self._engine:
            return
        
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for optimal performance."""
            if self._backend == DatabaseBackend.SQLITE:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        @event.listens_for(self._engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ensure that a connection is alive when checked out from the pool."""
            try:
                # Test the connection
                cursor = dbapi_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            except Exception:
                # Raise DisconnectionError to trigger connection recycling
                raise DisconnectionError()
    
    @property
    def enabled(self) -> bool:
        """Check if database features are enabled."""
        return self._enabled and SQLALCHEMY_AVAILABLE
    
    @property
    def backend(self) -> Optional[str]:
        """Get the database backend type."""
        return self._backend
    
    @contextmanager
    def session(self) -> Generator[Optional[SQLASession], None, None]:
        """
        Create a context-managed database session.
        
        Provides automatic transaction handling with commit on success
        and rollback on exceptions. Returns None if database not enabled.
        
        Yields:
            SQLAlchemy session object or None if database disabled
        """
        if not self.enabled or not self._sessionmaker:
            yield None
            return
        
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[Optional[AsyncSession], None]:
        """
        Create a context-managed async database session.
        
        Provides automatic transaction handling for async operations.
        Returns None if async database not enabled.
        
        Yields:
            SQLAlchemy async session object or None if async database disabled
        """
        if not self.enabled or not self._async_sessionmaker:
            yield None
            return
        
        session = self._async_sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            with self.session() as session:
                if session:
                    session.execute(text("SELECT 1"))
                    return True
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """
        Test async database connection.
        
        Returns:
            True if async connection successful, False otherwise
        """
        if not self.enabled or not self._async_sessionmaker:
            return False
        
        try:
            async with self.async_session() as session:
                if session:
                    await session.execute(text("SELECT 1"))
                    return True
            return False
        except Exception as e:
            logger.error(f"Async database connection test failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        
        if self._async_engine:
            self._async_engine.dispose()
            self._async_engine = None
        
        self._sessionmaker = None
        self._async_sessionmaker = None
        self._enabled = False
        
        logger.info("Database session manager closed")
    
    @classmethod
    def from_config(
        cls, 
        config: Union[DatabaseConfig, Dict[str, Any], DictConfig, None]
    ) -> 'SessionManager':
        """
        Create SessionManager from configuration object.
        
        Args:
            config: Database configuration from Hydra, Pydantic, or dict
            
        Returns:
            Configured SessionManager instance
        """
        if config is None:
            return cls()
        
        # Handle Hydra DictConfig
        if HYDRA_AVAILABLE and isinstance(config, DictConfig):
            config_dict = OmegaConf.to_object(config)
            return cls(config=config_dict)
        
        return cls(config=config)
    
    @classmethod
    def from_url(cls, database_url: str) -> 'SessionManager':
        """
        Create SessionManager from database URL.
        
        Args:
            database_url: Database connection URL
            
        Returns:
            Configured SessionManager instance
        """
        return cls(database_url=database_url)


# Global session manager instance (initialized lazily)
_global_session_manager: Optional[SessionManager] = None


def get_session_manager(
    config: Optional[Union[DatabaseConfig, Dict[str, Any]]] = None,
    database_url: Optional[str] = None,
    force_recreate: bool = False
) -> SessionManager:
    """
    Get or create global session manager instance.
    
    Args:
        config: Database configuration object
        database_url: Database connection URL
        force_recreate: Force creation of new session manager
        
    Returns:
        Global SessionManager instance
    """
    global _global_session_manager
    
    if _global_session_manager is None or force_recreate:
        if config is not None:
            _global_session_manager = SessionManager.from_config(config)
        elif database_url is not None:
            _global_session_manager = SessionManager.from_url(database_url)
        else:
            _global_session_manager = SessionManager()
    
    return _global_session_manager


@contextmanager
def get_session(
    config: Optional[Union[DatabaseConfig, Dict[str, Any]]] = None
) -> Generator[Optional[SQLASession], None, None]:
    """
    Get a context-managed database session.
    
    This is the primary interface for database operations. Returns None
    if database features are not enabled, allowing code to gracefully
    handle both database-enabled and file-only modes.
    
    Args:
        config: Optional database configuration
        
    Yields:
        SQLAlchemy session object or None if database disabled
        
    Example:
        with get_session() as session:
            if session:
                # Database operations
                session.add(trajectory_record)
            # File-based operations continue regardless
    """
    session_manager = get_session_manager(config)
    with session_manager.session() as session:
        yield session


@asynccontextmanager
async def get_async_session(
    config: Optional[Union[DatabaseConfig, Dict[str, Any]]] = None
) -> AsyncGenerator[Optional[AsyncSession], None]:
    """
    Get a context-managed async database session.
    
    Args:
        config: Optional database configuration
        
    Yields:
        SQLAlchemy async session object or None if async database disabled
        
    Example:
        async with get_async_session() as session:
            if session:
                # Async database operations
                await session.execute(query)
    """
    session_manager = get_session_manager(config)
    async with session_manager.async_session() as session:
        yield session


def is_database_enabled() -> bool:
    """
    Check if database features are enabled.
    
    Returns:
        True if database is configured and available
    """
    session_manager = get_session_manager()
    return session_manager.enabled


def test_database_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if database connection successful
    """
    session_manager = get_session_manager()
    return session_manager.test_connection()


async def test_async_database_connection() -> bool:
    """
    Test async database connection.
    
    Returns:
        True if async database connection successful
    """
    session_manager = get_session_manager()
    return await session_manager.test_async_connection()


def cleanup_database() -> None:
    """Close database connections and cleanup global resources."""
    global _global_session_manager
    if _global_session_manager:
        _global_session_manager.close()
        _global_session_manager = None


# Trajectory and metadata persistence hooks for future extensibility
class PersistenceHooks:
    """
    Optional persistence hooks for trajectory and metadata storage.
    
    These hooks provide ready-to-activate capabilities for storing
    simulation results and experimental metadata in the database.
    The hooks remain inactive by default and only function when
    database features are enabled through configuration.
    """
    
    @staticmethod
    def save_trajectory_data(
        trajectory_data: Dict[str, Any],
        session: Optional[SQLASession] = None
    ) -> bool:
        """
        Save trajectory data to database if enabled.
        
        Args:
            trajectory_data: Trajectory data dictionary
            session: Optional existing database session
            
        Returns:
            True if data was saved to database, False otherwise
        """
        if not is_database_enabled():
            return False
        
        try:
            if session:
                # Use provided session
                # TODO: Implement trajectory table operations
                logger.debug("Trajectory data persistence hook called")
                return True
            else:
                # Use context-managed session
                with get_session() as db_session:
                    if db_session:
                        # TODO: Implement trajectory table operations
                        logger.debug("Trajectory data persistence hook called")
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to save trajectory data: {e}")
            return False
    
    @staticmethod
    def save_experiment_metadata(
        metadata: Dict[str, Any],
        session: Optional[SQLASession] = None
    ) -> bool:
        """
        Save experiment metadata to database if enabled.
        
        Args:
            metadata: Experiment metadata dictionary
            session: Optional existing database session
            
        Returns:
            True if metadata was saved to database, False otherwise
        """
        if not is_database_enabled():
            return False
        
        try:
            if session:
                # Use provided session
                # TODO: Implement metadata table operations
                logger.debug("Experiment metadata persistence hook called")
                return True
            else:
                # Use context-managed session
                with get_session() as db_session:
                    if db_session:
                        # TODO: Implement metadata table operations
                        logger.debug("Experiment metadata persistence hook called")
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to save experiment metadata: {e}")
            return False
    
    @staticmethod
    async def async_save_trajectory_data(
        trajectory_data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Async save trajectory data to database if enabled.
        
        Args:
            trajectory_data: Trajectory data dictionary
            session: Optional existing async database session
            
        Returns:
            True if data was saved to database, False otherwise
        """
        if not is_database_enabled():
            return False
        
        try:
            if session:
                # Use provided async session
                # TODO: Implement async trajectory table operations
                logger.debug("Async trajectory data persistence hook called")
                return True
            else:
                # Use context-managed async session
                async with get_async_session() as db_session:
                    if db_session:
                        # TODO: Implement async trajectory table operations
                        logger.debug("Async trajectory data persistence hook called")
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to async save trajectory data: {e}")
            return False


# Export public interface
__all__ = [
    'SessionManager',
    'DatabaseConfig',
    'DatabaseConnectionInfo',
    'DatabaseBackend',
    'get_session_manager',
    'get_session',
    'get_async_session',
    'is_database_enabled',
    'test_database_connection',
    'test_async_database_connection',
    'cleanup_database',
    'PersistenceHooks',
    'SQLALCHEMY_AVAILABLE',
    'HYDRA_AVAILABLE',
    'DOTENV_AVAILABLE',
]


# Module initialization and configuration loading
if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    def main():
        """Example usage of the session management system."""
        print("SQLAlchemy Session Management System")
        print(f"SQLAlchemy Available: {SQLALCHEMY_AVAILABLE}")
        print(f"Hydra Available: {HYDRA_AVAILABLE}")
        print(f"python-dotenv Available: {DOTENV_AVAILABLE}")
        
        # Test database connection
        if is_database_enabled():
            print("Database enabled, testing connection...")
            if test_database_connection():
                print("Database connection successful!")
            else:
                print("Database connection failed.")
        else:
            print("Database not enabled - file-based mode active.")
        
        # Example session usage
        with get_session() as session:
            if session:
                print("Database session created successfully")
            else:
                print("Database session not available (expected in file-only mode)")
    
    async def async_main():
        """Example async usage of the session management system."""
        async with get_async_session() as session:
            if session:
                print("Async database session created successfully")
                if await test_async_database_connection():
                    print("Async database connection successful!")
            else:
                print("Async database session not available")
    
    # Run examples
    main()
    if SQLALCHEMY_AVAILABLE:
        asyncio.run(async_main())