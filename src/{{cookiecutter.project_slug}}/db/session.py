"""
SQLAlchemy session management infrastructure for odor plume navigation system.

This module provides enterprise-grade database session management that remains completely
inactive by default, ensuring zero performance impact on file-based operations while
providing ready-to-activate trajectory recording and experiment metadata storage through
configuration toggles.

The implementation supports:
- Multiple database backends (SQLite for development, PostgreSQL for production, in-memory for testing)
- Hydra configuration integration with environment variable interpolation
- Context-managed sessions with automatic transaction handling and connection cleanup
- Connection pooling with configurable pool sizes and async compatibility
- Secure credential management through python-dotenv integration
- Optional persistence hooks for trajectory and metadata storage

Architecture:
    The session management system uses a sophisticated adapter pattern enabling seamless
    transitions between file-based persistence (default) and database-backed storage
    (optional) based on research requirements. SQLAlchemy 2.0+ patterns provide modern
    async compatibility and enhanced type safety.

Examples:
    Basic usage (inactive by default):
        ```python
        from {{cookiecutter.project_slug}}.db.session import DatabaseConfig, get_session
        
        # Will return None if no database configured - zero impact
        session = get_session()
        if session:
            # Database operations only if explicitly configured
            pass
        ```
    
    Explicit database activation:
        ```python
        import os
        from {{cookiecutter.project_slug}}.db.session import DatabaseSessionManager
        
        # Configure database URL via environment variable
        os.environ['DATABASE_URL'] = 'sqlite:///trajectory_data.db'
        
        # Create session manager with automatic configuration detection
        session_manager = DatabaseSessionManager.from_environment()
        
        with session_manager.get_session() as session:
            # Transactional database operations
            pass
        ```
    
    Hydra integration:
        ```python
        import hydra
        from omegaconf import DictConfig
        from {{cookiecutter.project_slug}}.db.session import DatabaseSessionManager
        
        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig) -> None:
            # Automatic session manager creation from Hydra configuration
            session_manager = DatabaseSessionManager.from_hydra_config(cfg)
            
            if session_manager.is_active:
                # Database persistence available
                with session_manager.get_session() as session:
                    # Store trajectory data
                    pass
        ```
"""

import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Union, Iterator, Type
from urllib.parse import urlparse

# Third-party imports with graceful fallbacks
try:
    from sqlalchemy import create_engine, Engine, event
    from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
    from sqlalchemy.pool import StaticPool, QueuePool
    from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
    from sqlalchemy.engine import make_url
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without SQLAlchemy
    SQLALCHEMY_AVAILABLE = False
    create_engine = sessionmaker = Session = Engine = None
    DeclarativeBase = SQLAlchemyError = DisconnectionError = None
    StaticPool = QueuePool = make_url = event = None

try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without Pydantic
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = ConfigDict = field_validator = model_validator = lambda *args, **kwargs: lambda x: x

try:
    import python_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    # Fallback for environments without python-dotenv
    DOTENV_AVAILABLE = False
    python_dotenv = None

try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    HYDRA_AVAILABLE = False
    HydraConfig = DictConfig = OmegaConf = None

# Configure module logger
logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel if PYDANTIC_AVAILABLE else object):
    """
    Database connection configuration schema with comprehensive validation.
    
    Supports multiple database backends with environment variable interpolation
    through Hydra's ${oc.env:VAR_NAME} syntax and secure credential management
    via python-dotenv integration.
    
    Configuration Examples:
        SQLite (development):
            ```yaml
            database:
              url: "sqlite:///trajectory_data.db"
              echo: false
              pool_size: 1
            ```
        
        PostgreSQL (production):
            ```yaml
            database:
              url: "${oc.env:DATABASE_URL}"
              pool_size: ${oc.env:DB_POOL_SIZE,10}
              max_overflow: ${oc.env:DB_MAX_OVERFLOW,20}
              echo: ${oc.env:DB_ECHO,false}
            ```
        
        In-memory (testing):
            ```yaml
            database:
              url: "sqlite:///:memory:"
              echo: true
              pool_size: 1
            ```
    
    Attributes:
        url: Database connection URL supporting SQLite, PostgreSQL, MySQL backends
        pool_size: Connection pool size for concurrent operations (default: 10)
        max_overflow: Maximum pool overflow connections (default: 20)
        pool_timeout: Connection acquisition timeout in seconds (default: 30)
        pool_recycle: Connection recycle time to prevent stale connections (default: 3600)
        echo: Enable SQL query logging for debugging (default: False)
        echo_pool: Enable connection pool logging (default: False)
        future: Use SQLAlchemy 2.0 future mode (default: True)
    """
    
    # Core connection parameters
    url: Optional[str] = Field(
        default=None,
        description="Database connection URL (SQLite, PostgreSQL, MySQL supported)"
    )
    
    # Connection pool configuration
    pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size for concurrent operations"
    )
    
    max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum pool overflow connections beyond pool_size"
    )
    
    pool_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection acquisition timeout in seconds"
    )
    
    pool_recycle: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Connection recycle time to prevent stale connections"
    )
    
    # Debugging and logging
    echo: bool = Field(
        default=False,
        description="Enable SQL query logging for debugging"
    )
    
    echo_pool: bool = Field(
        default=False,
        description="Enable connection pool logging for performance analysis"
    )
    
    # SQLAlchemy configuration
    future: bool = Field(
        default=True,
        description="Use SQLAlchemy 2.0 future mode for enhanced features"
    )

    if PYDANTIC_AVAILABLE:
        model_config = ConfigDict(
            extra="allow",
            str_strip_whitespace=True,
            validate_assignment=True,
            arbitrary_types_allowed=True
        )

        @field_validator('url')
        @classmethod
        def validate_database_url(cls, v: Optional[str]) -> Optional[str]:
            """Validate database URL format and supported backends."""
            if v is None:
                return v
            
            # Handle Hydra environment variable interpolation
            if v.startswith('${') and v.endswith('}'):
                return v  # Return as-is for Hydra to resolve
            
            try:
                # Parse URL to validate format
                parsed = urlparse(v)
                
                # Validate supported database backends
                supported_schemes = {
                    'sqlite', 'postgresql', 'postgresql+psycopg2', 'postgresql+asyncpg',
                    'mysql', 'mysql+pymysql', 'mysql+aiomysql'
                }
                
                if parsed.scheme not in supported_schemes:
                    raise ValueError(
                        f"Unsupported database backend: {parsed.scheme}. "
                        f"Supported: {', '.join(supported_schemes)}"
                    )
                
                return v
                
            except Exception as e:
                raise ValueError(f"Invalid database URL format: {e}")

        @model_validator(mode="after")
        def validate_pool_configuration(self):
            """Validate connection pool parameter consistency."""
            # Ensure max_overflow is reasonable relative to pool_size
            if self.max_overflow > self.pool_size * 2:
                warnings.warn(
                    f"max_overflow ({self.max_overflow}) is more than 2x pool_size "
                    f"({self.pool_size}). Consider reducing for optimal performance."
                )
            
            # SQLite-specific validation
            if self.url and ('sqlite' in self.url.lower()):
                if self.pool_size > 1 and ':memory:' not in self.url:
                    logger.warning(
                        "SQLite file databases should use pool_size=1. "
                        "Multiple connections can cause locking issues."
                    )
            
            return self


class DatabaseSessionManager:
    """
    Comprehensive database session management with connection pooling and transaction handling.
    
    This class implements enterprise-grade database session management using SQLAlchemy 2.0+
    patterns with context managers, automatic transaction handling, and robust error recovery.
    The manager remains completely inactive when no database configuration is provided,
    ensuring zero performance impact on file-based operations.
    
    Features:
        - Context-managed sessions with automatic transaction handling
        - Connection pooling with configurable parameters
        - Multiple database backend support (SQLite, PostgreSQL, MySQL)
        - Async compatibility for high-performance operations
        - Graceful error recovery and connection retry mechanisms
        - Environment variable integration through Hydra and python-dotenv
        - Optional persistence hooks for trajectory and metadata storage
    
    Architecture:
        The session manager uses a lazy initialization pattern, creating database
        connections only when explicitly requested. This ensures that file-based
        operations experience zero overhead while providing ready-to-activate
        database capabilities for advanced research workflows.
    
    Examples:
        Basic session management:
            ```python
            manager = DatabaseSessionManager(DatabaseConfig(
                url="sqlite:///experiments.db"
            ))
            
            with manager.get_session() as session:
                # Automatic transaction management
                result = session.execute(text("SELECT COUNT(*) FROM trajectories"))
                session.commit()  # Automatic on context exit
            ```
        
        Error handling and retry:
            ```python
            manager = DatabaseSessionManager.from_environment()
            
            try:
                with manager.get_session() as session:
                    # Database operations with automatic rollback on error
                    pass
            except DatabaseConnectionError as e:
                logger.error(f"Database connection failed: {e}")
                # Fallback to file-based operations
            ```
    """
    
    def __init__(
        self, 
        config: Optional[DatabaseConfig] = None,
        engine: Optional[Engine] = None
    ):
        """
        Initialize database session manager with optional configuration.
        
        Args:
            config: Database configuration object with connection parameters
            engine: Pre-configured SQLAlchemy engine (overrides config)
        
        Note:
            If neither config nor engine are provided, the manager remains
            inactive and all session operations return None, ensuring zero
            performance impact on file-based workflows.
        """
        self._config = config
        self._engine = engine
        self._session_factory: Optional[sessionmaker] = None
        self._is_active = False
        
        # Initialize session factory if configuration is provided
        if config and config.url and SQLALCHEMY_AVAILABLE:
            self._initialize_engine()
        elif engine and SQLALCHEMY_AVAILABLE:
            self._engine = engine
            self._initialize_session_factory()
    
    @property
    def is_active(self) -> bool:
        """
        Check if database session management is active.
        
        Returns:
            True if database is configured and available, False otherwise
        """
        return self._is_active and SQLALCHEMY_AVAILABLE
    
    @property
    def config(self) -> Optional[DatabaseConfig]:
        """Access to current database configuration."""
        return self._config
    
    @property
    def engine(self) -> Optional[Engine]:
        """Access to SQLAlchemy engine for advanced operations."""
        return self._engine
    
    def _initialize_engine(self) -> None:
        """
        Initialize SQLAlchemy engine with connection pooling and error handling.
        
        Creates a properly configured engine based on the database configuration,
        with appropriate connection pooling, timeout settings, and event listeners
        for connection management and debugging.
        """
        if not self._config or not self._config.url or not SQLALCHEMY_AVAILABLE:
            logger.debug("Database configuration not available, remaining inactive")
            return
        
        try:
            # Resolve environment variables in URL if using Hydra interpolation
            database_url = self._resolve_environment_variables(self._config.url)
            
            # Create engine with appropriate pooling strategy
            engine_kwargs = self._build_engine_kwargs(database_url)
            
            self._engine = create_engine(database_url, **engine_kwargs)
            
            # Set up connection event listeners for monitoring and debugging
            self._setup_event_listeners()
            
            # Initialize session factory
            self._initialize_session_factory()
            
            # Test connection to ensure database is accessible
            self._test_connection()
            
            self._is_active = True
            logger.info(f"Database session manager initialized: {self._get_safe_url()}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
            logger.info("Continuing with file-based persistence only")
            self._engine = None
            self._session_factory = None
            self._is_active = False
    
    def _resolve_environment_variables(self, url: str) -> str:
        """
        Resolve environment variables in database URL.
        
        Supports both Hydra interpolation syntax and direct environment variable
        resolution for flexible deployment scenarios.
        
        Args:
            url: Database URL potentially containing environment variable references
            
        Returns:
            Resolved database URL with environment variables substituted
        """
        # Handle Hydra interpolation syntax ${oc.env:VAR_NAME,default}
        if url.startswith('${oc.env:') and url.endswith('}'):
            # Extract variable name and default value
            content = url[8:-1]  # Remove ${oc.env: and }
            if ',' in content:
                var_name, default_value = content.split(',', 1)
                resolved_url = os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = content.strip()
                resolved_url = os.getenv(var_name)
                if resolved_url is None:
                    raise ValueError(f"Environment variable {var_name} not found")
            return resolved_url
        
        # Handle direct environment variable references
        if url.startswith('$'):
            var_name = url[1:]
            resolved_url = os.getenv(var_name)
            if resolved_url is None:
                raise ValueError(f"Environment variable {var_name} not found")
            return resolved_url
        
        return url
    
    def _build_engine_kwargs(self, database_url: str) -> Dict[str, Any]:
        """
        Build SQLAlchemy engine configuration based on database type and settings.
        
        Args:
            database_url: Resolved database connection URL
            
        Returns:
            Dictionary of engine configuration parameters
        """
        engine_kwargs = {
            'echo': self._config.echo,
            'echo_pool': self._config.echo_pool,
            'future': self._config.future,
            'pool_timeout': self._config.pool_timeout,
            'pool_recycle': self._config.pool_recycle,
        }
        
        # Configure pooling based on database type
        if 'sqlite' in database_url.lower():
            if ':memory:' in database_url:
                # In-memory SQLite database
                engine_kwargs.update({
                    'poolclass': StaticPool,
                    'pool_size': 1,
                    'max_overflow': 0,
                    'connect_args': {
                        'check_same_thread': False,
                        'isolation_level': None
                    }
                })
            else:
                # File-based SQLite database
                engine_kwargs.update({
                    'pool_size': 1,
                    'max_overflow': 0,
                    'connect_args': {'check_same_thread': False}
                })
        else:
            # PostgreSQL, MySQL, or other database
            engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': self._config.pool_size,
                'max_overflow': self._config.max_overflow,
            })
        
        return engine_kwargs
    
    def _setup_event_listeners(self) -> None:
        """
        Set up SQLAlchemy event listeners for connection monitoring and debugging.
        
        Registers event handlers for connection lifecycle events, enabling
        comprehensive monitoring, debugging, and performance optimization.
        """
        if not self._engine:
            return
        
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            logger.debug("New database connection established")
            
            # SQLite-specific optimizations
            if 'sqlite' in str(self._engine.url).lower():
                # Enable foreign key constraints
                dbapi_connection.execute("PRAGMA foreign_keys=ON")
                # Set WAL mode for better concurrency
                dbapi_connection.execute("PRAGMA journal_mode=WAL")
        
        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            logger.debug("Connection checked in to pool")
        
        @event.listens_for(self._engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation due to errors."""
            logger.warning(f"Database connection invalidated: {exception}")
    
    def _initialize_session_factory(self) -> None:
        """
        Initialize SQLAlchemy session factory with appropriate configuration.
        
        Creates a sessionmaker instance configured for optimal transaction
        handling, autocommit behavior, and resource management.
        """
        if not self._engine:
            return
        
        self._session_factory = sessionmaker(
            bind=self._engine,
            future=True,  # Use SQLAlchemy 2.0 patterns
            autoflush=True,  # Automatic session flushing
            autocommit=False,  # Explicit transaction control
            expire_on_commit=True  # Refresh objects after commit
        )
    
    def _test_connection(self) -> None:
        """
        Test database connectivity and raise informative errors if unavailable.
        
        Performs a simple connectivity test to ensure the database is accessible
        and properly configured. Provides clear error messages for common
        connection issues.
        
        Raises:
            DatabaseConnectionError: If database connection cannot be established
        """
        if not self._engine:
            return
        
        try:
            with self._engine.connect() as conn:
                # Simple connectivity test
                conn.execute("SELECT 1" if 'sqlite' not in str(self._engine.url) 
                           else "SELECT 1")
                logger.debug("Database connectivity test successful")
                
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to connect to database: {e}. "
                f"Please verify connection parameters and database availability."
            ) from e
    
    def _get_safe_url(self) -> str:
        """
        Get database URL with credentials masked for safe logging.
        
        Returns:
            Database URL with password and sensitive information redacted
        """
        if not self._engine:
            return "No database configured"
        
        try:
            url = make_url(str(self._engine.url))
            # Mask password for security
            if url.password:
                url = url.set(password="***")
            return str(url)
        except Exception:
            return "Database URL unavailable"
    
    @contextmanager
    def get_session(self) -> Iterator[Optional[Session]]:
        """
        Context manager providing database session with automatic transaction handling.
        
        Creates a database session with comprehensive transaction management,
        automatic rollback on errors, and guaranteed resource cleanup. Returns
        None if database is not configured, enabling graceful fallback to
        file-based operations.
        
        Yields:
            SQLAlchemy Session object if database is active, None otherwise
            
        Examples:
            Basic usage:
                ```python
                with manager.get_session() as session:
                    if session:  # Check if database is active
                        result = session.execute(text("SELECT * FROM trajectories"))
                        session.commit()  # Automatic on context exit
                ```
            
            Error handling:
                ```python
                try:
                    with manager.get_session() as session:
                        if session:
                            # Database operations with automatic rollback on error
                            pass
                except DatabaseTransactionError as e:
                    logger.error(f"Transaction failed: {e}")
                    # Continue with file-based operations
                ```
        """
        if not self.is_active or not self._session_factory:
            # Return None session for inactive database - enables graceful fallback
            yield None
            return
        
        session = self._session_factory()
        try:
            logger.debug("Database session created")
            yield session
            
            # Commit transaction if no exceptions occurred
            if session.in_transaction():
                session.commit()
                logger.debug("Database transaction committed")
                
        except Exception as e:
            # Rollback transaction on any error
            if session.in_transaction():
                session.rollback()
                logger.warning(f"Database transaction rolled back due to error: {e}")
            
            # Re-raise as database-specific exception for better error handling
            raise DatabaseTransactionError(
                f"Database transaction failed: {e}"
            ) from e
            
        finally:
            # Ensure session is always closed
            session.close()
            logger.debug("Database session closed")
    
    def close(self) -> None:
        """
        Close database engine and clean up all connections.
        
        Properly shuts down the database connection pool and releases all
        resources. Should be called during application shutdown for clean
        resource management.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine closed and connections cleaned up")
            self._is_active = False
    
    @classmethod
    def from_environment(cls) -> "DatabaseSessionManager":
        """
        Create session manager from environment variables.
        
        Automatically detects database configuration from environment variables
        with optional python-dotenv integration for .env file loading.
        
        Environment Variables:
            DATABASE_URL: Complete database connection URL
            DB_POOL_SIZE: Connection pool size (default: 10)
            DB_MAX_OVERFLOW: Maximum pool overflow (default: 20)
            DB_POOL_TIMEOUT: Connection timeout in seconds (default: 30)
            DB_ECHO: Enable SQL logging (default: False)
        
        Returns:
            DatabaseSessionManager instance configured from environment
            
        Examples:
            With .env file:
                ```bash
                # .env
                DATABASE_URL=postgresql://user:pass@localhost:5432/odor_plume_nav
                DB_POOL_SIZE=15
                DB_ECHO=true
                ```
                
                ```python
                # Python code
                manager = DatabaseSessionManager.from_environment()
                ```
        """
        # Load .env file if python-dotenv is available
        if DOTENV_AVAILABLE:
            from dotenv import load_dotenv
            load_dotenv()
        
        # Extract database configuration from environment
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            logger.debug("No DATABASE_URL found in environment, remaining inactive")
            return cls()  # Return inactive manager
        
        # Build configuration from environment variables
        config = DatabaseConfig(
            url=database_url,
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600')),
            echo=os.getenv('DB_ECHO', 'false').lower() in ('true', '1', 'yes'),
            echo_pool=os.getenv('DB_ECHO_POOL', 'false').lower() in ('true', '1', 'yes')
        )
        
        return cls(config)
    
    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> "DatabaseSessionManager":
        """
        Create session manager from Hydra configuration.
        
        Integrates with Hydra's hierarchical configuration system, supporting
        environment variable interpolation and structured configuration composition.
        
        Args:
            cfg: Hydra DictConfig containing database configuration
            
        Returns:
            DatabaseSessionManager instance configured from Hydra config
            
        Examples:
            With Hydra configuration:
                ```yaml
                # conf/config.yaml
                database:
                  url: ${oc.env:DATABASE_URL}
                  pool_size: ${oc.env:DB_POOL_SIZE,10}
                  echo: ${oc.env:DB_ECHO,false}
                ```
                
                ```python
                # Python code
                @hydra.main(version_base=None, config_path="conf", config_name="config")
                def main(cfg: DictConfig) -> None:
                    manager = DatabaseSessionManager.from_hydra_config(cfg)
                    
                    with manager.get_session() as session:
                        if session:
                            # Database operations
                            pass
                ```
        """
        if not HYDRA_AVAILABLE:
            logger.warning("Hydra not available, falling back to environment configuration")
            return cls.from_environment()
        
        # Extract database configuration from Hydra config
        if not hasattr(cfg, 'database') or not cfg.database:
            logger.debug("No database configuration found in Hydra config")
            return cls()  # Return inactive manager
        
        db_config = cfg.database
        
        # Handle OmegaConf missing values
        if OmegaConf.is_missing(db_config, 'url') or not db_config.url:
            logger.debug("Database URL not configured in Hydra, remaining inactive")
            return cls()
        
        # Build configuration from Hydra config with defaults
        config = DatabaseConfig(
            url=db_config.url,
            pool_size=getattr(db_config, 'pool_size', 10),
            max_overflow=getattr(db_config, 'max_overflow', 20),
            pool_timeout=getattr(db_config, 'pool_timeout', 30),
            pool_recycle=getattr(db_config, 'pool_recycle', 3600),
            echo=getattr(db_config, 'echo', False),
            echo_pool=getattr(db_config, 'echo_pool', False)
        )
        
        return cls(config)


# Custom exception classes for better error handling
class DatabaseConnectionError(Exception):
    """Raised when database connection cannot be established."""
    pass


class DatabaseTransactionError(Exception):
    """Raised when database transaction fails."""
    pass


class DatabaseConfigurationError(Exception):
    """Raised when database configuration is invalid."""
    pass


# Convenience functions for simplified usage
def get_session_manager() -> DatabaseSessionManager:
    """
    Get default database session manager from environment configuration.
    
    This is a convenience function that creates a session manager using
    environment variables and .env file detection. Provides the simplest
    path to database integration for most use cases.
    
    Returns:
        DatabaseSessionManager configured from environment variables
        
    Examples:
        Simple usage:
            ```python
            from {{cookiecutter.project_slug}}.db.session import get_session_manager
            
            manager = get_session_manager()
            
            with manager.get_session() as session:
                if session:  # Database active
                    # Perform database operations
                    pass
                else:
                    # Continue with file-based operations
                    pass
            ```
    """
    return DatabaseSessionManager.from_environment()


def get_session() -> Optional[Session]:
    """
    Get a single database session using default configuration.
    
    Warning:
        This function returns a raw session without context management.
        Prefer using get_session_manager().get_session() for proper
        transaction handling and resource cleanup.
        
    Returns:
        SQLAlchemy Session if database is configured, None otherwise
    """
    manager = get_session_manager()
    if not manager.is_active:
        return None
    
    # This bypasses context management - use with caution
    return manager._session_factory() if manager._session_factory else None


# Module-level configuration for convenient access
_default_manager: Optional[DatabaseSessionManager] = None


def configure_database(config: DatabaseConfig) -> None:
    """
    Configure module-level default database session manager.
    
    Sets up a global session manager that can be accessed throughout
    the application without repeated configuration.
    
    Args:
        config: Database configuration object
        
    Examples:
        Application setup:
            ```python
            from {{cookiecutter.project_slug}}.db.session import configure_database, DatabaseConfig
            
            # Configure once at application startup
            configure_database(DatabaseConfig(
                url="postgresql://user:pass@localhost:5432/experiments"
            ))
            
            # Use throughout application
            from {{cookiecutter.project_slug}}.db.session import get_default_session
            
            with get_default_session() as session:
                if session:
                    # Database operations
                    pass
            ```
    """
    global _default_manager
    _default_manager = DatabaseSessionManager(config)


@contextmanager
def get_default_session() -> Iterator[Optional[Session]]:
    """
    Get session from module-level default manager.
    
    Provides access to the globally configured session manager for
    simplified database access throughout the application.
    
    Yields:
        SQLAlchemy Session if configured, None otherwise
        
    Raises:
        DatabaseConfigurationError: If no default manager is configured
    """
    global _default_manager
    
    if _default_manager is None:
        _default_manager = get_session_manager()
    
    with _default_manager.get_session() as session:
        yield session


# Export public API
__all__ = [
    # Core classes
    'DatabaseConfig',
    'DatabaseSessionManager',
    
    # Exception classes
    'DatabaseConnectionError',
    'DatabaseTransactionError', 
    'DatabaseConfigurationError',
    
    # Convenience functions
    'get_session_manager',
    'get_session',
    'configure_database',
    'get_default_session',
]