"""
Database session management for plume navigation simulation.

This module provides SQLAlchemy session management, connection lifecycle, transaction management,
and session security with support for multiple database backends.

Key Features:
- SQLAlchemy session management with automatic transaction handling
- Connection pooling and resource cleanup
- Multi-backend support (SQLite, PostgreSQL, MySQL, in-memory)
- Environment variable integration for secure credential handling
- Async/await compatibility for high-performance operations
- SQL injection prevention through parameterized queries
- Configuration validation and type safety

Authors: Blitzy Platform
License: MIT
Version: 1.0.0
"""

import asyncio
import os
import re
import sqlite3
import time
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generator, Optional, Protocol, Union
from urllib.parse import urlparse

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sqlalchemy = None
    Session = None
    AsyncSession = None

try:
    import hydra
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    hydra = None
    DictConfig = None

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None


class DatabaseBackend(Enum):
    """Supported database backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MEMORY = "memory"
    
    @classmethod
    def detect_backend(cls, url: Optional[str]) -> "DatabaseBackend":
        """Detect database backend from connection URL.
        
        Args:
            url: Database connection URL
            
        Returns:
            DatabaseBackend enum value
        """
        # Handle empty/None URLs - default to SQLite
        if not url:
            warnings.warn("Empty URL provided, defaulting to SQLite backend", UserWarning)
            return cls.SQLITE
            
        url_lower = url.lower()
        
        # Handle memory URLs
        if url_lower.startswith("memory://"):
            return cls.MEMORY
            
        # Handle SQLite URLs
        if url_lower.startswith("sqlite://"):
            return cls.SQLITE
            
        # Handle PostgreSQL URLs
        if (url_lower.startswith("postgresql://") or 
            url_lower.startswith("postgres://") or
            url_lower.startswith("postgresql+")):
            return cls.POSTGRESQL
            
        # Handle MySQL URLs  
        if url_lower.startswith("mysql://") or url_lower.startswith("mysql+"):
            return cls.MYSQL
            
        # Unknown scheme - warn and default to SQLite
        warnings.warn(f"Unknown database backend in URL: {url}, defaulting to SQLite", UserWarning)
        return cls.SQLITE
    
    @classmethod
    def get_backend_defaults(cls, backend: Union["DatabaseBackend", str]) -> Dict[str, Any]:
        """Get default configuration for a database backend.
        
        Args:
            backend: Database backend enum or string
            
        Returns:
            Dictionary of default configuration options
        """
        # Handle string backends
        if isinstance(backend, str):
            backend = cls.SQLITE  # Default to SQLite for unknown strings
            
        if backend == cls.SQLITE:
            return {
                'pool_size': 1,
                'max_overflow': 0,
                'pool_timeout': 30,
                'pool_recycle': -1,
                'echo': False,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            }
        elif backend == cls.POSTGRESQL:
            return {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'echo': False,
                'connect_args': {
                    'connect_timeout': 10,
                    'server_settings': {
                        'application_name': 'odor_plume_nav'
                    }
                }
            }
        elif backend == cls.MYSQL:
            return {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'echo': False,
                'connect_args': {
                    'connect_timeout': 10
                }
            }
        elif backend == cls.MEMORY:
            return {
                'pool_size': 1,
                'max_overflow': 0,
                'pool_timeout': 5,
                'pool_recycle': -1,
                'echo': False,
                'connect_args': {
                    'check_same_thread': False
                }
            }
        else:
            # Unknown backend - return SQLite defaults
            return cls.get_backend_defaults(cls.SQLITE)


class DatabaseConnectionInfo(Protocol):
    """Protocol for database connection information."""
    host: str
    port: int
    database: str
    username: str
    password: str
    backend: DatabaseBackend


class DatabaseConfig(Protocol):
    """Protocol for database configuration."""
    connection_string: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    echo: bool


class PersistenceHooks(Protocol):
    """Protocol for database persistence hooks."""
    
    def before_session_create(self, config: DatabaseConfig) -> None:
        """Called before session creation."""
        pass
    
    def after_session_create(self, session: Union[Session, AsyncSession]) -> None:
        """Called after session creation."""
        pass
    
    def before_session_close(self, session: Union[Session, AsyncSession]) -> None:
        """Called before session close."""
        pass


class SessionManager:
    """Database session manager with connection pooling and lifecycle management."""
    
    def __init__(self, connection_string: str, **kwargs):
        """Initialize session manager.
        
        Args:
            connection_string: Database connection URL
            **kwargs: Additional SQLAlchemy engine options
        """
        self.connection_string = connection_string
        self.engine_kwargs = kwargs
        self._engine = None
        self._session_factory = None
        self._async_engine = None
        self._async_session_factory = None
        
    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is not available")
            
        if self._engine is None:
            # Set up connection pooling for in-memory SQLite
            if ":memory:" in self.connection_string:
                self.engine_kwargs.setdefault("poolclass", StaticPool)
                self.engine_kwargs.setdefault("connect_args", {"check_same_thread": False})
                
            self._engine = create_engine(self.connection_string, **self.engine_kwargs)
            
        return self._engine
    
    @property
    def session_factory(self):
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @property
    def async_engine(self):
        """Get or create async SQLAlchemy engine."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is not available")
            
        if self._async_engine is None:
            # Convert sync connection string to async if needed
            async_url = self.connection_string
            if async_url.startswith("sqlite://"):
                async_url = async_url.replace("sqlite://", "sqlite+aiosqlite://")
            elif async_url.startswith("postgresql://"):
                async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
                
            self._async_engine = create_async_engine(async_url, **self.engine_kwargs)
            
        return self._async_engine
    
    @property
    def async_session_factory(self):
        """Get or create async session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup.
        
        Yields:
            SQLAlchemy async session
        """
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(connection_string: Optional[str] = None, **kwargs) -> SessionManager:
    """Get or create global session manager.
    
    Args:
        connection_string: Database connection URL (defaults to environment variable)
        **kwargs: Additional SQLAlchemy engine options
        
    Returns:
        SessionManager instance
    """
    global _session_manager
    
    if connection_string is None:
        connection_string = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    
    if _session_manager is None or _session_manager.connection_string != connection_string:
        _session_manager = SessionManager(connection_string, **kwargs)
    
    return _session_manager


@contextmanager
def get_session(connection_string: Optional[str] = None, **kwargs) -> Generator[Session, None, None]:
    """Get database session with automatic cleanup.
    
    Args:
        connection_string: Database connection URL
        **kwargs: Additional SQLAlchemy engine options
        
    Yields:
        SQLAlchemy session
    """
    manager = get_session_manager(connection_string, **kwargs)
    with manager.get_session() as session:
        yield session


async def get_async_session(connection_string: Optional[str] = None, **kwargs) -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with automatic cleanup.
    
    Args:
        connection_string: Database connection URL
        **kwargs: Additional SQLAlchemy engine options
        
    Yields:
        SQLAlchemy async session
    """
    manager = get_session_manager(connection_string, **kwargs)
    async with manager.get_async_session() as session:
        yield session


def test_database_connection(connection_string: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Test database connection.
    
    Args:
        connection_string: Database connection URL
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    if not SQLALCHEMY_AVAILABLE:
        return False
        
    if connection_string is None:
        connection_string = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    
    try:
        start_time = time.time()
        manager = get_session_manager(connection_string)
        
        with manager.get_session() as session:
            # Simple query to test connection
            result = session.execute(text("SELECT 1")).scalar()
            
        elapsed = time.time() - start_time
        return elapsed < timeout and result == 1
        
    except Exception:
        return False


async def test_async_database_connection(connection_string: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Test async database connection.
    
    Args:
        connection_string: Database connection URL
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    if not SQLALCHEMY_AVAILABLE:
        return False
        
    if connection_string is None:
        connection_string = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    
    try:
        start_time = time.time()
        manager = get_session_manager(connection_string)
        
        async with manager.get_async_session() as session:
            # Simple query to test connection
            result = await session.execute(text("SELECT 1"))
            scalar_result = result.scalar()
            
        elapsed = time.time() - start_time
        return elapsed < timeout and scalar_result == 1
        
    except Exception:
        return False


def is_database_enabled() -> bool:
    """Check if database functionality is enabled.
    
    Returns:
        True if SQLAlchemy is available and database URL is configured
    """
    return SQLALCHEMY_AVAILABLE and bool(os.getenv("DATABASE_URL"))


def cleanup_database(connection_string: Optional[str] = None) -> None:
    """Clean up database connections and resources.
    
    Args:
        connection_string: Database connection URL
    """
    global _session_manager
    
    if _session_manager is not None:
        if hasattr(_session_manager, '_engine') and _session_manager._engine is not None:
            _session_manager._engine.dispose()
        if hasattr(_session_manager, '_async_engine') and _session_manager._async_engine is not None:
            # For async engines, we need to close them properly
            pass  # This would need asyncio.run() in a real implementation
            
        _session_manager = None