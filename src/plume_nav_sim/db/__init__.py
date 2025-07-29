"""Database utilities and session management for plume_nav_sim.

This package provides database connection management, session handling,
and persistence utilities for the plume navigation simulation framework.
"""

from .session import (
    DatabaseBackend,
    DatabaseConfig,
    DatabaseConnectionInfo,
    PersistenceHooks,
    SessionManager,
    cleanup_database,
    create_session_manager,
    get_async_session,
    get_default_database_config,
    get_session,
    get_session_manager,
    is_database_enabled,
    test_async_database_connection,
    test_database_connection,
    DOTENV_AVAILABLE,
    HYDRA_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
)

__all__ = [
    "DatabaseBackend",
    "DatabaseConfig", 
    "DatabaseConnectionInfo",
    "PersistenceHooks",
    "SessionManager",
    "cleanup_database",
    "create_session_manager",
    "get_async_session",
    "get_default_database_config",
    "get_session",
    "get_session_manager",
    "is_database_enabled",
    "test_async_database_connection",
    "test_database_connection",
    "DOTENV_AVAILABLE",
    "HYDRA_AVAILABLE", 
    "SQLALCHEMY_AVAILABLE",
]