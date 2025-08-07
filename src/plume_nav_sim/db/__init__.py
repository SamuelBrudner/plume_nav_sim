"""
Database package for plume navigation simulation.

This package provides database session management, connection lifecycle, 
transaction management, and persistence hooks for the plume navigation simulation system.

Key Components:
- SessionManager: Database session management with connection pooling
- DatabaseBackend: Backend detection and configuration
- Database protocols: Type-safe interfaces for database operations
- Connection utilities: Helper functions for database connectivity

Authors: Blitzy Platform  
License: MIT
Version: 1.0.0
"""

from .session import (
    DatabaseBackend,
    DatabaseConfig,
    DatabaseConnectionInfo,
    PersistenceHooks,
    SessionManager,
    cleanup_database,
    get_async_session,
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
    "get_async_session", 
    "get_session",
    "get_session_manager",
    "is_database_enabled",
    "test_async_database_connection",
    "test_database_connection",
    "DOTENV_AVAILABLE",
    "HYDRA_AVAILABLE", 
    "SQLALCHEMY_AVAILABLE",
]