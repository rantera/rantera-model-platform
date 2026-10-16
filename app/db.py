"""PostgreSQL asynchronous database access layer for Rantera Model Platform.

Provides connection pooling, session management, FastAPI dependencies,
parameterized SQL execution, and connectivity readiness checks via SQLAlchemy 2.x
and asyncpg.

Security invariants:
- Least Privilege: Direct queries to website core tables ("User", "ApiKey",
  "Wallet", "Transaction", "UsageRecord") are strictly forbidden.
- View & Table Isolation: Application logic must query approved views
  ("mp_active_api_keys_v", "mp_active_users_v") and model platform tables
  ("mp_model_catalog", "mp_inference_logs") only.
- Parameterized SQL Only: All queries are executed using sqlalchemy.text with
  explicit parameter bindings. String concatenation/interpolation is banned.
- DDL Forbidden: No schema alterations, migrations, or DDL execution (CREATE,
  ALTER, DROP, TRUNCATE) from this service.
- Fail Closed: If the database is unreachable or unconfigured, access fails
  closed with HTTP 503.
- Secret Hygiene: Database credentials and raw URLs are never printed, logged,
  or exposed in API responses or exception traces.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, AsyncGenerator, Optional, Sequence

import anyio
from fastapi import HTTPException, Request, status
from sqlalchemy import Row, text

try:
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    _HAS_ASYNC_SQLALCHEMY = True
except ImportError:
    _HAS_ASYNC_SQLALCHEMY = False
    AsyncEngine = Any  # type: ignore[misc,assignment]
    AsyncSession = Any  # type: ignore[misc,assignment]
    async_sessionmaker = Any  # type: ignore[misc,assignment]
    create_async_engine = None  # type: ignore[assignment]

from app.config import Settings

logger = logging.getLogger(__name__)

# Disallowed DDL keywords indicating schema definitions or mutations that must never
# be executed by application runtime code.
_DISALLOWED_DDL_KEYWORDS = re.compile(
    r"\b(CREATE\s+(TABLE|VIEW|INDEX|TRIGGER|FUNCTION|ROLE|USER)|ALTER\s+|DROP\s+|TRUNCATE\s+|GRANT\s+|REVOKE\s+)\b",
    re.IGNORECASE,
)

# Core website tables that the model platform must never access directly.
_FORBIDDEN_WEBSITE_TABLES = re.compile(
    r'(?<!mp_)\b("?User"?|"?ApiKey"?|"?Wallet"?|"?Transaction"?|"?UsageRecord"?|"?Conversation"?|"?Message"?|"?BlogPost"?|"?WaitlistEntry"?|"?Announcement"?|"?AuditLog"?|"?SupportTicket"?)\b',
    re.IGNORECASE,
)


def validate_query_safety(statement: str) -> None:
    """Validate that SQL statements conform to architectural expectations.

    Defense-in-depth notice:
        Treat regex-based SQL validation as defense-in-depth only, never as
        a security boundary. It does not reliably prevent arbitrary SQL access.
        The actual authorization boundary is enforced at the database level
        via dedicated PostgreSQL service roles, grants, and approved views.

    Raises:
        ValueError: If DDL commands are detected.
        PermissionError: If direct access to protected website tables is attempted.
    """
    cleaned = statement.strip()
    if _DISALLOWED_DDL_KEYWORDS.search(cleaned):
        raise ValueError(
            "DDL and privilege operations (CREATE, ALTER, DROP, TRUNCATE, GRANT, REVOKE) "
            "are strictly forbidden in application queries."
        )

    match = _FORBIDDEN_WEBSITE_TABLES.search(cleaned)
    if match:
        raise PermissionError(
            f"Direct access to website table {match.group(0)!r} is forbidden by architectural boundary. "
            "Access must proceed through approved views (e.g. mp_active_api_keys_v)."
        )


def normalize_database_url(url: str) -> str:
    """Normalize a PostgreSQL connection URL to use the asyncpg driver.

    Converts 'postgresql://' or 'postgres://' schemes to 'postgresql+asyncpg://'.
    Translates 'sslmode' query parameters to 'ssl' for asyncpg driver compatibility.
    Leaves already-specified async drivers (e.g., 'postgresql+asyncpg://')
    and test database URLs (e.g., 'sqlite+aiosqlite://') unchanged.
    """
    stripped = url.strip()
    if stripped.startswith("postgres://"):
        stripped = "postgresql+asyncpg://" + stripped[len("postgres://"):]
    elif stripped.startswith("postgresql://"):
        stripped = "postgresql+asyncpg://" + stripped[len("postgresql://"):]

    if "asyncpg" in stripped and "sslmode" in stripped:
        try:
            parsed = urllib.parse.urlsplit(stripped)
            if parsed.query:
                params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                has_ssl = any(k == "ssl" for k, _ in params)
                new_params = []
                for k, v in params:
                    if k == "sslmode":
                        if not has_ssl:
                            new_params.append(("ssl", v))
                            has_ssl = True
                    else:
                        new_params.append((k, v))
                new_query = urllib.parse.urlencode(new_params)
                stripped = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))
        except Exception:
            pass

    return stripped


def create_db_engine(settings: Settings) -> AsyncEngine:
    """Create and configure a SQLAlchemy 2.x AsyncEngine with connection pooling.

    Configures connection pool limits, timeouts, connection recycling, and
    statement timeouts based on application settings.

    Args:
        settings: Application settings containing database configuration.

    Returns:
        AsyncEngine configured for asyncpg.

    Raises:
        ValueError: If database_url is not set in settings.
        ImportError: If async SQLAlchemy support is not available.
    """
    if not settings.database_url:
        raise ValueError("Cannot initialize database engine: database_url is not configured.")

    if not _HAS_ASYNC_SQLALCHEMY or create_async_engine is None:
        raise ImportError(
            "SQLAlchemy asyncio extension requires 'greenlet' ('pip install sqlalchemy[asyncio]')."
        )

    raw_url = normalize_database_url(settings.database_url.get_secret_value())

    # Connection pooling parameters per Phase 2 architecture specification
    pool_kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": settings.db_pool_recycle_seconds,
    }

    if "sqlite" not in raw_url:
        pool_kwargs["pool_size"] = settings.db_pool_size
        pool_kwargs["max_overflow"] = settings.db_max_overflow
        pool_kwargs["pool_timeout"] = settings.db_pool_timeout_seconds
        pool_kwargs["connect_args"] = {
            "command_timeout": settings.db_statement_timeout_ms / 1000.0,
            "timeout": settings.db_connect_timeout_seconds,
        }

    return create_async_engine(raw_url, **pool_kwargs)


def get_session_factory(engine: AsyncEngine) -> Any:
    """Create an async sessionmaker bound to the given AsyncEngine.

    Args:
        engine: AsyncEngine instance.

    Returns:
        async_sessionmaker producing AsyncSession instances.
    """
    if _HAS_ASYNC_SQLALCHEMY:
        return async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    # Fallback callable for mocked environments when greenlet is absent
    class _MockSessionFactory:
        def __call__(self) -> Any:
            return engine

    return _MockSessionFactory()


async def get_db_session(request: Request) -> AsyncGenerator[Any, None]:
    """FastAPI dependency yielding an AsyncSession.

    Fails closed with HTTP 503 if the database engine is not initialized or
    unavailable. Rollbacks transactions on unhandled exceptions.

    Yields:
        AsyncSession connected to the database.

    Raises:
        HTTPException: 503 if database service is not available.
    """
    engine: Optional[Any] = getattr(request.app.state, "db_engine", None)
    if engine is None:
        logger.error("Database access attempted but database engine is not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable.",
        )

    session_maker = getattr(request.app.state, "db_session_factory", None)
    if session_maker is None:
        session_maker = get_session_factory(engine)
        request.app.state.db_session_factory = session_maker

    async with session_maker() as session:
        try:
            yield session
        except BaseException:
            if hasattr(session, "rollback"):
                try:
                    await session.rollback()
                except Exception:
                    pass
            raise


async def execute_parameterized_query(
    session: Any,
    statement: str,
    params: Optional[dict[str, Any]] = None,
) -> Sequence[Row[Any]]:
    """Execute a parameterized SQL query and return fetched rows.

    Security invariants:
    - Queries must use parameter binding (:param_name).
    - DDL and schema mutations are forbidden.
    - Queries against core website tables are rejected.
    - No query parameters or credentials are logged.

    Args:
        session: Active AsyncSession or Mock session.
        statement: Parameterized SQL string.
        params: Dictionary of bound parameters.

    Returns:
        Sequence of result Rows.
    """
    validate_query_safety(statement)
    compiled = text(statement.strip())
    result = await session.execute(compiled, params or {})
    return result.fetchall()


async def check_database_readiness(
    engine: Optional[Any],
    timeout_seconds: float = 1.0,
) -> bool:
    """Execute a shallow 'SELECT 1' connectivity probe with a strict timeout.

    Security invariants:
    - Bounded execution time via anyio.fail_after.
    - Safe logging: never prints database URLs or credentials.
    - Returns False on any error or timeout (fails safe).

    Args:
        engine: AsyncEngine to probe.
        timeout_seconds: Maximum seconds before timing out (default 1.0s).

    Returns:
        True if connectivity probe returns 1, False otherwise.
    """
    if engine is None:
        return False

    try:
        with anyio.fail_after(timeout_seconds):
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                scalar = result.scalar()
                return scalar == 1
    except TimeoutError:
        logger.warning("Database readiness probe timed out after %.2fs", timeout_seconds)
        return False
    except Exception as exc:
        logger.warning(
            "Database readiness probe failed: %s",
            type(exc).__name__,
        )
        return False
