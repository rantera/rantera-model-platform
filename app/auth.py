"""Authentication interface for the Rantera Model Platform.

Provides SHA-256 API-key verification and caller identity resolution via
parameterized queries against the approved `mp_active_api_keys_v` view.

Security invariants:
- Least Privilege: Direct queries to website core tables ("User", "ApiKey",
  "Wallet", "Transaction", "UsageRecord") are strictly forbidden.
- View Isolation: Queries target the approved security-barrier view
  `mp_active_api_keys_v` only.
- Parameterized SQL Only: All database lookups bind the computed SHA-256
  key hash as a parameter (:key_hash); string interpolation is banned.
- Constant-Time Comparison: Verifies computed hash against stored hash via
  hmac.compare_digest to prevent side-channel timing attacks.
- Fail Closed: Missing database configuration or database query failures fail
  closed with HTTP 503.
- Distinction of States:
  * Missing key: HTTP 401 Unauthorized
  * Invalid key / hash mismatch: HTTP 401 Unauthorized ("Invalid API key.")
  * Expired key: HTTP 401 Unauthorized ("API key expired.")
  * Conflicting credentials: HTTP 400 Bad Request ("Conflicting authentication credentials provided.")
  * Banned user: HTTP 403 Forbidden ("Account suspended.")
  * Database/Infrastructure failure: HTTP 503 Service Unavailable
- No Trust in Client User ID: The caller's user_id is established strictly
  from the authenticated database record, never from client-supplied headers.
- Secret Hygiene: Raw API keys, authorization headers, and database URLs
  are never printed, logged, or leaked into API responses or exception messages.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from app.db import execute_parameterized_query, get_db_session
from app.logging import log_event

logger = logging.getLogger(__name__)

# Header schemes for API key authentication.
_API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authenticating requests.",
)

_BEARER_AUTH = HTTPBearer(
    auto_error=False,
    description="Bearer token authentication using an API key.",
)

# Parameterized query against the approved security-barrier view.
# Direct access to website core tables (User, ApiKey, Wallet) is strictly forbidden.
_AUTH_QUERY = """
SELECT
    api_key_id,
    user_id,
    key_hash,
    key_prefix,
    key_name,
    spend_limit_usd,
    expires_at,
    user_is_banned,
    user_email_verified,
    wallet_balance_usd
FROM mp_active_api_keys_v
WHERE key_hash = :key_hash
LIMIT 1
"""

# Parameterized query against the approved security-barrier users view for internal web calls.
_USER_AUTH_QUERY = """
SELECT
    user_id,
    is_banned,
    email_verified,
    wallet_balance_usd
FROM mp_active_users_v
WHERE user_id = :user_id
LIMIT 1
"""

# In-memory bounded cache for verified API keys: key_hash -> (AuthenticatedPrincipal, expire_time_mono)
# Maximum size is strictly bounded to prevent memory saturation.
_AUTH_CACHE_MAX_SIZE: int = 1000
_AUTH_CACHE: dict[str, tuple[AuthenticatedPrincipal, float]] = {}


class AuthenticatedPrincipal(BaseModel):
    """Authenticated entity representing the validated caller and billing context."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        ...,
        description="Logical user identifier from User table.",
        min_length=1,
        max_length=64,
    )
    api_key_id: str | None = Field(
        default=None,
        description="Logical API key identifier from ApiKey table.",
        max_length=64,
    )
    channel: str = Field(
        default="api",
        description="Access channel identifier (e.g. 'api').",
        max_length=32,
    )
    wallet_balance: Decimal = Field(
        default=Decimal("0.0"),
        description="Available wallet balance in USD.",
    )
    spend_limit: Decimal | None = Field(
        default=None,
        description="Optional spend limit in USD.",
    )
    key_prefix: str | None = Field(
        default=None,
        description="Key prefix for identification.",
        max_length=16,
    )
    key_name: str | None = Field(
        default=None,
        description="Descriptive name of the key.",
        max_length=128,
    )
    user_email_verified: bool = Field(
        default=True,
        description="Whether the user's email address has been verified.",
    )

    @property
    def name(self) -> str | None:
        """Alias for key_name matching ApiKey.name."""
        return self.key_name

    @property
    def wallet_balance_usd(self) -> Decimal:
        """Alias for wallet_balance matching view column name."""
        return self.wallet_balance

    @property
    def spend_limit_usd(self) -> Decimal | None:
        """Alias for spend_limit matching view column name."""
        return self.spend_limit

    @property
    def is_banned(self) -> bool:
        """Always False for an active AuthenticatedPrincipal."""
        return False

    @property
    def email_verified(self) -> bool:
        """Alias for user_email_verified."""
        return self.user_email_verified


def _set_cached_principal(
    key_hash: str,
    principal: AuthenticatedPrincipal,
    ttl_seconds: float,
) -> None:
    """Store an AuthenticatedPrincipal in the bounded cache with an enforced TTL.

    Cache policy:
    - Maximum capacity is bounded by _AUTH_CACHE_MAX_SIZE (1000).
    - When capacity is reached, expired entries are pruned first.
    - If still at capacity, the oldest entries (up to 200) are evicted.
    """
    now_mono = time.monotonic()
    if len(_AUTH_CACHE) >= _AUTH_CACHE_MAX_SIZE:
        expired = [k for k, (_, exp) in _AUTH_CACHE.items() if exp <= now_mono]
        for k in expired:
            _AUTH_CACHE.pop(k, None)
        if len(_AUTH_CACHE) >= _AUTH_CACHE_MAX_SIZE:
            oldest = list(_AUTH_CACHE.keys())[:200]
            for k in oldest:
                _AUTH_CACHE.pop(k, None)
    _AUTH_CACHE[key_hash] = (principal, now_mono + ttl_seconds)


def clear_auth_cache() -> None:
    """Clear the in-memory API key authentication cache.

    Useful in tests and configuration updates to prevent cache carryover.
    """
    _AUTH_CACHE.clear()


def hash_api_key(raw_key: str) -> str:
    """Compute a SHA-256 hash of an API key for storage/comparison.

    Args:
        raw_key: The plaintext API key.

    Returns:
        Hex-encoded SHA-256 hash (64 lowercase hex characters).
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_key_hash(raw_key: str, stored_hash: str) -> bool:
    """Constant-time comparison of a key's hash against a stored hash.

    Args:
        raw_key: The plaintext API key from the request.
        stored_hash: The hex-encoded hash stored server-side.

    Returns:
        True if the key matches the stored hash.
    """
    computed = hash_api_key(raw_key)
    return hmac.compare_digest(computed, stored_hash)


def _get_row_field(row: Any, *field_names: str, default: Any = None) -> Any:
    """Safely extract a field value from a database row, mapping, or dictionary using candidate names in order."""
    for field_name in field_names:
        if hasattr(row, "_mapping"):
            if field_name in row._mapping:
                return row._mapping[field_name]
        elif isinstance(row, dict):
            if field_name in row:
                return row[field_name]
        elif hasattr(row, field_name):
            return getattr(row, field_name)
    return default


async def authenticate_api_key(
    raw_key: str,
    request: Request,
    session: Optional[Any] = None,
) -> AuthenticatedPrincipal:
    """Authenticate an API key against mp_active_api_keys_v and return an AuthenticatedPrincipal.

    Security invariants:
    - Binds SHA-256 hash into a parameterized query against mp_active_api_keys_v.
    - Constant-time verification of computed hash against stored hash.
    - Accurately distinguishes invalid key, expired key, banned user, and database failure.
    - Never logs plaintext keys, passwords, or connection strings.
    - Client-supplied user ID is never consulted.

    Args:
        raw_key: Raw API key from the request.
        request: Incoming FastAPI Request.
        session: Optional active AsyncSession or mock session. If None, acquires one
            via the project's get_db_session dependency.

    Returns:
        AuthenticatedPrincipal with validated user_id, api_key_id, channel,
        wallet_balance, and spend_limit.

    Raises:
        HTTPException:
            - 401 if key is missing, invalid, or expired.
            - 403 if user account is banned.
            - 503 if database service is unavailable or query fails.
    """
    cleaned_key = raw_key.strip()
    if not cleaned_key:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.missing_key",
            "Authentication failed: empty API key",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a valid key in the X-API-Key or Authorization header.",
        )

    key_hash = hash_api_key(cleaned_key)

    # Check in-memory cache if enabled
    settings = getattr(request.app.state, "settings", None)
    cache_ttl = getattr(settings, "auth_cache_ttl_seconds", 60) if settings else 0
    if cache_ttl > 0 and key_hash in _AUTH_CACHE:
        cached_principal, expiry_mono = _AUTH_CACHE[key_hash]
        if time.monotonic() < expiry_mono:
            request.state.principal = cached_principal
            return cached_principal
        else:
            _AUTH_CACHE.pop(key_hash, None)

    async def _execute_lookup(active_session: Any) -> list[Any]:
        try:
            results = await execute_parameterized_query(
                session=active_session,
                statement=_AUTH_QUERY,
                params={"key_hash": key_hash},
            )
            return list(results)
        except HTTPException:
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "auth.failed.service_unavailable",
                f"Database error during API key authentication: {type(exc).__name__}",
                metadata={
                    "path": request.url.path,
                    "method": request.method,
                    "error_type": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is temporarily unavailable. Please retry.",
            )

    # Execute query using provided session or acquired session
    if session is not None:
        rows = await _execute_lookup(session)
    else:
        try:
            rows = None
            async for sess in get_db_session(request):
                rows = await _execute_lookup(sess)
                break
        except HTTPException:
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "auth.failed.service_unavailable",
                f"Failed to acquire database session for authentication: {type(exc).__name__}",
                metadata={
                    "path": request.url.path,
                    "method": request.method,
                    "error_type": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is temporarily unavailable. Please retry.",
            )

    # 1. Check row exists
    if not rows:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.invalid_key",
            "Authentication failed: invalid API key",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    row = rows[0]

    # 2. Constant-time verification of computed hash against stored hash
    stored_hash = _get_row_field(row, "key_hash")
    if not stored_hash or not hmac.compare_digest(key_hash, str(stored_hash)):
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.invalid_key",
            "Authentication failed: key hash mismatch",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    # 3. Check key expiry
    expires_at = _get_row_field(row, "expires_at")
    if expires_at is not None:
        now = datetime.now(timezone.utc)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            log_event(
                logger,
                logging.WARNING,
                "auth.failed.expired_key",
                "Authentication failed: expired API key",
                metadata={"path": request.url.path, "method": request.method},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key expired.",
            )

    # 4. Check banned user
    user_is_banned = _get_row_field(row, "user_is_banned", "is_banned", default=False)
    if bool(user_is_banned):
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.banned_user",
            "Authentication failed: user account suspended",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended.",
        )

    # 5. Extract authenticated user context and billing metrics
    user_id = str(_get_row_field(row, "user_id"))
    api_key_id = _get_row_field(row, "api_key_id")

    raw_balance = _get_row_field(row, "wallet_balance_usd", "wallet_balance", default=0.0)
    wallet_balance = Decimal(str(raw_balance))

    raw_spend_limit = _get_row_field(row, "spend_limit_usd", "spend_limit", default=None)
    spend_limit = Decimal(str(raw_spend_limit)) if raw_spend_limit is not None else None

    key_prefix = _get_row_field(row, "key_prefix")
    key_name = _get_row_field(row, "key_name", "name", default=None)
    user_email_verified = bool(_get_row_field(row, "user_email_verified", "email_verified", default=True))

    principal = AuthenticatedPrincipal(
        user_id=user_id,
        api_key_id=str(api_key_id) if api_key_id is not None else None,
        channel="api",
        wallet_balance=wallet_balance,
        spend_limit=spend_limit,
        key_prefix=str(key_prefix) if key_prefix else None,
        key_name=str(key_name) if key_name else None,
        user_email_verified=user_email_verified,
    )

    # Cache successful authentication result with clamped TTL
    if cache_ttl > 0:
        effective_ttl = float(cache_ttl)
        if expires_at is not None:
            now_utc = datetime.now(timezone.utc)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            seconds_until_expiry = (expires_at - now_utc).total_seconds()
            # Clamp TTL to remaining key lifetime so cache cannot bypass key expiration
            effective_ttl = min(effective_ttl, max(0.0, seconds_until_expiry))
        if effective_ttl > 0:
            _set_cached_principal(key_hash, principal, effective_ttl)

    request.state.principal = principal
    log_event(
        logger,
        logging.INFO,
        "auth.success",
        "API key authenticated successfully",
        metadata={"path": request.url.path, "method": request.method, "channel": principal.channel},
    )
    return principal


async def authenticate_web_user(
    user_id: str,
    request: Request,
    session: Optional[Any] = None,
) -> AuthenticatedPrincipal:
    """Authenticate a website session user via mp_active_users_v for internal web channel calls.

    Security invariants:
    - Binds user_id into a parameterized query against mp_active_users_v.
    - Validates account existence, suspension (is_banned), and loads wallet balance.
    - NEVER accepts client-supplied user IDs without verified server-to-server internal service key.
    """
    cleaned_uid = user_id.strip()
    if not cleaned_uid:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.missing_user_id",
            "Authentication failed: empty web user ID",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identifier for web channel delegation.",
        )

    async def _execute_user_lookup(active_session: Any) -> list[Any]:
        try:
            results = await execute_parameterized_query(
                session=active_session,
                statement=_USER_AUTH_QUERY,
                params={"user_id": cleaned_uid},
            )
            return list(results)
        except HTTPException:
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "auth.failed.service_unavailable",
                f"Database error during web user lookup: {type(exc).__name__}",
                metadata={"path": request.url.path, "method": request.method, "error_type": type(exc).__name__},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is temporarily unavailable. Please retry.",
            )

    if session is not None:
        rows = await _execute_user_lookup(session)
    else:
        rows = None
        async for sess in get_db_session(request):
            rows = await _execute_user_lookup(sess)
            break

    if not rows:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.user_not_found",
            f"Web channel authentication failed: user not found",
            metadata={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
        )

    row = rows[0]
    user_is_banned = _get_row_field(row, "is_banned", default=False)
    if bool(user_is_banned):
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.banned_user",
            "Web channel authentication failed: user account suspended",
            metadata={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended.",
        )

    raw_balance = _get_row_field(row, "wallet_balance_usd", default=0.0)
    wallet_balance = Decimal(str(raw_balance))
    email_verified = bool(_get_row_field(row, "email_verified", default=True))

    principal = AuthenticatedPrincipal(
        user_id=cleaned_uid,
        api_key_id=None,
        channel="web",
        wallet_balance=wallet_balance,
        user_email_verified=email_verified,
    )
    request.state.principal = principal
    log_event(
        logger,
        logging.INFO,
        "auth.success",
        "Web user authenticated successfully via internal service delegation",
        metadata={"path": request.url.path, "method": request.method, "channel": "web"},
    )
    return principal


async def require_valid_api_key(
    request: Request,
    api_key: Annotated[str | None, Depends(_API_KEY_HEADER)] = None,
    bearer_auth: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER_AUTH)] = None,
) -> AuthenticatedPrincipal:
    """FastAPI dependency that enforces API-key authentication or trusted web delegation.

    Accepts credentials via either 'X-API-Key' or 'Authorization: Bearer <key>'.
    Also supports internal website delegation via 'X-Rantera-Web-User-Id' when
    authorized by matching internal_service_key.
    """
    raw_x_key = api_key if api_key is not None else request.headers.get("X-API-Key")
    x_key = raw_x_key.strip() if raw_x_key and raw_x_key.strip() else None

    # Check for Authorization header
    raw_auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    bearer_key: str | None = None
    has_non_bearer_auth = False

    if bearer_auth and bearer_auth.credentials and bearer_auth.credentials.strip():
        bearer_key = bearer_auth.credentials.strip()
    elif raw_auth_header and raw_auth_header.strip():
        cleaned_auth = raw_auth_header.strip()
        if cleaned_auth.lower().startswith("bearer "):
            token_part = cleaned_auth[7:].strip()
            if token_part:
                bearer_key = token_part
        elif cleaned_auth.lower() == "bearer":
            pass
        else:
            has_non_bearer_auth = True

    # Safely detect and reject conflicting credentials
    if x_key and bearer_key and x_key != bearer_key:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.invalid_key",
            "Authentication failed: conflicting API keys in X-API-Key and Authorization headers",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting authentication credentials provided.",
        )

    if x_key and has_non_bearer_auth:
        log_event(
            logger,
            logging.WARNING,
            "auth.failed.invalid_key",
            "Authentication failed: conflicting authentication schemes in headers",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting authentication credentials provided.",
        )

    # Resolve active credential
    resolved_key = x_key or bearer_key

    # Check for trusted internal service delegation (e.g. Website server to Model Platform gateway)
    web_user_id = request.headers.get("X-Rantera-Web-User-Id")
    if web_user_id and web_user_id.strip():
        settings = getattr(request.app.state, "settings", None)
        internal_key = (
            settings.internal_service_key.get_secret_value()
            if (settings and settings.internal_service_key)
            else None
        )
        internal_header = request.headers.get("X-Internal-Service-Key")
        caller_key = internal_header or resolved_key

        if internal_key and caller_key and hmac.compare_digest(caller_key, internal_key):
            return await authenticate_web_user(user_id=web_user_id, request=request)
        else:
            log_event(
                logger,
                logging.WARNING,
                "auth.failed.invalid_service_key",
                "Internal web user delegation attempted without valid internal service key",
                metadata={"path": request.url.path},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service credentials for web channel delegation.",
            )

    if not resolved_key:
        if has_non_bearer_auth:
            log_event(
                logger,
                logging.WARNING,
                "auth.failed.invalid_key",
                "Authentication failed: invalid authorization scheme",
                metadata={"path": request.url.path, "method": request.method},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization scheme. Provide a Bearer token or X-API-Key.",
            )

        log_event(
            logger,
            logging.WARNING,
            "auth.failed.missing_key",
            "Authentication failed: missing API key",
            metadata={"path": request.url.path, "method": request.method},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a valid key in the X-API-Key or Authorization header.",
        )

    return await authenticate_api_key(raw_key=resolved_key, request=request)
