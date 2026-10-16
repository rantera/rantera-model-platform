"""Local, bounded, concurrency-safe in-memory token-bucket rate limiter.

Provides per-principal rate limiting for API requests prior to billing checks,
model catalog lookup, and provider invocation.

Architecture and Operational Invariants:
- LOCAL SINGLE-PROCESS ENFORCEMENT ONLY:
  This limiter operates entirely in memory within a single Python worker process.
  It does NOT enforce distributed rate limits across multiple container instances
  in a multi-instance deployment (e.g., Cloud Run).
- DISTRIBUTED BACKEND READINESS:
  Defines an abstract asynchronous protocol (BaseRateLimiter) to support future
  shared backends (e.g., Redis). If a distributed backend is configured in a future
  phase, backend failure must not silently degrade to an unmonitored local quota.
- SERVER-SIDE AUTHENTICATED IDENTITY:
  Limiter identity is established strictly from authenticated server-side credentials
  via AuthenticatedPrincipal:
    1. Primary: 'key:<principal.api_key_id>' when an API key ID is present.
    2. Fallback: 'user:<principal.user_id>' only if api_key_id is absent.
  Never uses raw API keys, request bodies, or client-supplied forwarding headers.
- ANTI-QUOTA-RESET INVARIANT:
  Pruning safely removes ONLY idle buckets that have completely refilled to full
  capacity. Depleted or actively consuming buckets are NEVER evicted, ensuring
  callers cannot circumvent their quota by forcing bucket recreation.
- MEMORY BOUNDING & CAPACITY SATURATION:
  Storage is bounded by max_keys. When storage capacity is reached and no inactive
  buckets can be safely pruned, the limiter deterministically fails closed for new
  identities (allowed=False, retry_after=60) to protect memory integrity.
- DETERMINISTIC TIME ARITHMETIC:
  Uses an injectable monotonic clock (default: time.monotonic) and wall clock
  (default: time.time) for precise, drift-free, and deterministic token math
  without real sleeps in automated tests.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Annotated, Callable, Protocol, runtime_checkable

import anyio
from fastapi import Depends, HTTPException, Request, status

from app.auth import AuthenticatedPrincipal, require_valid_api_key
from app.logging import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    """Result of an evaluated rate limit check.

    Attributes:
        allowed: True if the request is within quota and a token was consumed.
        limit: Configured maximum requests per minute (RPM).
        remaining: Integer count of remaining tokens available in this window.
        reset_seconds: Seconds until the bucket replenishes to full capacity.
        reset_at: Unix epoch timestamp when the bucket replenishes to full capacity.
        retry_after: Seconds the client must wait before retrying (0 if allowed).
    """

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int
    reset_at: int
    retry_after: int = 0

    @property
    def headers(self) -> dict[str, str]:
        """Generate standard HTTP rate limit headers.

        Includes:
        - X-RateLimit-Limit: configured quota ceiling
        - X-RateLimit-Remaining: tokens remaining in current bucket
        - X-RateLimit-Reset: Unix epoch timestamp for full bucket replenishment
        - Retry-After: integer seconds to wait (only when rejected)
        """
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_at),
        }
        if not self.allowed and self.retry_after > 0:
            h["Retry-After"] = str(self.retry_after)
        return h


@runtime_checkable
class BaseRateLimiter(Protocol):
    """Asynchronous rate limiter protocol.

    Defines the contract for rate limit enforcement backends
    (in-memory token bucket, future distributed Redis, etc.).
    """

    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit for the given key and consume one token if allowed.

        Args:
            key: Validated identity string (e.g. 'key:api_key_id').

        Returns:
            RateLimitResult containing allowed status, remaining tokens,
            and header metadata.
        """
        ...


@dataclass
class _BucketState:
    """Internal state for a single rate limit token bucket."""

    tokens: float
    last_updated: float
    last_access: float


class InMemoryRateLimiter:
    """Bounded, concurrency-safe in-memory token-bucket rate limiter.

    Operational Boundary and Scope:
    - LOCAL SINGLE-PROCESS ENFORCEMENT ONLY: Operates entirely in memory within
      a single Python process. Does NOT enforce distributed limits across multiple
      Cloud Run container instances.
    - Token bucket algorithm: refilled continuously at rpm / 60.0 tokens per second
      up to burst capacity.
    - Memory bounded: enforces max_keys to prevent memory exhaustion attacks.
    - Anti-quota-reset: only evicts fully-refilled idle buckets; never evicts
      active/depleted buckets to prevent granting free quota.
    - Injectable monotonic clock and wall clock for deterministic testing.
    """

    def __init__(
        self,
        rpm: int = 60,
        burst: int | None = None,
        max_keys: int = 10000,
        idle_prune_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the in-memory token-bucket rate limiter.

        Args:
            rpm: Maximum sustained requests per minute (must be >= 1).
            burst: Maximum token capacity for burst traffic (defaults to rpm).
            max_keys: Maximum number of tracked keys in memory before capacity failure.
            idle_prune_seconds: Inactive duration after which fully refilled buckets
                may be safely pruned from memory.
            clock: Monotonic clock callable (for deterministic unit tests).
            wall_clock: Wall clock callable returning epoch seconds.
        """
        if rpm < 1:
            raise ValueError(f"rpm must be >= 1, got {rpm}")
        if burst is not None and burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        if max_keys < 1:
            raise ValueError(f"max_keys must be >= 1, got {max_keys}")
        if idle_prune_seconds <= 0:
            raise ValueError(f"idle_prune_seconds must be > 0, got {idle_prune_seconds}")

        self.rpm = rpm
        self.capacity = float(burst if burst is not None else rpm)
        self.max_keys = max_keys
        self.idle_prune_seconds = idle_prune_seconds
        self._refill_rate = float(rpm) / 60.0  # tokens per second
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = anyio.Lock()
        self._buckets: dict[str, _BucketState] = {}

    async def check(self, key: str) -> RateLimitResult:
        """Evaluate rate limit for a key and consume 1 token if within quota.

        Args:
            key: Validated identity key. Cannot be empty or whitespace.

        Returns:
            RateLimitResult with allowed status, remaining tokens, and headers.

        Raises:
            ValueError: If key is empty or whitespace-only.
        """
        if not key or not key.strip():
            raise ValueError("Rate limit key cannot be empty or whitespace")
        cleaned_key = key.strip()

        async with self._lock:
            now_mono = self._clock()
            now_wall = self._wall_clock()

            bucket = self._buckets.get(cleaned_key)
            if bucket is None:
                # Key is not yet tracked in memory. Check capacity bounds first.
                if len(self._buckets) >= self.max_keys:
                    self._prune_inactive_buckets_locked(now_mono)
                    if len(self._buckets) >= self.max_keys:
                        # Storage capacity is saturated with active/unrefilled keys.
                        # Fail closed to protect memory bounds and prevent quota evasion.
                        log_event(
                            logger,
                            logging.WARNING,
                            "rate_limit.capacity_saturated",
                            f"Rate limiter memory capacity reached ({self.max_keys} keys); rejecting new identity",
                            metadata={"limit": self.rpm, "max_keys": self.max_keys, "retry_after": 60},
                        )
                        return RateLimitResult(
                            allowed=False,
                            limit=self.rpm,
                            remaining=0,
                            reset_seconds=60,
                            reset_at=int(now_wall + 60),
                            retry_after=60,
                        )

                # Initialize fresh bucket with full capacity
                bucket = _BucketState(
                    tokens=self.capacity,
                    last_updated=now_mono,
                    last_access=now_mono,
                )
                self._buckets[cleaned_key] = bucket
            else:
                # Refill tokens based on monotonic elapsed time
                elapsed = max(0.0, now_mono - bucket.last_updated)
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self._refill_rate)
                bucket.last_updated = now_mono
                bucket.last_access = now_mono

            # Determine whether 1 token is available for consumption
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                allowed = True
                retry_after = 0
            else:
                allowed = False
                needed = 1.0 - bucket.tokens
                retry_after = max(1, math.ceil(needed / self._refill_rate))

            remaining = max(0, int(bucket.tokens))
            missing = self.capacity - bucket.tokens
            reset_seconds = 0 if missing <= 0 else max(1, math.ceil(missing / self._refill_rate))
            reset_at = int(now_wall + reset_seconds)

            return RateLimitResult(
                allowed=allowed,
                limit=self.rpm,
                remaining=remaining,
                reset_seconds=reset_seconds,
                reset_at=reset_at,
                retry_after=retry_after,
            )

    def _prune_inactive_buckets_locked(self, now_mono: float) -> int:
        """Prune idle buckets that have fully refilled to capacity.

        CRITICAL SAFETY INVARIANT:
        Never prune active or token-depleted buckets. Evicting a depleted
        bucket would cause the caller to receive a new full-capacity bucket
        on their next request, silently granting quota beyond the policy.
        """
        prune_keys: list[str] = []
        for key, bucket in self._buckets.items():
            elapsed = max(0.0, now_mono - bucket.last_updated)
            effective_tokens = min(self.capacity, bucket.tokens + elapsed * self._refill_rate)
            idle_time = max(0.0, now_mono - bucket.last_access)
            # Safe to evict ONLY if:
            # 1. Bucket is fully refilled (has restored all tokens to capacity)
            # 2. Key has been idle for >= idle_prune_seconds
            if effective_tokens >= self.capacity and idle_time >= self.idle_prune_seconds:
                prune_keys.append(key)

        for key in prune_keys:
            self._buckets.pop(key, None)

        return len(prune_keys)

    async def prune(self, now_mono: float | None = None) -> int:
        """Manually trigger pruning of inactive buckets (for maintenance or tests)."""
        async with self._lock:
            now = now_mono if now_mono is not None else self._clock()
            return self._prune_inactive_buckets_locked(now)

    async def get_bucket_count(self) -> int:
        """Return the current number of tracked buckets in memory."""
        async with self._lock:
            return len(self._buckets)

    async def clear(self) -> None:
        """Clear all bucket states from memory."""
        async with self._lock:
            self._buckets.clear()


def resolve_rate_limit_identity(principal: AuthenticatedPrincipal) -> str:
    """Resolve the rate limit identity from an AuthenticatedPrincipal.

    Identity hierarchy:
    1. Primary: 'key:<api_key_id>' when api_key_id is present and non-empty.
    2. Fallback: 'user:<user_id>' only when api_key_id is absent.

    Security invariants:
    - Never uses raw API keys.
    - Never uses request bodies or client headers.
    - Rejects empty, whitespace-only, or invalid identities.
    """
    if not isinstance(principal, AuthenticatedPrincipal):
        raise TypeError(f"Expected AuthenticatedPrincipal, got {type(principal).__name__}")

    if principal.api_key_id and principal.api_key_id.strip():
        return f"key:{principal.api_key_id.strip()}"
    if principal.user_id and principal.user_id.strip():
        return f"user:{principal.user_id.strip()}"
    raise ValueError("Authenticated principal has neither a valid api_key_id nor user_id")


async def require_rate_limited_principal(
    request: Request,
    principal: Annotated[AuthenticatedPrincipal, Depends(require_valid_api_key)],
) -> AuthenticatedPrincipal:
    """FastAPI dependency that enforces per-principal rate limiting.

    Order of operations:
    1. Executed immediately after successful authentication (require_valid_api_key).
    2. Executed before wallet balance checks, catalog lookup, spend reservation,
       and provider invocation.

    Raises:
        HTTPException: HTTP 429 Too Many Requests if rate limit is exceeded.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is not None and not settings.rate_limit_enabled:
        return principal

    limiter: BaseRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return principal

    try:
        identity = resolve_rate_limit_identity(principal)
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "system.error",
            f"Rate limiter identity resolution failed: {type(exc).__name__}",
            metadata={"path": request.url.path, "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unable to establish caller identity for rate limiting.",
        )

    result = await limiter.check(identity)

    # Attach headers to request state for response middleware attachment
    request.state.rate_limit_headers = result.headers

    if not result.allowed:
        log_event(
            logger,
            logging.WARNING,
            "rate_limit.rejected",
            f"Rate limit exceeded (limit={result.limit}, retry_after={result.retry_after})",
            metadata={
                "path": request.url.path,
                "method": request.method,
                "limit": result.limit,
                "retry_after": result.retry_after,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers=result.headers,
        )

    return principal
