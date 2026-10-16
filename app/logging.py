"""Structured, security-conscious JSON audit logging for the Rantera Model Platform.

Provides:
- Single-line JSON formatted logging with stable schema:
  timestamp (UTC ISO-8601), level, logger, event, message, environment, request_id, metadata.
- Concurrency-safe request-ID correlation via contextvars.
- Defense-in-depth secret redaction:
  - Real API keys: sk-rnt-[0-9a-fA-F]{48} -> [REDACTED_API_KEY]
  - Fixture/test keys: sk-ran-[A-Za-z0-9_-]+ -> [REDACTED_API_KEY]
  - Bearer tokens: Bearer ... -> Bearer [REDACTED]
  - Database URLs: user:pass@host -> user:***@host (structured parsing)
  - Generic secrets: password, secret, token -> ***
- Strict event allowlist and metadata filtering:
  - Disallows arbitrary objects, headers, request dicts, or provider responses.
  - Excludes user IDs, API-key IDs, email addresses, and client IPs by default.
  - Forbids logging prompt contents (ChatMessage.content) and model responses.
- CRLF and control-character injection neutralization via strict JSON encoding.
- Exception sanitization without local variables.
- Idempotent logging configuration preventing duplicate handlers.
- Fail-safe fallback: formatting failures output safe static JSON without crashing the application.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import re
import sys
import traceback
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

# ── 1. Request ID Context Management ──────────────────────────

_CURRENT_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_current_request_id() -> str:
    """Return the active request correlation ID or empty string."""
    return _CURRENT_REQUEST_ID.get()


def set_current_request_id(request_id: str) -> contextvars.Token[str]:
    """Set the active request correlation ID. Returns token for reset."""
    return _CURRENT_REQUEST_ID.set(request_id)


def reset_current_request_id(token: contextvars.Token[str]) -> None:
    """Reset the active request correlation ID to its previous state."""
    _CURRENT_REQUEST_ID.reset(token)


# ── 2. Redaction Engine & Secret Sanitization ─────────────────

# Real production API key pattern: sk-rnt- followed by 48 hex characters
_PROD_API_KEY_REGEX = re.compile(r"sk-rnt-[0-9a-fA-F]{48}")

# Test/fixture key pattern: sk-ran- followed by alphanumeric characters
_TEST_API_KEY_REGEX = re.compile(r"\bsk-ran-[A-Za-z0-9_-]{4,}\b")

# Bearer token pattern
_BEARER_TOKEN_REGEX = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*\b")

# Database URL regex for credentials in authority
_DB_URL_REGEX = re.compile(
    r"\b((?:postgresql(?:\+[a-z0-9_]+)?|postgres|sqlite(?:\+[a-z0-9_]+)?)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# Database URL regex for locating complete database URLs in unstructured log text
_DB_URL_FINDER_REGEX = re.compile(
    r"\b(?:postgresql(?:\+[a-z0-9_]+)?|postgres|sqlite(?:\+[a-z0-9_]+)?)://[^\s'\"<>\)]+",
    re.IGNORECASE,
)

# Case-insensitive sensitive query parameter name matcher
_SENSITIVE_PARAM_REGEX = re.compile(
    r"(?i)^(.*[_-])?(password|passwd|pass|pwd|secret|token|credential|api[_-]?key|auth[_-]?key|access[_-]?key|private[_-]?key|priv[_-]?key|ssl[_-]?(?:key|pass|password))(s|[_-].*)?$"
)

# Generic credential key-value patterns (handles quoted multi-word strings, escaped quotes, and unquoted tokens)
_KEY_VALUE_SECRET_REGEX = re.compile(
    r"""(?i)(?:['"]|\b)(api[_-]?key|password|passwd|pwd|pass|secret|token|client_secret)(?:['"]|\b)\s*[:=]\s*(?:\"(?!\*\*\*|\[REDACTED)(?:[^\"\r\n\\]|\\.)*\"|'(?!\*\*\*|\[REDACTED)(?:[^'\r\n\\]|\\.)*'|(?!\*\*\*|\[REDACTED)([^\s,;'"\r\n\(\)\[\]\{\}]+))"""
)


def sanitize_database_url(url: str) -> str:
    """Sanitize database URL by masking credentials in authority, query parameters, and fragments.

    - Masks password in user:password@host authority.
    - Redacts sensitive query parameters (password, token, secret, api_key, private_key, ssl private-keys).
    - Preserves harmless query parameters (e.g. sslmode, connect_timeout).
    - Removes fragments to avoid leaking fragment credentials.
    - Handles percent-encoded credentials safely.
    - Fail-closed: If parsing fails or URL is invalid, returns [DATABASE_URL_REDACTED].
    """
    if not url:
        return url
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            # Handle SQLite URLs without netloc (e.g. sqlite:///:memory: or sqlite:///app.db)
            if parsed.scheme and (parsed.scheme == "sqlite" or parsed.scheme.startswith("sqlite+")):
                query = parsed.query
                if query:
                    raw_params = parse_qsl(query, keep_blank_values=True)
                    sanitized_params: list[tuple[str, str]] = []
                    for k, v in raw_params:
                        if _SENSITIVE_PARAM_REGEX.search(k):
                            sanitized_params.append((k, "***"))
                        else:
                            sanitized_params.append((k, v))
                    query = urlencode(sanitized_params, quote_via=quote, safe="/*")
                reconstructed = urlunsplit((parsed.scheme, "", parsed.path, query, ""))
                prefix = f"{parsed.scheme}://"
                if url.startswith(prefix) and not reconstructed.startswith(prefix):
                    reconstructed = prefix + reconstructed[len(parsed.scheme) + 1:]
                return reconstructed
            return "[DATABASE_URL_REDACTED]"

        # 1. Authority sanitization (username:password@host)
        netloc = parsed.netloc
        if "@" in netloc:
            userinfo, hostport = netloc.rsplit("@", 1)
            if ":" in userinfo:
                user_part, _ = userinfo.split(":", 1)
                netloc = f"{user_part}:***@{hostport}"

        # 2. Query parameter sanitization
        query = parsed.query
        if query:
            raw_params = parse_qsl(query, keep_blank_values=True)
            sanitized_params: list[tuple[str, str]] = []
            for k, v in raw_params:
                if _SENSITIVE_PARAM_REGEX.search(k):
                    sanitized_params.append((k, "***"))
                else:
                    sanitized_params.append((k, v))
            query = urlencode(sanitized_params, quote_via=quote, safe="/*")

        # 3. Fragment removal (fragments are not valid in DB URLs and may leak secrets)
        fragment = ""

        # Reconstruct URL preserving scheme:// for authority-less URLs (e.g. sqlite:///)
        reconstructed = urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))
        prefix = f"{parsed.scheme}://"
        if url.startswith(prefix) and not reconstructed.startswith(prefix):
            reconstructed = prefix + reconstructed[len(parsed.scheme) + 1:]

        return reconstructed
    except Exception:
        # Fail closed: never return partially sanitized URL or original URL
        return "[DATABASE_URL_REDACTED]"


def _sanitize_db_url_match(match: re.Match[str]) -> str:
    """Helper to sanitize DB URLs matched in text while preserving sentence punctuation."""
    raw = match.group(0)
    suffix = ""
    while raw and raw[-1] in ".,;":
        suffix = raw[-1] + suffix
        raw = raw[:-1]
    return sanitize_database_url(raw) + suffix


def sanitize_text(text: str) -> str:
    """Redact API keys, tokens, database passwords, and secrets from string."""
    if not text:
        return text

    # Redact real and fixture API keys completely without suffix leakage
    text = _PROD_API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)
    text = _TEST_API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)

    # Redact Bearer tokens
    text = _BEARER_TOKEN_REGEX.sub("Bearer [REDACTED]", text)

    # Redact database connection URLs completely (authority, query parameters, fragments)
    text = _DB_URL_FINDER_REGEX.sub(_sanitize_db_url_match, text)

    # Redact residual/raw database credentials if any matched raw format
    text = _DB_URL_REGEX.sub(r"\1***\3", text)

    # Redact generic secret assignments (quoted multi-word, escaped quotes, unquoted)
    text = _KEY_VALUE_SECRET_REGEX.sub(r"\1: ***", text)

    return text


# ── 3. Allowlisted Event Taxonomy & Metadata Schemas ───────────

_MAX_STRING_LENGTH: int = 256
_MAX_METADATA_KEYS: int = 16

_ALLOWED_EVENTS: frozenset[str] = frozenset({
    # HTTP lifecycle
    "http.request.completed",
    # Authentication
    "auth.success",
    "auth.failed.missing_key",
    "auth.failed.invalid_key",
    "auth.failed.expired_key",
    "auth.failed.banned_user",
    "auth.failed.service_unavailable",
    # Rate Limiting
    "rate_limit.rejected",
    "rate_limit.capacity_saturated",
    # Billing
    "billing.rejected.depleted_wallet",
    "billing.rejected.insufficient_for_prompt",
    "billing.rejected.concurrent_in_flight",
    # Inference
    "inference.started",
    "inference.completed",
    "inference.failed.provider_error",
    "inference.failed.bad_request",
    # Model Catalog
    "catalog.refresh.success",
    "catalog.refresh.failed",
    # Telemetry
    "telemetry.recorded",
    "telemetry.skipped_stateless",
    "telemetry.session_none",
    "telemetry.write_failed",
    # Service Lifecycle
    "service.startup",
    "service.shutdown",
    "database.pool_initialized",
    "database.pool_disposed",
    # Generic operational events
    "system.info",
    "system.warning",
    "system.error",
})

# Allowlisted metadata fields (strict primitive types only)
_ALLOWED_METADATA_FIELDS: frozenset[str] = frozenset({
    "method",
    "path",
    "status_code",
    "duration_ms",
    "channel",
    "error_type",
    "limit",
    "retry_after",
    "max_keys",
    "balance_usd",
    "estimated_cost_usd",
    "in_flight_usd",
    "model_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "latency_ms",
    "reason",
    "environment",
    "vertex_configured",
    "database_configured",
    "pool_size",
    "max_overflow",
})


# ── 4. Structured JSON Formatter ──────────────────────────────


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as bounded, single-line JSON with redacted secrets."""

    def __init__(self, environment: str = "development") -> None:
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        try:
            return self._format_record(record)
        except Exception:
            # Safe static fallback on formatting failure (Requirement 10)
            req_id = get_current_request_id() or getattr(record, "request_id", "") or ""
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "level": "ERROR",
                "logger": "app.logging",
                "event": "system.error",
                "message": "Log formatting failed safely",
                "environment": self.environment,
                "request_id": str(req_id)[:64],
            })

    def _format_record(self, record: logging.LogRecord) -> str:
        # Resolve request_id from record attribute or contextvar
        request_id = getattr(record, "request_id", None) or get_current_request_id() or ""

        # Validate event name against allowlist
        event_name = getattr(record, "event", None)
        if not event_name or not isinstance(event_name, str) or event_name not in _ALLOWED_EVENTS:
            event_name = "system.error" if record.levelno >= logging.ERROR else (
                "system.warning" if record.levelno >= logging.WARNING else "system.info"
            )

        # Sanitize and bound log message
        raw_msg = record.getMessage()
        sanitized_msg = sanitize_text(raw_msg)[:_MAX_STRING_LENGTH]

        # Stable UTC ISO-8601 timestamp ending in 'Z'
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        iso_ts = dt.isoformat().replace("+00:00", "Z")

        payload: dict[str, Any] = {
            "timestamp": iso_ts,
            "level": record.levelname,
            "logger": record.name,
            "event": event_name,
            "message": sanitized_msg,
            "environment": self.environment,
            "request_id": str(request_id),
        }

        # Filter allowlisted metadata
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            safe_meta: dict[str, Any] = {}
            for k, v in metadata.items():
                if len(safe_meta) >= _MAX_METADATA_KEYS:
                    break
                if k in _ALLOWED_METADATA_FIELDS:
                    if isinstance(v, (int, float, bool)):
                        safe_meta[k] = v
                    elif isinstance(v, Decimal):
                        safe_meta[k] = str(v)
                    elif isinstance(v, str):
                        safe_meta[k] = sanitize_text(v)[:_MAX_STRING_LENGTH]
            if safe_meta:
                payload["metadata"] = safe_meta

        # Sanitize exception and traceback without local variables (Requirement 7)
        if record.exc_info and record.exc_info[1] is not None:
            exc_type, exc_val, exc_tb = record.exc_info
            exc_type_name = getattr(exc_type, "__name__", "Exception")
            exc_msg = sanitize_text(str(exc_val))[:_MAX_STRING_LENGTH]
            tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
            tb_sanitized = sanitize_text("".join(tb_lines))
            payload["exception"] = {
                "type": exc_type_name,
                "message": exc_msg,
                "traceback": tb_sanitized[:2048],
            }

        return json.dumps(payload, ensure_ascii=False)


# ── 5. Redacting Logging Filter ───────────────────────────────


class RedactingFilter(logging.Filter):
    """Logging filter that redacts secrets directly from LogRecord message before handling."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_text(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                sanitize_text(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                k: sanitize_text(v) if isinstance(v, str) else v
                for k, v in record.args.items()
            }
        return True


# ── 6. Event Logging Helper ───────────────────────────────────


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    metadata: dict[str, Any] | None = None,
    request_id: str | None = None,
    exc_info: Any = None,
) -> None:
    """Emit a structured security or operational log event.

    Args:
        logger: Target standard logger.
        level: Logging severity level (logging.INFO, logging.WARNING, etc.).
        event: Allowlisted event identifier.
        message: Human-readable safe description.
        metadata: Allowlisted metadata dictionary.
        request_id: Optional explicit request correlation ID.
        exc_info: Optional exception tuple or boolean.
    """
    if event not in _ALLOWED_EVENTS:
        event = "system.warning"

    req_id = request_id or get_current_request_id() or ""
    extra: dict[str, Any] = {
        "event": event,
        "metadata": metadata or {},
        "request_id": req_id,
    }
    logger.log(level, message, extra=extra, exc_info=exc_info)


# ── 7. Idempotent Logging Configuration ───────────────────────

_CONFIGURED: bool = False


def configure_logging(
    settings: Any = None,
    force: bool = False,
) -> None:
    """Configure structured JSON logging for the application.

    Idempotent: Only configures once unless force=True.
    Preserves pytest caplog and foreign handlers.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    env_name = getattr(getattr(settings, "rantera_env", None), "value", "development")
    raw_level = str(getattr(settings, "log_level", "INFO")).upper()
    level = getattr(logging, raw_level, logging.INFO)

    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)

    # Ensure RedactingFilter is attached
    has_filter = any(isinstance(f, RedactingFilter) for f in app_logger.filters)
    if not has_filter:
        app_logger.addFilter(RedactingFilter())

    # Check if a StructuredJsonHandler is already attached
    for handler in app_logger.handlers:
        if getattr(handler, "_is_rantera_json_handler", False):
            handler.setLevel(level)
            handler.setFormatter(StructuredJsonFormatter(environment=env_name))
            _CONFIGURED = True
            return

    # Create new stdout handler with StructuredJsonFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(StructuredJsonFormatter(environment=env_name))
    setattr(handler, "_is_rantera_json_handler", True)
    app_logger.addHandler(handler)

    _CONFIGURED = True
