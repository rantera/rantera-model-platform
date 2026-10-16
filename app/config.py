"""Environment-driven application configuration.

Uses Pydantic Settings to load configuration from environment variables
(production) or a local .env file (development only).

Security invariants:
- No hardcoded credentials or secret defaults.
- Production fails closed when required settings are missing.
- Secret values are never printed in logs or exceptions.
- API keys are NOT part of this settings object.
- Database credentials use SecretStr and are never logged or leaked.
"""

from __future__ import annotations

import enum
import logging
import os
import re
import urllib.parse
from typing import Any, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Valid GCP project identifier pattern:
# - Standard project ID: 6-30 lowercase letters, digits, and hyphens; starts with a letter, ends with alphanumeric.
# - Numeric project number: 6-20 digits.
_GCP_PROJECT_ID_REGEX = re.compile(r"^([a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{6,20})$")

_PLACEHOLDER_PREFIXES = (
    "<",
    "pending",
    "todo",
    "fixme",
    "changeme",
    "change-me",
    "change_me",
)
_PLACEHOLDER_EXACT = {
    "none",
    "null",
    "undefined",
    "n/a",
    "na",
    "todo",
    "fixme",
    "changeme",
    "change_me",
    "change-me",
}


class Environment(str, enum.Enum):
    """Application deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    In development, values are loaded from a .env file.
    In production and staging, values MUST be set via environment variables
    or a managed secret mechanism — the .env file is not loaded.
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, *args: Any, _env_file: Any = ..., **values: Any) -> None:
        """Initialize settings with environment-dependent .env loading.

        In development, loads from .env if present.
        In production and staging, disables .env loading.
        Precedence of real environment variables over .env is preserved.
        """
        if _env_file is ...:
            raw_env = values.get("rantera_env") or os.getenv("RANTERA_ENV", "development")
            if isinstance(raw_env, Environment):
                is_dev = raw_env == Environment.DEVELOPMENT
            else:
                is_dev = str(raw_env).lower().strip() == "development"
            _env_file = ".env" if is_dev else None
        super().__init__(*args, _env_file=_env_file, **values)

    # ── Application ──────────────────────────────────────────
    rantera_env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment: development, staging, or production.",
    )
    app_host: str = Field(
        default="127.0.0.1",
        description="Host to bind the server to.",
    )
    app_port: int = Field(
        default=8000,
        description="Port to bind the server to.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    # ── Google Cloud Platform (server-side only) ─────────────
    gcp_project_id: Optional[str] = Field(
        default=None,
        description="GCP project hosting the Vertex AI endpoint(s). Required in production.",
    )
    gcp_region: Optional[str] = Field(
        default=None,
        description="GCP region for Vertex AI. Required in production.",
    )

    # ── Vertex AI (server-side only) ─────────────────────────
    vertex_endpoint_id: Optional[str] = Field(
        default=None,
        description="Vertex AI endpoint ID for rantera-base-v1. Required in production.",
    )

    # ── Rate Limiting ────────────────────────────────────────
    rate_limit_rpm: int = Field(
        default=60,
        description="Maximum requests per minute per API key.",
        ge=1,
        le=10000,
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable or disable rate limiting. In production, disabling requires rate_limit_allow_production_bypass=True.",
    )
    rate_limit_allow_production_bypass: bool = Field(
        default=False,
        description="Explicit operational override required to allow running in production with rate limiting disabled.",
    )

    # ── Database Configuration (Phase 2) ───────────────────
    database_url: Optional[SecretStr] = Field(
        default=None,
        description="Async PostgreSQL connection string (postgresql+asyncpg://...).",
    )
    db_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Base connection pool size per container instance.",
    )
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        le=50,
        description="Maximum burst connections above pool size.",
    )
    db_pool_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Timeout in seconds waiting for an available connection from the pool.",
    )
    db_pool_recycle_seconds: int = Field(
        default=1800,
        ge=60,
        description="Recycle connections older than this age in seconds.",
    )
    db_statement_timeout_ms: int = Field(
        default=3000,
        ge=500,
        le=30000,
        description="Server-side PostgreSQL statement timeout in milliseconds.",
    )
    db_connect_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="TCP/TLS connection establishment timeout in seconds.",
    )
    auth_cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        le=300,
        description="TTL for verified API key records in local memory cache (0 = disabled).",
    )

    # ── Durable Telemetry Configuration ──────────────────────
    telemetry_fail_closed: bool = Field(
        default=True,
        description="Fail closed (HTTP 503) if telemetry cannot be durably committed in production.",
    )
    telemetry_queue_enabled: bool = Field(
        default=False,
        description="Enable asynchronous durable queue publication for telemetry events.",
    )
    pubsub_topic_telemetry: Optional[str] = Field(
        default="mp-inference-telemetry",
        description="Google Cloud Pub/Sub topic name for inference telemetry.",
    )
    pubsub_dead_letter_topic: Optional[str] = Field(
        default="mp-inference-telemetry-dlq",
        description="Google Cloud Pub/Sub dead letter topic name for failed telemetry events.",
    )
    pubsub_push_audience: Optional[str] = Field(
        default=None,
        description="Expected audience for Google Cloud Pub/Sub push OIDC token verification.",
    )
    pubsub_push_service_account_email: Optional[str] = Field(
        default=None,
        description="Authorized service account email for Google Cloud Pub/Sub push delivery.",
    )

    # ── Internal Service Authentication (Web to Gateway) ─────
    internal_service_auth_enabled: bool = Field(
        default=False,
        description="Require internal service authentication for trusted server-to-server calls. When enabled, INTERNAL_SERVICE_KEY is mandatory in production/staging.",
    )
    internal_service_key: Optional[SecretStr] = Field(
        default=None,
        description="Pre-shared secret for trusted server-to-server calls from website to gateway.",
    )

    @field_validator(
        "gcp_project_id",
        "gcp_region",
        "vertex_endpoint_id",
        "pubsub_push_audience",
        "pubsub_push_service_account_email",
        mode="before",
    )
    @classmethod
    def _reject_placeholder_values(cls, v: Optional[str]) -> Optional[str]:
        """Normalize whitespace and reject empty, whitespace-only, and placeholder strings."""
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        stripped = v.strip()
        if not stripped:
            return None
        lower = stripped.lower()
        if lower in _PLACEHOLDER_EXACT:
            return None
        if any(lower.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
            return None
        return stripped

    @field_validator("gcp_project_id")
    @classmethod
    def _validate_gcp_project_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate GCP project identifier format (project ID or numeric project number)."""
        if v is None:
            return None
        if not _GCP_PROJECT_ID_REGEX.match(v):
            raise ValueError(
                f"Invalid GCP project identifier: {v!r}. Must be a valid GCP project ID "
                "(6-30 lowercase characters, digits, hyphens, starting with a letter) or a "
                "numeric project number (6-20 digits)."
            )
        return v

    @field_validator("internal_service_key", mode="before")
    @classmethod
    def _validate_internal_service_key_placeholder(cls, v: Any) -> Optional[SecretStr]:
        """Normalize whitespace and reject placeholder values for internal service key."""
        if v is None:
            return None
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        stripped = raw.strip()
        if not stripped:
            return None
        lower = stripped.lower()
        if lower in _PLACEHOLDER_EXACT or any(lower.startswith(p) for p in _PLACEHOLDER_PREFIXES):
            return None
        return SecretStr(stripped)

    @field_validator("database_url", mode="before")
    @classmethod
    def _validate_database_url_placeholder(cls, v: Any) -> Optional[SecretStr]:
        """Normalize and reject placeholder values for database URL."""
        if v is None:
            return None
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        stripped = raw.strip()
        if not stripped:
            return None
        lower = stripped.lower()
        if lower in _PLACEHOLDER_EXACT or any(lower.startswith(p) for p in _PLACEHOLDER_PREFIXES):
            return None
        # Normalize postgresql:// or postgres:// to postgresql+asyncpg://
        if stripped.startswith("postgres://"):
            stripped = "postgresql+asyncpg://" + stripped[len("postgres://"):]
        elif stripped.startswith("postgresql://"):
            stripped = "postgresql+asyncpg://" + stripped[len("postgresql://"):]

        # Translate sslmode to ssl for asyncpg compatibility
        if "asyncpg" in stripped and "sslmode" in stripped:
            try:
                parsed = urllib.parse.urlsplit(stripped)
                if parsed.query:
                    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                    has_ssl = any(k == "ssl" for k, _ in params)
                    new_params = []
                    for k, val in params:
                        if k == "sslmode":
                            if not has_ssl:
                                new_params.append(("ssl", val))
                                has_ssl = True
                        else:
                            new_params.append((k, val))
                    new_query = urllib.parse.urlencode(new_params)
                    stripped = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))
            except Exception:
                pass

        return SecretStr(stripped)

    @field_validator("database_url")
    @classmethod
    def _validate_database_url_scheme(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate database URL scheme."""
        if v is None:
            return None
        val = v.get_secret_value()
        allowed_schemes = (
            "postgresql+asyncpg://",
            "postgresql://",
            "postgres://",
            "sqlite+aiosqlite://",
        )
        if not any(val.startswith(s) for s in allowed_schemes):
            raise ValueError(
                "Invalid database URL scheme. Expected PostgreSQL URL (postgresql://, postgres://, or postgresql+asyncpg://) or SQLite test URL."
            )
        return v

    @property
    def is_development(self) -> bool:
        """True when running in development environment."""
        return self.rantera_env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """True when running in production or staging."""
        return self.rantera_env in (Environment.PRODUCTION, Environment.STAGING)

    @property
    def vertex_configured(self) -> bool:
        """True when all Vertex AI connection settings are present."""
        return all([
            self.gcp_project_id,
            self.gcp_region,
            self.vertex_endpoint_id,
        ])

    @property
    def database_configured(self) -> bool:
        """True when database_url is present and non-empty."""
        return self.database_url is not None

    def validate_production_readiness(
        self,
        require_internal_service_key: Optional[bool] = None,
    ) -> list[str]:
        """Return a list of missing production-critical settings.

        In production, the application MUST fail to start if this list
        is non-empty. In development, the warnings are logged.
        """
        missing: list[str] = []
        if not self.gcp_project_id:
            missing.append("GCP_PROJECT_ID")
        if not self.gcp_region:
            missing.append("GCP_REGION")
        if not self.vertex_endpoint_id:
            missing.append("VERTEX_ENDPOINT_ID")
        if not self.rate_limit_enabled and not self.rate_limit_allow_production_bypass:
            missing.append("RATE_LIMIT_ENABLED")

        check_internal_key = (
            require_internal_service_key
            if require_internal_service_key is not None
            else self.internal_service_auth_enabled
        )
        if check_internal_key and not self.internal_service_key:
            missing.append("INTERNAL_SERVICE_KEY")

        return missing

    def validate_database_configuration(self) -> list[str]:
        """Return list of missing database configuration settings."""
        missing: list[str] = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        return missing

    def __repr__(self) -> str:
        """Safe repr that never exposes secret-like values."""
        return (
            f"Settings(rantera_env={self.rantera_env.value!r}, "
            f"gcp_project_id={'[SET]' if self.gcp_project_id else '[NOT SET]'}, "
            f"vertex_configured={self.vertex_configured}, "
            f"database_configured={self.database_configured}, "
            f"telemetry_fail_closed={self.telemetry_fail_closed}, "
            f"internal_service_key={'[SET]' if self.internal_service_key else '[NOT SET]'})"
        )

    def __str__(self) -> str:
        """Safe str representation that delegates to __repr__."""
        return self.__repr__()
