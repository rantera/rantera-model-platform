"""Tests for application configuration."""

from __future__ import annotations

import pytest

from app.config import Environment, Settings


# ── Default development settings ─────────────────────────────


def test_default_environment_is_development():
    """Default environment should be development."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.rantera_env == Environment.DEVELOPMENT
    assert not settings.is_production


def test_development_allows_missing_config():
    """Development mode should not raise when Vertex config is missing."""
    settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    missing = settings.validate_production_readiness()
    assert len(missing) == 3
    assert "GCP_PROJECT_ID" in missing
    assert "GCP_REGION" in missing
    assert "VERTEX_ENDPOINT_ID" in missing


# ── Placeholder rejection ────────────────────────────────────


def test_placeholder_values_are_rejected():
    """Placeholder values like '<pending-*>' should be treated as None."""
    settings = Settings(
        gcp_project_id="<pending-project-id>",
        gcp_region="<pending-region>",
        vertex_endpoint_id="<pending-endpoint-id>",
    )
    assert settings.gcp_project_id is None
    assert settings.gcp_region is None
    assert settings.vertex_endpoint_id is None
    assert not settings.vertex_configured


def test_empty_strings_are_rejected():
    """Empty strings should be treated as None."""
    settings = Settings(
        gcp_project_id="",
        gcp_region="",
        vertex_endpoint_id="",
    )
    assert settings.gcp_project_id is None
    assert settings.gcp_region is None
    assert settings.vertex_endpoint_id is None


# ── Vertex configured ────────────────────────────────────────


def test_vertex_configured_when_all_set():
    """vertex_configured should be True when all Vertex settings are present."""
    settings = Settings(
        gcp_project_id="my-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
    )
    assert settings.vertex_configured
    assert settings.validate_production_readiness() == []


def test_vertex_not_configured_when_partial():
    """vertex_configured should be False when any Vertex setting is missing."""
    settings = Settings(
        gcp_project_id="my-project",
        gcp_region=None,
        vertex_endpoint_id="123456",
    )
    assert not settings.vertex_configured
    missing = settings.validate_production_readiness()
    assert "GCP_REGION" in missing


# ── Production fail-closed ───────────────────────────────────


def test_production_missing_config_raises():
    """Production app creation must fail when required config is missing."""
    from app.main import create_app

    settings = Settings(
        rantera_env="production",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    with pytest.raises(RuntimeError, match="missing required configuration"):
        create_app(settings=settings)


def test_staging_missing_config_raises():
    """Staging should also fail closed like production."""
    from app.main import create_app

    settings = Settings(
        rantera_env="staging",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    with pytest.raises(RuntimeError, match="missing required configuration"):
        create_app(settings=settings)


# ── Safe repr and str ────────────────────────────────────────


def test_repr_does_not_leak_secrets():
    """Settings repr must not contain actual project IDs or endpoint IDs."""
    settings = Settings(
        gcp_project_id="super-secret-project",
        gcp_region="us-central1",
        vertex_endpoint_id="9999999",
    )
    r = repr(settings)
    assert "super-secret-project" not in r
    assert "9999999" not in r
    assert "us-central1" not in r
    assert "[SET]" in r


def test_str_delegates_to_repr_and_does_not_leak_secrets():
    """Settings str must delegate to repr and never leak secrets."""
    settings = Settings(
        gcp_project_id="super-secret-project",
        gcp_region="us-central1",
        vertex_endpoint_id="9999999",
    )
    s = str(settings)
    assert s == repr(settings)
    assert "super-secret-project" not in s
    assert "9999999" not in s
    assert "us-central1" not in s


# ── Environment-dependent .env loading ────────────────────────


def test_dotenv_loaded_in_development(tmp_path, monkeypatch):
    """In development, Settings loads values from .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("RATE_LIMIT_RPM=120\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RATE_LIMIT_RPM", raising=False)
    monkeypatch.setenv("RANTERA_ENV", "development")

    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.rate_limit_rpm == 120


def test_dotenv_not_loaded_in_production(tmp_path, monkeypatch):
    """In production, .env file loading is disabled."""
    env_file = tmp_path / ".env"
    env_file.write_text("RATE_LIMIT_RPM=120\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RATE_LIMIT_RPM", raising=False)
    monkeypatch.setenv("RANTERA_ENV", "production")

    settings = Settings(
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
    )
    # Default is 60; the .env value (120) must be ignored
    assert settings.rate_limit_rpm == 60


def test_env_var_precedence_over_dotenv(tmp_path, monkeypatch):
    """Real environment variables take precedence over .env in development."""
    env_file = tmp_path / ".env"
    env_file.write_text("RATE_LIMIT_RPM=120\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANTERA_ENV", "development")
    monkeypatch.setenv("RATE_LIMIT_RPM", "200")

    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.rate_limit_rpm == 200


# ── Extended placeholder and whitespace validation ────────────


def test_whitespace_normalization():
    """Whitespace around valid values is normalized."""
    settings = Settings(
        gcp_project_id="  my-project  ",
        gcp_region="  us-central1  ",
        vertex_endpoint_id="  123456  ",
    )
    assert settings.gcp_project_id == "my-project"
    assert settings.gcp_region == "us-central1"
    assert settings.vertex_endpoint_id == "123456"


def test_whitespace_only_values_rejected():
    """Whitespace-only strings are treated as None."""
    settings = Settings(
        gcp_project_id="   ",
        gcp_region="\t\n  ",
        vertex_endpoint_id="   ",
    )
    assert settings.gcp_project_id is None
    assert settings.gcp_region is None
    assert settings.vertex_endpoint_id is None


@pytest.mark.parametrize(
    "placeholder",
    [
        "TODO",
        "todo",
        "TODO-fill",
        "FIXME",
        "fixme",
        "FIXME_endpoint",
        "changeme",
        "CHANGEME",
        "change-me",
        "change_me",
        "none",
        "NONE",
        "null",
        "NULL",
        "undefined",
        "n/a",
        "<pending-project>",
        "pending-value",
    ],
)
def test_common_placeholders_rejected(placeholder):
    """Common placeholder strings are rejected and treated as None."""
    settings = Settings(
        gcp_project_id=placeholder,
        gcp_region=placeholder,
        vertex_endpoint_id=placeholder,
    )
    assert settings.gcp_project_id is None
    assert settings.gcp_region is None
    assert settings.vertex_endpoint_id is None


# ── GCP Project ID validation ─────────────────────────────────


@pytest.mark.parametrize(
    "valid_pid",
    [
        "my-project",
        "test-project-01",
        "rantera-base-v1",
        "abcdef",  # min length 6
        "a" + "b" * 28 + "c",  # max length 30
    ],
)
def test_valid_gcp_project_ids(valid_pid):
    """Valid GCP project IDs conforming to format are accepted."""
    settings = Settings(gcp_project_id=valid_pid)
    assert settings.gcp_project_id == valid_pid


@pytest.mark.parametrize(
    "invalid_pid",
    [
        "short",  # length 5 (too short)
        "a" * 31,  # length 31 (too long)
        "1starts-with-digit",
        "has_underscore",
        "trailing-hyphen-",
        "-leading-hyphen",
        "UPPERCASE",
        "has.dot",
    ],
)
def test_invalid_gcp_project_ids_fail_closed(invalid_pid):
    """Invalid GCP project IDs are rejected with validation error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(gcp_project_id=invalid_pid)


def test_gcp_project_id_validation_not_applied_to_other_fields():
    """GCP project ID validation is not applied to vertex_endpoint_id or region."""
    # Vertex endpoint IDs are typically numeric digits (which would fail project ID regex)
    settings = Settings(
        vertex_endpoint_id="1234567890123456789",
        gcp_region="us-central1",
    )
    assert settings.vertex_endpoint_id == "1234567890123456789"
    assert settings.gcp_region == "us-central1"


# ── Rate limiting configuration ──────────────────────────────


def test_rate_limit_rpm_bounds_validation():
    """rate_limit_rpm must be between 1 and 10000 inclusive."""
    from pydantic import ValidationError

    s_min = Settings(rate_limit_rpm=1)
    assert s_min.rate_limit_rpm == 1

    s_max = Settings(rate_limit_rpm=10000)
    assert s_max.rate_limit_rpm == 10000

    with pytest.raises(ValidationError):
        Settings(rate_limit_rpm=0)

    with pytest.raises(ValidationError):
        Settings(rate_limit_rpm=10001)


def test_rate_limit_production_readiness_validation():
    """In production, rate_limit_enabled=False fails readiness unless bypass is enabled."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        rate_limit_enabled=False,
    )
    missing = settings.validate_production_readiness()
    assert "RATE_LIMIT_ENABLED" in missing

    settings_bypassed = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        rate_limit_enabled=False,
        rate_limit_allow_production_bypass=True,
    )
    assert settings_bypassed.validate_production_readiness() == []


# ── Secret Management & Environment Security ──────────────────


def test_database_url_loaded_from_environment(monkeypatch):
    """DATABASE_URL is loaded from environment variables and wrapped in SecretStr."""
    fake_url = "postgresql+asyncpg://app_user:secret_pass@127.0.0.1:5432/test_db"
    monkeypatch.setenv("DATABASE_URL", fake_url)

    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.database_configured is True
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == fake_url
    # Secret must never be exposed in str or repr
    assert "secret_pass" not in repr(settings)
    assert "secret_pass" not in str(settings)


def test_database_url_sslmode_translation(monkeypatch):
    """DATABASE_URL containing sslmode is normalized to ssl for asyncpg driver."""
    raw_url = "postgresql://app_user:secret_pass@127.0.0.1:5432/test_db?sslmode=require"
    monkeypatch.setenv("DATABASE_URL", raw_url)

    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.database_configured is True
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == "postgresql+asyncpg://app_user:secret_pass@127.0.0.1:5432/test_db?ssl=require"


def test_safe_development_defaults_have_no_hardcoded_secrets():
    """Settings defaults have no hardcoded secrets or production fallbacks."""
    settings = Settings(
        _env_file=None,
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.rantera_env == Environment.DEVELOPMENT
    assert settings.database_url is None
    assert settings.database_configured is False
    assert settings.validate_database_configuration() == ["DATABASE_URL"]


def test_env_file_and_credentials_are_git_ignored():
    """Verify that .gitignore properly excludes .env and cloud credential files."""
    from pathlib import Path
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    gitignore_path = repo_root / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"

    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert ".env.*" in content
    assert "service-account*.json" in content
    assert "*credentials*.json" in content

    # Verify that git does not track any real .env or credential files
    result = subprocess.run(
        ["git", "ls-files", ".env", "*.pem", "*.key", "service-account*.json"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_secrets = [f for f in result.stdout.strip().splitlines() if f]
    assert tracked_secrets == [], f"Secret files must not be tracked in git: {tracked_secrets}"


def test_env_example_contains_only_safe_placeholders():
    """Ensure .env.example contains only non-sensitive placeholders and no real secrets."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    env_example_path = repo_root / ".env.example"
    assert env_example_path.exists(), ".env.example must exist"

    lines = env_example_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, val = stripped.split("=", 1)
            # Keys with sensitive names must only contain placeholders (<...>)
            if any(s in key.lower() for s in ("secret", "key", "password", "token", "url")):
                assert val.startswith("<") and val.endswith(">"), (
                    f"Sensitive key {key} in .env.example must use <placeholder> format, got: {val}"
                )


# ── Production Configuration Validation ──────────────────────


def test_production_database_configuration_validation():
    """In production, missing database URL is detected and reported by validate_database_configuration."""
    settings_missing = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        database_url=None,
    )
    assert settings_missing.is_production is True
    assert settings_missing.database_configured is False
    assert settings_missing.validate_database_configuration() == ["DATABASE_URL"]

    settings_configured = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    assert settings_configured.database_configured is True
    assert settings_configured.validate_database_configuration() == []


def test_production_readiness_validation_reports_all_missing_vertex_settings():
    """validate_production_readiness accurately identifies all missing required GCP settings in production."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    missing = settings.validate_production_readiness()
    assert "GCP_PROJECT_ID" in missing
    assert "GCP_REGION" in missing
    assert "VERTEX_ENDPOINT_ID" in missing
    assert settings.vertex_configured is False





# ── Internal Service Key Production Readiness ────────────────


def test_production_rejects_missing_internal_service_key_when_auth_enabled():
    """Production readiness reports missing INTERNAL_SERVICE_KEY when internal service auth is enabled."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        internal_service_auth_enabled=True,
        internal_service_key=None,
    )
    missing = settings.validate_production_readiness()
    assert "INTERNAL_SERVICE_KEY" in missing


def test_staging_rejects_missing_internal_service_key_when_auth_enabled():
    """Staging readiness reports missing INTERNAL_SERVICE_KEY when internal service auth is enabled."""
    settings = Settings(
        rantera_env="staging",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        internal_service_auth_enabled=True,
        internal_service_key=None,
    )
    missing = settings.validate_production_readiness()
    assert "INTERNAL_SERVICE_KEY" in missing


def test_production_rejects_placeholder_internal_service_key_when_auth_enabled():
    """Placeholder values for internal_service_key are treated as None and reported as missing."""
    for placeholder in ["<pending-service-key>", "todo", "changeme", " "]:
        settings = Settings(
            rantera_env="production",
            gcp_project_id="valid-project",
            gcp_region="us-central1",
            vertex_endpoint_id="123456",
            internal_service_auth_enabled=True,
            internal_service_key=placeholder,
        )
        assert settings.internal_service_key is None
        missing = settings.validate_production_readiness()
        assert "INTERNAL_SERVICE_KEY" in missing


def test_production_accepts_valid_internal_service_key_when_auth_enabled():
    """Production readiness succeeds when a valid internal service key is provided."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        internal_service_auth_enabled=True,
        internal_service_key="secret-production-service-token-98765",
    )
    assert settings.internal_service_key is not None
    missing = settings.validate_production_readiness()
    assert "INTERNAL_SERVICE_KEY" not in missing
    assert missing == []


def test_production_allows_missing_internal_service_key_when_delegation_disabled():
    """Deployments that intentionally do not use internal delegation are not broken."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        internal_service_auth_enabled=False,
        internal_service_key=None,
    )
    missing = settings.validate_production_readiness()
    assert "INTERNAL_SERVICE_KEY" not in missing
    assert missing == []


def test_internal_service_key_never_exposed_in_repr_or_str():
    """Settings repr and str must never expose internal service key value."""
    secret_val = "super-secret-key-do-not-leak-12345"
    settings = Settings(
        internal_service_key=secret_val,
    )
    repr_str = repr(settings)
    str_str = str(settings)

    assert secret_val not in repr_str
    assert secret_val not in str_str
    assert "internal_service_key=[SET]" in repr_str
    assert "internal_service_key=[SET]" in str_str


def test_production_startup_fails_closed_when_internal_service_key_missing_and_auth_enabled():
    """create_app in production raises RuntimeError when required internal service key is absent."""
    from app.main import create_app

    settings = Settings(
        rantera_env="production",
        gcp_project_id="valid-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        internal_service_auth_enabled=True,
        internal_service_key=None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        create_app(settings)

    err_msg = str(exc_info.value)
    assert "INTERNAL_SERVICE_KEY" in err_msg
    assert "missing required configuration" in err_msg
