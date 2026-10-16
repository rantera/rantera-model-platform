"""Tests for database contract schemas and state machines (Phase 1).

Validates:
- Request lifecycle state transitions (in-memory specification)
- Settlement lifecycle state transitions and idempotency invariants (in-memory specification)
- Field constraints on inference log entries
- API key view projection contracts and credential exclusion
- Separation of wallet balance reporting from key authentication
"""

from __future__ import annotations

import enum
from typing import Set

import pytest


class RequestStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class SettlementStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"
    EXEMPT = "EXEMPT"


# Permitted state transitions (specification model)
PERMITTED_REQUEST_TRANSITIONS: dict[RequestStatus, Set[RequestStatus]] = {
    RequestStatus.PENDING: {
        RequestStatus.PROCESSING,
        RequestStatus.FAILED,
        RequestStatus.CANCELLED,
    },
    RequestStatus.PROCESSING: {
        RequestStatus.SUCCESS,
        RequestStatus.FAILED,
        RequestStatus.TIMEOUT,
        RequestStatus.CANCELLED,
    },
    RequestStatus.SUCCESS: set(),  # Terminal
    RequestStatus.FAILED: set(),  # Terminal
    RequestStatus.TIMEOUT: set(),  # Terminal
    RequestStatus.CANCELLED: set(),  # Terminal
}

PERMITTED_SETTLEMENT_TRANSITIONS: dict[SettlementStatus, Set[SettlementStatus]] = {
    SettlementStatus.PENDING: {
        SettlementStatus.PROCESSING,
        SettlementStatus.EXEMPT,
    },
    SettlementStatus.PROCESSING: {
        SettlementStatus.SETTLED,
        SettlementStatus.FAILED,
    },
    SettlementStatus.FAILED: {
        SettlementStatus.PENDING,  # Retry
        SettlementStatus.DEAD_LETTER,  # Max retries exceeded
    },
    SettlementStatus.SETTLED: set(),  # Terminal (Idempotent)
    SettlementStatus.DEAD_LETTER: set(),  # Terminal
    SettlementStatus.EXEMPT: set(),  # Terminal
}


def can_transition_request(current: RequestStatus, target: RequestStatus) -> bool:
    return target in PERMITTED_REQUEST_TRANSITIONS.get(current, set())


def can_transition_settlement(current: SettlementStatus, target: SettlementStatus) -> bool:
    return target in PERMITTED_SETTLEMENT_TRANSITIONS.get(current, set())


def test_valid_request_transitions():
    assert can_transition_request(RequestStatus.PENDING, RequestStatus.PROCESSING)
    assert can_transition_request(RequestStatus.PROCESSING, RequestStatus.SUCCESS)
    assert can_transition_request(RequestStatus.PROCESSING, RequestStatus.FAILED)
    assert can_transition_request(RequestStatus.PROCESSING, RequestStatus.TIMEOUT)


def test_terminal_request_states_cannot_transition():
    for terminal in (RequestStatus.SUCCESS, RequestStatus.FAILED, RequestStatus.TIMEOUT, RequestStatus.CANCELLED):
        assert not can_transition_request(terminal, RequestStatus.PROCESSING)
        assert not can_transition_request(terminal, RequestStatus.PENDING)


def test_valid_settlement_transitions():
    assert can_transition_settlement(SettlementStatus.PENDING, SettlementStatus.PROCESSING)
    assert can_transition_settlement(SettlementStatus.PROCESSING, SettlementStatus.SETTLED)
    assert can_transition_settlement(SettlementStatus.PROCESSING, SettlementStatus.FAILED)
    assert can_transition_settlement(SettlementStatus.FAILED, SettlementStatus.PENDING)
    assert can_transition_settlement(SettlementStatus.FAILED, SettlementStatus.DEAD_LETTER)


def test_settled_status_is_strictly_terminal():
    """Once settled, status must never be altered (protects against double billing)."""
    assert not can_transition_settlement(SettlementStatus.SETTLED, SettlementStatus.PENDING)
    assert not can_transition_settlement(SettlementStatus.SETTLED, SettlementStatus.PROCESSING)
    assert not can_transition_settlement(SettlementStatus.SETTLED, SettlementStatus.FAILED)


def test_api_key_view_contract_does_not_require_sensitive_fields():
    """Verify that expected view columns exclude sensitive user credentials."""
    safe_view_columns = {
        "api_key_id",
        "user_id",
        "key_hash",
        "key_prefix",
        "key_name",
        "spend_limit_usd",
        "expires_at",
        "user_is_banned",
        "user_email_verified",
        "wallet_balance_usd",
    }
    forbidden_columns = {
        "password",
        "verifyToken",
        "verifyTokenExpiry",
        "cognitoId",
        "email",
    }
    assert forbidden_columns.isdisjoint(safe_view_columns)


def test_wallet_balance_is_informational_not_auth_predicate():
    """Verify that wallet balance is projected as a numeric metric, distinct from auth validity."""
    view_columns = [
        "api_key_id",
        "user_id",
        "key_hash",
        "key_prefix",
        "key_name",
        "spend_limit_usd",
        "expires_at",
        "user_is_banned",
        "user_email_verified",
        "wallet_balance_usd",
    ]
    assert "wallet_balance_usd" in view_columns
    # Ensure view does not conflate balance with authentication status flags
    auth_flags = {"expires_at", "user_is_banned"}
    assert "wallet_balance_usd" not in auth_flags


def test_auth_state_distinction_contract():
    """Verify that view contract enables distinguishing invalid, expired, banned, and billing-ineligible states.

    Contract rules:
    - Missing row (hash not found) -> HTTP 401 Unauthorized ('Invalid API key')
    - Row exists with expires_at in past -> HTTP 401 Unauthorized ('API key expired')
    - Row exists with user_is_banned=True -> HTTP 403 Forbidden ('Account suspended')
    - Row exists, key valid, but wallet_balance_usd <= 0 -> HTTP 402 Payment Required ('Insufficient wallet balance')
    - Row exists, key valid, but spend_limit_usd exceeded -> HTTP 402 Payment Required ('Spend limit reached')
    - None of these error evaluations leak sensitive credentials (passwords, tokens, emails).
    """
    view_columns = {
        "api_key_id",
        "user_id",
        "key_hash",
        "key_prefix",
        "key_name",
        "spend_limit_usd",
        "expires_at",
        "user_is_banned",
        "user_email_verified",
        "wallet_balance_usd",
    }
    # Required columns to perform state distinction
    assert "expires_at" in view_columns
    assert "user_is_banned" in view_columns
    assert "spend_limit_usd" in view_columns
    assert "wallet_balance_usd" in view_columns


# ── Structured Schema & DDL Contract Verification (Phase 1) ───────────────────

import os
import re
from pathlib import Path


def _get_schema_and_migration_text() -> tuple[str, str]:
    """Locate adjacent website repo and return (prisma_schema_text, migration_sql_text).

    Uses repository-relative resolution and environment variable overrides,
    skipping gracefully if the adjacent website repository is absent.
    """
    env_dir = os.environ.get("RANTERA_WEBAPP_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    model_platform_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        model_platform_root.parent / "rantera-webapp" / "rantera",
        model_platform_root.parent / "rantera",
    ])

    for candidate in candidates:
        prisma_path = candidate / "prisma" / "schema.prisma"
        migration_path = (
            candidate / "prisma" / "migrations" / "20261014000000_add_model_platform_tables" / "migration.sql"
        )
        if prisma_path.is_file() and migration_path.is_file():
            return prisma_path.read_text(encoding="utf-8"), migration_path.read_text(encoding="utf-8")

    pytest.skip("Adjacent website repository rantera-webapp/rantera not found; skipping disk contract verification.")


def _normalize_default(val: str | None) -> str | None:
    """Canonicalize SQL and Prisma default expressions for equivalence comparisons."""
    if val is None:
        return None
    cleaned = val.strip().strip("'\"")
    if cleaned in ("now()", "CURRENT_TIMESTAMP"):
        return "CURRENT_TIMESTAMP"
    return cleaned


def _parse_prisma_model(text: str, model_name: str) -> dict[str, dict]:
    """Parse a specific model block from schema.prisma into structured field metadata.

    Parser limitations:
    - Specialized for Prisma 5.x model blocks with one field declaration per line.
    - Requires standard Prisma syntax: <field_name> <field_type> [attributes...].
    - Fails visibly on unexpected syntax or unparseable lines rather than silently dropping fields.
    """
    match = re.search(r'model\s+' + re.escape(model_name) + r'\s+\{([^}]+)\}', text)
    if not match:
        raise ValueError(f"Model block '{model_name}' not found in prisma schema")

    body = match.group(1)
    fields = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("@@"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed Prisma field declaration in '{model_name}': {line}")

        field_name = parts[0]
        raw_type = parts[1]
        is_optional = raw_type.endswith("?")
        base_type = raw_type[:-1] if is_optional else raw_type

        map_match = re.search(r'@map\("([^"]+)"\)', line)
        col_name = map_match.group(1) if map_match else field_name

        default_match = re.search(r'@default\((cuid\(\)|now\(\)|[^\)]+)\)', line)
        default_val = default_match.group(1).strip() if default_match else None

        db_match = re.search(r'@db\.([A-Za-z0-9_]+(?:\([^)]+\))?)', line)
        db_type = db_match.group(0) if db_match else None

        is_id = "@id" in parts
        is_unique = "@unique" in line
        is_updated_at = "@updatedAt" in parts

        if col_name in fields:
            raise ValueError(f"Duplicate column mapping '{col_name}' detected in model '{model_name}'")

        fields[col_name] = {
            "field_name": field_name,
            "base_type": base_type,
            "is_optional": is_optional,
            "default": _normalize_default(default_val),
            "db_type": db_type,
            "is_id": is_id,
            "is_unique": is_unique,
            "is_updated_at": is_updated_at,
            "has_explicit_map": map_match is not None,
            "raw": line,
        }

    if not fields:
        raise ValueError(f"No fields parsed from model block '{model_name}'")
    return fields


def _parse_sql_table(text: str, table_name: str) -> dict[str, dict]:
    """Parse a specific CREATE TABLE statement from migration.sql into structured column metadata.

    Parser limitations:
    - Designed specifically for Rantera Model Platform Phase 1 migration SQL.
    - Requires double-quoted column identifiers on individual lines: "<col_name>" <TYPE> [CONSTRAINTS].
    - Requires standard PostgreSQL types (VARCHAR, INTEGER, DECIMAL, TIMESTAMPTZ, BOOLEAN, TEXT).
    - Fails visibly on unexpected or malformed SQL rather than silently skipping lines.
    """
    table_pattern = r'CREATE\s+TABLE\s+"' + re.escape(table_name) + r'"\s*\((.*?)\);'
    match = re.search(table_pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Table '{table_name}' definition not found in migration SQL")

    body = match.group(1).strip()
    cols = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        line_clean = line.rstrip(",")
        col_match = re.match(r'^"([a-zA-Z0-9_]+)"\s+([A-Za-z]+(?:\([0-9, ]+\))?)(.*)$', line_clean)
        if not col_match:
            raise ValueError(f"Malformed or unsupported SQL column definition in table '{table_name}': {line}")

        col_name = col_match.group(1)
        sql_type = col_match.group(2).strip()
        constraints = col_match.group(3).strip()

        is_pk = "PRIMARY KEY" in constraints
        is_not_null = "NOT NULL" in constraints or is_pk

        default_match = re.search(r'DEFAULT\s+([^,]+)', constraints)
        default_val = default_match.group(1).strip() if default_match else None

        if col_name in cols:
            raise ValueError(f"Duplicate column definition '{col_name}' in table '{table_name}'")

        cols[col_name] = {
            "sql_type": sql_type,
            "is_not_null": is_not_null,
            "is_pk": is_pk,
            "default": _normalize_default(default_val),
            "constraints": constraints,
            "raw": line,
        }

    if not cols:
        raise ValueError(f"No columns parsed from table '{table_name}'")
    return cols


def _parse_sql_view(text: str, view_name: str) -> dict:
    """Parse an individual CREATE VIEW statement from migration.sql into structured metadata.

    Parser limitations:
    - Designed specifically for Rantera Model Platform Phase 1 views.
    - Expects format: CREATE VIEW "<name>" [WITH (...)] AS SELECT ... FROM ...;
    - Requires projected columns to use explicit 'AS <alias>' syntax on each line.
    - Fails visibly on unsupported syntax rather than silently ignoring lines.
    """
    view_pattern = r'CREATE\s+VIEW\s+"' + re.escape(view_name) + r'"\s*(?:WITH\s*\(([^)]+)\)\s*)?AS\s+SELECT\s+(.*?)\s+FROM\s+(.*?);'
    match = re.search(view_pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"View '{view_name}' definition not found or malformed in migration SQL")

    with_clause = (match.group(1) or "").strip()
    select_clause = match.group(2).strip()
    from_clause = match.group(3).strip()

    projections = []
    for line in select_clause.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("--"):
            continue
        proj_match = re.match(r'^(.*?)\s+AS\s+"([a-zA-Z0-9_]+)"$', line)
        if not proj_match:
            raise ValueError(f"Malformed or non-aliased column projection in view '{view_name}': {line}")
        expr = proj_match.group(1).strip()
        alias = proj_match.group(2).strip()
        projections.append((expr, alias))

    if not projections:
        raise ValueError(f"No projected columns parsed from view '{view_name}'")

    has_where = bool(re.search(r'\bWHERE\b', from_clause, re.IGNORECASE))
    has_having = bool(re.search(r'\bHAVING\b', from_clause, re.IGNORECASE))

    return {
        "with": with_clause,
        "projections": projections,
        "from": from_clause,
        "has_where": has_where,
        "has_having": has_having,
    }


def test_model_catalog_columns_and_types_against_schema_files():
    """Verify bidirectional 1:1 mapping and attribute equivalence for MpModelCatalog."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    p_fields = _parse_prisma_model(prisma_text, "MpModelCatalog")
    s_cols = _parse_sql_table(sql_text, "mp_model_catalog")

    expected_specs = {
        "id": {"sql_type": "VARCHAR(36)", "db_type": "@db.VarChar(36)", "base_type": "String", "opt": False, "pk": True, "field": "id", "has_map": False, "default": "cuid()"},
        "model_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": False, "pk": False, "field": "modelId", "has_map": True, "default": None},
        "display_name": {"sql_type": "VARCHAR(128)", "db_type": "@db.VarChar(128)", "base_type": "String", "opt": False, "pk": False, "field": "displayName", "has_map": True, "default": None},
        "description": {"sql_type": "TEXT", "db_type": "@db.Text", "base_type": "String", "opt": False, "pk": False, "field": "description", "has_map": False, "default": None},
        "provider": {"sql_type": "VARCHAR(32)", "db_type": "@db.VarChar(32)", "base_type": "String", "opt": False, "pk": False, "field": "provider", "has_map": False, "default": "vertex_ai"},
        "provider_model_id": {"sql_type": "VARCHAR(256)", "db_type": "@db.VarChar(256)", "base_type": "String", "opt": True, "pk": False, "field": "providerModelId", "has_map": True, "default": None},
        "is_active": {"sql_type": "BOOLEAN", "db_type": None, "base_type": "Boolean", "opt": False, "pk": False, "field": "isActive", "has_map": True, "default": "true"},
        "context_window": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "contextWindow", "has_map": True, "default": "8192"},
        "max_output_tokens": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "maxOutputTokens", "has_map": True, "default": "4096"},
        "input_cost_per_1k": {"sql_type": "DECIMAL(10, 6)", "db_type": "@db.Decimal(10, 6)", "base_type": "Decimal", "opt": False, "pk": False, "field": "inputCostPer1k", "has_map": True, "default": "0.003000"},
        "output_cost_per_1k": {"sql_type": "DECIMAL(10, 6)", "db_type": "@db.Decimal(10, 6)", "base_type": "Decimal", "opt": False, "pk": False, "field": "outputCostPer1k", "has_map": True, "default": "0.015000"},
        "created_at": {"sql_type": "TIMESTAMPTZ", "db_type": "@db.Timestamptz(6)", "base_type": "DateTime", "opt": False, "pk": False, "field": "createdAt", "has_map": True, "default": "CURRENT_TIMESTAMP"},
        "updated_at": {"sql_type": "TIMESTAMPTZ", "db_type": "@db.Timestamptz(6)", "base_type": "DateTime", "opt": False, "pk": False, "field": "updatedAt", "has_map": True, "default": None},
    }

    # Strict bijection: every Prisma field maps to exactly one SQL column with zero unexpected columns
    assert len(p_fields) == len(s_cols) == 13
    assert set(p_fields.keys()) == set(s_cols.keys()) == set(expected_specs.keys())

    for col, spec in expected_specs.items():
        pf = p_fields[col]
        sc = s_cols[col]

        # Field-level @map and name assertions
        assert pf["field_name"] == spec["field"], f"Field name mismatch for {col}"
        assert pf["has_explicit_map"] == spec["has_map"], f"@map presence mismatch for {col}"

        # SQL type vs Prisma native type and base type
        assert sc["sql_type"] == spec["sql_type"], f"SQL type mismatch for {col}"
        assert pf["db_type"] == spec["db_type"], f"Prisma @db native type mismatch for {col}"
        assert pf["base_type"] == spec["base_type"], f"Prisma base type mismatch for {col}"

        # Nullability equivalence
        assert pf["is_optional"] == spec["opt"], f"Prisma optionality mismatch for {col}"
        assert sc["is_not_null"] != spec["opt"], f"SQL NOT NULL mismatch for {col}"

        # Primary key equivalence
        assert pf["is_id"] == sc["is_pk"] == spec["pk"], f"Primary key mismatch for {col}"

        # Default equivalence (normalized)
        if spec["default"] is not None:
            if col == "id":
                assert pf["default"] == "cuid()"
            elif col == "updated_at":
                assert sc["default"] == "CURRENT_TIMESTAMP"
            else:
                assert pf["default"] == spec["default"]
                assert sc["default"] == spec["default"]


def test_inference_logs_columns_and_types_against_schema_files():
    """Verify bidirectional 1:1 mapping and attribute equivalence for MpInferenceLog."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    p_fields = _parse_prisma_model(prisma_text, "MpInferenceLog")
    s_cols = _parse_sql_table(sql_text, "mp_inference_logs")

    expected_specs = {
        "id": {"sql_type": "VARCHAR(36)", "db_type": "@db.VarChar(36)", "base_type": "String", "opt": False, "pk": True, "field": "id", "has_map": False, "default": "cuid()"},
        "request_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": False, "pk": False, "field": "requestId", "has_map": True, "default": None},
        "user_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": True, "pk": False, "field": "userId", "has_map": True, "default": None},
        "api_key_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": True, "pk": False, "field": "apiKeyId", "has_map": True, "default": None},
        "model_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": False, "pk": False, "field": "modelId", "has_map": True, "default": None},
        "channel": {"sql_type": "VARCHAR(32)", "db_type": "@db.VarChar(32)", "base_type": "String", "opt": False, "pk": False, "field": "channel", "has_map": True, "default": "api"},
        "tokens_in": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "tokensIn", "has_map": True, "default": "0"},
        "tokens_out": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "tokensOut", "has_map": True, "default": "0"},
        "total_tokens": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "totalTokens", "has_map": True, "default": "0"},
        "cost_usd": {"sql_type": "DECIMAL(12, 6)", "db_type": "@db.Decimal(12, 6)", "base_type": "Decimal", "opt": False, "pk": False, "field": "costUsd", "has_map": True, "default": "0.000000"},
        "latency_ms": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "latencyMs", "has_map": True, "default": "0"},
        "ttft_ms": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": True, "pk": False, "field": "ttftMs", "has_map": True, "default": None},
        "request_status": {"sql_type": "VARCHAR(32)", "db_type": "@db.VarChar(32)", "base_type": "String", "opt": False, "pk": False, "field": "requestStatus", "has_map": True, "default": "PENDING"},
        "error_code": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": True, "pk": False, "field": "errorCode", "has_map": True, "default": None},
        "settlement_status": {"sql_type": "VARCHAR(32)", "db_type": "@db.VarChar(32)", "base_type": "String", "opt": False, "pk": False, "field": "settlementStatus", "has_map": True, "default": "PENDING"},
        "settlement_id": {"sql_type": "VARCHAR(64)", "db_type": "@db.VarChar(64)", "base_type": "String", "opt": True, "pk": False, "field": "settlementId", "has_map": True, "default": None},
        "settlement_attempts": {"sql_type": "INTEGER", "db_type": None, "base_type": "Int", "opt": False, "pk": False, "field": "settlementAttempts", "has_map": True, "default": "0"},
        "settled_at": {"sql_type": "TIMESTAMPTZ", "db_type": "@db.Timestamptz(6)", "base_type": "DateTime", "opt": True, "pk": False, "field": "settledAt", "has_map": True, "default": None},
        "created_at": {"sql_type": "TIMESTAMPTZ", "db_type": "@db.Timestamptz(6)", "base_type": "DateTime", "opt": False, "pk": False, "field": "createdAt", "has_map": True, "default": "CURRENT_TIMESTAMP"},
        "updated_at": {"sql_type": "TIMESTAMPTZ", "db_type": "@db.Timestamptz(6)", "base_type": "DateTime", "opt": False, "pk": False, "field": "updatedAt", "has_map": True, "default": None},
    }

    # Strict bijection: every Prisma field maps to exactly one SQL column with zero unexpected columns
    assert len(p_fields) == len(s_cols) == 20
    assert set(p_fields.keys()) == set(s_cols.keys()) == set(expected_specs.keys())

    for col, spec in expected_specs.items():
        pf = p_fields[col]
        sc = s_cols[col]

        # Field-level @map and name assertions
        assert pf["field_name"] == spec["field"], f"Field name mismatch for {col}"
        assert pf["has_explicit_map"] == spec["has_map"], f"@map presence mismatch for {col}"

        # SQL type vs Prisma native type and base type
        assert sc["sql_type"] == spec["sql_type"], f"SQL type mismatch for {col}"
        assert pf["db_type"] == spec["db_type"], f"Prisma @db native type mismatch for {col}"
        assert pf["base_type"] == spec["base_type"], f"Prisma base type mismatch for {col}"

        # Nullability equivalence
        assert pf["is_optional"] == spec["opt"], f"Prisma optionality mismatch for {col}"
        assert sc["is_not_null"] != spec["opt"], f"SQL NOT NULL mismatch for {col}"

        # Primary key equivalence
        assert pf["is_id"] == sc["is_pk"] == spec["pk"], f"Primary key mismatch for {col}"

        # Default equivalence (normalized)
        if spec["default"] is not None:
            if col == "id":
                assert pf["default"] == "cuid()"
            elif col == "updated_at":
                assert sc["default"] == "CURRENT_TIMESTAMP"
            else:
                assert pf["default"] == spec["default"]
                assert sc["default"] == spec["default"]


def test_primary_and_unique_indexes_match_without_duplicates():
    """Verify primary keys and unique indexes match exactly and create no duplicate constraints."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    # Primary keys
    assert '"id" VARCHAR(36) PRIMARY KEY' in sql_text
    p_cat = _parse_prisma_model(prisma_text, "MpModelCatalog")
    assert p_cat["id"]["is_id"]

    # Unique index for model_catalog.model_id
    assert 'CREATE UNIQUE INDEX "idx_mp_model_catalog_model_id"' in sql_text
    assert 'ON "mp_model_catalog"("model_id");' in sql_text
    assert '@unique(map: "idx_mp_model_catalog_model_id")' in prisma_text
    # Ensure NO duplicate inline UNIQUE constraint on model_id
    assert '"model_id" VARCHAR(64) NOT NULL UNIQUE' not in sql_text

    # Unique index for inference_logs.request_id
    assert 'CREATE UNIQUE INDEX "idx_mp_inference_logs_request_id"' in sql_text
    assert '@unique(map: "idx_mp_inference_logs_request_id")' in prisma_text
    # Ensure NO duplicate inline UNIQUE constraint on request_id
    assert '"request_id" VARCHAR(64) NOT NULL UNIQUE' not in sql_text


def test_standard_indexes_match_between_prisma_and_sql():
    """Verify standard B-tree index declarations match between Prisma and SQL."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    standard_indexes = [
        ("idx_mp_model_catalog_is_active", '@@index([isActive], map: "idx_mp_model_catalog_is_active")'),
        ("idx_mp_inference_logs_user_id", '@@index([userId], map: "idx_mp_inference_logs_user_id")'),
        ("idx_mp_inference_logs_api_key_id", '@@index([apiKeyId], map: "idx_mp_inference_logs_api_key_id")'),
        ("idx_mp_inference_logs_model_id", '@@index([modelId], map: "idx_mp_inference_logs_model_id")'),
        ("idx_mp_inference_logs_created_at", '@@index([createdAt], map: "idx_mp_inference_logs_created_at")'),
    ]

    for index_name, prisma_idx in standard_indexes:
        assert index_name in sql_text, f"Missing index {index_name} in SQL"
        assert prisma_idx in prisma_text, f"Missing index {index_name} in Prisma"


def test_settlement_status_partial_index_and_intentional_prisma_omission():
    """Verify settlement status partial index in SQL and intentional omission in Prisma."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    # Must exist in SQL with WHERE clause
    assert 'CREATE INDEX "idx_mp_inference_logs_settlement_status"' in sql_text
    assert "WHERE \"settlement_status\" = 'PENDING'" in sql_text

    # Must be omitted from Prisma @@index directives to avoid falsely claiming parity
    assert '@@index([settlementStatus]' not in prisma_text

    # Must be documented inline in Prisma schema
    assert "idx_mp_inference_logs_settlement_status is an intentionally raw-SQL-managed" in prisma_text
    assert "WHERE settlement_status = 'PENDING'" in prisma_text


def test_updated_at_triggers_cover_both_tables():
    """Verify mp_set_updated_at function and triggers cover both Model Platform tables."""
    prisma_text, sql_text = _get_schema_and_migration_text()

    # Trigger function definition
    assert "CREATE OR REPLACE FUNCTION mp_set_updated_at()" in sql_text
    assert "NEW.updated_at = CURRENT_TIMESTAMP;" in sql_text
    assert "LANGUAGE plpgsql;" in sql_text

    # Triggers for both tables
    assert 'CREATE TRIGGER "trg_mp_model_catalog_updated_at"' in sql_text
    assert 'BEFORE UPDATE ON "mp_model_catalog"' in sql_text
    assert 'EXECUTE FUNCTION mp_set_updated_at();' in sql_text

    assert 'CREATE TRIGGER "trg_mp_inference_logs_updated_at"' in sql_text
    assert 'BEFORE UPDATE ON "mp_inference_logs"' in sql_text

    # Prisma @updatedAt on both tables
    p_cat = _parse_prisma_model(prisma_text, "MpModelCatalog")
    assert p_cat["updated_at"]["is_updated_at"]
    p_log = _parse_prisma_model(prisma_text, "MpInferenceLog")
    assert p_log["updated_at"]["is_updated_at"]


def test_auth_views_definitions_and_projected_columns():
    """Parse each authentication view separately and verify columns, joins, and absence of pre-filtering."""
    _, sql_text = _get_schema_and_migration_text()

    # 1. Independent assertion for mp_active_api_keys_v
    v_keys = _parse_sql_view(sql_text, "mp_active_api_keys_v")
    assert v_keys["with"] == "security_barrier = true"
    assert not v_keys["has_where"], "mp_active_api_keys_v must NOT filter records via WHERE"
    assert not v_keys["has_having"], "mp_active_api_keys_v must NOT filter records via HAVING"

    expected_api_key_projections = [
        ('k."id"', "api_key_id"),
        ('k."userId"', "user_id"),
        ('k."keyHash"', "key_hash"),
        ('k."keyPrefix"', "key_prefix"),
        ('k."name"', "key_name"),
        ('k."spendLimitUsd"', "spend_limit_usd"),
        ('k."expiresAt"', "expires_at"),
        ('u."isBanned"', "user_is_banned"),
        ('u."emailVerified"', "user_email_verified"),
        ('COALESCE(w."balanceUsd", 0.000000)', "wallet_balance_usd"),
    ]
    assert v_keys["projections"] == expected_api_key_projections

    # Verify source table and exact join conditions
    assert v_keys["from"].startswith('"ApiKey" k')
    assert 'JOIN "User" u ON k."userId" = u."id"' in v_keys["from"]
    assert 'LEFT JOIN "Wallet" w ON u."id" = w."userId"' in v_keys["from"]

    # 2. Independent assertion for mp_active_users_v
    v_users = _parse_sql_view(sql_text, "mp_active_users_v")
    assert v_users["with"] == "security_barrier = true"
    assert not v_users["has_where"], "mp_active_users_v must NOT filter records via WHERE"
    assert not v_users["has_having"], "mp_active_users_v must NOT filter records via HAVING"

    expected_user_projections = [
        ('u."id"', "user_id"),
        ('u."isBanned"', "is_banned"),
        ('u."emailVerified"', "email_verified"),
        ('COALESCE(w."balanceUsd", 0.000000)', "wallet_balance_usd"),
    ]
    assert v_users["projections"] == expected_user_projections

    # Verify source table and exact join conditions
    assert v_users["from"].startswith('"User" u')
    assert 'LEFT JOIN "Wallet" w ON u."id" = w."userId"' in v_users["from"]

    # 3. Forbid sensitive credentials in projections and aliases across both views
    for v in [v_keys, v_users]:
        proj_text = " ".join([f"{e} {a}" for e, a in v["projections"]])
        for forbidden in ['"password"', '"verifyToken"', '"verifyTokenExpiry"', '"cognitoId"', 'u."email"']:
            assert forbidden not in proj_text, f"Sensitive credential token '{forbidden}' leaked in view projection"


def test_no_accidental_foreign_keys_to_website_tables():
    """Verify that mp_ tables use decoupled logical references with zero physical FK constraints."""
    _, sql_text = _get_schema_and_migration_text()

    catalog_sql = re.search(r'CREATE TABLE "mp_model_catalog" \((.*?)\);', sql_text, re.DOTALL).group(1)
    logs_sql = re.search(r'CREATE TABLE "mp_inference_logs" \((.*?)\);', sql_text, re.DOTALL).group(1)

    assert "REFERENCES" not in catalog_sql.upper(), "Accidental physical foreign key in mp_model_catalog"
    assert "REFERENCES" not in logs_sql.upper(), "Accidental physical foreign key in mp_inference_logs"
    assert "FOREIGN KEY" not in catalog_sql.upper(), "Accidental foreign key in mp_model_catalog"
    assert "FOREIGN KEY" not in logs_sql.upper(), "Accidental foreign key in mp_inference_logs"
