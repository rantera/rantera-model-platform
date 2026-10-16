# Rantera Model Platform — Phase 2 Database Integration Plan

> **Status:** Integration Design & Verification Blueprint (Part 1 Staged Delivery).
> **Target Service:** `rantera-model-platform` (FastAPI).
> **Target Database:** Shared PostgreSQL instance managed by `rantera-webapp/rantera` (Prisma 5.16.1).
> **Implementation State:** Design & Contract Verified. Zero Phase 2 code, database connections, or migrations executed.

---

## 1. Executive Summary & Architectural Invariants

This document establishes the technical design, security boundary, and operational procedure for Phase 2: connecting the Rantera Model Platform (FastAPI) to the existing Rantera PostgreSQL database.

### Strict Architectural Invariants
1. **DDL Ownership:** The website repository (`rantera-webapp/rantera`) via Prisma versioned migrations strictly owns all database DDL. The Python service must **never** execute migrations, run `ALTER`/`CREATE` statements, or create tables automatically.
2. **Authority Decoupling:** The website remains solely authoritative for user lifecycle, API-key creation, revocation, rotation, wallet balances, payment transactions, and billing receipts. The Model Platform operates as a consumer of authentication data and producer of inference telemetry.
3. **Inference & Routing Ownership:** FastAPI owns model request routing, provider integration, latency tracking, and writes to `mp_inference_logs`.
4. **Least Privilege & Role Isolation:** The Model Platform must never connect using superuser or website application credentials (`rantera_app_owner`). It connects via a dedicated, unprivileged PostgreSQL role (`rantera_mp_service`) restricted to approved views and tables.
5. **Fail-Closed Principle:** The platform strictly fails closed. If authentication data cannot be verified or the database is unreachable, requests must be rejected with appropriate HTTP status codes (401/402/403/503). Under no circumstances may requests pass through unauthenticated.
6. **No Client Credential Exposure:** Internal database credentials, Google Cloud credentials, raw API keys, and sensitive user records must never be logged, printed in stack traces, or returned in API responses.

---

## 2. Database Connection & Pooling Strategy

### 2.1 Async Driver & ORM Stack
- **Engine:** SQLAlchemy 2.0 Async Engine with `asyncpg` (`postgresql+asyncpg://...`).
- **Rationale:** Native asynchronous I/O cleanly integrates with FastAPI's `asyncio` event loop without blocking threadpools. `asyncpg` provides high-throughput binary protocol execution and robust PostgreSQL type support.

### 2.2 Connection Pooling Configuration
Cloud Run auto-scales container instances based on request concurrency. The connection pool must balance container concurrency against database server connection limits:

| Parameter | Recommended Value | Purpose & Rationale |
|---|---|---|
| `pool_size` | `10` | Steady-state open connections per Cloud Run container instance. |
| `max_overflow` | `20` | Temporary burst capacity under load spikes (total max: 30 connections/instance). |
| `pool_timeout` | `10.0s` | Maximum wait time for an available connection before raising pool starvation error. |
| `pool_recycle` | `1800s` (30 min) | Periodically recycles connections to prevent stale sockets across NAT/load balancers. |
| `pool_pre_ping` | `True` | Tests connection liveness with a lightweight test before handing it to a request. |
| `statement_timeout` | `3000ms` (3.0s) | Hard server-side query timeout preventing rogue queries from hanging indefinitely. |
| `connect_timeout` | `5.0s` | Timeout for establishing new TCP/TLS connections to PostgreSQL. |

### 2.3 Application Lifespan & Health Probes
- **Startup:** Initialize `create_async_engine()` during FastAPI lifespan startup in `app/main.py`. Attach engine to `app.state.db_engine`.
- **Shutdown:** Execute `await app.state.db_engine.dispose()` during lifespan teardown to drain and close pooled connections cleanly.
- **Readiness Probe (`/health/ready`):**
  - Executes a shallow `SELECT 1` with a strict `1.0s` timeout.
  - If the database query succeeds, reports `status: "ready"`.
  - If the database query fails or times out, reports `status: "degraded"` (in development) or `status: "degraded"` without internal exception details (in production).
  - Liveness probe (`/health/live`) remains purely in-process (`200 OK`) and must **never** depend on database connectivity.

---

## 3. Approved Database Role & View Privilege Matrix

### 3.1 Why `security_barrier` Alone is NOT Sufficient Access Control
`WITH (security_barrier = true)` is an optimization fence in PostgreSQL:
- It instructs the PostgreSQL query optimizer **not** to push user-supplied functions or where clauses below the view. This prevents malicious queries from inferring values from filtered rows through timing attacks or custom functions.
- **Critical Limitation:** `security_barrier` provides **zero** privilege isolation. If the connected database user possesses `SELECT` privileges on the underlying physical tables (`User`, `ApiKey`, `Wallet`), that user can query those tables directly and bypass the view completely.
- True security requires combining `security_barrier` with **strict PostgreSQL Role-Based Access Control (RBAC)**.

### 3.2 Dedicated Least-Privilege Role (`rantera_mp_service`)
The platform must connect as `rantera_mp_service`. The role permissions are strictly confined as follows:

```sql
-- 1. Create dedicated service role (executed by DBA)
CREATE ROLE rantera_mp_service WITH LOGIN PASSWORD '<SECURE_MANAGED_PASSWORD>';

-- 2. Restrict public schema access
REVOKE ALL ON SCHEMA public FROM rantera_mp_service;
GRANT USAGE ON SCHEMA public TO rantera_mp_service;

-- 3. Explicitly grant access to Model Platform objects
GRANT SELECT ON "mp_model_catalog" TO rantera_mp_service;
GRANT SELECT, INSERT, UPDATE ON "mp_inference_logs" TO rantera_mp_service;
GRANT SELECT ON "mp_active_api_keys_v" TO rantera_mp_service;
GRANT SELECT ON "mp_active_users_v" TO rantera_mp_service;

-- 4. Strictly forbid access to website tables and sensitive data
REVOKE ALL ON "User" FROM rantera_mp_service;
REVOKE ALL ON "ApiKey" FROM rantera_mp_service;
REVOKE ALL ON "Wallet" FROM rantera_mp_service;
REVOKE ALL ON "Transaction" FROM rantera_mp_service;
REVOKE ALL ON "UsageRecord" FROM rantera_mp_service;
REVOKE ALL ON "Conversation" FROM rantera_mp_service;
REVOKE ALL ON "Message" FROM rantera_mp_service;
REVOKE ALL ON "BlogPost" FROM rantera_mp_service;
REVOKE ALL ON "WaitlistEntry" FROM rantera_mp_service;
REVOKE ALL ON "Announcement" FROM rantera_mp_service;
REVOKE ALL ON "AuditLog" FROM rantera_mp_service;
REVOKE ALL ON "SupportTicket" FROM rantera_mp_service;
```

### 3.3 View Execution Semantics (`security_invoker = false`)
- The views `mp_active_api_keys_v` and `mp_active_users_v` must maintain the default PostgreSQL behavior (`security_invoker = false`).
- Under this default, PostgreSQL checks privileges on the underlying `ApiKey`, `User`, and `Wallet` tables against the **view owner** (the DBA / migration runner `postgres` or `rantera_app_owner`), while requiring `rantera_mp_service` to have permissions only on the view itself.
- Setting `security_invoker = true` would require granting `SELECT` on `User` and `ApiKey` directly to `rantera_mp_service`, defeating the least-privilege boundary.

---

## 4. API-Key and User Verification Flow

### 4.1 Step-by-Step Gateway Evaluation Pipeline
When an API request arrives at FastAPI:

```
Incoming Request (X-API-Key header)
        │
        ▼
[Check Header Present] ──────── No ───────▶ HTTP 401 Unauthorized ("Missing API key")
        │ Yes
        ▼
[Compute SHA-256 Hash] (hashlib.sha256)
        │
        ▼
[Query mp_active_api_keys_v] (Parameterized WHERE key_hash = :hash)
        │
        ├── 0 rows returned ──────────────▶ HTTP 401 Unauthorized ("Invalid API key")
        │
        ▼ 1 row returned
[Constant-Time Hash Check] ─── Fail ─────▶ HTTP 401 Unauthorized ("Invalid API key")
        │ Pass
        ▼
[Check Key Expiry] (expires_at < now) ───▶ HTTP 401 Unauthorized ("API key expired")
        │ Not expired
        ▼
[Check User Banned] (user_is_banned=true) ▶ HTTP 403 Forbidden ("Account suspended")
        │ Not banned
        ▼
[Check Wallet Balance] (balance <= 0) ───▶ HTTP 402 Payment Required ("Insufficient wallet balance")
        │ Positive balance
        ▼
[Check Spend Limit] (limit exceeded) ────▶ HTTP 402 Payment Required ("Spend limit reached")
        │ Within limit
        ▼
[Authentication Succeeded] ──────────────▶ Proceed to Model Routing & Vertex Execution
```

### 4.2 Key Invariants & Safeguards
- **Constant-Time Comparison:** Even after a SQL match, verify `hmac.compare_digest(computed_hash, row.key_hash)` to protect against database index side channels.
- **Deterministic Error Cascade:** Clear distinction between 401 (invalid/expired), 403 (banned account), and 402 (billing issues) without ever leaking account credentials or email addresses.
- **In-Memory TTL Caching (Phase 2 Enhancement):** To prevent database saturation at high queries-per-second (QPS), key authentication results should be cached in an in-memory LRU cache with short TTL (e.g., 60 seconds). Cache invalidation occurs naturally via TTL expiration.

---

## 5. Configuration Variables & Schema Extensions

The following settings will extend `app/config.py` in Phase 2:

```python
class Settings(BaseSettings):
    # Existing Phase 1 configuration preserved...

    # ── Database Configuration (Phase 2) ───────────────────
    database_url: Optional[SecretStr] = Field(
        default=None,
        description="Async PostgreSQL connection string (postgresql+asyncpg://...)",
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
    db_statement_timeout_ms: int = Field(
        default=3000,
        ge=500,
        le=30000,
        description="Server-side PostgreSQL statement timeout in milliseconds.",
    )
    db_connect_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        description="TCP/TLS connection establishment timeout in seconds.",
    )
    auth_cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        le=300,
        description="TTL for verified API key records in local memory cache (0 = disabled).",
    )
```

### Security Rules for Configuration
- `database_url` must use Pydantic `SecretStr`. It is strictly masked in logs and string representations.
- In production/staging (`RANTERA_ENV != "development"`), `database_url` is mandatory and fails closed if missing or set to placeholder values (`change-me`, `todo`, etc.).

---

## 6. Timeout, Fallback & Failure Behavior

### 6.1 Database Unreachability
- If PostgreSQL connection fails or statement times out (>3.0s):
  - Request aborts immediately.
  - Return **HTTP 503 Service Unavailable** with standardized body:
    ```json
    {
      "error": "Service Unavailable",
      "message": "Authentication service is temporarily unavailable. Please retry.",
      "request_id": "<request_id>"
    }
    ```
  - Log structured error with `request_id`, error category, and execution latency. Never log database passwords or raw connection strings.

### 6.2 Connection Pool Starvation
- If all connections are in use and `db_pool_timeout_seconds` (10s) expires:
  - Return **HTTP 503 Service Unavailable** (`"Service under heavy load; please retry"`).
  - Increment starvation telemetry counter to trigger infrastructure auto-scaling.

### 6.3 Inference Log Ingestion Failure
- If writing to `mp_inference_logs` fails *after* a successful model generation:
  - The client request has already succeeded. Do **not** fail the customer response.
  - Emit an `EMERGENCY` structured log record containing metadata (`request_id`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, `settlement_status = 'PENDING'`).
  - Raw prompts, responses, and keys are omitted from the log.
  - Future (Phase 3): Enqueue to local disk queue or Pub/Sub dead-letter queue for background replay.

---

## 7. Comprehensive Test Strategy

The Phase 2 testing strategy spans three distinct layers:

### 7.1 Unit & Contract Tests (Mocked DB)
- **State Machine Unit Tests:** Validate that all branches of the authentication pipeline (invalid key, expired, banned, zero balance, negative balance, spend limit exceeded) trigger exact HTTP codes and error models without DB dependencies.
- **Contract Verification:** Continuous execution of `tests/test_schema_contracts.py` ensuring that Python expectations match `schema.prisma` and `migration.sql`.
- **Settings & Config Validation:** Ensure malformed connection strings, placeholder values, or missing credentials fail closed during container startup.

### 7.2 Integration Tests (Ephemeral PostgreSQL Container)
- Spin up an ephemeral PostgreSQL instance (via testcontainers or GitHub Actions service container).
- Apply migration `20261014000000_add_model_platform_tables/migration.sql`.
- Create role `rantera_mp_service` and apply the exact privilege grants.
- Insert synthetic test data into `User`, `ApiKey`, and `Wallet`.
- Execute real async queries via `app/auth.py` and verify:
  - Correct row resolution and status evaluation.
  - Automatic `updated_at` trigger execution on `mp_inference_logs` updates.
  - Query performance using the `WHERE settlement_status = 'PENDING'` partial index.

### 7.3 Negative Security Tests (Privilege Verification)
- Connect using `rantera_mp_service` and assert that:
  - `SELECT * FROM "User"` $
ightarrow$ raises `InsufficientPrivilegeError` / `42501`.
  - `SELECT * FROM "ApiKey"` $
ightarrow$ raises `InsufficientPrivilegeError` / `42501`.
  - `SELECT * FROM "Wallet"` $
ightarrow$ raises `InsufficientPrivilegeError` / `42501`.
  - `DELETE FROM "mp_inference_logs"` $
ightarrow$ raises `InsufficientPrivilegeError`.
  - Only approved views and tables are accessible.

---

## 8. Migration Validation & Approval Gates

Before Phase 2 implementation or live deployment begins, the following mandatory gates must be satisfied:

```
[Gate 1: Baselining Gate]
Verify target PostgreSQL instance has _prisma_migrations records for
20260902000000, 20260912000000, 20260912120000.
Apply `prisma migrate resolve --applied` if missing.
        │
        ▼
[Gate 2: Migration Deployment Gate]
Execute `prisma migrate deploy` in staging environment using official Prisma 5.16.1 CLI.
Confirm clean zero-error exit code.
        │
        ▼
[Gate 3: RBAC Script Execution Gate]
DBA executes `rantera_mp_service` role creation and privilege grant script.
Verify role cannot read User/ApiKey tables.
        │
        ▼
[Gate 4: Model Catalog Seed Gate]
Execute approved seed SQL script to insert `rantera-base-v1` into `mp_model_catalog`.
        │
        ▼
[Gate 5: Rollback Script Validation Gate]
Rehearse down migration script in isolated environment:
DROP VIEW, DROP TRIGGER, DROP FUNCTION, DROP TABLE in reverse order.
```

---

## 9. Identified Inconsistencies, Risks & Unresolved Questions

### Inconsistency 1: Model Catalog Seed Strategy
- **Finding:** `migration.sql` creates `mp_model_catalog` but includes zero `INSERT` statements. `app/models.py` currently holds `rantera-base-v1` in an in-memory dictionary.
- **Material Question:** In Phase 2, will FastAPI query `mp_model_catalog` in PostgreSQL for active model metadata and pricing, or continue using the code-based registry with database verification?
- **Recommendation:** Add an explicit seed script or migration data step to populate `rantera-base-v1` in `mp_model_catalog` prior to cutover.

### Inconsistency 2: Decimal Precision Discrepancies
- **Finding:**
  - `ApiKey.spendLimitUsd`: `DECIMAL(10, 4)` (4 decimal places).
  - `Wallet.balanceUsd`: `DECIMAL(12, 6)` (6 decimal places).
  - `mp_inference_logs.cost_usd`: `DECIMAL(12, 6)` (6 decimal places).
- **Material Question:** How should cumulative spend calculations reconcile 4-decimal spend limits against 6-decimal inference costs?
- **Recommendation:** Normalize all monetary calculations in Python using `decimal.Decimal` with explicit `ROUND_HALF_UP` to prevent precision mismatches.

### Inconsistency 3: Historical Migrations & Fresh Environment Baselining
- **Finding:** `prisma/migrations` lacks an initial schema creation migration (e.g., `0_init`). The core tables (`User`, `ApiKey`, `Wallet`) were created before migration tracking was introduced.
- **Material Question:** How will local development and CI testing environments create the baseline schema without connecting to production?
- **Recommendation:** Provide a sanitized baseline schema SQL dump (`tests/fixtures/baseline_schema.sql`) for local and automated testing.

### Risk 1: View Query Overhead on High Concurrency
- `mp_active_api_keys_v` executes a 3-table join (`ApiKey` $owtie$ `User` $owtie$ `Wallet`) on every API request.
- **Mitigation:** Ensure indexes exist on `ApiKey(keyHash)`, `User(id)`, and `Wallet(userId)`. Implement a short-lived in-memory cache (60s TTL) in FastAPI to eliminate redundant database roundtrips.

### Risk 2: Wallet Deductions & Settlement Decoupling
- In Phase 1 and Phase 2, `mp_inference_logs` records requests with `settlement_status = 'PENDING'`.
- The website is authoritative for wallet balances, but the website has no automated cron or worker currently polling `mp_inference_logs` where `settlement_status = 'PENDING'`.
- **Material Question:** Will Phase 3 implement a Python settlement worker or a Next.js cron route to execute batch wallet deductions?
- **Recommendation:** Keep settlement decoupled; define a dedicated settlement worker specification for Phase 3 before enabling billable public traffic.

---

## 10. Conclusion & Part 1 Sign-Off

Part 1 has established:
1. Complete, bidirectional contract verification between Prisma 5.16.1 schema, PostgreSQL migration DDL, and Python test contracts.
2. 100% test pass rate (114/114 passing tests, zero compilation errors).
3. Thorough defense-in-depth security model combining `security_barrier` with dedicated role privileges.
4. Clean, uncommitted Git state across both repositories.

Part 1 is **complete and ready to close**. Phase 2 implementation should commence only after explicit user authorization and completion of the approval gates.
