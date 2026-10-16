# Rantera Model Platform — Phase 1 Database & Schema Integration

> **Document Status:** Architectural specification, local schema contract, and migration design for Phase 1.
> **Implementation Status:** Schema definitions and SQL migration script prepared locally. Live database migration, role provisioning, persistent auth backend, and billing settlement workers are pending DBA review and are NOT yet deployed.
> **Date:** October 13, 2026

---

## 1. Architecture & Ownership Boundaries

Phase 1 establishes the database contract connecting the FastAPI Model Platform to the existing PostgreSQL database owned by the Next.js website application.

### Key Governance Principles
1. **Single Schema Owner:** The website repository's Prisma schema and migration pipeline own all database objects. The Model Platform runs zero DDL.
2. **Namespace Isolation:** All Model Platform-owned tables use the `mp_` prefix (`mp_model_catalog`, `mp_inference_logs`).
3. **Decoupled Audit Logs:** All foreign references (`user_id`, `api_key_id`) are logical `VARCHAR(64)` fields without physical foreign keys (`REFERENCES`). Operational inference logs remain intact even if a website user or API key is rotated or deleted.
4. **Strict Least Privilege:** The Model Platform gateway connects with a dedicated role (`mp_gateway_user`) holding narrowly scoped DML privileges on `mp_` tables and `SELECT` on authorized views. It holds zero privileges on website tables (`User`, `ApiKey`, `Wallet`, `Transaction`, `UsageRecord`, etc.).

---

## 2. Least-Privilege Database Role & Permissions

### 2.1 Separation of Privileges
- **Website Role (`rantera_web` / `rantera_admin`):** Retains full CRUD permissions on existing website tables and owns the database objects and views.
- **Gateway Role (`mp_gateway_user`):** Granted only the minimal necessary privileges required to authenticate requests, log inference metadata, and query the model catalog.

### 2.2 Narrowly Scoped Provisioning Script
```sql
-- 1. Create dedicated gateway role with NOINHERIT to prevent inheriting group privileges
CREATE ROLE mp_gateway_user WITH LOGIN PASSWORD '<STRONG_PASSWORD_FROM_SECRETS>' NOINHERIT;

-- 2. Grant namespace lookup on public schema (does NOT grant table access)
GRANT USAGE ON SCHEMA public TO mp_gateway_user;

-- 3. Explicit table DML permissions (no DELETE or TRUNCATE)
GRANT SELECT ON TABLE "mp_model_catalog" TO mp_gateway_user;
GRANT SELECT, INSERT, UPDATE ON TABLE "mp_inference_logs" TO mp_gateway_user;

-- 4. Explicit view SELECT permissions
GRANT SELECT ON "mp_active_api_keys_v" TO mp_gateway_user;
GRANT SELECT ON "mp_active_users_v" TO mp_gateway_user;
```

> **Note on Permissions Scope:**
> The proposed configuration does **not** include broad statements such as `REVOKE ALL ON SCHEMA public FROM PUBLIC`, as modifying `PUBLIC` permissions can unintentionally disrupt existing website application roles or standard PostgreSQL extensions. All privilege boundaries are enforced by scoping grants directly to `mp_gateway_user`.

### 2.3 Permission Assumptions for Database Owner Verification
The database owner must verify:
1. **No Group Role Inheritance:** `mp_gateway_user` is not a member of the DBA role, `postgres`, or the website application role (`SELECT rolname FROM pg_roles WHERE ...`).
2. **Table Isolation:** Confirm `mp_gateway_user` cannot access website tables:
   ```sql
   SELECT grantee, table_name, privilege_type
   FROM information_schema.table_privileges
   WHERE grantee = 'mp_gateway_user';
   ```
   Should list only `mp_model_catalog`, `mp_inference_logs`, `mp_active_api_keys_v`, and `mp_active_users_v`.
3. **Default Privileges:** Confirm no existing `ALTER DEFAULT PRIVILEGES` rules grant automatic table permissions to newly created roles.

---

## 3. Migration Safety, Baselining & Drift Prevention

### 3.1 Why `prisma migrate dev` Must NEVER Be Used in Production
`prisma migrate dev` is designed for development environments:
- It creates and drops shadow databases to detect drift.
- If it detects that the database schema does not match the migration history from empty, it prompts to reset the database (`DROP SCHEMA` / wipe all tables).
- In automated or non-interactive environments, it fails immediately.

### 3.2 Pre-Flight Alignment Check by Database Owner
Because the initial website schema was applied via `prisma db push`, the database owner must verify whether the live database and Prisma's `_prisma_migrations` table are aligned:

```sql
-- Step 1: Check if _prisma_migrations table exists and what has been applied
SELECT migration_name, finished_at, rolled_back_at
FROM _prisma_migrations
ORDER BY started_at;

-- Step 2: Check whether historical migration columns already exist on live tables
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_name IN ('User', 'ApiKey')
  AND column_name IN ('password', 'emailVerified', 'verifyToken', 'spendLimitUsd', 'expiresAt');
```

- **If historical columns exist but are NOT recorded in `_prisma_migrations`:**
  Running `prisma migrate deploy` directly would fail because earlier migrations (such as `20260912000000_add_api_key_spend_limit_expiry`) would attempt to execute `ALTER TABLE "ApiKey" ADD COLUMN "spendLimitUsd"...`, which fails if the column already exists.
  - **Remediation:** The database owner must baseline historical migrations before applying new migrations:
    ```bash
    npx prisma migrate resolve --applied 20260902000000_add_email_verification --schema=../prisma/schema.prisma
    npx prisma migrate resolve --applied 20260912000000_add_api_key_spend_limit_expiry --schema=../prisma/schema.prisma
    npx prisma migrate resolve --applied 20260912120000_add_tera_model_id --schema=../prisma/schema.prisma
    ```
- **Once baselined:** The new migration `20261014000000_add_model_platform_tables` can be applied cleanly via `prisma migrate deploy`.

### 3.3 Failing Visibly on Drift (Removal of `IF NOT EXISTS`)
The migration SQL deliberately avoids `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`:
- If an object named `mp_model_catalog` or `mp_inference_logs` already existed in the target database with different columns or constraints, `IF NOT EXISTS` would silently skip creation, concealing schema drift and leaving an invalid table in place.
- Using strict DDL (`CREATE TABLE`, `CREATE INDEX`, `CREATE VIEW`) ensures the migration **fails visibly and rolls back atomically** if any conflicting object exists, alerting the DBA immediately.

---

## 4. Migration Tracking in Version Control

### 4.1 Smallest Targeted Exception for
In , line 40 previously contained a blanket ignore:
```gitignore
prisma/migrations/
```
This caused all migrations to be ignored by Git and required manual `git add -f`.

**Safe Resolution Implemented:**
Rather than unignoring all past or future migrations broadly, a minimal, targeted exception was added to line 40 to track specifically the verified Phase 1 migration directory while strictly preserving the existing ignore behavior for unrelated or legacy migrations:
```gitignore
# Prisma
prisma/migrations/*
!prisma/migrations/20261014000000_add_model_platform_tables/
```
**Verification:**
- `git check-ignore -v prisma/migrations/20261014000000_add_model_platform_tables/migration.sql` returns exit code 1 (file is tracked).
- `git check-ignore -v prisma/migrations/other_folder/migration.sql` returns exit code 0 (`.gitignore:40:prisma/migrations/*`, strictly ignored).
- `git status --short --ignored` cleanly lists `?? prisma/migrations/20261014000000_add_model_platform_tables/` as untracked for inclusion without exposing unrelated artifacts.

---

## 5. Schema Contract & Authentication View

### 5.1 Prisma-to-SQL Mapping and Representation Boundaries

Prisma 5.16.1 is in use. While Prisma represents basic relational entities, columns, primary keys, and standard indexes, it cannot represent:
1. PostgreSQL partial indexes (`WHERE` clauses).
2. Database triggers and stored functions (`BEFORE UPDATE FOR EACH ROW`).
3. Security-barrier views (`WITH (security_barrier = true)`).

Therefore, **versioned SQL migrations (`migration.sql`) own the raw database schema and advanced objects**, while `schema.prisma` serves as the typed application mapping for Next.js. We do **not** claim full Prisma migration parity for objects that Prisma 5.16.1 cannot represent.

#### Model Platform Tables & Native Types

| Table / Feature | PostgreSQL Table (`migration.sql`) | Prisma Model (`schema.prisma`) | Notes & Ownership |
|---|---|---|---|
| **`mp_model_catalog`** | | | |
| `id` | `VARCHAR(36) PRIMARY KEY` | `String @id @default(cuid()) @db.VarChar(36)` | Aligned string bound |
| `model_id` | `VARCHAR(64) NOT NULL` | `String @unique(...) @map("model_id") @db.VarChar(64)` | Aligned string bound |
| Unique index on `model_id` | `CREATE UNIQUE INDEX "idx_mp_model_catalog_model_id"` | `@unique(map: "idx_mp_model_catalog_model_id")` | Explicit unified name; no duplicate constraints |
| `display_name` | `VARCHAR(128) NOT NULL` | `String @map("display_name") @db.VarChar(128)` | Aligned string bound |
| `description` | `TEXT NOT NULL` | `String @db.Text` | Explicit PostgreSQL `TEXT` |
| `provider` | `VARCHAR(32) NOT NULL DEFAULT 'vertex_ai'` | `String @default("vertex_ai") @db.VarChar(32)` | Aligned string bound & default |
| `provider_model_id` | `VARCHAR(256)` (nullable) | `String? @map("provider_model_id") @db.VarChar(256)` | Aligned string bound |
| `is_active` | `BOOLEAN NOT NULL DEFAULT true` | `Boolean @default(true) @map("is_active")` | Aligned boolean default |
| Index on `is_active` | `CREATE INDEX "idx_mp_model_catalog_is_active"` | `@@index([isActive], map: "idx_mp_model_catalog_is_active")` | Aligned standard index |
| `context_window` | `INTEGER NOT NULL DEFAULT 8192` | `Int @default(8192) @map("context_window")` | Aligned integer default |
| `max_output_tokens` | `INTEGER NOT NULL DEFAULT 4096` | `Int @default(4096) @map("max_output_tokens")` | Aligned integer default |
| `input_cost_per_1k` | `DECIMAL(10, 6) NOT NULL DEFAULT 0.003000` | `Decimal @default(0.003000) @map(...) @db.Decimal(10, 6)` | Aligned decimal precision |
| `output_cost_per_1k` | `DECIMAL(10, 6) NOT NULL DEFAULT 0.015000` | `Decimal @default(0.015000) @map(...) @db.Decimal(10, 6)` | Aligned decimal precision |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @default(now()) @map("created_at") @db.Timestamptz(6)` | Aligned timestamp with timezone |
| `updated_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @updatedAt @map("updated_at") @db.Timestamptz(6)` | Dual strategy: Prisma `@updatedAt` + SQL trigger |
| **`mp_inference_logs`** | | | |
| `id` | `VARCHAR(36) PRIMARY KEY` | `String @id @default(cuid()) @db.VarChar(36)` | Aligned string bound |
| `request_id` | `VARCHAR(64) NOT NULL` | `String @unique(...) @map("request_id") @db.VarChar(64)` | Aligned string bound |
| Unique index on `request_id` | `CREATE UNIQUE INDEX "idx_mp_inference_logs_request_id"` | `@unique(map: "idx_mp_inference_logs_request_id")` | Aligned unique constraint |
| `user_id` | `VARCHAR(64)` (logical ref, no FK) | `String? @map("user_id") @db.VarChar(64)` | Aligned string bound |
| `api_key_id` | `VARCHAR(64)` (logical ref, no FK) | `String? @map("api_key_id") @db.VarChar(64)` | Aligned string bound |
| `model_id` | `VARCHAR(64) NOT NULL` | `String @map("model_id") @db.VarChar(64)` | Aligned string bound |
| `channel` | `VARCHAR(32) NOT NULL DEFAULT 'api'` | `String @default("api") @map("channel") @db.VarChar(32)` | Aligned string bound & default |
| `tokens_in` / `tokens_out` | `INTEGER NOT NULL DEFAULT 0` | `Int @default(0) @map(...)` | Aligned integer defaults |
| `total_tokens` | `INTEGER NOT NULL DEFAULT 0` | `Int @default(0) @map("total_tokens")` | Aligned integer default |
| `cost_usd` | `DECIMAL(12, 6) NOT NULL DEFAULT 0.000000` | `Decimal @default(0.000000) @map(...) @db.Decimal(12, 6)` | Aligned decimal precision |
| `latency_ms` | `INTEGER NOT NULL DEFAULT 0` | `Int @default(0) @map("latency_ms")` | Aligned integer default |
| `ttft_ms` | `INTEGER` (nullable) | `Int? @map("ttft_ms")` | Aligned nullable integer |
| `request_status` | `VARCHAR(32) NOT NULL DEFAULT 'PENDING'` | `String @default("PENDING") @map(...) @db.VarChar(32)` | Aligned string bound & default |
| `error_code` | `VARCHAR(64)` (nullable) | `String? @map("error_code") @db.VarChar(64)` | Aligned string bound |
| `settlement_status` | `VARCHAR(32) NOT NULL DEFAULT 'PENDING'` | `String @default("PENDING") @map(...) @db.VarChar(32)` | Aligned string bound & default |
| `settlement_id` | `VARCHAR(64)` (nullable) | `String? @map("settlement_id") @db.VarChar(64)` | Aligned string bound |
| `settlement_attempts` | `INTEGER NOT NULL DEFAULT 0` | `Int @default(0) @map("settlement_attempts")` | Aligned integer default |
| `settled_at` | `TIMESTAMPTZ` (nullable) | `DateTime? @map("settled_at") @db.Timestamptz(6)` | Aligned timestamp with timezone |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @default(now()) @map("created_at") @db.Timestamptz(6)` | Aligned timestamp with timezone |
| `updated_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @updatedAt @map("updated_at") @db.Timestamptz(6)` | Dual strategy: Prisma `@updatedAt` + SQL trigger |
| Standard indexes (`user_id`, `api_key_id`, `model_id`, `created_at`) | `CREATE INDEX "idx_mp_inference_logs_<col>"` | `@@index([<col>], map: "idx_mp_inference_logs_<col>")` | Synchronized standard B-tree indexes |

---

### 5.2 Resolution of Raw-SQL-Managed Objects & Migration Drift Implications

#### 1. `settlement_status` Partial Index
- **PostgreSQL Migration (`migration.sql`):**
  ```sql
  CREATE INDEX "idx_mp_inference_logs_settlement_status"
      ON "mp_inference_logs"("settlement_status") WHERE "settlement_status" = 'PENDING';
  ```
- **Prisma Representation (`schema.prisma`):** Intentionally **omitted** from `schema.prisma`.
- **Rationale:** The query pattern for billing settlement workers is to periodically poll records requiring settlement (`WHERE settlement_status = 'PENDING'`). Because 99.9% of inference records eventually become `SETTLED`, indexing all settled records creates severe index bloat and unnecessary write overhead. Prisma 5.16.1 does not support partial indexes with `WHERE` predicates. Declaring a full `@@index([settlementStatus])` in Prisma would falsely claim schema parity and could induce DB tooling to recreate a bloat-prone full index.
- **Migration Drift Implication:** Any Prisma schema comparison command (such as `prisma migrate diff` or `prisma db push`) will report that `idx_mp_inference_logs_settlement_status` is not defined in `schema.prisma`. **This drift is intentional.** Versioned raw SQL migrations manage this index. Automated tools must not drop or alter this index.

#### 2. Explicit `updated_at` Strategy & PostgreSQL Triggers
- **Prisma `@updatedAt`:** Only updates the timestamp when mutations are performed via the Prisma Client in the Next.js application.
- **Direct SQL / FastAPI Updates:** When the FastAPI Model Platform or batch settlement processes execute direct SQL `UPDATE` statements (e.g. `UPDATE mp_inference_logs SET settlement_status = 'SETTLED' WHERE request_id = `), Prisma hooks do not fire.
- **Unified Resolution:** A safe, reusable PostgreSQL trigger function `mp_set_updated_at()` and explicit triggers are created in `migration.sql`:
  ```sql
  CREATE OR REPLACE FUNCTION mp_set_updated_at()
  RETURNS TRIGGER AS 62781
  BEGIN
      NEW.updated_at = CURRENT_TIMESTAMP;
      RETURN NEW;
  END;
  62781 LANGUAGE plpgsql;

  CREATE TRIGGER "trg_mp_model_catalog_updated_at"
      BEFORE UPDATE ON "mp_model_catalog"
      FOR EACH ROW
      EXECUTE FUNCTION mp_set_updated_at();

  CREATE TRIGGER "trg_mp_inference_logs_updated_at"
      BEFORE UPDATE ON "mp_inference_logs"
      FOR EACH ROW
      EXECUTE FUNCTION mp_set_updated_at();
  ```
- **Migration Drift Implication:** Prisma schema has no mechanism to define or track triggers. Prisma diff tools will report these triggers as extraneous database objects. Versioned SQL migrations own them.

#### 3. Unique Index on `model_id`
- In `mp_model_catalog`, the inline `UNIQUE` keyword on `"model_id" VARCHAR(64) NOT NULL UNIQUE` was replaced with an explicit index definition:
  ```sql
  CREATE UNIQUE INDEX "idx_mp_model_catalog_model_id"
      ON "mp_model_catalog"("model_id");
  ```
  This eliminates duplicate constraint creation and aligns exactly with Prisma's `@unique(map: "idx_mp_model_catalog_model_id")`.

---

### 5.3 Authentication View (`mp_active_api_keys_v`)
```sql
CREATE VIEW "mp_active_api_keys_v"
WITH (security_barrier = true) AS
SELECT
    k."id" AS "api_key_id",
    k."userId" AS "user_id",
    k."keyHash" AS "key_hash",
    k."keyPrefix" AS "key_prefix",
    k."name" AS "key_name",
    k."spendLimitUsd" AS "spend_limit_usd",
    k."expiresAt" AS "expires_at",
    u."isBanned" AS "user_is_banned",
    u."emailVerified" AS "user_email_verified",
    COALESCE(w."balanceUsd", 0.000000) AS "wallet_balance_usd"
FROM "ApiKey" k
JOIN "User" u ON k."userId" = u."id"
LEFT JOIN "Wallet" w ON u."id" = w."userId";
```

#### Key Guarantees
1. **Clean Distinction of Auth & Billing States:** By projecting `expires_at`, `user_is_banned`, and `wallet_balance_usd` without pre-filtering rows in the SQL `WHERE` clause, the gateway can cleanly distinguish failure modes without masking reasons:
   - **Invalid API Key:** Key hash lookup returns 0 rows $\rightarrow$ returns `401 Unauthorized` (`"Invalid API key"`).
   - **Expired API Key:** Row found, `expires_at` is in the past $\rightarrow$ returns `401 Unauthorized` (`"API key expired"`).
   - **Banned User Account:** Row found, `user_is_banned = true` $\rightarrow$ returns `403 Forbidden` (`"Account suspended"`).
   - **Billing-Ineligible Account:** Row found, active key, but `wallet_balance_usd <= 0` or spend limit exceeded $\rightarrow$ returns `402 Payment Required` (`"Insufficient wallet balance"" or `"Spend limit reached"`).
2. **Zero Sensitive Data Exposure:** Password hashes, email verification tokens, Cognito IDs, user emails, and payment transaction details are completely excluded from the view projection.
3. **No Side-Channel Leaks:** The view enforces `WITH (security_barrier = true)` to prevent query optimizer leakage through user-defined functions during query planning.

---

## 6. Implementation Status & Billing Disclaimers

### 6.1 What is Implemented
- Local Prisma schema models in `schema.prisma` with explicit native types (`@db.VarChar`, `@db.Timestamptz(6)`).
- Local strict migration DDL script in `prisma/migrations/20261014000000_add_model_platform_tables/migration.sql` with partial index `idx_mp_inference_logs_settlement_status`, `mp_set_updated_at()` triggers, explicit unique index `idx_mp_model_catalog_model_id`, and security barrier views.
- In-memory lifecycle and state transition specification tests in `tests/test_schema_contracts.py` including explicit verification of column bounds and raw-SQL managed objects.
- Schema-level `request_id` unique constraint on `mp_inference_logs` for operational request deduplication.

### 6.2 Unimplemented Functionality (Pending Future Phases)
The following functionality is explicitly **unimplemented** in the current Phase 1 codebase:
- **Live Database Authentication:** No live database connection pool or persistent query lookup is active in FastAPI; `app/auth.py` continues to fail closed (`503 Service Unavailable`) for key verification until Phase 2 database integration is reviewed and executed.
- **Vertex Inference:** No Google Cloud Vertex AI client SDK, endpoint routing, or prediction pipeline is active in this service; model metadata queries return static registry information.
- **Rate Limiting:** `RATE_LIMIT_RPM` is defined as a configuration setting in `app/config.py`, but runtime token-bucket / sliding-window rate limiting middleware and backing storage (e.g., Redis) are not yet implemented.
- **Settlement Retries:** No background retry worker, exponential backoff mechanism, or dead-letter queue processor is implemented for failed settlement attempts.
- **Atomic Billing:** Deductions from `Wallet.balanceUsd` are not transactionally coupled to inference delivery in the model platform codebase.
- **Exactly-Once Settlement:** Although structural idempotency fields (`request_id`, `settlement_status`, `settlement_id`) exist in the schema, application-level distributed two-phase commit or exactly-once settlement guarantees are not yet implemented.
- **Website Billing Behavior:** The existing website billing routes (`chat/route.ts`, `tera/route.ts`) continue using their independent local logic and remain unchanged.
