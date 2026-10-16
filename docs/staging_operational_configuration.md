# Rantera — Staging Operational Configuration & Deployment Guide

> **Document Classification:** Production & Staging Operations Runbook
> **Target Services:** `rantera-model-platform` (FastAPI / Cloud Run) & `rantera-webapp/rantera` (Next.js / Vercel & PostgreSQL)
> **Status:** PRE-STAGING READY (Strict Fail-Closed Enforcement)

---

## 1. Intended Telemetry Ingestion Architecture (Push vs. Pull)

The Rantera architecture separates fast inference response paths from durable telemetry persistence and asynchronous financial settlement.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Telemetry Ingestion Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Model Platform Gateway]                                                   │
│           │                                                                 │
│           ├── (1) Fast In-Transaction Direct Ingest (Primary Path)          │
│           │   └── INSERT mp_inference_logs (PENDING, settlement_attempts=0)  │
│           │                                                                 │
│           └── (2) Asynchronous Queue Backup (Fallback / High Concurrency)   │
│               └── Publish to Google Cloud Pub/Sub Topic                     │
│                   ("mp-inference-telemetry")                                │
│                              │                                              │
│                              ▼                                              │
│           ┌───────────────────────────────────────┐                         │
│           │      Pub/Sub Delivery Modes           │                         │
│           │                                       │                         │
│           │  Mode A: PUSH Webhook (RECOMMENDED)   │                         │
│           │  Cloud Run Gateway receives POST to   │                         │
│           │  /v1/telemetry/pubsub/push            │                         │
│           │  (Authenticated via Google OIDC)      │                         │
│           │                                       │                         │
│           │  Mode B: PULL Consumer (Daemon Worker)│                         │
│           │  Background process running:          │                         │
│           │  python -m app.telemetry_worker       │                         │
│           └───────────────────────────────────────┘                         │
│                              │                                              │
│                              ▼                                              │
│           [PostgreSQL: mp_inference_logs (PENDING)]                         │
│                              │                                              │
│                              ▼                                              │
│           [Website Settlement Worker (Every Minute)]                        │
│           SELECT FOR UPDATE SKIP LOCKED ...                                 │
│           └── Atomic Wallet Debit, UsageRecord, Transaction                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Recommended Mode: Authenticated Push Webhook
- **Recommendation:** **Mode A: Google Authenticated Pub/Sub Push Webhook**.
- **Rationale:**
  - Rantera Model Platform is deployed on **Google Cloud Run** (serverless, scale-to-zero capability).
  - Push subscriptions leverage Cloud Run's native auto-scaling and avoid running a continuously polling pull daemon container when traffic is quiet.
  - Pub/Sub natively guarantees bounded retries with exponential backoff on HTTP 503 and automatic routing to Dead-Letter Topic (DLQ) upon exceeding max delivery attempts.
- **Alternative Mode: Pull Consumer Daemon:**
  - If deploying to a persistent VM or GKE cluster, or if inbound webhook latency/firewall restrictions exist, the `PubSubTelemetryConsumer` class can run as an independent background daemon worker (`python -m app.telemetry_worker`).

---

## 2. Authentication Configuration for Pub/Sub Delivery

> [!NOTE]
> **Provisioning Status Notice:**
> The GCP IAM service accounts, Pub/Sub topics, and push subscriptions described below are configuration specifications. They have **not** yet been verified as provisioned in staging.

### 2.1 Mode A: Authenticated Push Configuration (Google Cloud IAM + OIDC)

In accordance with strict security standards:
- Query-string token authentication is **strictly forbidden and rejected** (`HTTP 401`).
- Arbitrary untrusted bearer tokens are **strictly rejected** (`HTTP 401`).
- Shared internal service keys are **never conflated with Google identity**.

#### Step 1: Create Dedicated Service Account
```bash
gcloud iam service-accounts create pubsub-telemetry-invoker \
  --display-name="PubSub Telemetry Push Invoker" \
  --description="Dedicated SA for authenticated Pub/Sub push delivery to Rantera Gateway"
```

#### Step 2: Grant Cloud Run Invoker Role
```bash
gcloud run services add-iam-policy-binding rantera-model-platform \
  --region=us-central1 \
  --member="serviceAccount:pubsub-telemetry-invoker@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

#### Step 3: Create Dead Letter Topic and Push Subscription
```bash
# 1. Create DLQ topic
gcloud pubsub topics create mp-inference-telemetry-dlq

# 2. Create primary topic
gcloud pubsub topics create mp-inference-telemetry

# 3. Create authenticated push subscription
gcloud pubsub subscriptions create mp-inference-telemetry-push-sub \
  --topic=mp-inference-telemetry \
  --push-endpoint="https://gateway.rantera.com/v1/telemetry/pubsub/push" \
  --push-auth-service-account="pubsub-telemetry-invoker@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --push-auth-token-audience="https://gateway.rantera.com/v1/telemetry/pubsub/push" \
  --ack-deadline=30 \
  --min-retry-delay=10s \
  --max-retry-delay=600s \
  --dead-letter-topic=mp-inference-telemetry-dlq \
  --max-delivery-attempts=5
```

#### Step 4: Model Platform Cloud Run Environment Variables
Inject via Cloud Secret Manager / Cloud Run configuration:
```env
PUBSUB_PUSH_AUDIENCE=https://gateway.rantera.com/v1/telemetry/pubsub/push
PUBSUB_PUSH_SERVICE_ACCOUNT_EMAIL=pubsub-telemetry-invoker@${GCP_PROJECT_ID}.iam.gserviceaccount.com
```

### 2.2 Mode B: Pull Consumer Configuration (GCP IAM)
If running the pull worker daemon:
```bash
# Grant pubsub.subscriber role to the Cloud Run / VM service account
gcloud pubsub subscriptions add-iam-policy-binding mp-inference-telemetry-sub \
  --member="serviceAccount:model-platform-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/pubsub.subscriber"
```

---

## 3. Least-Privilege Database Grants

> [!NOTE]
> **Template / Verification Notice:**
> The role names (`rantera_model_platform_app`, `rantera_webapp_app`), database name (`rantera`), and grants shown below are administrative templates. In staging, verify existing database roles, the actual database name, and administrative privileges before executing. Database roles and grants have not yet been verified or provisioned in staging.

Both services share the same PostgreSQL database instance (AWS RDS / Cloud SQL) but MUST operate under distinct, least-privilege PostgreSQL roles.

### 3.1 Model Platform Database Role: `rantera_model_platform_app`
The Model Platform gateway requires access ONLY to read active models/keys and insert telemetry logs. It must NEVER have write/update access to user wallets, financial transactions, or billing records.

```sql
-- Create role
CREATE ROLE rantera_model_platform_app WITH LOGIN PASSWORD 'STRONG_RANDOMLY_GENERATED_PASSWORD';

-- Connect to target database
\c rantera;

-- Schema usage
GRANT USAGE ON SCHEMA public TO rantera_model_platform_app;

-- Table permissions
-- 1. Model Catalog: Read-only access
GRANT SELECT ON TABLE public.mp_model_catalog TO rantera_model_platform_app;

-- 2. Authentication: Read-only view access (API keys and Web user delegation)
GRANT SELECT ON TABLE public.mp_active_api_keys_v TO rantera_model_platform_app;
GRANT SELECT ON TABLE public.mp_active_users_v TO rantera_model_platform_app;

-- 3. Telemetry Logs: Insert and Select only (NO UPDATE, NO DELETE)
-- Model Platform inserts rows with settlement_status = 'PENDING'; it never settles them.
GRANT SELECT, INSERT ON TABLE public.mp_inference_logs TO rantera_model_platform_app;

-- Explicitly revoke sensitive table access
REVOKE ALL ON TABLE public."Wallet" FROM rantera_model_platform_app;
REVOKE ALL ON TABLE public."UsageRecord" FROM rantera_model_platform_app;
REVOKE ALL ON TABLE public."Transaction" FROM rantera_model_platform_app;
REVOKE ALL ON TABLE public."User" FROM rantera_model_platform_app;
```

### 3.2 Website Settlement Worker Database Role: `rantera_webapp_app`
The website application and its background settlement worker own the financial ledger and wallet deductions.

```sql
-- Permissions required for website settlement worker
GRANT SELECT, UPDATE ON TABLE public.mp_inference_logs TO rantera_webapp_app;
GRANT SELECT, UPDATE ON TABLE public."Wallet" TO rantera_webapp_app;
GRANT SELECT, INSERT ON TABLE public."UsageRecord" TO rantera_webapp_app;
GRANT SELECT, INSERT ON TABLE public."Transaction" TO rantera_webapp_app;
GRANT SELECT ON TABLE public.mp_model_catalog TO rantera_webapp_app;
```

---

## 4. Website Settlement Cron & Scheduler Strategy

### 4.1 Deployment Compatibility & Limitations
The website settlement process runs `processPendingInferenceSettlements()` in `web/src/lib/billing/settlement.ts`, exposed via endpoint:
`POST /api/cron/settlement` (secured by `Authorization: Bearer ${CRON_SECRET}`).
*Note: The settlement route supports both POST (standard for Google Cloud Scheduler) and GET (legacy / Vercel Cron) requests.*

- **Vercel Pro / Enterprise Accounts:**
  Vercel supports cron schedules defined in `vercel.json`:
  ```json
  {
    "crons": [
      {
        "path": "/api/cron/settlement",
        "schedule": "* * * * *"
      }
    ]
  }
  ```
  This triggers once every minute as intended.

- **Vercel Hobby (Free) Accounts:**
  **Limitation Warning:** Vercel Hobby accounts strictly enforce a limit of **1 cron execution per day**.
  Deploying a 1-minute cron on Hobby accounts will fail to deploy or will silently execute at 00:00 UTC only.

### 4.2 Production External Scheduler Alternative (Google Cloud Scheduler)
If deploying to Vercel Hobby, custom domains, or decoupling schedulers from web hosting:
```bash
gcloud scheduler jobs create http rantera-settlement-cron \
  --schedule="* * * * *" \
  --uri="https://rantera.ai/api/cron/settlement" \
  --http-method=POST \
  --headers="Authorization=Bearer ${CRON_SECRET}" \
  --time-zone="UTC" \
  --attempt-deadline=60s
```

---

## 5. Required Staging Database Migrations & Model Catalog Seed SQL

> [!WARNING]
> **Pre-Execution Notice:**
> Migrations have **not** yet been applied to the staging database; that has not been verified. Before any migration or seed execution, staging database connectivity and migration status must be inspected via `npx prisma migrate status`.
> DO NOT EXECUTE THESE STATEMENTS DURING AUDIT OR PRE-STAGING CHECKS.
> These are documented for the staging database migration release window.

### 5.1 Prisma Migration SQL (`prisma/migrations/20261014000000_add_model_platform_tables/migration.sql`)

Below is the exact checked-in SQL from migration `20261014000000_add_model_platform_tables`.
*Note: `settlement_status` is stored as `VARCHAR(32)` (no custom enum type `MpSettlementStatus` exists in the database schema).*

```sql
-- ─── Model Platform Integration: Phase 1 Schema Migration ───────
-- Creates Model Platform tables and narrow views for API gateway.
-- All objects use the "mp_" prefix.
-- References to User and ApiKey are logical (no physical foreign keys).
-- Strict DDL without IF NOT EXISTS to fail visibly on unexpected schema drift.

-- 1. Model Catalog Table
CREATE TABLE "mp_model_catalog" (
    "id" VARCHAR(36) PRIMARY KEY,
    "model_id" VARCHAR(64) NOT NULL,
    "display_name" VARCHAR(128) NOT NULL,
    "description" TEXT NOT NULL,
    "provider" VARCHAR(32) NOT NULL DEFAULT 'vertex_ai',
    "provider_model_id" VARCHAR(256),
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "context_window" INTEGER NOT NULL DEFAULT 8192,
    "max_output_tokens" INTEGER NOT NULL DEFAULT 4096,
    "input_cost_per_1k" DECIMAL(10, 6) NOT NULL DEFAULT 0.003000,
    "output_cost_per_1k" DECIMAL(10, 6) NOT NULL DEFAULT 0.015000,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX "idx_mp_model_catalog_model_id"
    ON "mp_model_catalog"("model_id");
CREATE INDEX "idx_mp_model_catalog_is_active"
    ON "mp_model_catalog"("is_active");

-- 2. Inference Logs Table
CREATE TABLE "mp_inference_logs" (
    "id" VARCHAR(36) PRIMARY KEY,
    "request_id" VARCHAR(64) NOT NULL,
    "user_id" VARCHAR(64),
    "api_key_id" VARCHAR(64),
    "model_id" VARCHAR(64) NOT NULL,
    "channel" VARCHAR(32) NOT NULL DEFAULT 'api',
    "tokens_in" INTEGER NOT NULL DEFAULT 0,
    "tokens_out" INTEGER NOT NULL DEFAULT 0,
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "cost_usd" DECIMAL(12, 6) NOT NULL DEFAULT 0.000000,
    "latency_ms" INTEGER NOT NULL DEFAULT 0,
    "ttft_ms" INTEGER,
    "request_status" VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    "error_code" VARCHAR(64),
    "settlement_status" VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    "settlement_id" VARCHAR(64),
    "settlement_attempts" INTEGER NOT NULL DEFAULT 0,
    "settled_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX "idx_mp_inference_logs_request_id"
    ON "mp_inference_logs"("request_id");
CREATE INDEX "idx_mp_inference_logs_user_id"
    ON "mp_inference_logs"("user_id");
CREATE INDEX "idx_mp_inference_logs_api_key_id"
    ON "mp_inference_logs"("api_key_id");
CREATE INDEX "idx_mp_inference_logs_model_id"
    ON "mp_inference_logs"("model_id");
CREATE INDEX "idx_mp_inference_logs_created_at"
    ON "mp_inference_logs"("created_at");
CREATE INDEX "idx_mp_inference_logs_settlement_status"
    ON "mp_inference_logs"("settlement_status") WHERE "settlement_status" = 'PENDING';

-- 3. Automatic Updated-At Triggers (Raw-SQL Managed)
-- Enforces updated_at = CURRENT_TIMESTAMP on direct SQL / FastAPI DML updates.
-- Raw SQL migration owns these triggers; Prisma schema cannot represent triggers.
CREATE OR REPLACE FUNCTION mp_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER "trg_mp_model_catalog_updated_at"
    BEFORE UPDATE ON "mp_model_catalog"
    FOR EACH ROW
    EXECUTE FUNCTION mp_set_updated_at();

CREATE TRIGGER "trg_mp_inference_logs_updated_at"
    BEFORE UPDATE ON "mp_inference_logs"
    FOR EACH ROW
    EXECUTE FUNCTION mp_set_updated_at();

-- 4. Narrow API Keys Authentication View
-- Used by FastAPI to validate sha256 key hash and check eligibility.
-- Does NOT expose sensitive user credentials (password, verifyToken, cognitoId, email).
-- Projects status flags (expiresAt, user_is_banned) and wallet_balance_usd so the gateway
-- can cleanly distinguish invalid, expired, banned, and billing-ineligible accounts.
-- Uses security_barrier to prevent function side-channel leaks during query optimization.
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

-- 5. Narrow Users View (Defense-in-depth for web channel calls)
CREATE VIEW "mp_active_users_v"
WITH (security_barrier = true) AS
SELECT
    u."id" AS "user_id",
    u."isBanned" AS "is_banned",
    u."emailVerified" AS "email_verified",
    COALESCE(w."balanceUsd", 0.000000) AS "wallet_balance_usd"
FROM "User" u
LEFT JOIN "Wallet" w ON u."id" = w."userId";
```

### 5.2 Model Enum Migration SQL (`prisma/migrations/20261016000000_add_rantera_base_v1_model_id/migration.sql`)

```sql
-- AlterEnum
ALTER TYPE "ModelId" ADD VALUE IF NOT EXISTS 'RANTERA_BASE_V1';
```

> [!NOTE]
> **Prisma Enum Conventions:**
> The PostgreSQL enum type `"ModelId"` uses uppercase identifiers (`RANT_AI`, `AKILI`, `ILIMI`, `ONA`, `OGBON`, `HIRA`, `TERA`, `RANTERA_BASE_V1`).
> The public model identifier in API requests and `mp_model_catalog.model_id` is lowercase kebab-case (`rantera-base-v1`), mapped at the application boundaries via `TS_TO_PRISMA` in the website and `MODEL_ID_MAP` in the settlement worker.

### 5.3 Initial Model Catalog Seed SQL (`rantera-base-v1`)

> [!IMPORTANT]
> **Model Identifier Distinctions & Confirmed Values:**
> - **Public Model ID (`model_id`)**: `rantera-base-v1` (the public logical identifier used in client API requests: `{"model": "rantera-base-v1", ...}`).
> - **Provider Model Identifier (`provider_model_id`)**: `/tmp/afriqueqwen` (the confirmed model identifier expected by the deployed vLLM container at the Vertex AI endpoint).
> - **Vertex Endpoint ID (`VERTEX_ENDPOINT_ID` env var)**: The numeric/short endpoint identifier (e.g., `6728110112210157568`) used by `app/vertex.py` to construct the `:rawPredict` URL. **Never** place the full Vertex resource path or endpoint ID in `provider_model_id`.
> - **Agreed Pricing**:
>   - Input cost: `$0.003000` per 1,000 prompt tokens (`input_cost_per_1k = 0.003000`)
>   - Output cost: `$0.015000` per 1,000 completion tokens (`output_cost_per_1k = 0.015000`)
> - **Token Limits**: `context_window = 8192`, `max_output_tokens = 4096`.
> - **Template Execution**: Execute this seed statement only after applying migrations `20261014000000` and `20261016000000`.

```sql
-- Seed rantera-base-v1 into mp_model_catalog (Template)
INSERT INTO "mp_model_catalog" (
    "id",
    "model_id",
    "display_name",
    "description",
    "provider",
    "provider_model_id",
    "is_active",
    "context_window",
    "max_output_tokens",
    "input_cost_per_1k",
    "output_cost_per_1k",
    "created_at",
    "updated_at"
) VALUES (
    'cl_rantera_base_v1_seed',
    'rantera-base-v1',
    'Rantera Base v1',
    'Rantera base model for internal evaluation and experimentation.',
    'vertex_ai',
    '/tmp/afriqueqwen',
    true,
    8192,
    4096,
    0.003000,
    0.015000,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
)
ON CONFLICT ("model_id") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "description" = EXCLUDED."description",
    "provider" = EXCLUDED."provider",
    "provider_model_id" = EXCLUDED."provider_model_id",
    "is_active" = EXCLUDED."is_active",
    "context_window" = EXCLUDED."context_window",
    "max_output_tokens" = EXCLUDED."max_output_tokens",
    "input_cost_per_1k" = EXCLUDED."input_cost_per_1k",
    "output_cost_per_1k" = EXCLUDED."output_cost_per_1k",
    "updated_at" = CURRENT_TIMESTAMP;
```

---

## 6. Staging Deployment Checklist

1. [ ] Inspect staging database migration status (`npx prisma migrate status`).
2. [ ] Apply PostgreSQL DDL migrations (`20261014000000` and `20261016000000`) to staging.
3. [ ] Verify `mp_active_api_keys_v` and `mp_active_users_v` views and apply least-privilege role grants.
4. [ ] Execute `rantera-base-v1` catalog seed SQL template with confirmed provider model ID (`/tmp/afriqueqwen`).
5. [ ] Verify or provision GCP Pub/Sub topic `mp-inference-telemetry` and dead-letter topic `mp-inference-telemetry-dlq`.
6. [ ] Verify or provision GCP Service Account `pubsub-telemetry-invoker` and bind Cloud Run invoker role.
7. [ ] Configure push subscription with audience `https://<staging-gateway-url>/v1/telemetry/pubsub/push`.
8. [ ] Deploy Cloud Run service with environment variables (`INTERNAL_SERVICE_KEY`, `PUBSUB_PUSH_AUDIENCE`, `PUBSUB_PUSH_SERVICE_ACCOUNT_EMAIL`).
9. [ ] Deploy website to staging with `MODEL_PLATFORM_URL`, `MODEL_PLATFORM_INTERNAL_KEY`, and `CRON_SECRET`.
10. [ ] Configure Cloud Scheduler (or verify Vercel Pro cron) for 1-minute settlement interval (`POST /api/cron/settlement`).
