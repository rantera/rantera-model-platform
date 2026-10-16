# Rantera Model Platform — Model Catalog Seed & Phased Strategy Specification

> **Document Status:** Architectural Specification & Seed Contract
> **Target Service:** `rantera-model-platform` (FastAPI) & `rantera-webapp/rantera` (PostgreSQL / Prisma 5.16.1)
> **Phase:** Part 8 — Production Integration Design
> **Approval Status:** DRAFT — Values marked `PENDING APPROVAL` require CEO / Engineering Owner Sign-off

---

## 1. Executive Summary & Phased Strategy

The Rantera Model Platform provides an internal gateway to foundation models deployed on Google Cloud Vertex AI. To ensure rapid, safe, and verifiable delivery, the platform adopts a **two-phase model strategy**:

1. **Phase 1 (Immediate Target — End-to-End Baseline):**
   Focus exclusively on making the inference workflow for the initial model, **`rantera-base-v1`**, functional end-to-end. This spans client authentication, dynamic catalog lookup, rate limiting, wallet eligibility checks, upstream Vertex AI prediction dispatch, inference telemetry logging, and handoff to billing settlement.
2. **Phase 2 (Future Expansion — Catalog-Driven Growth):**
   Support future fine-tuned model versions (e.g., domain-specific checkpoints, task-optimized variants) and additional Rantera models entirely through **catalog database entries and server-side configuration**, avoiding gateway rewrites or structural architectural changes.

---

## 2. Model Catalog Schema Contract (`mp_model_catalog`)

All model metadata is stored in the PostgreSQL table `mp_model_catalog` (owned by Prisma schema and migration `20261014000000_add_model_platform_tables`):

| Column Name | PostgreSQL Type | Prisma Type | Description |
|---|---|---|---|
| `id` | `VARCHAR(36) PRIMARY KEY` | `String @id @default(cuid())` | Unique catalog row identifier (CUID). |
| `model_id` | `VARCHAR(64) UNIQUE NOT NULL` | `String @unique @map("model_id")` | Logical model identifier presented to API consumers (e.g., `rantera-base-v1`). |
| `display_name` | `VARCHAR(128) NOT NULL` | `String @map("display_name")` | Human-readable name displayed in dashboard/UI. |
| `description` | `TEXT NOT NULL` | `String @db.Text` | Functional description of the model. |
| `provider` | `VARCHAR(32) NOT NULL DEFAULT 'vertex_ai'` | `String @default("vertex_ai")` | Upstream provider backend (default: `vertex_ai`). |
| `provider_model_id` | `VARCHAR(256)` (Nullable) | `String? @map("provider_model_id")` | Internal upstream identifier (Vertex AI Endpoint Resource Name or Publisher Path). |
| `is_active` | `BOOLEAN NOT NULL DEFAULT true` | `Boolean @default(true) @map("is_active")` | Availability flag. If `false`, gateway returns HTTP 400. |
| `context_window` | `INTEGER NOT NULL DEFAULT 8192` | `Int @default(8192) @map("context_window")` | Maximum total tokens allowed (prompt + completion). |
| `max_output_tokens` | `INTEGER NOT NULL DEFAULT 4096` | `Int @default(4096) @map("max_output_tokens")` | Maximum generation completion tokens allowed. |
| `input_cost_per_1k` | `DECIMAL(10, 6) NOT NULL DEFAULT 0.003000` | `Decimal @db.Decimal(10, 6)` | Exact USD cost per 1,000 prompt tokens. |
| `output_cost_per_1k`| `DECIMAL(10, 6) NOT NULL DEFAULT 0.015000` | `Decimal @db.Decimal(10, 6)` | Exact USD cost per 1,000 completion tokens. |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @default(now())` | Creation audit timestamp. |
| `updated_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` | `DateTime @updatedAt` | Auto-updated via `mp_set_updated_at()` trigger. |

---

## 3. Initial Model Specification: `rantera-base-v1`

The initial model deployment establishes the primary pipeline. No values are assumed or invented; commercial and infrastructure specifics are designated as `PENDING APPROVAL`.

### 3.1 Metadata & Upstream Provider Mapping

| Attribute | Specification / Value | Status | Rationale & Notes |
|---|---|---|---|
| **Public Model ID** | `rantera-base-v1` | **APPROVED** | Stable public identifier matching gateway registry and API documentation. |
| **Display Name** | `Rantera Base v1` | **APPROVED** | Customer-facing display name. |
| **Description** | `Rantera base foundation model for general evaluation, text reasoning, and experimentation.` | **APPROVED** | Clear description for model discovery (`GET /v1/models`). |
| **Provider** | `vertex_ai` | **APPROVED** | Target prediction infrastructure is Google Cloud Vertex AI. |
| **Provider Model ID** | `projects/${GCP_PROJECT_ID}/locations/${GCP_REGION}/endpoints/${VERTEX_ENDPOINT_ID}` | `PENDING APPROVAL` | Fully qualified Vertex AI Endpoint resource name. Specific GCP Project, Region, and Endpoint ID must be provided by infrastructure owners. |
| **Is Active** | `true` | **APPROVED** | Initial state must be active to permit authorized client traffic. |

### 3.2 Token Limits & Cost Schedule

| Constraint / Metric | Code / Schema Default | Production Commercial Value | Status |
|---|---|---|---|
| **Context Window** | `8192` tokens | `PENDING APPROVAL` | Maximum context limit. Gateway enforces prompt token length against this ceiling prior to dispatch. |
| **Max Output Tokens** | `4096` tokens | `PENDING APPROVAL` | Maximum completion length per generation request. |
| **Input Cost ($ / 1k tokens)** | `$0.003000` | `PENDING APPROVAL` | Rate for prompt tokens. Must be finalized with finance/commercial leadership. |
| **Output Cost ($ / 1k tokens)**| `$0.015000` | `PENDING APPROVAL` | Rate for completion tokens. Must be finalized with finance/commercial leadership. |

### 3.3 Website Billing & Database Enum Mapping

| Integration Point | Value / Mapping | Status | Gap & Required Resolution |
|---|---|---|---|
| **Website `UsageRecord.modelId`** | `PENDING APPROVAL` | `PENDING APPROVAL` | Website Prisma schema defines `enum ModelId { RANT_AI, AKILI, ILIMI, ONA, OGBON, HIRA, TERA }`. Because `'rantera-base-v1'` is not an enum member, leadership must decide whether to: (a) Add `RANTERA_BASE_V1` to the Prisma enum and run DDL migration, (b) Map `rantera-base-v1` to `RANT_AI` in the settlement worker, or (c) Alter `UsageRecord.modelId` to `VARCHAR(64)`. |
| **Settlement Payment Reference** | `usage-req_<hex>` | **APPROVED** | Deterministic prefix guaranteeing idempotency in `Transaction.paymentReference`. |

---

## 4. Initial Seed SQL Script (Phase 1 Baseline)

Once the `PENDING APPROVAL` parameters are resolved, the following parameterized SQL script must be executed by the database owner to seed `mp_model_catalog`:

```sql
-- ─── Model Platform Seed: Initial Model (rantera-base-v1) ───────────
-- Inserts the initial base model into mp_model_catalog.
-- Execution requires DBA privileges; values below reflect approved parameters.

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
    'cat_rantera_base_v1_001',
    'rantera-base-v1',
    'Rantera Base v1',
    'Rantera base foundation model for general evaluation, text reasoning, and experimentation.',
    'vertex_ai',
    '<PENDING_APPROVAL_VERTEX_ENDPOINT_RESOURCE_NAME>',
    true,
    8192,         -- PENDING APPROVAL (Context window limit)
    4096,         -- PENDING APPROVAL (Max output token limit)
    0.003000,     -- PENDING APPROVAL (Input cost per 1k USD)
    0.015000,     -- PENDING APPROVAL (Output cost per 1k USD)
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
)
ON CONFLICT ("model_id") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "description" = EXCLUDED."description",
    "provider_model_id" = EXCLUDED."provider_model_id",
    "context_window" = EXCLUDED."context_window",
    "max_output_tokens" = EXCLUDED."max_output_tokens",
    "input_cost_per_1k" = EXCLUDED."input_cost_per_1k",
    "output_cost_per_1k" = EXCLUDED."output_cost_per_1k",
    "updated_at" = CURRENT_TIMESTAMP;
```

---

## 5. End-to-End Inference Workflow for `rantera-base-v1`

The complete execution lifecycle for `rantera-base-v1` requests consists of seven discrete stages:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              End-to-End Execution Sequence for rantera-base-v1                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. Ingress & Auth Check                                                         │
│     POST /v1/chat/completions (model="rantera-base-v1", X-API-Key: sk-rnt-...)   │
│     ├── Compute SHA-256 hash                                                     │
│     ├── Query mp_active_api_keys_v WHERE key_hash = :hash (with 60s memory cache)│
│     └── Verify: not expired, not banned, emailVerified=true                      │
│                                                                                  │
│  2. Rate Limiting                                                                │
│     Consume token for key:<api_key_id> (60 RPM default)                          │
│     └── Exceeded ──▶ Return HTTP 429 Too Many Requests                           │
│                                                                                  │
│  3. Model Catalog Resolution                                                     │
│     Query mp_model_catalog WHERE model_id = 'rantera-base-v1' AND is_active = true│
│     └── Cached (<60s TTL); fail-closed (HTTP 503) if stale > 300s or DB down    │
│                                                                                  │
│  4. Pre-Inference Balance & Context Validation                                   │
│     ├── Reject prompt tokens > context_window (8192) ──▶ HTTP 400 Bad Request    │
│     ├── Verify wallet_balance > 0.0 ──▶ HTTP 402 Payment Required                │
│     ├── Verify wallet_balance >= estimated_input_cost ──▶ HTTP 402               │
│     └── InFlightSpendTracker.reserve(user_id, estimated_cost)                    │
│                                                                                  │
│  5. Upstream Vertex AI Execution                                                 │
│     VertexModelProvider invokes endpoint using Cloud Run Service Account         │
│     └── Receive: generated text, tokens_in, tokens_out, latency_ms               │
│                                                                                  │
│  6. Telemetry & Cost Accounting                                                  │
│     ├── Exact cost: (tokens_in * input_cost + tokens_out * output_cost) / 1000   │
│     ├── INSERT mp_inference_logs (request_id, cost_usd, settlement_status=       │
│     │   'PENDING') ON CONFLICT (request_id) DO NOTHING                           │
│     └── Release in-flight spend reservation                                      │
│                                                                                  │
│  7. Client Response & Downstream Settlement                                      │
│     ├── Return HTTP 200 ChatCompletionResponse with TokenUsage                   │
│     └── Website background worker polls PENDING logs and deducts Wallet          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Architecture for Future Model Expansion (Zero Gateway Redesign)

The platform architecture is designed to support additional foundation models and fine-tuned checkpoints without requiring gateway recoding or redeployment:

### 6.1 Principles for Zero-Redesign Expansion
1. **Catalog-Driven Routing:**
   The gateway routes requests dynamically using `mp_model_catalog.provider_model_id`. Adding a fine-tuned checkpoint (e.g., `rantera-legal-v1` or `rantera-base-v1-ft202611`) requires only inserting a new row in `mp_model_catalog` with its unique `provider_model_id`.
2. **Autonomous Pricing & Limits:**
   Each row in `mp_model_catalog` encapsulates its own `context_window`, `max_output_tokens`, and token pricing (`input_cost_per_1k`, `output_cost_per_1k`). The billing calculation function in `app/models.py` uses the model's attributes directly:
   $$\text{Cost} = \frac{\text{tokens}_{\text{in}} \times \text{input\_cost} + \text{tokens}_{\text{out}} \times \text{output\_cost}}{1000}$$
3. **Decoupled Upstream Targets:**
   Future models can target dedicated Vertex AI endpoints, regional endpoints, or Google publisher model paths (`publishers/google/models/...`) as configured in `provider_model_id`, without changing gateway code.
4. **Soft Deactivation:**
   Deprecated models are deactivated by setting `is_active = false` in `mp_model_catalog`. Within 60 seconds (cache TTL), the gateway automatically stops routing traffic to the model and returns clean `HTTP 400 Bad Request` (`"Model is currently inactive"`) responses.

### 6.2 Anticipated Future Catalog Candidates (Illustrative Template)

*Note: The following entries represent anticipated catalog additions for future phases. All parameters are placeholder templates awaiting operational readiness.*

| Model ID | Target Purpose | Provider Model ID | Context Limit | Pricing Status |
|---|---|---|---|---|
| `rantera-base-v1` | General Purpose / Baseline | `<VERTEX_BASE_ENDPOINT>` | 8,192 | `PENDING APPROVAL` |
| `rantera-reasoning-v1` | Complex Multi-Step Reasoning | `PENDING APPROVAL` | `PENDING APPROVAL` | `PENDING APPROVAL` |
| `rantera-fast-v1` | Low-Latency / Summarization | `PENDING APPROVAL` | `PENDING APPROVAL` | `PENDING APPROVAL` |
| `rantera-base-v1-<checkpoint>` | Customer-Specific Fine-Tunes | `PENDING APPROVAL` | 8,192 | `PENDING APPROVAL` |

---

## 7. Open Decisions Awaiting Approval

The following items are blocking production deployment and must be resolved by the designated stakeholders:

1. **GCP Vertex AI Endpoint Resource Mapping:**
   *Stakeholder:* Infrastructure / GCP Owner
   *Requirement:* Provide the complete Vertex AI Endpoint Resource Name or Publisher ID for `rantera-base-v1` (`projects/.../locations/.../endpoints/...`).
2. **Commercial Pricing Schedule:**
   *Stakeholder:* CEO / Finance
   *Requirement:* Confirm whether the Phase 1 schema defaults ($0.003000 input / $0.015000 output per 1,000 tokens) represent approved customer pricing or if updated commercial rates apply.
3. **Context Window Ceiling:**
   *Stakeholder:* Model / ML Engineering
   *Requirement:* Confirm whether `rantera-base-v1` context window is strictly 8,192 tokens or requires higher capacity (e.g., 16,384 or 32,768).
4. **Website Billing Schema Harmonization:**
   *Stakeholder:* Lead Backend Engineer
   *Requirement:* Approve strategy for reconciling `mp_inference_logs.model_id` (`VARCHAR(64)`) with website `UsageRecord.modelId` (`enum ModelId`).
5. **Wallet Overdraft & Negative Balance Policy:**
   *Stakeholder:* Product / Finance
   *Requirement:* Confirm whether settlement deductions can leave a wallet with a negative balance (overdraft) if completion token cost exceeds the pre-inference balance check.
