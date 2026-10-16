# Part 9 — Rantera Base Model Integration Plan: `rantera-base-v1`

**Document Status:** Architecture & Implementation Specification (Pending Stakeholder Approvals)
**Scope:** Model Platform (`/Users/jeevankumar/rantera-model-platform`) & Website (`/Users/jeevankumar/rantera-webapp/rantera`)
**Target Model:** `rantera-base-v1`
**Upstream Provider:** Custom Model deployed on Google Cloud Vertex AI
**Reported Deployment Region:** `europe-north1` (Hamina, Finland — Unverified)
**Endpoint Resource ID:** `UNVERIFIED / PENDING DISCOVERY`
**Commercial Pricing Status:** `UNAPPROVED / PENDING COMMERCIAL SIGN-OFF`
**Model Context & Limits:** `UNAPPROVED / PENDING ML ENGINEERING SIGN-OFF`
**Overdraft & Negative Balance Policy:** `UNRESOLVED / BLOCKING PRODUCTION BILLING`

---

## Executive Summary & Architectural Invariants

This document specifies the smallest safe, production-grade integration sequence to connect `rantera-base-v1` to the Rantera Model Platform gateway and bind it to the Rantera website billing system.

It synthesizes audited findings from:
1. The **Model Platform Part 8 Audit & Catalog Seed Specification** ([`docs/model_catalog_seed_spec.md`](file:///Users/jeevankumar/rantera-model-platform/docs/model_catalog_seed_spec.md)).
2. The **Rantera Website Billing Implementation Audit** ([`rantera-webapp/rantera`](file:///Users/jeevankumar/rantera-webapp/rantera)).

### Architectural Invariants:
* **Zero Gateway Redesign for Future Models:** The gateway routes dynamically based on database configuration in `mp_model_catalog`. Future checkpoints (e.g., `rantera-base-v1-ft*`, `rantera-reasoning-v1`) must require only row insertions in `mp_model_catalog`, never gateway code modifications or redeployments.
* **Separation of Domains (Least Privilege):** The Model Platform is an inference gateway; it owns telemetry recording (`mp_inference_logs`) but has **zero write permissions** to core billing tables (`Wallet`, `Transaction`, `UsageRecord`). The website repository owns financial settlement.
* **Deterministic Idempotency:** Financial settlement must strictly enforce exactly-once execution using deterministic keys (`mp-<request_id>`) tied to database unique constraints.
* **Fail-Closed on Unverified Infrastructure:** The model remains marked `is_active = false` in catalog configuration until the physical Vertex AI endpoint is discovered, contract-tested, and commercially signed off.
* **No Speculative Balances or Overdrafts:** Creating negative wallet balances on-the-fly is strictly forbidden. An approved, transactionally enforced balance policy must be ratified prior to live billing.

---

## 1. Smallest Implementation Sequence (End-to-End)

The transition from the current development stub to live production inference requires five disciplined phases:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        End-to-End Implementation Sequence                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  Phase 1: Upstream Endpoint Discovery & Protocol Contract Verification                 │
│  ├── Discover deployed endpoint in europe-north1 via Cloud SDK                          │
│  ├── Probe endpoint protocol: determine predict (gRPC) vs rawPredict (HTTP REST)       │
│  ├── Document exact request/response JSON schema (instances, parameters, predictions) │
│  └── Verify authentication via Cloud Run Service Account (ADC)                         │
│                                                                                        │
│  Phase 2: Model Platform Provider Implementation                                       │
│  ├── Replace VertexModelProvider stub in app/vertex.py with verified protocol client   │
│  ├── Implement payload translation matching discovered upstream protocol               │
│  └── Capture verified token metrics, latency, and upstream error classification        │
│                                                                                        │
│  Phase 3: Secure Catalog Registration                                                  │
│  ├── Populate mp_model_catalog entry for 'rantera-base-v1' with verified endpoint URI   │
│  └── Keep is_active=false until staging smoke test passes; activate upon sign-off      │
│                                                                                        │
│  Phase 4: Website-Owned Settlement Worker                                              │
│  ├── Implement safe concurrency claiming (FOR UPDATE SKIP LOCKED) in website worker   │
│  ├── Expose authenticated cron route in web/src/app/api/cron/settle-inference/route.ts │
│  ├── Execute exact Decimal arithmetic with non-negative balance enforcement            │
│  └── Reconcile ModelId enum mismatch via approved website schema migration proposal   │
│                                                                                        │
│  Phase 5: Staging Verification & Production Cutover                                    │
│  ├── Execute automated test suite (371+ passing tests)                                │
│  ├── Execute live end-to-end staging smoke test with real sk-rnt-* API key             │
│  └── Verify ledger parity across Wallet, Transaction, and User Dashboard               │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Vertex Endpoint Discovery & Protocol Verification

### 2.1 Locating the Deployed Endpoint
Because the Endpoint Resource ID is not yet provided, operations must locate the active resource in `europe-north1` without assumptions or invented values.

Run the following read-only Cloud SDK command:
```bash
gcloud ai endpoints list \
  --project="<GCP_PROJECT_ID>" \
  --region="europe-north1" \
  --filter="displayName:rantera*" \
  --format="table(name,displayName,deployedModels[0].id,deployedModels[0].model,createTime)"
```

**Expected Endpoint Resource Name Format:**
```
projects/{PROJECT_NUMBER_OR_ID}/locations/europe-north1/endpoints/{ENDPOINT_ID}
```

### 2.2 Prediction Protocol Distinction: `predict` vs `rawPredict`
In Google Cloud Vertex AI, **`predict` and `rawPredict` are fundamentally distinct APIs and are NOT interchangeable**:

1. **`PredictionService.Predict` (gRPC / GAPIC):**
   * Requires inputs wrapped in Google protobuf `Value` structs matching `{"instances": [...], "parameters": {...}}`.
   * Standard for models deployed using official Google pre-built prediction containers.
   * Expects structured prediction objects in response.
2. **`PredictionService.RawPredict` (HTTP REST):**
   * Bypasses Google protobuf conversion; streams raw HTTP request bytes directly to the container's internal HTTP server.
   * Standard for modern open-source LLM serving engines (vLLM, TGI, Triton) running on Vertex AI as custom containers exposing native OpenAI-compatible endpoints (e.g., `/v1/chat/completions`).
   * Returns raw HTTP response bytes directly from the container.

**Rule:** The gateway provider must **not** assume `predict` or `rawPredict` prematurely. The exact client API must be selected based on the output of the Phase 1 discovery probe.

### 2.3 Verification Probe Protocol
Execute an isolated probe to test both protocol variants against the deployed endpoint:

```bash
# Probe 1: Testing Standard Vertex Predict Schema
cat << 'EOF' > probe_predict.json
{
  "instances": [
    {
      "messages": [{"role": "user", "content": "Rantera ping test"}]
    }
  ],
  "parameters": {
    "temperature": 0.1,
    "max_output_tokens": 10
  }
}
EOF

gcloud ai endpoints predict <ENDPOINT_ID> \
  --project="<GCP_PROJECT_ID>" \
  --region="europe-north1" \
  --json-request=probe_predict.json

# Probe 2: Testing Raw Predict Schema (OpenAI / vLLM REST Format)
cat << 'EOF' > probe_raw.json
{
  "model": "rantera-base-v1",
  "messages": [{"role": "user", "content": "Rantera ping test"}],
  "temperature": 0.1,
  "max_tokens": 10
}
EOF

gcloud ai endpoints raw-predict <ENDPOINT_ID> \
  --project="<GCP_PROJECT_ID>" \
  --region="europe-north1" \
  --request=@probe_raw.json \
  --headers="Content-Type=application/json"
```

**Contract Checklist to Record from Probe Output:**
* [ ] Which API succeeded: `predict` (gRPC structured instances) or `rawPredict` (direct HTTP pass-through)?
* [ ] Does the response include native token metrics (`tokens_in`, `tokens_out`, `prompt_tokens`), or does it return raw generated text only?
* [ ] If raw text only, which tokenizer model (e.g., Llama-3, Mistral, SentencePiece) matches `rantera-base-v1`'s vocabulary for deterministic client-side token counting?
* [ ] What is the exact response JSON path for generated content (e.g., `predictions[0].content`, `choices[0].message.content`)?
* [ ] What are the exact upstream HTTP error status codes on rate limiting / GPU saturation (e.g., 429 vs 503)?

---

## 3. Upstream Provider Implementation Pattern: `VertexModelProvider`

The existing provider stub in [`app/vertex.py`](file:///Users/jeevankumar/rantera-model-platform/app/vertex.py#L125-L149) raises `HTTP 503 Service Unavailable`. It must be replaced with a concrete client tailored to the verified protocol.

> [!NOTE]
> **ILLUSTRATIVE ARCHITECTURAL PATTERN ONLY**
> The following code outlines the integration architecture using `PredictionServiceAsyncClient`. It must be finalized only after the Phase 1 probe confirms whether `predict` or `rawPredict` is expected by the deployed container.

```python
# app/vertex.py — Illustrative Provider Pattern
from google.cloud import aiplatform_v1
from google.api_core.client_options import ClientOptions
from app.models import RegisteredModel
from app.schemas import ChatMessage
from app.vertex import PredictionResult

class VertexModelProvider:
    """Production Vertex AI provider communicating with regional custom endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Regional endpoint routing strictly required for europe-north1
        client_options = ClientOptions(
            api_endpoint=f"{settings.vertex_location}-aiplatform.googleapis.com"
        )
        self._client = aiplatform_v1.PredictionServiceAsyncClient(
            client_options=client_options
        )

    async def predict(
        self,
        model: RegisteredModel,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> PredictionResult:
        if not model.provider_model_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{model.model_id}' has no configured upstream endpoint.",
            )

        # 1. Translate ChatMessage[] to verified upstream instances
        payload = self._format_instances(messages, temperature, max_tokens)

        # 2. Invoke endpoint using regional client (predict or raw_predict per contract)
        start_time = time.monotonic()
        try:
            response = await self._client.predict(
                endpoint=model.provider_model_id,
                instances=payload["instances"],
                parameters=payload["parameters"],
                timeout=30.0,
            )
        except Exception as exc:
            # Classify upstream error into 502 Bad Gateway or 504 Gateway Timeout
            logger.error("Upstream Vertex prediction failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream model prediction service failed.",
            ) from exc

        latency_ms = int((time.monotonic() - start_time) * 1000)

        # 3. Extract text and verified token counts
        content, tokens_in, tokens_out = self._extract_result(response, messages)

        return PredictionResult(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            finish_reason="stop",
        )
```

### 3.1 Implementation Invariants
1. **Server-Side Authentication Only:** Authentication uses Application Default Credentials (ADC) attached to the Cloud Run execution service account (`rantera-model-platform-sa@...`). Never mount or load service account JSON keys.
2. **Regional Endpoint Affinity:** The client must explicitly specify `ClientOptions(api_endpoint="europe-north1-aiplatform.googleapis.com")`. Directing Europe calls to default US central endpoints causes authentication or network failures.
3. **Accurate Token Metrics:** If the upstream container provides usage, parse it directly into `tokens_in` and `tokens_out`. If not provided, invoke a pinned client-side tokenizer rather than the development heuristic (`chars // 4`).

---

## 4. Secure Model Catalog Configuration

The gateway retrieves model metadata from `mp_model_catalog` via [`app/models.py:ModelCatalogService`](file:///Users/jeevankumar/rantera-model-platform/app/models.py).

Per [`docs/model_catalog_seed_spec.md`](file:///Users/jeevankumar/rantera-model-platform/docs/model_catalog_seed_spec.md), no mock or unverified endpoint IDs are permitted. All pricing and limits remain unapproved placeholders until certified.

### 4.1 Production Seed Specification Template

```sql
-- Migration/Seed: Register rantera-base-v1 in mp_model_catalog
-- INVARIANT: is_active is set to FALSE until endpoint discovery and contract testing complete.
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
    "output_cost_per_1k"
) VALUES (
    'clx_rantera_base_v1_001',
    'rantera-base-v1',
    'Rantera Base v1',
    'Rantera foundational language model optimized for African linguistic contexts and enterprise workloads.',
    'vertex_ai',
    NULL,                -- UNVERIFIED: Populated only after Phase 1 discovery
    false,               -- INVARIANT: Dormant until staging smoke test passes
    8192,                -- UNAPPROVED PLACEHOLDER: Context ceiling
    4096,                -- UNAPPROVED PLACEHOLDER: Max output tokens
    0.003000,            -- UNAPPROVED PLACEHOLDER: Commercial input price
    0.015000             -- UNAPPROVED PLACEHOLDER: Commercial output price
)
ON CONFLICT ("model_id") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "description" = EXCLUDED."description",
    "updated_at" = CURRENT_TIMESTAMP;
```

### 4.2 Zero-Redesign Expansion for Future Models
The gateway uses `mp_model_catalog.provider_model_id` dynamically. To introduce future models (e.g., `rantera-base-v1-ft202611` or `rantera-reasoning-v1`):
1. Deploy the new checkpoint to a Vertex AI Endpoint.
2. Insert a new row in `mp_model_catalog` with the new `model_id` and `provider_model_id`.
3. Within 60 seconds (the `ModelCatalogService` memory cache TTL), the gateway recognizes and routes the new model without rebuilding or redeploying code.

---

## 5. Website-Owned Billing Settlement Component

The audit of [`/Users/jeevankumar/rantera-webapp/rantera`](file:///Users/jeevankumar/rantera-webapp/rantera) verified that core billing resides in `Wallet`, `Transaction`, and `UsageRecord`. The Model Platform records usage in `mp_inference_logs` with `settlement_status = 'PENDING'`.

To preserve domain isolation and least privilege, the settlement processor must be owned and executed by the **website application**, not the FastAPI gateway.

### 5.1 Concurrency Claim Architecture (`FOR UPDATE SKIP LOCKED`)
In a distributed deployment with multiple worker pods or concurrent cron invocations, standard `findMany` queries suffer from race conditions.

To guarantee that each pending log is processed **exactly once by a single worker**:
1. The settlement query must lock candidate rows using `FOR UPDATE SKIP LOCKED`.
2. Any concurrent worker executing the query automatically skips locked rows and processes a distinct batch without contention.
3. State transitions on failure must be guarded so that a failed worker can never overwrite a successful `'SETTLED'` state.

### 5.2 Illustrative Settlement Processor Pseudocode

> [!WARNING]
> **ILLUSTRATIVE PSEUDOCODE ONLY — NOT VERIFIED PRODUCTION CODE**
> The following TypeScript implementation demonstrates the required transaction and concurrency semantics (row locking, exact decimal arithmetic, non-negative balance checking, and state-transition safety). It must be thoroughly tested against a staging database before deployment.

```typescript
// web/src/lib/billing/settlement.ts — ILLUSTRATIVE PSEUDOCODE
import { prisma } from "@/lib/prisma"
import { Prisma } from "@prisma/client"

const MAX_BATCH_SIZE = 50
const MAX_ATTEMPTS = 5

export async function processPendingInferenceSettlements(): Promise<{
  settled: number
  failed: number
}> {
  let settled = 0
  let failed = 0

  // Execute settlement batch within an explicit interactive transaction
  await prisma.$transaction(async (tx) => {
    // 1. Concurrency Design: Atomically claim pending logs using FOR UPDATE SKIP LOCKED
    // Raw SQL required because Prisma cannot represent SKIP LOCKED in high-level API
    const candidateLogs = await tx.$queryRaw<Array<{
      id: string
      request_id: string
      user_id: string | null
      api_key_id: string | null
      model_id: string
      channel: string
      tokens_in: number
      tokens_out: number
      cost_usd: Prisma.Decimal
      settlement_attempts: number
    }>>`
      SELECT id, request_id, user_id, api_key_id, model_id, channel,
             tokens_in, tokens_out, cost_usd, settlement_attempts
      FROM "mp_inference_logs"
      WHERE "settlement_status" = 'PENDING'
        AND "settlement_attempts" < ${MAX_ATTEMPTS}
      ORDER BY "created_at" ASC
      LIMIT ${MAX_BATCH_SIZE}
      FOR UPDATE SKIP LOCKED
    `

    for (const log of candidateLogs) {
      if (!log.user_id) {
        continue
      }

      const settlementId = `mp-${log.request_id}`
      // Strict Decimal arithmetic: NEVER cast database decimals to Number
      const costUsd = log.cost_usd

      try {
        // A. Idempotency Check: Verify if Transaction reference already exists
        const existingTx = await tx.transaction.findUnique({
          where: { paymentReference: settlementId },
        })

        if (!existingTx) {
          // B. Balance Verification: Reject negative balance creation
          const wallet = await tx.wallet.findUnique({
            where: { userId: log.user_id },
          })

          if (!wallet) {
            throw new Error(`Wallet not found for userId: ${log.user_id}`)
          }

          // Enforce non-negative balance policy (Unresolved overdraft gate)
          if (wallet.balanceUsd.lessThan(costUsd)) {
            throw new Error(
              `Insufficient wallet balance: available ${wallet.balanceUsd}, required ${costUsd}`
            )
          }

          // C. Atomic Debit
          await tx.wallet.update({
            where: { userId: log.user_id },
            data: {
              balanceUsd: wallet.balanceUsd.minus(costUsd),
            },
          })

          // D. Create Transaction record
          await tx.transaction.create({
            data: {
              userId: log.user_id,
              type: "USAGE",
              amountUsd: costUsd,
              currency: "USD",
              paymentProvider: "model_platform",
              paymentReference: settlementId,
              status: "COMPLETED",
              metadata: {
                requestId: log.request_id,
                modelId: log.model_id,
                tokensIn: log.tokens_in,
                tokensOut: log.tokens_out,
                channel: log.channel,
              },
            },
          })

          // E. Create UsageRecord (Reconciling ModelId per Section 7 Proposal)
          await tx.usageRecord.create({
            data: {
              userId: log.user_id,
              modelId: "RANT_AI", // PENDING APPROVAL: Schema harmonization proposal
              tokensIn: log.tokens_in,
              tokensOut: log.tokens_out,
              apiKeyId: log.api_key_id,
            },
          })
        }

        // F. Transition log state to SETTLED
        await tx.mpInferenceLog.update({
          where: { id: log.id },
          data: {
            settlementStatus: "SETTLED",
            settlementId: settlementId,
            settledAt: new Date(),
          },
        })

        settled++
      } catch (err) {
        failed++
        console.error(`[settlement] Failed to settle log ${log.id}:`, err)

        // G. Guard against overwriting SETTLED state on failure
        // Raw SQL conditional update ensures late-failing worker cannot overwrite SETTLED
        await tx.$executeRaw`
          UPDATE "mp_inference_logs"
          SET "settlement_attempts" = "settlement_attempts" + 1,
              "settlement_status" = CASE
                WHEN "settlement_attempts" + 1 >= ${MAX_ATTEMPTS} THEN 'FAILED'
                ELSE 'PENDING'
              END,
              "updated_at" = CURRENT_TIMESTAMP
          WHERE "id" = ${log.id}
            AND "settlement_status" != 'SETTLED'
        `
      }
    }
  })

  return { settled, failed }
}
```

### 5.3 Scheduled Settlement Route Security & Authentication
The settlement worker must be invoked securely via a protected route:
* **Route:** `web/src/app/api/cron/settle-inference/route.ts`

**Strict Authentication & Authorization Requirements:**
1. **Bearer Token Validation:** The incoming request must present `Authorization: Bearer <CRON_SECRET>`.
2. **Timing-Safe Comparison:** Authentication checks must use `crypto.timingSafeEqual` with utf8-encoded Buffers to prevent timing attacks.
3. **Fail-Closed Configuration:** If `process.env.CRON_SECRET` is unset, empty, or shorter than 32 characters, the endpoint must immediately abort with `HTTP 503 Service Unavailable` without evaluating credentials.
4. **Platform Headers:** On hosting platforms (e.g., Vercel, Google Cloud Scheduler), verify expected platform headers (e.g., `x-vercel-cron` or Cloud Scheduler OIDC tokens).
5. **Method Guard:** Restrict strictly to `POST` (or `GET` if platform requires) and reject all other HTTP verbs with `405 Method Not Allowed`.

---

## 6. Financial Integrity: Idempotency, Balances & Concurrency

### 6.1 Deterministic Idempotency Key
* **Reference Format:** `mp-<request_id>`
* **Mechanism:** The website's `Transaction` table enforces `@unique` on `paymentReference` ([`schema.prisma:L112`](file:///Users/jeevankumar/rantera-webapp/rantera/prisma/schema.prisma#L112)).
* **Guarantee:** If a settlement run is interrupted, retried, or executed concurrently, duplicate debits are physically blocked by PostgreSQL unique index constraints.

### 6.2 Pre-Inference Gate vs. Unresolved Overdraft Policy
The audit exposed that the existing website `/api/chat` routes permit negative wallet balances by default. For the Model Platform, strict boundaries apply:

1. **Gateway Pre-Inference Gate (Hard Defense):**
   * Implemented in [`app/billing.py:require_billing_eligible_principal`](file:///Users/jeevankumar/rantera-model-platform/app/billing.py#L32-L68).
   * Reads `wallet_balance_usd` from `mp_active_api_keys_v`.
   * Rejects requests with `HTTP 402 Payment Required` if balance $\le 0.0$ USD or if estimated prompt cost exceeds available balance.
2. **Unresolved Overdraft Policy (BLOCKING PRODUCTION BILLING):**
   * *The Problem:* Output generation length cannot be known prior to inference. If a user with $0.02 balance generates completion tokens costing $0.05, deducting the full cost results in a negative balance (-$0.03).
   * *Status:* **UNRESOLVED.** Creating negative balances on-the-fly is prohibited in the settlement component until an official commercial and legal policy is signed off:
     * **Policy Option A (Strict Floor / No Overdraft):** Never allow wallet balance $< 0$. If settlement would breach zero, deduct only down to $0.00 and flag the account as delinquent.
     * **Policy Option B (Controlled Overdraft with Auto-Freeze):** Deduct the full actual cost into negative balance, while the gateway's pre-inference gate automatically locks out all subsequent requests across web and API until the deficit is cleared via Paystack.
   * *Requirement:* Product, Finance, and Legal must formally ratify Option A or Option B before enabling live billing.

### 6.3 Concurrency Protection
* **FastAPI Gateway:** The in-memory [`InFlightSpendTracker`](file:///Users/jeevankumar/rantera-model-platform/app/billing.py#L106-L167) dampens rapid intra-process bursts from the same user.
* **Database Settlement:** Row-level locks (`FOR UPDATE SKIP LOCKED`) prevent concurrent settlement workers from colliding or interleaving ledger entries.

---

## 7. ModelId Enum Mismatch & Proposed Harmonization

### 7.1 The Conflict
* **Website Schema:** [`prisma/schema.prisma:L221-L229`](file:///Users/jeevankumar/rantera-webapp/rantera/prisma/schema.prisma#L221-L229) defines a strict PostgreSQL enum:
  ```prisma
  enum ModelId {
    RANT_AI
    AKILI
    ILIMI
    ONA
    OGBON
    HIRA
    TERA
  }
  ```
  `UsageRecord.modelId` is strictly typed as `ModelId`.
* **Model Platform:** Uses dynamic strings (`VARCHAR(64)` in `mp_model_catalog` and `mp_inference_logs`). Initial model is `rantera-base-v1`.
* **Result:** Direct insertion of `'rantera-base-v1'` into `UsageRecord` throws PostgreSQL error:
  `invalid input value for enum "ModelId": "rantera-base-v1"`.

### 7.2 Evaluated Proposals (Requiring Website Backend Approval)

| Proposal | Architecture | Pros | Cons | Status |
|---|---|---|---|---|
| **Proposal 1 (Recommended): Migrate `UsageRecord.modelId` to VARCHAR(64)** | Change `UsageRecord.modelId` from enum to string in website schema. | Fully future-proof; permits unlimited model additions without DDL migrations. | Requires running a Prisma schema migration in the website repository. | **Proposal Submitted for Backend Approval** |
| **Proposal 2 (Fallback): Add Value to Enum** | Run `ALTER TYPE "ModelId" ADD VALUE 'RANTERA_BASE_V1'`. | Retains enum typing in website. | High friction; requires a database migration for every future checkpoint. | **Fallback Proposal** |
| **Proposal 3: Virtual Mapping to `RANT_AI`** | Insert `RANT_AI` in `UsageRecord`, record real ID in `Transaction.metadata`. | Zero website schema changes. | Corrupts Rant AI usage metrics in admin analytics dashboards. | **Rejected** |

> [!NOTE]
> No database schemas, migrations, or application code have been altered for this proposal. Proposal 1 remains under review by the Website Backend Lead.

---

## 8. Pricing & Spend-Limit Decisions Requiring Confirmation

The website audit verified that the default pricing ($0.003 / $0.015) was hardcoded as a temporary placeholder pending confirmed Vertex AI billing.

### 8.1 Required Commercial Sign-Offs

| Decision Item | Audited Default | Operational Reality | Required Sign-Off |
|---|---|---|---|
| **Input Token Price** | $0.003000 / 1,000 | Must cover regional europe-north1 GPU hosting + gross margin. | CEO / Finance (UNAPPROVED) |
| **Output Token Price** | $0.015000 / 1,000 | Must cover regional europe-north1 GPU hosting + gross margin. | CEO / Finance (UNAPPROVED) |
| **Context Window Ceiling** | 8,192 tokens | Must match physical model architecture context training. | ML Engineering (UNAPPROVED) |
| **Max Output Tokens** | 4,096 tokens | Enforced upper bound on single generation. | ML Engineering (UNAPPROVED) |

### 8.2 API Key Spend-Limit Semantics
* The website stores `ApiKey.spendLimitUsd` (`Decimal(10, 4)?`).
* Currently, the website evaluates spend limits on the fly without a rolling reset window.
* **Decision Required:** Is `spendLimitUsd` a **lifetime cap** or a **monthly billing cycle limit**?
  * *Status:* UNAPPROVED pending Finance decision.

---

## 9. Least-Privilege Database Permissions

To guarantee that gateway vulnerabilities cannot compromise financial ledgers, database privileges must be strictly partitioned between the FastAPI Gateway and the Website Settlement Worker.

### 9.1 Database Role Matrix

```sql
-- ─── 1. FastAPI Model Platform Service User ─────────────────────────
CREATE ROLE rantera_model_platform_user WITH LOGIN PASSWORD '<STRONG_PASSWORD>';

REVOKE ALL ON SCHEMA public FROM rantera_model_platform_user;
GRANT USAGE ON SCHEMA public TO rantera_model_platform_user;

-- Security Views Only (NO access to base User, ApiKey, or Wallet tables)
GRANT SELECT ON "mp_active_api_keys_v" TO rantera_model_platform_user;
GRANT SELECT ON "mp_active_users_v" TO rantera_model_platform_user;

-- Model Catalog: Read-Only
GRANT SELECT ON "mp_model_catalog" TO rantera_model_platform_user;

-- Inference Logs: INSERT and SELECT Only (Strictly NO UPDATE or DELETE)
GRANT SELECT, INSERT ON "mp_inference_logs" TO rantera_model_platform_user;


-- ─── 2. Website Settlement Worker Service User ──────────────────────
CREATE ROLE rantera_settlement_worker WITH LOGIN PASSWORD '<STRONG_PASSWORD>';

REVOKE ALL ON SCHEMA public FROM rantera_settlement_worker;
GRANT USAGE ON SCHEMA public TO rantera_settlement_worker;

-- Settlement State Updates on mp_inference_logs
GRANT SELECT, UPDATE ("settlement_status", "settlement_id", "settlement_attempts", "settled_at", "updated_at")
    ON "mp_inference_logs" TO rantera_settlement_worker;

-- Core Billing Table Access (Controlled DML)
GRANT SELECT, UPDATE ("balanceUsd", "updatedAt") ON "Wallet" TO rantera_settlement_worker;
GRANT SELECT, INSERT ON "Transaction" TO rantera_settlement_worker;
GRANT SELECT, INSERT ON "UsageRecord" TO rantera_settlement_worker;
```

---

## 10. Automated Tests & Staging Validation Checklist

### 10.1 Automated Test Suite Additions
1. `tests/test_vertex_provider.py`: Unit test custom model serialization, raw/predict protocol compliance, and timeout mapping.
2. `tests/test_settlement_worker.py`: Verify `SKIP LOCKED` batch claiming, idempotency replay, decimal math precision, attempt incrementing, and overdraft rejection.
3. `tests/test_catalog_resolution.py`: Verify gateway correctly serves `rantera-base-v1` when active and returns 400 when inactive.

### 10.2 Staging Validation Checklist

```markdown
### Staging Gate 1: GCP Infrastructure & Identity
- [ ] Vertex AI Endpoint verified in `europe-north1` via `gcloud ai endpoints list`.
- [ ] Protocol probe confirms whether `predict` or `rawPredict` is expected.
- [ ] Cloud Run Service Account granted `roles/aiplatform.user` on the specific endpoint.

### Staging Gate 2: Database Grants & Catalog
- [ ] Database user `rantera_model_platform_user` verified with restricted grants.
- [ ] Direct `UPDATE` or `INSERT` to `Wallet` by `rantera_model_platform_user` fails with permission denied.
- [ ] `mp_model_catalog` populated with verified `provider_model_id` and signed-off pricing.

### Staging Gate 3: End-to-End Inference & Telemetry
- [ ] Request to `/v1/chat/completions` with model `rantera-base-v1` returns HTTP 200.
- [ ] Row appears in `mp_inference_logs` with `request_id`, accurate tokens, and `settlement_status = 'PENDING'`.

### Staging Gate 4: Financial Settlement Parity
- [ ] Settlement cron route invoked with valid Bearer token; unauthorized calls return 401.
- [ ] Settlement worker processes pending log using `SKIP LOCKED`.
- [ ] `mp_inference_logs.settlement_status` transitions to `'SETTLED'`.
- [ ] User's `Wallet.balanceUsd` decrements by exact decimal cost without floating-point conversion.
- [ ] Row created in `Transaction` with `paymentReference = 'mp-<request_id>'`.
- [ ] Settlement worker re-runs; verifies no duplicate debit occurs (Idempotency confirmed).
- [ ] Website user dashboard displays transaction accurately in billing history.
```

---

## 11. Work Breakdown, Blockers, Approvals & Acceptance Criteria

### 11.1 Work Breakdown Structure (WBS)

```
Track 1: Infrastructure & Protocol Discovery (Owner: GCP / DevOps Lead)
├── Task 1.1: Locate Endpoint Resource ID in europe-north1
├── Task 1.2: Execute protocol probe (predict vs rawPredict) and document instance schema
└── Task 1.3: Bind roles/aiplatform.user to Cloud Run Service Account

Track 2: Gateway Provider Implementation (Owner: Model Platform Lead)
├── Task 2.1: Implement verified client in app/vertex.py (gRPC or REST per probe)
├── Task 2.2: Implement instance serialization and error translation
├── Task 2.3: Unit test provider with mock responses
└── Task 2.4: Seed mp_model_catalog with discovered endpoint and is_active=false

Track 3: Website Settlement Component (Owner: Website Backend Lead)
├── Task 3.1: Review and approve Enum harmonization migration (UsageRecord.modelId -> VARCHAR)
├── Task 3.2: Implement web/src/lib/billing/settlement.ts with SKIP LOCKED & Decimal arithmetic
├── Task 3.3: Implement authenticated web/src/app/api/cron/settle-inference/route.ts
└── Task 3.4: Ratify and enforce non-negative balance / overdraft policy

Track 4: Verification & Deployment (Owner: QA / Security Lead)
├── Task 4.1: Verify PostgreSQL least-privilege role grants
├── Task 4.2: Execute Staging Validation Checklist
└── Task 4.3: Toggle mp_model_catalog.is_active = true for rantera-base-v1
```

### 11.2 Blockers & Approvals Matrix

| Item | Required Sign-Off Stakeholder | Blocking Impact | Resolution Action |
|---|---|---|---|
| **Endpoint Resource ID** | Infrastructure / GCP Admin | Blocks Phase 2 & 3 | Provide output of Section 2.1 `gcloud` command. |
| **Prediction Protocol Contract** | ML Serving / GCP Admin | Blocks Task 2.1 | Confirm `predict` vs `rawPredict` from Section 2.3 probe. |
| **Commercial Token Pricing** | CEO / Finance Lead | Blocks Phase 3 & 4 | Confirm whether $0.003 / $0.015 applies to `rantera-base-v1`. |
| **ModelId Enum Resolution** | Website Backend Lead | Blocks Task 3.1 | Approve Proposal 1 (`UsageRecord.modelId` migration). |
| **Overdraft & Balance Policy** | Product / Finance / Legal | Blocks Task 3.2 | Formally approve Option A or Option B balance policy. |

### 11.3 Acceptance Criteria
1. **Zero Redesign:** A subsequent model (e.g., `rantera-base-v2`) can be onboarded solely by adding a row to `mp_model_catalog`.
2. **Deterministic Ledger:** Every completed inference generates exactly one `Transaction` with reference `mp-<request_id>` and decrements the wallet balance by the exact decimal cost.
3. **No Duplicate Charges:** Running settlement multiple times on the same batch produces zero redundant deductions.
4. **No Overwrites:** A failed concurrent settlement worker cannot overwrite a row already marked `'SETTLED'`.
5. **Leak-Proof Gateway:** The Model Platform cannot mutate `Wallet` or `Transaction` tables directly even under arbitrary code execution within the gateway container.
6. **Fail-Closed Availability:** If the upstream Vertex AI endpoint returns errors or exhausts capacity, the gateway returns clean `502 Bad Gateway` / `504 Gateway Timeout` responses with zero corrupted billing deductions.
