# Rantera Model Platform — Architecture

> **Status:** Current implementation provides the minimal FastAPI service foundation (`app/` modules, health endpoints, read-only model registry, and auth boundary). Upstream Vertex AI integration, persistent key storage, and Cloud Run deployment are planned for future phases.

## Overview

Rantera Model Platform is a secure internal API service that gives authorized teammates access to Rantera's foundation models. Clients interact with a clean HTTPS API and never receive direct access to Google Cloud, Vertex AI, or the underlying infrastructure.

## Request Flow

```
┌────────────┐       HTTPS        ┌──────────────────────────────┐        gRPC/REST       ┌────────────────┐
│            │  ──────────────▶   │   Rantera Model Platform     │  ──────────────────▶    │   Vertex AI    │
│   Client   │   API key auth    │                              │   Service account       │   Endpoint     │
│            │  ◀──────────────   │   (FastAPI on Cloud Run)     │  ◀──────────────────    │                │
└────────────┘    JSON response   └──────────────────────────────┘    Prediction response  └────────────────┘
```

### Step-by-step

1. **Client sends request** — an internal teammate or service calls the platform's HTTPS API, including their individual API key in the request header.
2. **Authentication** — the platform hashes the presented key and looks up the hash to identify the caller. Invalid or missing keys are rejected immediately.
3. **Authorization** — the platform checks whether the authenticated client is authorized to use the requested model.
4. **Request validation** — input payload is validated: prompt length, token limits, and required fields are enforced.
5. **Model routing** — the logical model ID (e.g. `rantera-base-v1`) is resolved to a Vertex AI endpoint through server-side configuration. The client never sees or controls the endpoint ID.
6. **Vertex AI call** — the platform forwards the validated request to the Vertex AI prediction endpoint using server-side Google Cloud credentials (Application Default Credentials or the Cloud Run service account).
7. **Response** — the prediction result is returned to the client. Internal identifiers, project details, and infrastructure information are stripped from the response.

## Key Design Decisions

### Model Registry

- Each model is identified by a logical ID (e.g. `rantera-base-v1`).
- The mapping from logical ID → Vertex AI endpoint is stored in server-side configuration and is not exposed to clients.
- Future model IDs can be added through a controlled registry update. Multi-model routing is not implemented in the initial version.

### Credential Isolation

- Google Cloud credentials remain entirely server-side.
- The service authenticates to Vertex AI using Application Default Credentials (local development) or the Cloud Run service account (deployed).
- No service-account JSON keys are stored in the repository.
- Clients authenticate with their own API keys, which are independent of any Google Cloud identity.

### Deployment Target

- The planned deployment target is **Google Cloud Run**.
- Specific project ID, region, and endpoint ID are **pending** and will be configured at deployment time via environment variables.
- Model training and fine-tuning are completely separate from this API service.

## Application Modules

The application is structured into implemented baseline modules and planned future components:

### Implemented Modules

| Module | Purpose | Status |
|---|---|---|
| `app/config.py` | Environment-driven configuration and settings (Pydantic Settings) | Implemented |
| `app/auth.py` | API-key authentication boundary and key hashing utilities | Implemented (view-based auth) |
| `app/billing.py` | Pre-inference wallet balance verification and local in-flight tracker | Implemented |
| `app/rate_limit.py` | Bounded in-memory token-bucket rate limiter & async limiter protocol | Implemented (Part 5.1 local) |
| `app/models.py` | Dynamic model catalog service with local caching and validation | Implemented |
| `app/schemas.py` | Strict request/response validation (Pydantic) | Implemented |
| `app/telemetry.py` | Post-inference telemetry insertion to mp_inference_logs | Implemented |
| `app/logging.py` | Structured audit and security event logging (Part 6) | Implemented |
| `app/main.py` | FastAPI application entrypoint, middleware, and route handlers | Implemented |

### Planned Modules

| Module | Purpose | Status |
|---|---|---|
| `app/vertex.py` | Vertex AI client integration and live prediction routing | Planned (container contract pending) |
| Distributed Limiter | Shared multi-instance rate limiting backend (Redis) | Planned (Part 5.2) |
| Persistent Key Store | Database-backed API key storage, issuance, and verification | Implemented (via mp_active_api_keys_v) |

## Structured Security and Audit Logging (Part 6)

The platform implements single-line structured JSON logging (`app/logging.py`) with strict defense-in-depth sanitization, PII minimization, and crash-resilient fallbacks:

- **Stable Schema:** Every record is emitted as single-line JSON with:
  `timestamp` (UTC ISO-8601 with `Z`), `level`, `logger`, `event`, `message`, `environment`, `request_id`, and optional `metadata`.
- **Request Correlation via ContextVar:**
  Request IDs are propagated across asynchronous coroutines via `contextvars.ContextVar`. The HTTP middleware sets the contextvar and strictly resets it in a `finally` block, ensuring no correlation bleeding across concurrent tasks or exception/cancellation flows.
- **Strict Allowlisted Events & Metadata:**
  Only predefined events (`_ALLOWED_EVENTS`) and bounded metadata fields (`_ALLOWED_METADATA_FIELDS`) are serialized. Arbitrary dictionaries, request bodies, prompt messages, and model completions are strictly forbidden.
- **Secret Redaction Engine:**
  - Production API keys: `sk-rnt-[0-9a-fA-F]{48}` are completely redacted to `[REDACTED_API_KEY]` with no suffix preserved.
  - Test/fixture keys: `sk-ran-[A-Za-z0-9_-]+` are redacted to `[REDACTED_API_KEY]`.
  - Bearer tokens: `Bearer ...` are redacted to `Bearer [REDACTED]`.
  - Database URLs: Credential components in `postgresql+asyncpg://`, `postgres://`, and `sqlite+aiosqlite://` URLs (including percent-encoded passwords) are parsed via `urllib.parse.urlsplit` and masked with `***`.
  - Generic secrets: Key-value assignments like `password: ...` or `secret: ...` are masked with `***`.
- **PII Minimization:**
  User IDs, API-key IDs, key prefixes, email addresses, and client IP addresses are excluded from logs by default. Only operational parameters (token counts, execution latencies, status codes) are included where justified.
- **Exception Sanitization:**
  Tracebacks are rendered without local variables (`f_locals`), and both exception messages and stack frames are scrubbed of secrets and bounded in size.
- **Idempotent Handler Configuration:**
  `configure_logging()` configures the `app` logger without removing existing foreign or `pytest` handlers (such as `caplog`) and avoids duplicate record emission.
- **Fail-Safe Formatting Fallback:**
  If JSON formatting encounters an unhandled failure, the formatter catches the error and emits a safe static JSON record with `event="system.error"` and `message="Log formatting failed safely"`, preventing logger failures from interrupting inference.

## Rate Limiting Architecture: Local vs. Distributed Enforcement

### Local Rate Limiting (Part 5.1 — Current Implementation)
- **Mechanism:** Bounded, concurrency-safe in-memory token bucket (`app/rate_limit.py`, `InMemoryRateLimiter`).
- **Refill Semantics:** Continuous refill at `RATE_LIMIT_RPM / 60.0` tokens per second up to burst capacity (`RATE_LIMIT_RPM`).
- **Identity Hierarchy:** Primary identity is `key:<principal.api_key_id>`. Fallback is `user:<principal.user_id>` only when `api_key_id` is absent. Raw API keys, request bodies, and forwarding headers are never used.
- **Pipeline Order:** Enforced immediately after successful authentication and strictly before wallet checks, catalog lookup, spend reservations, and provider calls.
- **Header Contract:** Preserves and emits `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` on both 200 OK and 429 Too Many Requests responses.
- **Memory Bounding & Anti-Quota-Reset:** Bounded by `max_keys` (default 10,000). Pruning evicts ONLY fully refilled, idle buckets ($>60\text{s}$). Depleted or actively consuming buckets are never evicted, preventing quota circumvention via bucket recreation. If capacity saturates, new keys fail closed deterministically.
- **Production Safety:** Disabling rate limiting (`RATE_LIMIT_ENABLED=false`) in production fails closed at startup unless an explicit operational override (`RATE_LIMIT_ALLOW_PRODUCTION_BYPASS=true`) is configured.

### Distributed Rate Limiting (Part 5.2 — Future Phase)
- **Single-Process Scope Limitation:** The in-memory limiter governs requests strictly within a single Python process. On multi-instance deployments (e.g., Google Cloud Run), each worker instance tracks its own local bucket.
- **Shared Backend Interface:** `BaseRateLimiter` protocol defines the asynchronous check interface to allow seamless plugging of a distributed backend (e.g., Redis).
- **Production Failure Policy Requirement:** When a shared store (Redis) is introduced as a required production backend, store failures must NOT silently fall back or degrade to an unmonitored per-process quota without an explicit, documented, and approved operational policy.

## Unimplemented Functionality (Phase 1 Baseline)

The following components require Phase 2/3 development before production deployment:

1. **Live database authentication:** Read queries operate against `mp_active_api_keys_v`; website core tables remain inaccessible.
2. **Vertex inference:** Upstream Google Cloud Vertex AI client execution (`app/vertex.py`), model endpoint invocations, and prediction streaming are planned for future phases; mock provider used locally.
3. **Distributed rate limiting:** While local single-process in-memory rate limiting is implemented in `app/rate_limit.py` (Part 5.1), distributed multi-instance rate limiting across Cloud Run containers requires a shared store (e.g., Redis, planned for Part 5.2).
4. **Settlement retries:** No background worker, queue-based consumer, or exponential retry mechanism exists for re-processing failed inference settlements.
5. **Atomic billing:** Inference logging and user wallet deductions are not transactionally coupled in the service code.
6. **Exactly-once settlement:** While structural idempotency columns (`request_id`, `settlement_status`, `settlement_id`) exist in schema contracts, distributed exactly-once settlement guarantees are pending application-level implementation.

## Unresolved Configuration

| Setting | Status |
|---|---|
| GCP Project ID | Pending |
| GCP Region | Pending |
| Vertex AI Endpoint ID for `rantera-base-v1` | Pending |
| Cloud Run service name and configuration | Pending |
| API-key storage backend (in-memory, database, Secret Manager) | To be decided |
| Rate-limit storage backend | To be decided |
