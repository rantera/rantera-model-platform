# Rantera Model Platform — Security Requirements

> **Status:** This document lists *requirements* for the platform's security posture. It does **not** establish that these controls are implemented or verified. Each requirement must be validated through code review, testing, and operational audit before the platform is considered production-ready.

## API Key Management

- **Cryptographically random keys** — API keys must be generated using a cryptographically secure random generator (e.g. `secrets.token_urlsafe`).
- **Hash-only storage** — only the secure hash (e.g. SHA-256 or Argon2) of each key is stored. Plaintext keys are never persisted.
- **Show once** — a newly issued API key is displayed to the administrator exactly once at creation time. It cannot be retrieved after that.
- **Revocation** — any key can be revoked immediately, disabling all access for that key.
- **Rotation** — keys can be rotated (issue a new key, deprecate the old one) without service downtime.
- **Access auditing** — all key issuance, revocation, and rotation events must be logged with timestamps and the identity of the operator.

## Authentication and Authorization

- **Authenticate every protected request** — every request to a protected endpoint must present a valid API key. Unauthenticated requests are rejected with a `401 Unauthorized` response.
- **Model-level authorization** — after authentication, the platform must verify that the caller is authorized to use the specific model requested. Unauthorized requests are rejected with a `403 Forbidden` response.
- **Constant-time comparison** — key hash lookups must use constant-time comparison to prevent timing attacks.

## Google Cloud Credential Security

- **Least-privilege IAM** — the service account used by the platform must have only the minimum IAM permissions required to call the Vertex AI prediction endpoint and access required secrets. No project-wide editor or owner roles.
- **No service-account JSON keys in the repository** — the service authenticates using Application Default Credentials (local development) or the attached Cloud Run service account (deployed). JSON key files are never committed.
- **No secrets in Git** — no API keys, credentials, tokens, or sensitive configuration may be committed to the repository. `.env` and `.env.*` files are excluded by `.gitignore`.
- **No secrets in container images** — secrets are injected at runtime via environment variables or Secret Manager, never baked into Docker images.

## Production Secret Management (Google Cloud Secret Manager)

In production and staging deployments (Google Cloud Run), all sensitive credentials must be managed via Google Cloud Secret Manager rather than local files:

### Secret Storage & Injection
1. **Secret Resources:** Sensitive values such as the PostgreSQL connection string (`DATABASE_URL`) are created as versioned Secret Manager resources:
   - `projects/${GCP_PROJECT_ID}/secrets/model-platform-database-url`
2. **Container Injection:** Google Cloud Run mounts secrets directly as environment variables at container startup:
   ```bash
   gcloud run deploy rantera-model-platform \
     --set-secrets=DATABASE_URL=model-platform-database-url:latest \
     --service-account=model-platform-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com ...
   ```
3. **Application Decoupling:** The application reads secrets via standard environment variables through Pydantic `Settings(BaseSettings)`. This keeps runtime code decoupled from cloud SDK calls and ensures local development behaves identically using `.env`.
4. **Environment Isolation:** In production and staging (`RANTERA_ENV=production` or `staging`), `.env` file loading is disabled in `Settings`. Values must originate from the container environment.

### Least-Privilege IAM Configuration
- **Dedicated Service Account:** The service runs under a dedicated, minimal service account (`model-platform-sa@<project-id>.iam.gserviceaccount.com`).
- **Resource-Scoped Access:**
  - `roles/secretmanager.secretAccessor` is granted **only** on the specific secret resource (`projects/${PROJECT_ID}/secrets/model-platform-database-url`), never at the project or folder level.
  - `roles/aiplatform.user` is granted **only** on the specific Vertex AI model endpoint resource.
  - Administrative roles (`roles/secretmanager.admin`), viewer roles (`roles/secretmanager.viewer`), and broad roles (`roles/editor`, `roles/owner`) are strictly forbidden.
- **Database Role Isolation:** The PostgreSQL connection string connects as `mp_service_role`, restricted solely to approved views (`mp_active_api_keys_v`, `mp_active_users_v`) and tables (`mp_model_catalog`, `mp_inference_logs`), with no DDL privileges or access to core website tables.
- **Fail-Closed Redaction:** If database URL parsing fails or is malformed, redaction fails closed by emitting `[DATABASE_URL_REDACTED]` rather than a partially masked URL.

## Request Validation and Input Safety

- **Strict schema validation** — all incoming request bodies must conform to a defined Pydantic schema. Unknown fields are rejected.
- **Bounded input sizes** — maximum prompt length and token limits are enforced server-side.
- **Timeouts** — upstream Vertex AI calls must have explicit timeouts to prevent hanging requests.
- **Rate limiting** — per-key rate limits are enforced to prevent abuse and protect upstream quotas. Rate limit values are configurable.

## Error Handling

- **Safe error responses** — error responses must not reveal:
  - Stack traces or internal exception details.
  - API keys, credentials, or tokens.
  - Vertex AI endpoint IDs, GCP project IDs, or internal infrastructure details.
  - Raw upstream error messages that might leak configuration.
- **Structured error format** — errors are returned in a consistent JSON format with a request ID for correlation.

## Logging and Audit Trail

- **Request IDs** — every request is assigned a unique request ID, included in the response and in all log entries for that request.
- **Audit logs** — authentication successes, failures, authorization decisions, and key lifecycle events are logged.
- **No sensitive data in logs** — API keys, credentials, raw prompts, and raw model responses are **not** logged by default. If prompt logging is enabled for debugging, it must be an explicit opt-in with appropriate access controls.
- **Structured logging** — logs are emitted in structured JSON format for integration with Cloud Logging.

## Transport Security

- **HTTPS for deployed traffic** — all client-facing traffic must be encrypted with TLS. In a Cloud Run deployment, this is handled by the managed ingress.
- **No plaintext credentials in transit** — API keys are transmitted in HTTPS request headers, never in query strings or request bodies.

## Testing Requirements

Automated tests must cover:

- Authentication failure (missing key, invalid key, revoked key).
- Authorization bypass (valid key, unauthorized model).
- Invalid inputs (malformed payload, oversized prompt, missing fields).
- Rate limit enforcement (exceeding configured limits).
- Upstream error handling (Vertex AI timeout, 5xx, invalid response).
- Error response safety (no leakage of internal details).

## Compliance and Review

- All security requirements must be reviewed and approved before production deployment.
- Security controls must be validated through automated tests, code review, and manual audit.
- This document will be updated as the implementation progresses and new threats are identified.
