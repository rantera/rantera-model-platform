# Rantera Model Platform

Secure internal API platform that gives authorized Rantera teammates access to foundation models without requiring direct Google Cloud or Vertex AI credentials.

## Status

> **⚠️ In progress** — the FastAPI application foundation is implemented with health endpoints, model registry, and authentication boundaries. Vertex AI integration, persistent key storage, and deployment are not yet implemented.

### What works now

| Endpoint | Auth | Description |
|---|---|---|
| `GET /health/live` | Public | Process liveness probe |
| `GET /health/ready` | Public | Configuration readiness (does not verify upstream) |
| `GET /v1/models` | Protected | Lists registered models (returns 503 until key storage is implemented) |

### Not yet implemented

- Live Vertex AI upstream integration (managed prediction routing gate)
- Cloud Run production deployment


## Architecture

- **FastAPI** service exposing a REST/HTTPS endpoint for model inference.
- **API-key authentication** — each client receives a unique key; only secure hashes are stored server-side.
- **Model registry** — logical model IDs (e.g. `rantera-base-v1`) map to Vertex AI endpoints via server-side configuration.
- **Google Cloud credentials** are never exposed to API clients; the service uses Application Default Credentials or the deployed Cloud Run service account.
- **Rate limiting, input validation, and audit logging** protect the service and track usage.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/security.md`](docs/security.md) for details.

## Local Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the project with dev dependencies
pip install -e ".[dev]"

# 3. Copy environment template and fill in values
cp .env.example .env
# Edit .env with your configuration (Vertex AI settings are optional for local dev)

# 4. Run the development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 5. Run tests
pytest
```

> **Note:** The `app` variable in `app.main` is created via `create_app()` at module level. For the development server, the app factory runs with settings loaded from `.env`.

## Container Deployment (Docker & Cloud Run)

The service is packaged using a multi-stage Docker build with non-root runtime execution (`uid=10001`):

### 1. Build Container Image

```bash
docker build -t rantera-model-platform:local .
```

### 2. Local Container Smoke Test (Offline / Development Mode)

```bash
# Run container in development mode with unconfigured Vertex/DB fallbacks
docker run -d --name rmp-smoke -p 8000:8000 -e RANTERA_ENV=development rantera-model-platform:local

# Verify process liveness
curl -fsS http://127.0.0.1:8000/health/live

# Verify readiness (returns 200 with degraded state indicating unconfigured external resources)
curl -fsS http://127.0.0.1:8000/health/ready

# Clean up
docker stop rmp-smoke
docker rm rmp-smoke
```

### 3. Production Deployment Notes (Cloud Run)

In production (`RANTERA_ENV=production`), secrets and configuration MUST be injected via environment variables or Google Cloud Secret Manager at container runtime. Missing required configuration (`GCP_PROJECT_ID`, `GCP_REGION`, `VERTEX_ENDPOINT_ID`) causes the service to fail closed on startup.

```bash
gcloud run deploy rantera-model-platform \
  --image gcr.io/${GCP_PROJECT_ID}/rantera-model-platform:latest \
  --set-env-vars RANTERA_ENV=production,GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_REGION=${GCP_REGION},VERTEX_ENDPOINT_ID=${VERTEX_ENDPOINT_ID} \
  --set-secrets DATABASE_URL=model-platform-database-url:latest \
  --service-account model-platform-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --port 8080
```

## Project Structure

```
rantera-model-platform/
├── app/
│   ├── __init__.py       # Package initializer
│   ├── auth.py           # Authentication interface
│   ├── billing.py        # Balance verification & in-flight tracker
│   ├── config.py         # Pydantic Settings configuration
│   ├── db.py             # Database connectivity & pooling
│   ├── logging.py        # Structured JSON audit logging
│   ├── main.py           # FastAPI application factory
│   ├── models.py         # Dynamic model catalog service
│   ├── prompts.py        # HIRA system prompt, identity protection, message composition
│   ├── rate_limit.py     # Token-bucket rate limiter
│   ├── schemas.py        # Request/response Pydantic models
│   ├── telemetry.py      # Inference telemetry recording
│   └── vertex.py         # Upstream model provider interface
├── tests/
├── docs/
├── .dockerignore
├── .env.example          # Environment variable template (no secrets)
├── .gitignore
├── Dockerfile            # Multi-stage production container image
├── pyproject.toml        # Project metadata and dependencies
└── README.md
```

## License

Proprietary — Rantera internal use only.
