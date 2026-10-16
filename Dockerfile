# ── Stage 1: Build virtual environment ────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package metadata and source to install runtime dependencies only
COPY pyproject.toml README.md ./
COPY app/ ./app/

# Upgrade pip and install package with runtime dependencies (excluding dev dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Stage 2: Minimal runtime image ────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create a dedicated non-root user and group (UID/GID 10001) with no login shell
RUN groupadd -g 10001 rantera && \
    useradd -u 10001 -g rantera -s /usr/sbin/nologin -M rantera

WORKDIR /app

# Copy virtualenv containing installed dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY --chown=rantera:rantera app/ /app/app

# Enforce read-only permissions for runtime security
RUN chmod -R 555 /app /opt/venv

# Run as unprivileged user
USER 10001:10001

# Document container port
EXPOSE 8000

# Container liveness check using standard library urllib (no external tools needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health/live')" || exit 1

# Start Uvicorn via exec to ensure PID 1 receives SIGTERM for graceful shutdown
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --lifespan on --access-log"]
