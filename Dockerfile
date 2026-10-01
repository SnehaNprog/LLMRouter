# syntax=docker/dockerfile:1

# ---- build: install the third-party dependencies into a virtualenv ----
FROM python:3.12-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Only pyproject.toml is copied here, so this (slow) layer is rebuilt when the dependencies
# change and not when the code does. The pinned versions are read straight from it.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt

# ---- runtime: the virtualenv and the code, nothing else ----
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# An unprivileged user: a bug in the gateway must not be a bug in the container as root.
RUN groupadd --system --gid 10001 llmrouter \
    && useradd --system --uid 10001 --gid llmrouter --home-dir /app --shell /usr/sbin/nologin llmrouter

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=llmrouter:llmrouter app ./app
COPY --chown=llmrouter:llmrouter migrations ./migrations
COPY --chown=llmrouter:llmrouter alembic.ini config.example.yaml ./

USER llmrouter
EXPOSE 8000

# Liveness only (the process answers); /ready also checks the database and Redis.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# One worker per container: scale by running more containers, which also gives each its own
# failure domain. Secrets and settings come from the environment (see .env.example).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
