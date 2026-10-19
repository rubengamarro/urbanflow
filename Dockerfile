FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.9.8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PREFECT_HOME=/app/.prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}" \
    && useradd --create-home --uid 10001 urbanflow
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra spark --no-editable \
    && mkdir -p data artifacts reports .prefect \
    && chown -R urbanflow:urbanflow /app
COPY --chown=urbanflow:urbanflow apps ./apps
USER urbanflow
EXPOSE 8000 8501 5000 4200
CMD ["urbanflow", "serve", "--host", "0.0.0.0"]
