# One image, two entrypoints (api / worker) selected by LRA_SERVICE.
# Build from the repo root:  docker build -f services/Dockerfile -t lra .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 LRA_SERVICE=worker
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[gcp,services]"

COPY services ./services

# Cloud Run sends SIGTERM and waits 10s; uvicorn drains in-flight requests.
CMD ["sh", "-c", "exec uvicorn services.${LRA_SERVICE}.main:app --host 0.0.0.0 --port ${PORT} --timeout-graceful-shutdown 25"]
