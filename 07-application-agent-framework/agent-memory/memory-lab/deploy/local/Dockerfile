# The lab as an image: the memory service by default; the same image runs the fake OpenAI-compatible server
# (compose) and the consolidation job (Cloud Run jobs, deploy/gcp). Build from the lab root:
#   docker build -f deploy/local/Dockerfile -t memlab:0.1.0 .
# python:3.12-slim is a moving tag: pin a digest before production (verify).
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY memlab ./memlab
RUN pip install --no-cache-dir ".[pg]" \
 && useradd --uid 10001 --create-home memlab \
 && mkdir -p /data && chown memlab:memlab /data

USER 10001
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
VOLUME ["/data"]
CMD ["python", "-m", "memlab", "serve", "--host", "0.0.0.0", "--port", "8080", "--db", "/data/memory.db", "--audit", "/data/audit.jsonl"]
