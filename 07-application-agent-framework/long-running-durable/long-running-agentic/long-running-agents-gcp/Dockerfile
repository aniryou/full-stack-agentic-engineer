FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . 
# library service by default; the ADK service overrides the command (see infra/deploy.sh)
CMD ["python", "-m", "lragents.service.app"]
