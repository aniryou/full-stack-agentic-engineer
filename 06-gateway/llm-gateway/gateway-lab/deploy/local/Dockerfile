# The lab image: the gateway and the fake providers (build context = the lab root).
#   docker build -f deploy/local/Dockerfile -t gwlab:local .
FROM python:3.11-slim
WORKDIR /lab
COPY pyproject.toml README.md LICENSE ./
COPY gwlab ./gwlab
RUN pip install --no-cache-dir ".[dpop]"
EXPOSE 8080 8101 8102
CMD ["python", "-m", "gwlab", "--help"]
