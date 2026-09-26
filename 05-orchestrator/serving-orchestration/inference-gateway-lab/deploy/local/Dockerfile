# The lab image: the router and the fake backend (build context = the lab root).
#   docker build -f deploy/local/Dockerfile -t igwlab:local .
FROM python:3.11-slim
WORKDIR /lab
COPY pyproject.toml README.md LICENSE ./
COPY igwlab ./igwlab
RUN pip install --no-cache-dir .
EXPOSE 8000 9000
CMD ["python", "-m", "igwlab.router", "--help"]
