# Public research demo of the classifier (Phase 3): FastAPI on CPU. Not for medical use.
# Build context: the repository, with the serving bundle copied to dist/bundle beforehand
# (never committed). Runs as user 1000 on port 7860, as Hugging Face Spaces expects.
FROM python:3.11.17-slim

# TRUSTED_PROXY_HOPS=1: Hugging Face Spaces puts one proxy in front (see app/api/main.py).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 BUNDLE_DIR=/app/bundle \
    TRUSTED_PROXY_HOPS=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0

WORKDIR /app
COPY pyproject.toml README.md docker/constraints.txt ./
COPY src ./src
RUN pip install -c constraints.txt ".[api]" scikit-learn
COPY app/api ./app/api
COPY dist/bundle ./bundle

# Code and bundle stay root-owned: the app only reads them.
RUN useradd --uid 1000 --create-home demo
USER demo
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
# No access log: it would record client addresses; /health keeps counts and timing only.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
