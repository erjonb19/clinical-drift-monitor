# Public research demo of the classifier (Phase 3): FastAPI on CPU. Not for medical use.
# Build context: the repository, with the serving bundle copied to dist/bundle beforehand
# (never committed). Runs as user 1000 on port 7860, as Hugging Face Spaces expects.
# Pinned by digest (backlog item 35): the multi-platform index digest of python:3.11.17-slim,
# read from Docker Hub on 2026-11-11. Dependabot's docker updates keep it current.
FROM python:3.11.17-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

# TRUSTED_PROXY_HOPS=1: Hugging Face Spaces puts one proxy in front (see app/api/main.py).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 BUNDLE_DIR=/app/bundle \
    TRUSTED_PROXY_HOPS=1
# Debian security fixes released after the base image was built (Trivy, item 39: apt and zlib1g
# had fixable HIGH findings), taken from fixed snapshot.debian.org imports instead of the moving
# mirror, so every build of this file installs the same Debian package versions (the pip layers
# are item 27). Release files are still signature-checked with the image's Debian keyring. Raise
# both timestamps (existing imports, listed on snapshot.debian.org) and the base digest together,
# then rebuild and re-run Trivy.
ARG DEBIAN_SNAPSHOT=20261111T083534Z
ARG DEBIAN_SECURITY_SNAPSHOT=20261111T085431Z
RUN sed -i \
      -e "s|^URIs: http://deb.debian.org/debian-security$|URIs: https://snapshot.debian.org/archive/debian-security/${DEBIAN_SECURITY_SNAPSHOT}|" \
      -e "s|^URIs: http://deb.debian.org/debian$|URIs: https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|" \
      /etc/apt/sources.list.d/debian.sources \
    && grep -q "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" /etc/apt/sources.list.d/debian.sources \
    && grep -q "snapshot.debian.org/archive/debian-security/${DEBIAN_SECURITY_SNAPSHOT}" /etc/apt/sources.list.d/debian.sources \
    && ! grep -rqs "deb.debian.org" /etc/apt/sources.list /etc/apt/sources.list.d/ \
    && apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Retries=3 update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Backlog item 27: every Python package comes from docker/requirements.txt, exported from uv.lock
# with hashes (the CPU torch wheels included), so pip refuses any file whose hash differs and
# installs nothing that is not listed (--no-deps). PyPI stays the main index; the PyTorch CPU index
# is searched too, for the +cpu wheels only the hashes accept. The project itself is then built
# offline with the hashed setuptools from that file (--no-build-isolation, --no-index).
COPY docker/requirements.txt ./
RUN pip install --require-hashes --no-deps --only-binary=:all: \
      --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip check
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation --no-index .
# The base image's setuptools carried old jaraco.context and wheel copies with fixable HIGH CVEs
# (Trivy, item 39), and nothing in the API imports setuptools at run time, so the build backend
# and wheel are removed once the project is installed.
RUN pip uninstall -y setuptools wheel
COPY app/api ./app/api
COPY dist/bundle ./bundle

# Code and bundle stay root-owned: the app only reads them.
RUN useradd --uid 1000 --create-home demo
USER demo
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
# No access log: it would record client addresses; /health keeps counts and timing only.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
