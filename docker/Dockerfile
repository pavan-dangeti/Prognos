# Pinned by digest so the base image cannot change underneath a passing build.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# No system HDF5 needed: the h5py wheels ship their own HDF5 build, so the image
# stays free of a native library dependency that would need distro pinning.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: this layer only rebuilds when the lockfile changes, so a
# code change does not re-resolve the whole dependency tree.
COPY pyproject.toml uv.lock README.md ./
COPY src/prognos/__init__.py src/prognos/__init__.py
RUN uv sync --frozen --no-install-project --all-groups

COPY src ./src
RUN uv sync --frozen --all-groups

# Run as a non-root user: a data pipeline that reads a 15 GB archive has no
# business running as root.
RUN useradd --create-home --uid 10001 prognos && chown -R prognos:prognos /app /opt/venv
USER prognos

# No ENTRYPOINT: the image runs the CLI, the test suite or a shell depending on
# the task. A fixed entrypoint would have to be overridden for every one of them.
CMD ["prognos", "--help"]