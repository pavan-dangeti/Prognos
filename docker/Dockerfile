# Pinned by digest so the base image cannot change underneath a passing build.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_VENV_SEED=false

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

# Strip pip from the virtualenv. uv resolves and installs dependencies, so nothing
# at runtime needs pip — but pip bundles its own vendored copies of setuptools,
# urllib3 and msgpack, and the image scanner reports those vendored copies as
# project dependencies. Deleting pip here rather than in the final stage matters:
# the final stage copies the venv from this one, so a deletion applied afterwards
# would leave the files in an earlier layer where the scanner still finds them.
RUN rm -rf /opt/venv/lib/python3.12/site-packages/pip* \
    /opt/venv/lib/python3.12/site-packages/setuptools* \
    /opt/venv/bin/pip* \
    /root/.cache

# Runtime stage: same base, but only the venv crosses over. Nothing built here
# carries the builder's pip.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# LightGBM links against the GNU OpenMP runtime, which the slim image does not
# ship. The h5py wheels bundle their own HDF5, so that one needs nothing; this one
# does need a system library.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The base image's own pip also vendors those libraries, so it goes too.
RUN rm -rf /usr/local/lib/python3.12/site-packages/pip* \
    /usr/local/lib/python3.12/site-packages/setuptools* \
    /usr/local/lib/python3.12/site-packages/urllib3* \
    /usr/local/bin/pip*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src
# pyproject.toml carries the pytest configuration, including the custom markers.
# Without it, `pytest -m smoke` runs everything and warns about unknown marks.
COPY pyproject.toml /app/pyproject.toml

WORKDIR /app

# Run as a non-root user: a data pipeline that reads a 15 GB archive has no
# business running as root.
RUN useradd --create-home --uid 10001 prognos && chown -R prognos:prognos /app /opt/venv
USER prognos

# No ENTRYPOINT: the image runs the CLI, the test suite or a shell depending on
# the task. A fixed entrypoint would have to be overridden for every one of them.
CMD ["prognos", "--help"]