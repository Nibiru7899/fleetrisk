# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm AS build
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.7
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY apps ./apps
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12.12-slim-bookworm AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fleetrisk && useradd --uid 10001 --gid 10001 --create-home fleetrisk
COPY --from=build /app/.venv /app/.venv
COPY --chown=10001:10001 db ./db
COPY --chown=10001:10001 alembic.ini ./
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts /app/reports /app/mlruns && chown -R 10001:10001 /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
USER 10001:10001
EXPOSE 8000
