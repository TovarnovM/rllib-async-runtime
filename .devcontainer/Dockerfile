ARG UV_VERSION=0.11.29
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/rllib-async-venv \
    PATH="/opt/rllib-async-venv/bin:${PATH}"

RUN mkdir -p "${UV_PROJECT_ENVIRONMENT}" \
    && chown -R vscode:vscode "${UV_PROJECT_ENVIRONMENT}"
