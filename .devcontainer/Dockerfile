FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ARG UV_VERSION=0.11.29
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/rllib-async-venv \
    PATH="/opt/rllib-async-venv/bin:${PATH}"

RUN mkdir -p "${UV_PROJECT_ENVIRONMENT}" \
    && chown -R vscode:vscode "${UV_PROJECT_ENVIRONMENT}"
