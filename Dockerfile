FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH='/app/.venv/bin:/usr/local/bin:/usr/bin:/bin'

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.16 \
    /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
RUN uv sync --no-dev


FROM python:3.12-slim AS runtime

ENV HOME='/tmp' \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME='/app/pretrained_models/cache' \
    HF_HOME='/app/pretrained_models/huggingface' \
    PATH='/app/.venv/bin:/usr/local/bin:/usr/bin:/bin'

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder --chown=10001:10001 /app /app
RUN mkdir -p /app/pretrained_models \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    --retries=3 \
    CMD python -c "import urllib.request; \
urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)"

CMD ["voice-match"]
