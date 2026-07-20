#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'uv не установлен: https://docs.astral.sh/uv/' >&2
    exit 1
fi

uv sync --frozen
exec uv run voice-match
