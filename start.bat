@echo off
where uv >nul 2>&1
if errorlevel 1 (
    echo uv не установлен: https://docs.astral.sh/uv/
    exit /b 1
)

uv sync --frozen
if errorlevel 1 exit /b 1
uv run voice-match
