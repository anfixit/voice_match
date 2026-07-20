"""Тесты конфигурации."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from voice_match.config import Settings


def test_ensure_dirs_creates_runtime_directories(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        logs_dir=tmp_path / 'logs',
        models_dir=tmp_path / 'models',
        temp_dir=tmp_path / 'tmp',
    )

    settings.ensure_dirs()

    assert settings.logs_dir.is_dir()
    assert settings.models_dir.is_dir()
    assert settings.temp_dir.is_dir()


def test_gradio_auth_returns_credentials() -> None:
    settings = Settings(
        gradio_auth_user='anfi',
        gradio_auth_password=SecretStr('secret'),
    )

    assert settings.gradio_auth() == ('anfi', 'secret')


def test_gradio_auth_requires_password() -> None:
    settings = Settings(gradio_auth_user='anfi')

    with pytest.raises(ValueError, match='PASSWORD обязателен'):
        settings.gradio_auth()
