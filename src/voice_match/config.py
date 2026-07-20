"""Конфигурация voice_match через переменные окружения."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='VM_',
        extra='ignore',
    )

    base_dir: Path = _BASE_DIR
    logs_dir: Path = _BASE_DIR / 'logs'
    models_dir: Path = _BASE_DIR / 'pretrained_models'
    temp_dir: Path = _BASE_DIR / 'tmp'

    gradio_host: str = '127.0.0.1'
    gradio_port: int = 7860
    gradio_auth_user: str | None = None
    gradio_auth_password: SecretStr | None = None

    log_level: str = 'INFO'
    use_gpu: bool = False
    num_threads: int = 4

    max_file_size_mb: int = 20
    min_audio_duration: float = 5.0
    max_audio_duration: float = 300.0
    min_speech_seconds: float = 3.0
    min_speech_ratio: float = 0.25
    max_clipping_ratio: float = 0.01
    segment_count: int = 8
    segment_duration: float = 4.0
    min_segment_duration: float = 2.0

    antispoofing_enabled: bool = False
    modification_detection_enabled: bool = False

    def ensure_dirs(self) -> None:
        """Создать каталоги, необходимые приложению."""
        for directory in (
            self.logs_dir,
            self.models_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def gradio_auth(self) -> tuple[str, str] | None:
        """Вернуть пару логин/пароль для Gradio."""
        if self.gradio_auth_user is None:
            return None
        if self.gradio_auth_password is None:
            raise ValueError(
                'VM_GRADIO_AUTH_PASSWORD обязателен, если задан '
                'VM_GRADIO_AUTH_USER.'
            )
        return (
            self.gradio_auth_user,
            self.gradio_auth_password.get_secret_value(),
        )


settings = Settings()
