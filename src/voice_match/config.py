"""Конфигурация приложения через переменные окружения.

Инфраструктурные параметры и настраиваемые пороги/веса.
Физические константы (форманты, анатомия) остаются
в constants.py — они не должны меняться через .env.

Использование::

    from voice_match.config import settings

    settings.gradio_port       # 7860
    settings.thresholds        # {'very_high': 0.95, ...}
    settings.model_weights     # {'ecapa': 1.5, ...}
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Настройки приложения voice_match."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='VM_',
        extra='ignore',
    )

    # ── Пути ─────────────────────────────────────
    base_dir: Path = _BASE_DIR
    logs_dir: Path = _BASE_DIR / 'logs'
    models_dir: Path = _BASE_DIR / 'pretrained_models'
    temp_dir: Path = _BASE_DIR / 'tmp'
    weights_path: Path = _BASE_DIR / 'weights.json'

    # ── Gradio UI ────────────────────────────────
    gradio_host: str = '0.0.0.0'
    gradio_port: int = 7860

    # ── Логирование ──────────────────────────────
    log_level: str = 'INFO'

    # ── Производительность ───────────────────────
    use_gpu: bool = False
    num_threads: int = 4

    # ── Аудио ────────────────────────────────────
    max_file_size_mb: int = 20
    min_audio_duration: float = 3.0
    max_audio_duration: float = 300.0
    segment_count: int = 5
    segment_duration: float = 30.0

    # ── Пороги вердиктов ─────────────────────────
    threshold_very_high: float = 0.95
    threshold_high: float = 0.88
    threshold_probable: float = 0.80
    threshold_similar: float = 0.70

    # ── Весовые коэффициенты моделей ─────────────
    weight_ecapa: float = 1.5
    weight_xvec: float = 1.5
    weight_res: float = 1.2
    weight_formant: float = 1.0
    weight_formant_dynamics: float = 1.5
    weight_fricative: float = 1.2
    weight_nasal: float = 1.3
    weight_vocal_tract: float = 1.8
    weight_jitter_shimmer: float = 1.1
    weight_yamnet: float = 0.8

    # ── Детекция ─────────────────────────────────
    antispoofing_enabled: bool = True
    modification_detection_enabled: bool = True
    confidence_threshold: float = 0.7

    # ── Отчёты ───────────────────────────────────
    include_visualizations: bool = True
    include_technical_details: bool = True

    @property
    def thresholds(self) -> dict[str, float]:
        """Пороги вердиктов как словарь."""
        return {
            'very_high': self.threshold_very_high,
            'high': self.threshold_high,
            'probable': self.threshold_probable,
            'similar': self.threshold_similar,
            'different': self.threshold_similar,
        }

    @property
    def model_weights(self) -> dict[str, float]:
        """Весовые коэффициенты моделей как словарь."""
        return {
            'ecapa': self.weight_ecapa,
            'xvec': self.weight_xvec,
            'res': self.weight_res,
            'formant': self.weight_formant,
            'formant_dynamics': self.weight_formant_dynamics,
            'fricative': self.weight_fricative,
            'nasal': self.weight_nasal,
            'vocal_tract': self.weight_vocal_tract,
            'jitter_shimmer': self.weight_jitter_shimmer,
            'yamnet': self.weight_yamnet,
        }

    def ensure_dirs(self) -> None:
        """Создаёт необходимые директории."""
        for d in (self.logs_dir, self.models_dir, self.temp_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
