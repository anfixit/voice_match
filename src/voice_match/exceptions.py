"""Исключения приложения voice_match."""


class VoiceMatchError(Exception):
    """Базовое исключение приложения."""


class AudioValidationError(VoiceMatchError):
    """Аудиофайл не подходит для анализа."""


class ModelUnavailableError(VoiceMatchError):
    """Необходимая модель недоступна или не загружена."""


class UncalibratedScoringError(VoiceMatchError):
    """Запрошена вероятностная оценка без калибровки."""
