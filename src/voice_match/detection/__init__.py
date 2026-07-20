"""Детекторы подлинности аудио."""

from voice_match.detection.antispoofing import (
    AntiSpoofingDetector,
    AntiSpoofingResult,
    get_antispoofing_detector,
)

__all__ = [
    'AntiSpoofingDetector',
    'AntiSpoofingResult',
    'get_antispoofing_detector',
]
