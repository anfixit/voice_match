"""Сервисный слой voice_match."""

from voice_match.services.audio import load_audio_signal
from voice_match.services.comparison import compare_voices_dual

__all__ = ['compare_voices_dual', 'load_audio_signal']
