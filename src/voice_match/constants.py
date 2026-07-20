"""Неизменяемые технические константы voice_match."""

SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
SUPPORTED_EXTENSIONS = frozenset(
    {'.flac', '.m4a', '.mp3', '.ogg', '.wav'},
)
VAD_FRAME_MS = 30
