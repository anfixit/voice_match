"""
Детектор модификаций аудио (вырезки, склейки, изменения темпа и т.д.).
TODO: Реализовать детекцию цифровых модификаций аудио.
"""


import numpy as np

from voice_match.constants import (
    SAMPLE_RATE,
)
from voice_match.log import setup_logger

log = setup_logger("modification_detector")


def detect_modifications(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """
    Обнаруживает признаки цифровых модификаций в аудио.

    Args:
        audio: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Словарь с вероятностями различных типов модификаций
    """
    log.warning("Детекция модификаций еще не реализована")

    return {
        "is_modified": 0.0,
        "has_cuts": 0.0,
        "has_splicing": 0.0,
        "has_tempo_change": 0.0,
        "confidence": 0.0
    }
