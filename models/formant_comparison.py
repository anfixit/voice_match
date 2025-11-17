"""
Сравнение формантных профилей голосов.
TODO: Реализовать детальное сравнение формант с учетом временной динамики.
"""

import numpy as np
from typing import Dict, List
from app.log import setup_logger

log = setup_logger("formant_comparison")


def compare_formants(
    formants1: Dict[str, np.ndarray],
    formants2: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Сравнивает формантные характеристики двух голосов.

    Args:
        formants1: Форманты первого голоса
        formants2: Форманты второго голоса

    Returns:
        Словарь с метриками сходства формант
    """
    log.warning("Детальное сравнение формант еще не реализовано")

    # Заглушка - базовое сравнение
    similarity = {
        "overall_similarity": 0.5,
        "F1_similarity": 0.5,
        "F2_similarity": 0.5,
        "F3_similarity": 0.5
    }

    return similarity
