"""
Визуализация формант для анализа и отчетов.
TODO: Реализовать графики формантных треков, F1-F2 диаграммы и др.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from app.log import setup_logger

log = setup_logger("formant_visualization")


def visualize_formant_tracks(
    formants: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Создает визуализацию формантных треков.

    Args:
        formants: Словарь с формантными треками
        save_path: Путь для сохранения изображения (опционально)

    Returns:
        Путь к сохраненному изображению или None
    """
    log.warning("Визуализация формант еще не реализована")
    return None
