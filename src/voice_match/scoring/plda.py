"""
PLDA (Probabilistic Linear Discriminant Analysis) скоринг.
TODO: Реализовать PLDA для улучшенного сравнения эмбеддингов.
"""

import numpy as np
from typing import Dict
from voice_match.log import setup_logger

log = setup_logger("plda_scoring")


def compute_plda_score(embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет PLDA оценку сходства двух эмбеддингов.

    Args:
        embedding1: Первый эмбеддинг
        embedding2: Второй эмбеддинг

    Returns:
        Словарь с PLDA оценками
    """
    # Заглушка - пока используем косинусное сходство
    log.warning("PLDA скоринг еще не реализован, используется косинусное сходство")

    # Косинусное сходство как заглушка
    cosine_sim = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

    return {
        "plda_score": float(cosine_sim),
        "log_likelihood_ratio": float(cosine_sim * 10)  # Масштабированное значение
    }
