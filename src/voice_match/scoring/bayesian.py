"""
Байесовский скоринг для оценки вероятности совпадения голосов.
TODO: Реализовать байесовский подход к оценке сходства голосов.
"""


import numpy as np

from voice_match.log import setup_logger

log = setup_logger("bayesian_scoring")


def compute_bayesian_score(
    similarity_scores: dict[str, float],
    priors: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Вычисляет байесовскую оценку вероятности совпадения голосов.

    Args:
        similarity_scores: Словарь с оценками сходства от разных моделей
        priors: Априорные вероятности (опционально)

    Returns:
        Словарь с байесовскими оценками
    """
    # Заглушка - пока возвращаем среднее
    log.warning("Байесовский скоринг еще не реализован, используется среднее значение")

    scores = list(similarity_scores.values())
    mean_score = np.mean(scores) if scores else 0.5

    return {
        "posterior_probability": mean_score,
        "confidence": np.std(scores) if len(scores) > 1 else 0.0
    }
