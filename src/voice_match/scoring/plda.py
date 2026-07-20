"""Контракт для будущего обученного PLDA backend."""

import numpy as np

from voice_match.exceptions import UncalibratedScoringError


def compute_plda_score(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> dict[str, float]:
    """Запретить подмену PLDA косинусным сходством.

    Реализация появится только вместе с обученными параметрами,
    описанием training cohort и воспроизводимым benchmark.
    """
    del embedding1, embedding2
    raise UncalibratedScoringError(
        'PLDA не обучена. Сырой cosine score нельзя называть '
        'PLDA score или log-likelihood ratio.'
    )


__all__ = ['compute_plda_score']
