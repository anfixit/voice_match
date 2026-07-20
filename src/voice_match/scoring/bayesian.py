"""Контракт для будущей вероятностной калибровки."""

from collections.abc import Mapping

from voice_match.exceptions import UncalibratedScoringError


def compute_bayesian_score(
    similarity_scores: Mapping[str, float],
    priors: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Запретить выдачу среднего score за posterior probability."""
    del similarity_scores, priors
    raise UncalibratedScoringError(
        'Вероятностная калибровка не обучена. Для posterior '
        'probability нужны размеченные target/non-target trials.'
    )


__all__ = ['compute_bayesian_score']
