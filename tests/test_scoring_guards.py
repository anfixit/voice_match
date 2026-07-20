"""Тесты запрета некалиброванных вероятностных оценок."""

import numpy as np
import pytest

from voice_match.exceptions import UncalibratedScoringError
from voice_match.scoring.bayesian import compute_bayesian_score
from voice_match.scoring.plda import compute_plda_score


def test_plda_fails_closed_without_trained_backend() -> None:
    with pytest.raises(UncalibratedScoringError, match='PLDA не обучена'):
        compute_plda_score(np.ones(2), np.ones(2))


def test_bayesian_score_fails_closed_without_calibration() -> None:
    with pytest.raises(
        UncalibratedScoringError,
        match='калибровка не обучена',
    ):
        compute_bayesian_score({'ecapa': 0.5})
