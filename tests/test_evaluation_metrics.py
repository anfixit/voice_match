"""Тесты метрик speaker verification benchmark."""

import pytest

from voice_match.evaluation.metrics import (
    DetectionCostModel,
    calculate_benchmark,
    calculate_error_rates,
)
from voice_match.evaluation.protocol import Trial, TrialLabel


def _trial(
    label: TrialLabel,
    score: float,
    index: int,
) -> Trial:
    return Trial(
        enrollment_id=f'enroll-{index}',
        test_id=f'test-{index}',
        label=label,
        score=score,
    )


def test_error_rates_accept_score_equal_to_threshold() -> None:
    trials = [
        _trial(TrialLabel.TARGET, 0.5, 1),
        _trial(TrialLabel.TARGET, 0.4, 2),
        _trial(TrialLabel.NONTARGET, 0.5, 3),
        _trial(TrialLabel.NONTARGET, 0.2, 4),
    ]

    rates = calculate_error_rates(trials, threshold=0.5)

    assert rates.false_reject_rate == pytest.approx(0.5)
    assert rates.false_accept_rate == pytest.approx(0.5)


def test_benchmark_reports_zero_eer_for_separable_scores() -> None:
    trials = [
        _trial(TrialLabel.TARGET, 0.9, 1),
        _trial(TrialLabel.TARGET, 0.8, 2),
        _trial(TrialLabel.NONTARGET, 0.2, 3),
        _trial(TrialLabel.NONTARGET, 0.1, 4),
    ]

    summary = calculate_benchmark(trials)

    assert summary.equal_error_rate == pytest.approx(0.0)
    assert summary.minimum_normalized_dcf == pytest.approx(0.0)
    assert summary.minimum_dcf_threshold == pytest.approx(0.8)


def test_benchmark_interpolates_equal_error_rate() -> None:
    trials = [
        _trial(TrialLabel.TARGET, 0.9, 1),
        _trial(TrialLabel.TARGET, 0.1, 2),
        _trial(TrialLabel.NONTARGET, 0.8, 3),
        _trial(TrialLabel.NONTARGET, 0.2, 4),
    ]

    summary = calculate_benchmark(trials)

    assert summary.equal_error_rate == pytest.approx(0.5)


def test_minimum_dcf_uses_requested_cost_model() -> None:
    trials = [
        _trial(TrialLabel.TARGET, 0.9, 1),
        _trial(TrialLabel.TARGET, 0.4, 2),
        _trial(TrialLabel.NONTARGET, 0.8, 3),
        _trial(TrialLabel.NONTARGET, 0.3, 4),
    ]
    cost_model = DetectionCostModel(
        target_prior=0.5,
        miss_cost=2.0,
        false_alarm_cost=1.0,
    )

    summary = calculate_benchmark(trials, cost_model=cost_model)

    assert summary.minimum_dcf_threshold == pytest.approx(0.4)
    assert summary.minimum_raw_dcf == pytest.approx(0.25)
    assert summary.minimum_normalized_dcf == pytest.approx(0.5)


def test_detection_cost_model_rejects_invalid_prior() -> None:
    with pytest.raises(ValueError, match='между 0 и 1'):
        DetectionCostModel(target_prior=1.0)


def test_error_rates_reject_non_finite_threshold() -> None:
    trials = [
        _trial(TrialLabel.TARGET, 0.9, 1),
        _trial(TrialLabel.NONTARGET, 0.1, 2),
    ]

    with pytest.raises(ValueError, match='конечным числом'):
        calculate_error_rates(trials, threshold=float('nan'))


def test_detection_cost_model_rejects_non_finite_cost() -> None:
    with pytest.raises(ValueError, match='конечной'):
        DetectionCostModel(miss_cost=float('inf'))
