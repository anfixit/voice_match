"""Воспроизводимые метрики speaker verification benchmark."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from voice_match.evaluation.protocol import Trial, TrialLabel


@dataclass(frozen=True, slots=True)
class DetectionCostModel:
    """Параметры detection cost function."""

    target_prior: float = 0.01
    miss_cost: float = 1.0
    false_alarm_cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_prior < 1.0:
            raise ValueError(
                'Априорная вероятность target должна быть между 0 и 1.',
            )
        if not math.isfinite(self.miss_cost) or self.miss_cost <= 0.0:
            raise ValueError(
                'Стоимость miss должна быть конечной и положительной.',
            )
        if (
            not math.isfinite(self.false_alarm_cost)
            or self.false_alarm_cost <= 0.0
        ):
            raise ValueError(
                'Стоимость false alarm должна быть конечной и положительной.',
            )

    @property
    def normalizer(self) -> float:
        """Вернуть стоимость лучшего тривиального решения."""
        reject_all = self.miss_cost * self.target_prior
        accept_all = self.false_alarm_cost * (1.0 - self.target_prior)
        return min(reject_all, accept_all)


@dataclass(frozen=True, slots=True)
class ErrorRates:
    """Ошибки при одном пороге принятия решения."""

    threshold: float
    false_reject_rate: float
    false_accept_rate: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Основные результаты evaluation protocol."""

    trial_count: int
    target_trial_count: int
    nontarget_trial_count: int
    equal_error_rate: float
    equal_error_rate_threshold: float
    minimum_raw_dcf: float
    minimum_normalized_dcf: float
    minimum_dcf_threshold: float
    cost_model: DetectionCostModel
    operating_point: ErrorRates | None = None


def calculate_error_rates(
    trials: Sequence[Trial],
    *,
    threshold: float,
) -> ErrorRates:
    """Рассчитать FAR и FRR для правила ``score >= threshold``."""
    if not math.isfinite(threshold):
        raise ValueError('Рабочий порог должен быть конечным числом.')
    target_scores, nontarget_scores = _split_scores(trials)
    false_rejects = sum(score < threshold for score in target_scores)
    false_accepts = sum(score >= threshold for score in nontarget_scores)
    return ErrorRates(
        threshold=threshold,
        false_reject_rate=false_rejects / len(target_scores),
        false_accept_rate=false_accepts / len(nontarget_scores),
    )


def calculate_benchmark(
    trials: Sequence[Trial],
    *,
    cost_model: DetectionCostModel | None = None,
    threshold: float | None = None,
) -> BenchmarkSummary:
    """Рассчитать EER, minDCF и опциональный operating point."""
    model = cost_model or DetectionCostModel()
    target_scores, nontarget_scores = _split_scores(trials)
    points = _build_operating_points(
        target_scores,
        nontarget_scores,
    )
    eer, eer_threshold = _interpolate_eer(points)
    minimum_point, raw_dcf, normalized_dcf = _find_minimum_dcf(
        points,
        model,
    )
    operating_point = None
    if threshold is not None:
        if not math.isfinite(threshold):
            raise ValueError('Рабочий порог должен быть конечным числом.')
        operating_point = calculate_error_rates(
            trials,
            threshold=threshold,
        )

    return BenchmarkSummary(
        trial_count=len(trials),
        target_trial_count=len(target_scores),
        nontarget_trial_count=len(nontarget_scores),
        equal_error_rate=eer,
        equal_error_rate_threshold=eer_threshold,
        minimum_raw_dcf=raw_dcf,
        minimum_normalized_dcf=normalized_dcf,
        minimum_dcf_threshold=minimum_point.threshold,
        cost_model=model,
        operating_point=operating_point,
    )


def calculate_detection_cost(
    rates: ErrorRates,
    cost_model: DetectionCostModel,
) -> tuple[float, float]:
    """Вернуть raw и normalized detection cost."""
    raw_cost = (
        cost_model.miss_cost
        * cost_model.target_prior
        * rates.false_reject_rate
        + cost_model.false_alarm_cost
        * (1.0 - cost_model.target_prior)
        * rates.false_accept_rate
    )
    return raw_cost, raw_cost / cost_model.normalizer


def _split_scores(
    trials: Sequence[Trial],
) -> tuple[list[float], list[float]]:
    target_scores = [
        trial.score for trial in trials if trial.label is TrialLabel.TARGET
    ]
    nontarget_scores = [
        trial.score for trial in trials if trial.label is TrialLabel.NONTARGET
    ]
    if not target_scores or not nontarget_scores:
        raise ValueError(
            'Benchmark должен содержать target и nontarget trials.',
        )
    if not all(
        math.isfinite(score) for score in target_scores + nontarget_scores
    ):
        raise ValueError('Все system score должны быть конечными.')
    return target_scores, nontarget_scores


def _build_operating_points(
    target_scores: Sequence[float],
    nontarget_scores: Sequence[float],
) -> list[ErrorRates]:
    labelled_scores = [
        (score, TrialLabel.TARGET) for score in target_scores
    ] + [(score, TrialLabel.NONTARGET) for score in nontarget_scores]
    labelled_scores.sort(key=lambda item: item[0], reverse=True)

    target_count = len(target_scores)
    nontarget_count = len(nontarget_scores)
    false_reject_count = target_count
    false_accept_count = 0
    points = [
        ErrorRates(
            threshold=math.inf,
            false_reject_rate=1.0,
            false_accept_rate=0.0,
        )
    ]

    for score, group in groupby(
        labelled_scores,
        key=lambda item: item[0],
    ):
        for _, label in group:
            if label is TrialLabel.TARGET:
                false_reject_count -= 1
            else:
                false_accept_count += 1
        points.append(
            ErrorRates(
                threshold=score,
                false_reject_rate=(false_reject_count / target_count),
                false_accept_rate=(false_accept_count / nontarget_count),
            )
        )

    return points


def _interpolate_eer(
    points: Sequence[ErrorRates],
) -> tuple[float, float]:
    previous = points[0]
    previous_difference = (
        previous.false_reject_rate - previous.false_accept_rate
    )

    for point in points:
        difference = point.false_reject_rate - point.false_accept_rate
        if difference == 0.0:
            return point.false_accept_rate, _finite_threshold(
                point.threshold,
            )
        if difference < 0.0 <= previous_difference:
            weight = previous_difference / (previous_difference - difference)
            eer = previous.false_accept_rate + weight * (
                point.false_accept_rate - previous.false_accept_rate
            )
            threshold = _interpolate_threshold(
                previous.threshold,
                point.threshold,
                weight,
            )
            return eer, threshold
        previous = point
        previous_difference = difference

    raise RuntimeError('Не удалось найти пересечение FAR и FRR.')


def _interpolate_threshold(
    upper: float,
    lower: float,
    weight: float,
) -> float:
    if math.isfinite(upper):
        return upper + weight * (lower - upper)
    return lower


def _find_minimum_dcf(
    points: Sequence[ErrorRates],
    cost_model: DetectionCostModel,
) -> tuple[ErrorRates, float, float]:
    candidates = [
        (*calculate_detection_cost(point, cost_model), point)
        for point in points
    ]
    raw_cost, normalized_cost, point = min(
        candidates,
        key=lambda item: item[1],
    )
    return point, raw_cost, normalized_cost


def _finite_threshold(threshold: float) -> float:
    if math.isfinite(threshold):
        return threshold
    raise RuntimeError('EER не должен требовать бесконечного порога.')


__all__ = [
    'BenchmarkSummary',
    'DetectionCostModel',
    'ErrorRates',
    'calculate_benchmark',
    'calculate_detection_cost',
    'calculate_error_rates',
]
