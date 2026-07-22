"""Сериализация результатов speaker verification benchmark."""

import math
from typing import TypedDict

from voice_match.evaluation.metrics import BenchmarkSummary, ErrorRates


class ErrorRatesPayload(TypedDict):
    threshold: float | None
    false_reject_rate: float
    false_accept_rate: float


class BenchmarkPayload(TypedDict):
    trial_counts: dict[str, int]
    equal_error_rate: dict[str, float]
    minimum_dcf: dict[str, float | None]
    cost_model: dict[str, float]
    operating_point: ErrorRatesPayload | None
    disclaimer: str


_DISCLAIMER = (
    'Метрики описывают только переданный evaluation protocol. '
    'Они не являются вероятностью личности и не переносятся на '
    'другой домен без отдельной проверки.'
)


def build_payload(summary: BenchmarkSummary) -> BenchmarkPayload:
    """Преобразовать summary в JSON-совместимую структуру."""
    return BenchmarkPayload(
        trial_counts={
            'all': summary.trial_count,
            'target': summary.target_trial_count,
            'nontarget': summary.nontarget_trial_count,
        },
        equal_error_rate={
            'value': summary.equal_error_rate,
            'threshold': summary.equal_error_rate_threshold,
        },
        minimum_dcf={
            'raw': summary.minimum_raw_dcf,
            'normalized': summary.minimum_normalized_dcf,
            'threshold': _optional_threshold(
                summary.minimum_dcf_threshold,
            ),
        },
        cost_model={
            'target_prior': summary.cost_model.target_prior,
            'miss_cost': summary.cost_model.miss_cost,
            'false_alarm_cost': summary.cost_model.false_alarm_cost,
        },
        operating_point=(
            _rates_payload(summary.operating_point)
            if summary.operating_point is not None
            else None
        ),
        disclaimer=_DISCLAIMER,
    )


def render_markdown(summary: BenchmarkSummary) -> str:
    """Сформировать компактный человекочитаемый отчёт."""
    lines = [
        '# Speaker verification benchmark',
        '',
        '## Protocol',
        '',
        '| Trials | Count |',
        '| --- | ---: |',
        f'| Target | {summary.target_trial_count} |',
        f'| Nontarget | {summary.nontarget_trial_count} |',
        f'| Total | {summary.trial_count} |',
        '',
        '## Metrics',
        '',
        '| Metric | Value | Threshold |',
        '| --- | ---: | ---: |',
        (
            '| EER | '
            f'{summary.equal_error_rate:.4%} | '
            f'{summary.equal_error_rate_threshold:.6f} |'
        ),
        (
            '| minDCF, normalized | '
            f'{summary.minimum_normalized_dcf:.6f} | '
            f'{_format_threshold(summary.minimum_dcf_threshold)} |'
        ),
        (
            '| minDCF, raw | '
            f'{summary.minimum_raw_dcf:.6f} | '
            f'{_format_threshold(summary.minimum_dcf_threshold)} |'
        ),
        '',
        'Cost model: '
        f'P(target)={summary.cost_model.target_prior:g}, '
        f'C(miss)={summary.cost_model.miss_cost:g}, '
        'C(false alarm)='
        f'{summary.cost_model.false_alarm_cost:g}.',
    ]
    if summary.operating_point is not None:
        lines.extend(
            [
                '',
                '## Requested operating point',
                '',
                '| Threshold | FRR | FAR |',
                '| ---: | ---: | ---: |',
                (
                    f'| {summary.operating_point.threshold:.6f} | '
                    f'{summary.operating_point.false_reject_rate:.4%} | '
                    f'{summary.operating_point.false_accept_rate:.4%} |'
                ),
            ]
        )
    lines.extend(['', _DISCLAIMER, ''])
    return '\n'.join(lines)


def _rates_payload(rates: ErrorRates) -> ErrorRatesPayload:
    return ErrorRatesPayload(
        threshold=_optional_threshold(rates.threshold),
        false_reject_rate=rates.false_reject_rate,
        false_accept_rate=rates.false_accept_rate,
    )


def _optional_threshold(threshold: float) -> float | None:
    return threshold if math.isfinite(threshold) else None


def _format_threshold(threshold: float) -> str:
    return f'{threshold:.6f}' if math.isfinite(threshold) else 'reject all'


__all__ = ['BenchmarkPayload', 'build_payload', 'render_markdown']
