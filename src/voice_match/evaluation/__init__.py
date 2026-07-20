"""Оценка качества speaker verification system score."""

from voice_match.evaluation.metrics import (
    BenchmarkSummary,
    DetectionCostModel,
    ErrorRates,
    calculate_benchmark,
    calculate_detection_cost,
    calculate_error_rates,
)
from voice_match.evaluation.protocol import (
    Trial,
    TrialLabel,
    load_trials_csv,
)
from voice_match.evaluation.report import build_payload, render_markdown

__all__ = [
    'BenchmarkSummary',
    'DetectionCostModel',
    'ErrorRates',
    'Trial',
    'TrialLabel',
    'build_payload',
    'calculate_benchmark',
    'calculate_detection_cost',
    'calculate_error_rates',
    'load_trials_csv',
    'render_markdown',
]
