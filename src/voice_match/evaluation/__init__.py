"""Оценка и подготовка speaker verification protocol."""

from voice_match.evaluation.dataset import (
    DatasetSplit,
    Recording,
    load_recordings_csv,
)
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
from voice_match.evaluation.scoring import (
    EcapaRecordingEncoder,
    RecordingEncoder,
    score_trial_definitions,
    write_scored_trials_csv,
)
from voice_match.evaluation.trial_generation import (
    TrialDefinition,
    TrialGenerationConfig,
    count_trial_labels,
    generate_trial_definitions,
    write_trial_definitions_csv,
)
from voice_match.evaluation.trial_plan import (
    load_trial_definitions_csv,
)

__all__ = [
    'BenchmarkSummary',
    'DatasetSplit',
    'DetectionCostModel',
    'EcapaRecordingEncoder',
    'ErrorRates',
    'Recording',
    'RecordingEncoder',
    'Trial',
    'TrialDefinition',
    'TrialGenerationConfig',
    'TrialLabel',
    'build_payload',
    'calculate_benchmark',
    'calculate_detection_cost',
    'calculate_error_rates',
    'count_trial_labels',
    'generate_trial_definitions',
    'load_recordings_csv',
    'load_trial_definitions_csv',
    'load_trials_csv',
    'render_markdown',
    'score_trial_definitions',
    'write_scored_trials_csv',
    'write_trial_definitions_csv',
]
