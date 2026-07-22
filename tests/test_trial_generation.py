"""Тесты генерации speaker verification trial plan."""

import csv
from pathlib import Path

import pytest

from voice_match.evaluation.dataset import DatasetSplit, Recording
from voice_match.evaluation.generate_trials import main
from voice_match.evaluation.protocol import TrialLabel
from voice_match.evaluation.trial_generation import (
    TrialGenerationConfig,
    count_trial_labels,
    generate_trial_definitions,
    write_trial_definitions_csv,
)


def _recording(
    recording_id: str,
    speaker_id: str,
    session_id: str,
    *,
    split: DatasetSplit = DatasetSplit.CALIBRATION,
    condition: str | None = None,
) -> Recording:
    return Recording(
        recording_id=recording_id,
        speaker_id=speaker_id,
        session_id=session_id,
        path=f'audio/{recording_id}.wav',
        split=split,
        condition=condition,
    )


def _dataset() -> list[Recording]:
    return [
        _recording('a-s1-u1', 'a', 's1', condition='clean'),
        _recording('a-s1-u2', 'a', 's1', condition='clean'),
        _recording('a-s2-u1', 'a', 's2', condition='phone'),
        _recording('b-s1-u1', 'b', 's1', condition='clean'),
        _recording('b-s2-u1', 'b', 's2', condition='clean'),
    ]


def test_generate_trials_uses_cross_session_targets_only() -> None:
    trials = generate_trial_definitions(
        _dataset(),
        config=TrialGenerationConfig(nontarget_ratio=1),
    )

    target_pairs = {
        (trial.enrollment_id, trial.test_id)
        for trial in trials
        if trial.label is TrialLabel.TARGET
    }
    assert ('a-s1-u1', 'a-s1-u2') not in target_pairs
    assert ('a-s1-u1', 'a-s2-u1') in target_pairs
    assert ('a-s1-u2', 'a-s2-u1') in target_pairs
    assert ('b-s1-u1', 'b-s2-u1') in target_pairs


def test_generate_trials_respects_nontarget_ratio() -> None:
    trials = generate_trial_definitions(
        _dataset(),
        config=TrialGenerationConfig(nontarget_ratio=2),
    )

    counts = count_trial_labels(trials)

    assert counts[TrialLabel.TARGET] == 3
    assert counts[TrialLabel.NONTARGET] == 6


def test_generate_trials_is_deterministic_for_seed() -> None:
    config = TrialGenerationConfig(nontarget_ratio=1, seed=42)

    first = generate_trial_definitions(_dataset(), config=config)
    second = generate_trial_definitions(
        list(reversed(_dataset())),
        config=config,
    )

    assert first == second


def test_generate_trials_combines_conditions() -> None:
    trials = generate_trial_definitions(
        _dataset(),
        config=TrialGenerationConfig(nontarget_ratio=1),
    )

    mixed = next(
        trial
        for trial in trials
        if trial.enrollment_id == 'a-s1-u1'
        and trial.test_id == 'a-s2-u1'
    )
    assert mixed.condition == 'clean+phone'


def test_generate_trials_rejects_single_session_targets() -> None:
    recordings = [
        _recording('a-1', 'a', 's1'),
        _recording('a-2', 'a', 's1'),
        _recording('b-1', 'b', 's1'),
    ]

    with pytest.raises(ValueError, match='двух разных сессий'):
        generate_trial_definitions(recordings)


def test_write_trial_definitions_csv_writes_scoreless_plan(
    tmp_path: Path,
) -> None:
    trials = generate_trial_definitions(
        _dataset(),
        config=TrialGenerationConfig(nontarget_ratio=1),
    )
    path = tmp_path / 'trials.csv'

    write_trial_definitions_csv(trials, path)

    with path.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    assert rows
    assert 'score' not in rows[0]
    assert set(rows[0]) == {
        'enrollment_id',
        'test_id',
        'label',
        'condition',
    }


def test_generate_trials_cli_writes_output(tmp_path: Path) -> None:
    manifest = tmp_path / 'recordings.csv'
    output = tmp_path / 'trials.csv'
    manifest.write_text(
        'recording_id,speaker_id,session_id,path,split\n'
        'a-s1,a,s1,a/1.wav,calibration\n'
        'a-s2,a,s2,a/2.wav,calibration\n'
        'b-s1,b,s1,b/1.wav,calibration\n'
        'b-s2,b,s2,b/2.wav,calibration\n',
        encoding='utf-8',
    )

    exit_code = main(
        [
            str(manifest),
            '--output',
            str(output),
            '--nontarget-ratio',
            '1',
        ]
    )

    assert exit_code == 0
    assert output.is_file()
