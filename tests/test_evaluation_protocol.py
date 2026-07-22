"""Тесты загрузки benchmark trials."""

from pathlib import Path

import pytest

from voice_match.evaluation.protocol import TrialLabel, load_trials_csv


def test_load_trials_csv_reads_valid_protocol(tmp_path: Path) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label,score,condition\n'
        'speaker-a-1,speaker-a-2,target,0.82,clean\n'
        'speaker-a-1,speaker-b-1,nontarget,0.21,codec\n',
        encoding='utf-8',
    )

    trials = load_trials_csv(path)

    assert len(trials) == 2
    assert trials[0].label is TrialLabel.TARGET
    assert trials[0].condition == 'clean'
    assert trials[1].score == pytest.approx(0.21)


def test_load_trials_csv_rejects_duplicate_trial(tmp_path: Path) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,target,0.82\n'
        'speaker-a-1,speaker-a-2,target,0.83\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='дублируется'):
        load_trials_csv(path)


def test_load_trials_csv_requires_both_trial_classes(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,target,0.82\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='target и nontarget'):
        load_trials_csv(path)


def test_load_trials_csv_rejects_non_finite_score(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,target,nan\n'
        'speaker-a-1,speaker-b-1,nontarget,0.21\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='конечным числом'):
        load_trials_csv(path)


def test_load_trials_csv_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label\nspeaker-a-1,speaker-a-2,target\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='score'):
        load_trials_csv(path)


def test_load_trials_csv_rejects_invalid_label(tmp_path: Path) -> None:
    path = tmp_path / 'trials.csv'
    path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,unknown,0.82\n'
        'speaker-a-1,speaker-b-1,nontarget,0.21\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='target или nontarget'):
        load_trials_csv(path)
