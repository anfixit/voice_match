"""Тесты загрузки scoreless speaker trial plan."""

from pathlib import Path

import pytest

from voice_match.evaluation.protocol import TrialLabel
from voice_match.evaluation.trial_plan import (
    load_trial_definitions_csv,
)


def test_load_trial_definitions_reads_valid_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trial-plan.csv'
    path.write_text(
        'enrollment_id,test_id,label,condition\n'
        'a-1,a-2,target,clean\n'
        'a-1,b-1,nontarget,cross-device\n',
        encoding='utf-8',
    )

    trials = load_trial_definitions_csv(path)

    assert len(trials) == 2
    assert trials[0].label is TrialLabel.TARGET
    assert trials[0].condition == 'clean'
    assert trials[1].label is TrialLabel.NONTARGET


def test_load_trial_definitions_rejects_reversed_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trial-plan.csv'
    path.write_text(
        'enrollment_id,test_id,label\n'
        'a-1,a-2,target\n'
        'a-2,a-1,target\n'
        'a-1,b-1,nontarget\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='дублируется'):
        load_trial_definitions_csv(path)


def test_load_trial_definitions_rejects_self_comparison(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trial-plan.csv'
    path.write_text(
        'enrollment_id,test_id,label\n'
        'a-1,a-1,target\n'
        'a-1,b-1,nontarget\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='с собой'):
        load_trial_definitions_csv(path)


def test_load_trial_definitions_requires_both_classes(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trial-plan.csv'
    path.write_text(
        'enrollment_id,test_id,label\n'
        'a-1,a-2,target\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='target и nontarget'):
        load_trial_definitions_csv(path)


def test_load_trial_definitions_rejects_missing_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trial-plan.csv'
    path.write_text(
        'enrollment_id,test_id\n'
        'a-1,a-2\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='label'):
        load_trial_definitions_csv(path)
