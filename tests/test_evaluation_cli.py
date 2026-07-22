"""Тесты CLI benchmark evaluator."""

import json
from pathlib import Path

from voice_match.evaluation.__main__ import main


def test_cli_writes_machine_readable_json(
    tmp_path: Path,
) -> None:
    trials_path = tmp_path / 'trials.csv'
    output_path = tmp_path / 'report.json'
    trials_path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,target,0.9\n'
        'speaker-a-1,speaker-b-1,nontarget,0.1\n',
        encoding='utf-8',
    )

    exit_code = main(
        [
            str(trials_path),
            '--format',
            'json',
            '--output',
            str(output_path),
            '--threshold',
            '0.5',
        ]
    )

    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert exit_code == 0
    assert payload['trial_counts']['target'] == 1
    assert payload['equal_error_rate']['value'] == 0.0
    assert payload['operating_point']['threshold'] == 0.5


def test_cli_serializes_reject_all_threshold_as_null(
    tmp_path: Path,
) -> None:
    trials_path = tmp_path / 'trials.csv'
    output_path = tmp_path / 'report.json'
    trials_path.write_text(
        'enrollment_id,test_id,label,score\n'
        'speaker-a-1,speaker-a-2,target,0.1\n'
        'speaker-a-1,speaker-b-1,nontarget,0.9\n',
        encoding='utf-8',
    )

    exit_code = main(
        [
            str(trials_path),
            '--format',
            'json',
            '--output',
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert exit_code == 0
    assert payload['minimum_dcf']['threshold'] is None
