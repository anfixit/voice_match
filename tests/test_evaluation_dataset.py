"""Тесты manifest голосовых записей."""

from pathlib import Path

import pytest

from voice_match.evaluation.dataset import (
    DatasetSplit,
    load_recordings_csv,
)


def test_load_recordings_csv_reads_valid_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'recordings.csv'
    path.write_text(
        'recording_id,speaker_id,session_id,path,split,condition\n'
        'spk-a-s1-u1,spk-a,s1,a/s1/u1.wav,calibration,clean\n'
        'spk-a-s2-u1,spk-a,s2,a/s2/u1.wav,calibration,phone\n',
        encoding='utf-8',
    )

    recordings = load_recordings_csv(path)

    assert len(recordings) == 2
    assert recordings[0].split is DatasetSplit.CALIBRATION
    assert recordings[1].condition == 'phone'


def test_load_recordings_csv_rejects_speaker_split_leakage(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'recordings.csv'
    path.write_text(
        'recording_id,speaker_id,session_id,path,split\n'
        'spk-a-s1-u1,spk-a,s1,a/1.wav,calibration\n'
        'spk-a-s2-u1,spk-a,s2,a/2.wav,test\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='встречается в split'):
        load_recordings_csv(path)


def test_load_recordings_csv_rejects_duplicate_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'recordings.csv'
    path.write_text(
        'recording_id,speaker_id,session_id,path,split\n'
        'spk-a-s1-u1,spk-a,s1,a/1.wav,calibration\n'
        'spk-b-s1-u1,spk-b,s1,a/1.wav,calibration\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='path'):
        load_recordings_csv(path)


@pytest.mark.parametrize(
    'unsafe_path',
    [
        '/absolute/file.wav',
        '../outside.wav',
        'speaker/../../outside.wav',
        r'speaker\file.wav',
        '~/voice.wav',
    ],
)
def test_load_recordings_csv_rejects_unsafe_path(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    path = tmp_path / 'recordings.csv'
    path.write_text(
        'recording_id,speaker_id,session_id,path,split\n'
        f'spk-a-s1-u1,spk-a,s1,{unsafe_path},calibration\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='относительным POSIX-путём'):
        load_recordings_csv(path)


def test_load_recordings_csv_rejects_nonportable_identifier(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'recordings.csv'
    path.write_text(
        'recording_id,speaker_id,session_id,path,split\n'
        'spk-a-s1-u1,Иван Иванов,s1,a/1.wav,calibration\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='ASCII-идентификатором'):
        load_recordings_csv(path)
