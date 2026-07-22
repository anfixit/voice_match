"""Загрузка и валидация manifest голосовых записей."""

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

_REQUIRED_COLUMNS = frozenset(
    {
        'recording_id',
        'speaker_id',
        'session_id',
        'path',
        'split',
    }
)
_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


class DatasetSplit(StrEnum):
    """Назначение записи в speaker-disjoint dataset split."""

    TRAIN = 'train'
    CALIBRATION = 'calibration'
    TEST = 'test'


@dataclass(frozen=True, slots=True)
class Recording:
    """Одна псевдонимизированная голосовая запись."""

    recording_id: str
    speaker_id: str
    session_id: str
    path: str
    split: DatasetSplit
    condition: str | None = None


def load_recordings_csv(path: str | Path) -> list[Recording]:
    """Загрузить manifest и проверить отсутствие split leakage.

    Raises:
        FileNotFoundError: Если manifest отсутствует.
        ValueError: Если manifest повреждён или неоднозначен.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Файл manifest не найден: {csv_path}.',
        )

    with csv_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        _validate_header(reader.fieldnames)

        recordings: list[Recording] = []
        seen_recording_ids: set[str] = set()
        seen_paths: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            recording = _parse_recording(row, line_number)
            _reject_duplicate(
                recording.recording_id,
                seen_recording_ids,
                'recording_id',
                line_number,
            )
            _reject_duplicate(
                recording.path,
                seen_paths,
                'path',
                line_number,
            )
            recordings.append(recording)

    if not recordings:
        raise ValueError('Manifest не содержит записей.')
    _validate_speaker_splits(recordings)
    return recordings


def _validate_header(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise ValueError('CSV не содержит строку заголовка.')

    normalized = {name.strip() for name in fieldnames if name}
    missing = sorted(_REQUIRED_COLUMNS - normalized)
    if missing:
        raise ValueError(
            'CSV не содержит обязательные столбцы: '
            + ', '.join(missing)
            + '.',
        )


def _parse_recording(
    row: dict[str, str | None],
    line_number: int,
) -> Recording:
    recording_id = _identifier_value(
        row,
        'recording_id',
        line_number,
    )
    speaker_id = _identifier_value(row, 'speaker_id', line_number)
    session_id = _identifier_value(row, 'session_id', line_number)
    relative_path = _relative_path_value(row, line_number)
    split_value = _required_value(row, 'split', line_number).lower()

    try:
        split = DatasetSplit(split_value)
    except ValueError as exc:
        raise ValueError(
            f'Строка {line_number}: split должен быть train, '
            f'calibration или test, получено {split_value!r}.',
        ) from exc

    condition = _optional_identifier_value(
        row,
        'condition',
        line_number,
    )
    return Recording(
        recording_id=recording_id,
        speaker_id=speaker_id,
        session_id=session_id,
        path=relative_path,
        split=split,
        condition=condition or None,
    )


def _identifier_value(
    row: dict[str, str | None],
    field: str,
    line_number: int,
) -> str:
    value = _required_value(row, field, line_number)
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f'Строка {line_number}: поле {field!r} должно быть '
            'переносимым ASCII-идентификатором без пробелов.',
        )
    return value


def _optional_identifier_value(
    row: dict[str, str | None],
    field: str,
    line_number: int,
) -> str | None:
    raw_value = row.get(field)
    value = raw_value.strip() if raw_value else ''
    if not value:
        return None
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f'Строка {line_number}: поле {field!r} должно быть '
            'переносимым ASCII-идентификатором без пробелов.',
        )
    return value


def _relative_path_value(
    row: dict[str, str | None],
    line_number: int,
) -> str:
    value = _required_value(row, 'path', line_number)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or '..' in path.parts
        or '\\' in value
        or value.startswith('~')
        or path == PurePosixPath('.')
    ):
        raise ValueError(
            f'Строка {line_number}: path должен быть безопасным '
            'относительным POSIX-путём.',
        )
    return path.as_posix()


def _required_value(
    row: dict[str, str | None],
    field: str,
    line_number: int,
) -> str:
    value = row.get(field)
    normalized = value.strip() if value else ''
    if not normalized:
        raise ValueError(
            f'Строка {line_number}: поле {field!r} не заполнено.',
        )
    return normalized


def _reject_duplicate(
    value: str,
    seen: set[str],
    field: str,
    line_number: int,
) -> None:
    if value in seen:
        raise ValueError(
            f'Строка {line_number}: {field} {value!r} дублируется.',
        )
    seen.add(value)


def _validate_speaker_splits(recordings: Sequence[Recording]) -> None:
    speaker_splits: dict[str, DatasetSplit] = {}
    for recording in recordings:
        previous = speaker_splits.setdefault(
            recording.speaker_id,
            recording.split,
        )
        if previous is not recording.split:
            raise ValueError(
                f'Диктор {recording.speaker_id!r} встречается в split '
                f'{previous.value!r} и {recording.split.value!r}.',
            )


__all__ = ['DatasetSplit', 'Recording', 'load_recordings_csv']
