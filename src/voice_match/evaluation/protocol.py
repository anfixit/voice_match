"""Загрузка и валидация speaker verification trials."""

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_REQUIRED_COLUMNS = frozenset(
    {'enrollment_id', 'test_id', 'label', 'score'},
)


class TrialLabel(StrEnum):
    """Класс trial в speaker verification protocol."""

    TARGET = 'target'
    NONTARGET = 'nontarget'


@dataclass(frozen=True, slots=True)
class Trial:
    """Одна размеченная пара записей и сырой system score."""

    enrollment_id: str
    test_id: str
    label: TrialLabel
    score: float
    condition: str | None = None


def load_trials_csv(path: str | Path) -> list[Trial]:
    """Загрузить строгий CSV protocol.

    Обязательные столбцы: ``enrollment_id``, ``test_id``, ``label``
    и ``score``. Решение считается target при ``score >= threshold``.

    Raises:
        FileNotFoundError: Если CSV-файл отсутствует.
        ValueError: Если protocol пуст, повреждён или неоднозначен.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Файл trials не найден: {csv_path}.',
        )

    with csv_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        _validate_header(reader.fieldnames)

        trials: list[Trial] = []
        seen_pairs: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            trial = _parse_trial(row, line_number)
            pair = (trial.enrollment_id, trial.test_id)
            if pair in seen_pairs:
                raise ValueError(
                    f'Trial {pair!r} дублируется в строке {line_number}.',
                )
            seen_pairs.add(pair)
            trials.append(trial)

    _validate_trial_classes(trials)
    return trials


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


def _parse_trial(row: dict[str, str | None], line_number: int) -> Trial:
    enrollment_id = _required_value(
        row,
        'enrollment_id',
        line_number,
    )
    test_id = _required_value(row, 'test_id', line_number)
    label_value = _required_value(row, 'label', line_number).lower()
    score_value = _required_value(row, 'score', line_number)

    try:
        label = TrialLabel(label_value)
    except ValueError as exc:
        raise ValueError(
            f'Строка {line_number}: label должен быть target или '
            f'nontarget, получено {label_value!r}.',
        ) from exc

    try:
        score = float(score_value)
    except ValueError as exc:
        raise ValueError(
            f'Строка {line_number}: score должен быть числом.',
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            f'Строка {line_number}: score должен быть конечным числом.',
        )

    condition_value = row.get('condition')
    condition = condition_value.strip() if condition_value else None
    return Trial(
        enrollment_id=enrollment_id,
        test_id=test_id,
        label=label,
        score=score,
        condition=condition or None,
    )


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


def _validate_trial_classes(trials: list[Trial]) -> None:
    labels = {trial.label for trial in trials}
    if labels != {TrialLabel.TARGET, TrialLabel.NONTARGET}:
        raise ValueError(
            'Benchmark должен содержать target и nontarget trials.',
        )


__all__ = ['Trial', 'TrialLabel', 'load_trials_csv']
