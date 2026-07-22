"""Загрузка и валидация scoreless speaker trial plan."""

import csv
from collections.abc import Sequence
from pathlib import Path

from voice_match.evaluation.protocol import TrialLabel
from voice_match.evaluation.trial_generation import TrialDefinition

_REQUIRED_COLUMNS = frozenset(
    {'enrollment_id', 'test_id', 'label'},
)


def load_trial_definitions_csv(
    path: str | Path,
) -> list[TrialDefinition]:
    """Загрузить строгий trial plan без system score.

    Raises:
        FileNotFoundError: Если trial plan отсутствует.
        ValueError: Если CSV пуст, повреждён или неоднозначен.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Файл trial plan не найден: {csv_path}.',
        )

    with csv_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        _validate_header(reader.fieldnames)

        trials: list[TrialDefinition] = []
        seen_pairs: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            trial = _parse_trial_definition(row, line_number)
            pair = tuple(
                sorted((trial.enrollment_id, trial.test_id)),
            )
            if pair in seen_pairs:
                raise ValueError(
                    f'Пара {pair!r} дублируется в строке '
                    f'{line_number}.',
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


def _parse_trial_definition(
    row: dict[str, str | None],
    line_number: int,
) -> TrialDefinition:
    enrollment_id = _required_value(
        row,
        'enrollment_id',
        line_number,
    )
    test_id = _required_value(row, 'test_id', line_number)
    if enrollment_id == test_id:
        raise ValueError(
            f'Строка {line_number}: запись нельзя сравнивать с собой.',
        )

    label_value = _required_value(row, 'label', line_number).lower()
    try:
        label = TrialLabel(label_value)
    except ValueError as exc:
        raise ValueError(
            f'Строка {line_number}: label должен быть target или '
            f'nontarget, получено {label_value!r}.',
        ) from exc

    raw_condition = row.get('condition')
    condition = raw_condition.strip() if raw_condition else None
    return TrialDefinition(
        enrollment_id=enrollment_id,
        test_id=test_id,
        label=label,
        condition=condition or None,
    )


def _required_value(
    row: dict[str, str | None],
    field: str,
    line_number: int,
) -> str:
    raw_value = row.get(field)
    value = raw_value.strip() if raw_value else ''
    if not value:
        raise ValueError(
            f'Строка {line_number}: поле {field!r} не заполнено.',
        )
    return value


def _validate_trial_classes(
    trials: Sequence[TrialDefinition],
) -> None:
    labels = {trial.label for trial in trials}
    if labels != {TrialLabel.TARGET, TrialLabel.NONTARGET}:
        raise ValueError(
            'Trial plan должен содержать target и nontarget пары.',
        )


__all__ = ['load_trial_definitions_csv']
