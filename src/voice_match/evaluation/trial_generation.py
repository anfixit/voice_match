"""Детерминированная генерация speaker verification trial plan."""

import csv
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from voice_match.evaluation.dataset import DatasetSplit, Recording
from voice_match.evaluation.protocol import TrialLabel


@dataclass(frozen=True, slots=True)
class TrialDefinition:
    """Пара записей до расчёта system score."""

    enrollment_id: str
    test_id: str
    label: TrialLabel
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class TrialGenerationConfig:
    """Ограничения генератора trial plan."""

    split: DatasetSplit = DatasetSplit.CALIBRATION
    nontarget_ratio: int = 5
    max_target_trials_per_speaker: int | None = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.nontarget_ratio < 1:
            raise ValueError('nontarget_ratio должен быть не меньше 1.')
        if (
            self.max_target_trials_per_speaker is not None
            and self.max_target_trials_per_speaker < 1
        ):
            raise ValueError(
                'max_target_trials_per_speaker должен быть положительным.',
            )


def generate_trial_definitions(
    recordings: Sequence[Recording],
    *,
    config: TrialGenerationConfig | None = None,
) -> list[TrialDefinition]:
    """Сформировать target и nontarget trials для одного split.

    Target trials строятся только между разными сессиями одного
    диктора. Nontarget trials выбираются детерминированно из пар
    разных дикторов.

    Raises:
        ValueError: Если данных недостаточно для обоих классов.
    """
    generation_config = config or TrialGenerationConfig()
    selected = sorted(
        (
            recording
            for recording in recordings
            if recording.split is generation_config.split
        ),
        key=lambda recording: recording.recording_id,
    )
    if not selected:
        raise ValueError(
            f'В manifest нет записей split '
            f'{generation_config.split.value!r}.',
        )

    randomizer = random.Random(generation_config.seed)
    targets = _build_target_trials(
        selected,
        generation_config.max_target_trials_per_speaker,
        randomizer,
    )
    if not targets:
        raise ValueError(
            'Для target trials нужны записи одного диктора '
            'минимум из двух разных сессий.',
        )

    requested_nontarget_count = len(targets) * (
        generation_config.nontarget_ratio
    )
    nontargets = _reservoir_sample(
        _iter_nontarget_trials(selected),
        requested_nontarget_count,
        randomizer,
    )
    if not nontargets:
        raise ValueError(
            'Для nontarget trials нужны минимум два разных диктора.',
        )

    return sorted(
        targets + nontargets,
        key=lambda trial: (
            trial.label.value,
            trial.enrollment_id,
            trial.test_id,
        ),
    )


def count_trial_labels(
    trials: Sequence[TrialDefinition],
) -> Counter[TrialLabel]:
    """Подсчитать классы в trial plan."""
    return Counter(trial.label for trial in trials)


def write_trial_definitions_csv(
    trials: Sequence[TrialDefinition],
    path: str | Path,
) -> None:
    """Атомарно записать trial plan без system score."""
    if not trials:
        raise ValueError('Нельзя записать пустой trial plan.')

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=(
                    'enrollment_id',
                    'test_id',
                    'label',
                    'condition',
                ),
            )
            writer.writeheader()
            for trial in trials:
                writer.writerow(
                    {
                        'enrollment_id': trial.enrollment_id,
                        'test_id': trial.test_id,
                        'label': trial.label.value,
                        'condition': trial.condition or '',
                    }
                )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_target_trials(
    recordings: Sequence[Recording],
    limit_per_speaker: int | None,
    randomizer: random.Random,
) -> list[TrialDefinition]:
    by_speaker: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_speaker[recording.speaker_id].append(recording)

    trials: list[TrialDefinition] = []
    for speaker_id in sorted(by_speaker):
        candidates = [
            _make_trial(first, second, TrialLabel.TARGET)
            for first, second in combinations(
                sorted(
                    by_speaker[speaker_id],
                    key=lambda recording: recording.recording_id,
                ),
                2,
            )
            if first.session_id != second.session_id
        ]
        if limit_per_speaker is not None:
            candidates = _sample_trials(
                candidates,
                min(limit_per_speaker, len(candidates)),
                randomizer,
            )
        trials.extend(candidates)
    return trials


def _iter_nontarget_trials(
    recordings: Sequence[Recording],
) -> Iterable[TrialDefinition]:
    for first, second in combinations(recordings, 2):
        if first.speaker_id != second.speaker_id:
            yield _make_trial(first, second, TrialLabel.NONTARGET)


def _make_trial(
    first: Recording,
    second: Recording,
    label: TrialLabel,
) -> TrialDefinition:
    enrollment, test = sorted(
        (first, second),
        key=lambda recording: recording.recording_id,
    )
    return TrialDefinition(
        enrollment_id=enrollment.recording_id,
        test_id=test.recording_id,
        label=label,
        condition=_pair_condition(enrollment, test),
    )


def _pair_condition(
    first: Recording,
    second: Recording,
) -> str | None:
    conditions = sorted(
        {
            condition
            for condition in (first.condition, second.condition)
            if condition
        }
    )
    if not conditions:
        return None
    return '+'.join(conditions)


def _reservoir_sample(
    trials: Iterable[TrialDefinition],
    count: int,
    randomizer: random.Random,
) -> list[TrialDefinition]:
    reservoir: list[TrialDefinition] = []
    for index, trial in enumerate(trials):
        if index < count:
            reservoir.append(trial)
            continue
        replacement = randomizer.randrange(index + 1)
        if replacement < count:
            reservoir[replacement] = trial
    return reservoir


def _sample_trials(
    trials: Sequence[TrialDefinition],
    count: int,
    randomizer: random.Random,
) -> list[TrialDefinition]:
    if count >= len(trials):
        return list(trials)
    indices = sorted(randomizer.sample(range(len(trials)), count))
    return [trials[index] for index in indices]


__all__ = [
    'TrialDefinition',
    'TrialGenerationConfig',
    'count_trial_labels',
    'generate_trial_definitions',
    'write_trial_definitions_csv',
]
