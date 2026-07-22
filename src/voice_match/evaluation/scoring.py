"""Пакетный расчёт ECAPA score для speaker trial plan."""

import csv
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from voice_match.config import settings
from voice_match.constants import SAMPLE_RATE
from voice_match.evaluation.dataset import Recording
from voice_match.evaluation.protocol import Trial, TrialLabel
from voice_match.evaluation.trial_generation import TrialDefinition
from voice_match.exceptions import AudioValidationError
from voice_match.models.ecapa import get_ecapa
from voice_match.scoring.similarity import (
    cosine_similarity,
    normalize_embedding,
)
from voice_match.services.audio import load_audio_signal
from voice_match.services.quality import (
    analyze_audio_quality,
    extract_speech_segments,
)


class SegmentEncoder(Protocol):
    """Минимальный интерфейс speaker embedding модели."""

    def encode_segments(
        self,
        segments: list[np.ndarray],
    ) -> np.ndarray: ...


class RecordingEncoder(Protocol):
    """Интерфейс кодирования одной manifest-записи."""

    def encode(self, recording: Recording) -> np.ndarray: ...


class EcapaRecordingEncoder:
    """Преобразовать manifest-запись в один ECAPA centroid embedding."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        segment_encoder: SegmentEncoder | None = None,
    ) -> None:
        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(
                f'Каталог датасета не найден: {root}.',
            )
        self._dataset_root = root
        self._segment_encoder = segment_encoder or get_ecapa()

    def encode(self, recording: Recording) -> np.ndarray:
        """Извлечь нормализованный centroid embedding записи."""
        path = self._resolve_recording_path(recording)
        audio = load_audio_signal(path)
        quality = analyze_audio_quality(
            audio,
            SAMPLE_RATE,
            min_duration_seconds=settings.min_audio_duration,
            min_speech_seconds=settings.min_speech_seconds,
            min_speech_ratio=settings.min_speech_ratio,
            max_clipping_ratio=settings.max_clipping_ratio,
        )
        if quality.issues:
            details = ' '.join(quality.issues)
            raise AudioValidationError(
                f'Запись {recording.recording_id!r}: {details}',
            )

        segments = extract_speech_segments(
            audio,
            SAMPLE_RATE,
            segment_seconds=settings.segment_duration,
            max_segments=settings.segment_count,
            min_segment_seconds=settings.min_segment_duration,
        )
        if not segments:
            raise AudioValidationError(
                f'Запись {recording.recording_id!r}: '
                'не удалось выделить речевые сегменты.',
            )

        embeddings = np.asarray(
            self._segment_encoder.encode_segments(segments),
            dtype=np.float64,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                'Speaker encoder должен вернуть непустую матрицу.',
            )
        normalized = np.vstack(
            [normalize_embedding(embedding) for embedding in embeddings],
        )
        return normalize_embedding(normalized.mean(axis=0))

    def _resolve_recording_path(self, recording: Recording) -> Path:
        relative = PurePosixPath(recording.path)
        candidate = self._dataset_root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise AudioValidationError(
                f'Запись {recording.recording_id!r}: файл не найден.',
            ) from exc

        if not resolved.is_relative_to(self._dataset_root):
            raise AudioValidationError(
                f'Запись {recording.recording_id!r}: '
                'путь выходит за пределы датасета.',
            )
        if not resolved.is_file():
            raise AudioValidationError(
                f'Запись {recording.recording_id!r}: путь не является '
                'обычным файлом.',
            )
        return resolved


def score_trial_definitions(
    recordings: Sequence[Recording],
    definitions: Sequence[TrialDefinition],
    encoder: RecordingEncoder,
) -> list[Trial]:
    """Рассчитать cosine score, кодируя каждую запись один раз.

    Raises:
        ValueError: Если trial plan противоречит manifest.
        AudioValidationError: Если запись непригодна для анализа.
    """
    if not definitions:
        raise ValueError('Trial plan не содержит пар.')

    recording_by_id = _index_recordings(recordings)
    seen_pairs: set[tuple[str, str]] = set()
    for definition in definitions:
        _validate_definition(
            definition,
            recording_by_id,
            seen_pairs,
        )

    referenced_ids = sorted(
        {
            recording_id
            for definition in definitions
            for recording_id in (
                definition.enrollment_id,
                definition.test_id,
            )
        }
    )
    embeddings = {
        recording_id: normalize_embedding(
            encoder.encode(recording_by_id[recording_id]),
        )
        for recording_id in referenced_ids
    }

    return [
        Trial(
            enrollment_id=definition.enrollment_id,
            test_id=definition.test_id,
            label=definition.label,
            score=cosine_similarity(
                embeddings[definition.enrollment_id],
                embeddings[definition.test_id],
            ),
            condition=definition.condition,
        )
        for definition in definitions
    ]


def write_scored_trials_csv(
    trials: Sequence[Trial],
    path: str | Path,
) -> None:
    """Атомарно записать scored protocol для benchmark evaluator."""
    if not trials:
        raise ValueError('Нельзя записать пустой scored protocol.')
    if not all(math.isfinite(trial.score) for trial in trials):
        raise ValueError('Все system score должны быть конечными.')

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
                    'score',
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
                        'score': format(trial.score, '.17g'),
                        'condition': trial.condition or '',
                    }
                )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _index_recordings(
    recordings: Sequence[Recording],
) -> dict[str, Recording]:
    recording_by_id: dict[str, Recording] = {}
    for recording in recordings:
        if recording.recording_id in recording_by_id:
            raise ValueError(
                f'recording_id {recording.recording_id!r} дублируется.',
            )
        recording_by_id[recording.recording_id] = recording
    if not recording_by_id:
        raise ValueError('Manifest не содержит записей.')
    return recording_by_id


def _validate_definition(
    definition: TrialDefinition,
    recording_by_id: dict[str, Recording],
    seen_pairs: set[tuple[str, str]],
) -> None:
    if definition.enrollment_id == definition.test_id:
        raise ValueError('Запись нельзя сравнивать с собой.')

    pair = tuple(
        sorted((definition.enrollment_id, definition.test_id)),
    )
    if pair in seen_pairs:
        raise ValueError(f'Пара {pair!r} дублируется.')
    seen_pairs.add(pair)

    try:
        enrollment = recording_by_id[definition.enrollment_id]
        test = recording_by_id[definition.test_id]
    except KeyError as exc:
        raise ValueError(
            f'Trial ссылается на неизвестную запись {exc.args[0]!r}.',
        ) from exc

    if enrollment.split is not test.split:
        raise ValueError(
            f'Пара {pair!r} пересекает dataset split.',
        )

    same_speaker = enrollment.speaker_id == test.speaker_id
    if definition.label is TrialLabel.TARGET:
        if not same_speaker:
            raise ValueError(
                f'Target-пара {pair!r} содержит разных дикторов.',
            )
        if enrollment.session_id == test.session_id:
            raise ValueError(
                f'Target-пара {pair!r} должна содержать разные сессии.',
            )
    elif same_speaker:
        raise ValueError(
            f'Nontarget-пара {pair!r} содержит одного диктора.',
        )


__all__ = [
    'EcapaRecordingEncoder',
    'RecordingEncoder',
    'SegmentEncoder',
    'score_trial_definitions',
    'write_scored_trials_csv',
]
