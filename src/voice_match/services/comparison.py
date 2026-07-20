"""Оркестрация честного baseline-сравнения голосов."""

from pathlib import Path

import librosa
import numpy as np

from voice_match.config import settings
from voice_match.constants import SAMPLE_RATE
from voice_match.detection.antispoofing import (
    get_antispoofing_detector,
)
from voice_match.exceptions import (
    AudioValidationError,
    ModelUnavailableError,
)
from voice_match.log import setup_logger
from voice_match.models.ecapa import get_ecapa
from voice_match.scoring.similarity import (
    SimilaritySummary,
    summarize_embeddings,
)
from voice_match.services.quality import (
    AudioQuality,
    analyze_audio_quality,
    extract_speech_segments,
)

log = setup_logger('comparison')

_DISCLAIMER = (
    'Результат является исследовательской инструментальной оценкой. '
    'Он не является вероятностью принадлежности голоса, LLR или '
    'заключением судебного эксперта.'
)


def compare_voices_dual(file1: str, file2: str) -> tuple[str, str]:
    """Сравнить две записи через ECAPA speaker embeddings.

    Автоматическое решение same/different не выносится до появления
    калибровки на целевом наборе данных.
    """
    first_path = Path(file1)
    second_path = Path(file2)
    log.info(
        'Запущено сравнение файлов %s и %s',
        first_path.name,
        second_path.name,
    )

    try:
        first_audio = _load_audio(first_path)
        second_audio = _load_audio(second_path)
        first_quality = _analyze_quality(first_audio)
        second_quality = _analyze_quality(second_audio)
        _validate_quality(first_quality, second_quality)

        first_segments = _extract_segments(first_audio)
        second_segments = _extract_segments(second_audio)
        if not first_segments or not second_segments:
            raise AudioValidationError(
                'Не удалось выделить пригодные речевые сегменты.'
            )

        encoder = get_ecapa()
        first_embeddings = encoder.encode_segments(first_segments)
        second_embeddings = encoder.encode_segments(second_segments)
        similarity = summarize_embeddings(
            first_embeddings,
            second_embeddings,
        )
        anti_spoofing = _run_antispoofing(
            first_audio,
            second_audio,
        )
    except AudioValidationError as exc:
        return (
            'Недостаточно данных для надёжного сравнения.',
            f'### Анализ остановлен\n\n{exc}\n\n{_DISCLAIMER}',
        )
    except ModelUnavailableError as exc:
        log.warning('Модель недоступна: %s', exc)
        return (
            'Модель сравнения недоступна.',
            f'### Ошибка модели\n\n{exc}\n\n{_DISCLAIMER}',
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log.exception('Сравнение завершилось ошибкой')
        return (
            'Не удалось выполнить сравнение.',
            f'### Техническая ошибка\n\n{exc}\n\n{_DISCLAIMER}',
        )

    verdict = (
        'Автоматическое решение same/different не вынесено. '
        f'Сырой ECAPA cosine score: {similarity.centroid_score:.3f}.'
    )
    report = _build_report(
        first_quality=first_quality,
        second_quality=second_quality,
        first_segment_count=len(first_segments),
        second_segment_count=len(second_segments),
        similarity=similarity,
        anti_spoofing=anti_spoofing,
    )
    return verdict, report


def _load_audio(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AudioValidationError(f'Файл не найден: {path.name}.')

    try:
        signal, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioValidationError(
            f'Не удалось декодировать файл {path.name}.'
        ) from exc

    audio = np.asarray(signal, dtype=np.float32)
    if audio.size == 0:
        raise AudioValidationError(f'Файл {path.name} пуст.')
    if not np.all(np.isfinite(audio)):
        raise AudioValidationError(
            f'Файл {path.name} содержит некорректные значения.'
        )

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _analyze_quality(audio: np.ndarray) -> AudioQuality:
    return analyze_audio_quality(
        audio,
        SAMPLE_RATE,
        min_duration_seconds=settings.min_audio_duration,
        min_speech_seconds=settings.min_speech_seconds,
        min_speech_ratio=settings.min_speech_ratio,
        max_clipping_ratio=settings.max_clipping_ratio,
    )


def _validate_quality(
    first: AudioQuality,
    second: AudioQuality,
) -> None:
    issues: list[str] = []
    if first.issues:
        issues.append('Файл 1: ' + ' '.join(first.issues))
    if second.issues:
        issues.append('Файл 2: ' + ' '.join(second.issues))
    if issues:
        raise AudioValidationError('\n\n'.join(issues))


def _extract_segments(audio: np.ndarray) -> list[np.ndarray]:
    return extract_speech_segments(
        audio,
        SAMPLE_RATE,
        segment_seconds=settings.segment_duration,
        max_segments=settings.segment_count,
        min_segment_seconds=settings.min_segment_duration,
    )


def _run_antispoofing(
    first_audio: np.ndarray,
    second_audio: np.ndarray,
) -> str:
    if not settings.antispoofing_enabled:
        return 'отключён до подключения валидированной модели'

    detector = get_antispoofing_detector()
    try:
        first = detector.detect(first_audio, SAMPLE_RATE)
        second = detector.detect(second_audio, SAMPLE_RATE)
    except ModelUnavailableError as exc:
        return f'недоступен: {exc}'

    return (
        'сырые model score: '
        f'файл 1 = {first["spoof_score"]:.3f}, '
        f'файл 2 = {second["spoof_score"]:.3f}. '
        'Это не калиброванные вероятности.'
    )


def _build_report(
    *,
    first_quality: AudioQuality,
    second_quality: AudioQuality,
    first_segment_count: int,
    second_segment_count: int,
    similarity: SimilaritySummary,
    anti_spoofing: str,
) -> str:
    return '\n'.join(
        [
            '### Качество входных данных',
            '',
            '| Показатель | Файл 1 | Файл 2 |',
            '| --- | ---: | ---: |',
            (
                '| Длительность | '
                f'{first_quality.duration_seconds:.1f} с | '
                f'{second_quality.duration_seconds:.1f} с |'
            ),
            (
                '| Обнаруженная речь | '
                f'{first_quality.speech_seconds:.1f} с | '
                f'{second_quality.speech_seconds:.1f} с |'
            ),
            (
                '| Доля речи | '
                f'{first_quality.speech_ratio:.0%} | '
                f'{second_quality.speech_ratio:.0%} |'
            ),
            (
                '| Клиппинг | '
                f'{first_quality.clipping_ratio:.2%} | '
                f'{second_quality.clipping_ratio:.2%} |'
            ),
            (
                '| Речевые сегменты | '
                f'{first_segment_count} | {second_segment_count} |'
            ),
            '',
            '### Speaker embedding score',
            '',
            '| Метрика | Значение |',
            '| --- | ---: |',
            f'| ECAPA cosine по центроидам | '
            f'{similarity.centroid_score:.4f} |',
            f'| Медиана попарных score | '
            f'{similarity.median_pair_score:.4f} |',
            f'| Среднее попарных score | '
            f'{similarity.mean_pair_score:.4f} |',
            f'| Стандартное отклонение score | '
            f'{similarity.standard_deviation:.4f} |',
            f'| Минимум / максимум | '
            f'{similarity.minimum:.4f} / {similarity.maximum:.4f} |',
            f'| Число сравнений сегментов | '
            f'{similarity.pair_count} |',
            '',
            '### Anti-spoofing',
            '',
            anti_spoofing,
            '',
            '### Интерпретация',
            '',
            (
                'Cosine score показывает геометрическую близость '
                'эмбеддингов ECAPA. Он не является процентом '
                'совпадения и не имеет универсального порога.'
            ),
            '',
            _DISCLAIMER,
        ]
    )


__all__ = ['compare_voices_dual']
