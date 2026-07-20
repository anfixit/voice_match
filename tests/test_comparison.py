"""Тесты оркестрации сравнения без загрузки ML-модели."""

from pathlib import Path

import numpy as np

from voice_match.services import comparison
from voice_match.services.quality import AudioQuality


class FakeEncoder:
    """Детерминированный encoder для unit-теста."""

    def __init__(self) -> None:
        self._call_count = 0

    def encode_segments(
        self,
        segments: list[np.ndarray],
    ) -> np.ndarray:
        self._call_count += 1
        if self._call_count == 1:
            return np.array([[1.0, 0.0], [0.9, 0.1]])
        return np.array([[1.0, 0.0], [0.8, 0.2]])


def _usable_quality() -> AudioQuality:
    return AudioQuality(
        duration_seconds=10.0,
        speech_seconds=8.0,
        speech_ratio=0.8,
        clipping_ratio=0.0,
        rms=0.1,
        issues=(),
    )


def test_compare_voices_reports_raw_score_without_probability(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    first.touch()
    second.touch()
    audio = np.ones(16000 * 5, dtype=np.float32) * 0.1

    monkeypatch.setattr(comparison, '_load_audio', lambda _: audio)
    monkeypatch.setattr(
        comparison,
        '_analyze_quality',
        lambda _: _usable_quality(),
    )
    monkeypatch.setattr(
        comparison,
        '_extract_segments',
        lambda _: [audio, audio],
    )
    monkeypatch.setattr(comparison, 'get_ecapa', FakeEncoder)
    monkeypatch.setattr(
        comparison,
        '_run_antispoofing',
        lambda *_: 'отключён',
    )

    verdict, report = comparison.compare_voices_dual(
        str(first),
        str(second),
    )

    assert 'решение same/different не вынесено' in verdict
    assert 'ECAPA cosine' in verdict
    assert 'не является процентом' in report
    assert 'вероятностью принадлежности' in report


def test_compare_voices_stops_on_quality_issues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    first.touch()
    second.touch()
    audio = np.ones(16000, dtype=np.float32)
    poor_quality = AudioQuality(
        duration_seconds=1.0,
        speech_seconds=0.0,
        speech_ratio=0.0,
        clipping_ratio=0.0,
        rms=0.1,
        issues=('Недостаточно речи.',),
    )

    monkeypatch.setattr(comparison, '_load_audio', lambda _: audio)
    monkeypatch.setattr(
        comparison,
        '_analyze_quality',
        lambda _: poor_quality,
    )

    verdict, report = comparison.compare_voices_dual(
        str(first),
        str(second),
    )

    assert verdict == 'Недостаточно данных для надёжного сравнения.'
    assert 'Недостаточно речи' in report
