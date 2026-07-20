"""Тесты quality gate и сегментации речи."""

import numpy as np

from voice_match.constants import SAMPLE_RATE, VAD_FRAME_MS
from voice_match.services import quality


def test_analyze_audio_quality_accepts_sufficient_speech(
    monkeypatch,
) -> None:
    duration_seconds = 6
    signal = np.full(SAMPLE_RATE * duration_seconds, 0.1)
    frame_count = duration_seconds * 1_000 // VAD_FRAME_MS
    monkeypatch.setattr(
        quality,
        '_voiced_frame_mask',
        lambda *_: np.ones(frame_count, dtype=bool),
    )

    result = quality.analyze_audio_quality(
        signal,
        SAMPLE_RATE,
        min_duration_seconds=5.0,
        min_speech_seconds=3.0,
        min_speech_ratio=0.25,
        max_clipping_ratio=0.01,
    )

    assert result.is_usable
    assert result.speech_ratio > 0.95
    assert result.issues == ()


def test_analyze_audio_quality_rejects_short_silent_signal(
    monkeypatch,
) -> None:
    signal = np.zeros(SAMPLE_RATE * 2)
    frame_count = 2_000 // VAD_FRAME_MS
    monkeypatch.setattr(
        quality,
        '_voiced_frame_mask',
        lambda *_: np.zeros(frame_count, dtype=bool),
    )

    result = quality.analyze_audio_quality(
        signal,
        SAMPLE_RATE,
        min_duration_seconds=5.0,
        min_speech_seconds=3.0,
        min_speech_ratio=0.25,
        max_clipping_ratio=0.01,
    )

    assert not result.is_usable
    assert len(result.issues) == 4


def test_extract_speech_segments_limits_count(monkeypatch) -> None:
    duration_seconds = 12
    signal = np.linspace(
        -0.5,
        0.5,
        SAMPLE_RATE * duration_seconds,
        dtype=np.float32,
    )
    frame_count = duration_seconds * 1_000 // VAD_FRAME_MS
    monkeypatch.setattr(
        quality,
        '_voiced_frame_mask',
        lambda *_: np.ones(frame_count, dtype=bool),
    )

    segments = quality.extract_speech_segments(
        signal,
        SAMPLE_RATE,
        segment_seconds=4.0,
        max_segments=2,
        min_segment_seconds=2.0,
    )

    assert len(segments) == 2
    assert all(segment.size == SAMPLE_RATE * 4 for segment in segments)
