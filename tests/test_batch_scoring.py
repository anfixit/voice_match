"""Тесты пакетного расчёта speaker verification score."""

from pathlib import Path

import numpy as np
import pytest

from voice_match.evaluation import scoring
from voice_match.evaluation.dataset import DatasetSplit, Recording
from voice_match.evaluation.protocol import (
    Trial,
    TrialLabel,
    load_trials_csv,
)
from voice_match.evaluation.scoring import (
    EcapaRecordingEncoder,
    score_trial_definitions,
    write_scored_trials_csv,
)
from voice_match.evaluation.trial_generation import TrialDefinition
from voice_match.exceptions import AudioValidationError
from voice_match.services.quality import AudioQuality


class FakeRecordingEncoder:
    """Детерминированный recording encoder с учётом вызовов."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors
        self.calls: list[str] = []

    def encode(self, recording: Recording) -> np.ndarray:
        self.calls.append(recording.recording_id)
        return self._vectors[recording.recording_id]


class FakeSegmentEncoder:
    """Детерминированный segment encoder."""

    def encode_segments(
        self,
        segments: list[np.ndarray],
    ) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.8, 0.2]])


def _recording(
    recording_id: str,
    speaker_id: str,
    session_id: str,
    *,
    split: DatasetSplit = DatasetSplit.CALIBRATION,
    path: str | None = None,
) -> Recording:
    return Recording(
        recording_id=recording_id,
        speaker_id=speaker_id,
        session_id=session_id,
        path=path or f'audio/{recording_id}.wav',
        split=split,
        condition='clean',
    )


def _usable_quality() -> AudioQuality:
    return AudioQuality(
        duration_seconds=10.0,
        speech_seconds=8.0,
        speech_ratio=0.8,
        clipping_ratio=0.0,
        rms=0.1,
        issues=(),
    )


def test_score_trial_definitions_encodes_each_recording_once() -> None:
    recordings = [
        _recording('a-1', 'speaker-a', 'session-1'),
        _recording('a-2', 'speaker-a', 'session-2'),
        _recording('b-1', 'speaker-b', 'session-1'),
    ]
    definitions = [
        TrialDefinition('a-1', 'a-2', TrialLabel.TARGET, 'clean'),
        TrialDefinition(
            'a-1',
            'b-1',
            TrialLabel.NONTARGET,
            'cross-speaker',
        ),
    ]
    encoder = FakeRecordingEncoder(
        {
            'a-1': np.array([1.0, 0.0]),
            'a-2': np.array([0.9, 0.1]),
            'b-1': np.array([0.0, 1.0]),
        }
    )

    trials = score_trial_definitions(recordings, definitions, encoder)

    assert encoder.calls == ['a-1', 'a-2', 'b-1']
    assert trials[0].score > 0.9
    assert trials[1].score == pytest.approx(0.0)
    assert trials[0].condition == 'clean'


def test_score_trial_definitions_rejects_wrong_target_label() -> None:
    recordings = [
        _recording('a-1', 'speaker-a', 'session-1'),
        _recording('b-1', 'speaker-b', 'session-1'),
    ]
    definitions = [
        TrialDefinition('a-1', 'b-1', TrialLabel.TARGET),
    ]
    encoder = FakeRecordingEncoder({})

    with pytest.raises(ValueError, match='разных дикторов'):
        score_trial_definitions(recordings, definitions, encoder)

    assert encoder.calls == []


def test_score_trial_definitions_rejects_same_session_target() -> None:
    recordings = [
        _recording('a-1', 'speaker-a', 'session-1'),
        _recording('a-2', 'speaker-a', 'session-1'),
    ]
    definitions = [
        TrialDefinition('a-1', 'a-2', TrialLabel.TARGET),
    ]

    with pytest.raises(ValueError, match='разные сессии'):
        score_trial_definitions(
            recordings,
            definitions,
            FakeRecordingEncoder({}),
        )


def test_score_trial_definitions_rejects_cross_split_pair() -> None:
    recordings = [
        _recording('a-1', 'speaker-a', 'session-1'),
        _recording(
            'b-1',
            'speaker-b',
            'session-1',
            split=DatasetSplit.TEST,
        ),
    ]
    definitions = [
        TrialDefinition('a-1', 'b-1', TrialLabel.NONTARGET),
    ]

    with pytest.raises(ValueError, match='dataset split'):
        score_trial_definitions(
            recordings,
            definitions,
            FakeRecordingEncoder({}),
        )


def test_ecapa_recording_encoder_builds_centroid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / 'audio' / 'a-1.wav'
    audio_path.parent.mkdir()
    audio_path.touch()
    audio = np.ones(16000 * 5, dtype=np.float32) * 0.1
    monkeypatch.setattr(scoring, 'load_audio_signal', lambda _: audio)
    monkeypatch.setattr(
        scoring,
        'analyze_audio_quality',
        lambda *args, **kwargs: _usable_quality(),
    )
    monkeypatch.setattr(
        scoring,
        'extract_speech_segments',
        lambda *args, **kwargs: [audio, audio],
    )
    encoder = EcapaRecordingEncoder(
        tmp_path,
        segment_encoder=FakeSegmentEncoder(),
    )

    embedding = encoder.encode(
        _recording('a-1', 'speaker-a', 'session-1'),
    )

    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert embedding[0] > embedding[1]


def test_ecapa_recording_encoder_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / 'dataset'
    dataset_root.mkdir()
    outside = tmp_path / 'outside.wav'
    outside.touch()
    link = dataset_root / 'escape.wav'
    link.symlink_to(outside)
    encoder = EcapaRecordingEncoder(
        dataset_root,
        segment_encoder=FakeSegmentEncoder(),
    )
    recording = _recording(
        'escape',
        'speaker-a',
        'session-1',
        path='escape.wav',
    )

    with pytest.raises(AudioValidationError, match='за пределы'):
        encoder.encode(recording)


def test_write_scored_trials_creates_benchmark_protocol(
    tmp_path: Path,
) -> None:
    output = tmp_path / 'scored.csv'
    trials = [
        Trial('a-1', 'a-2', TrialLabel.TARGET, 0.9, 'clean'),
        Trial('a-1', 'b-1', TrialLabel.NONTARGET, 0.1, 'clean'),
    ]

    write_scored_trials_csv(trials, output)

    loaded = load_trials_csv(output)
    assert loaded == trials
