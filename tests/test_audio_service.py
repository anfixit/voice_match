"""Тесты общего декодирования аудиосигнала."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_match.constants import SAMPLE_RATE
from voice_match.exceptions import AudioValidationError
from voice_match.services.audio import load_audio_signal


def test_load_audio_signal_centers_and_normalizes_waveform(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'sample.wav'
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    signal = 0.2 + 0.5 * np.sin(2 * np.pi * 220 * time)
    sf.write(path, signal, SAMPLE_RATE)

    audio = load_audio_signal(path)

    assert audio.dtype == np.float32
    assert audio.shape == (SAMPLE_RATE,)
    assert float(np.mean(audio)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(np.abs(audio))) == pytest.approx(1.0)


def test_load_audio_signal_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioValidationError, match='Файл не найден'):
        load_audio_signal(tmp_path / 'missing.wav')
