"""Тесты безопасной конвертации аудио."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_match.constants import SAMPLE_RATE
from voice_match.services.preprocessing import (
    convert_audio_to_wav,
    get_audio_info,
)


def test_get_audio_info_reads_pcm_wav(tmp_path: Path) -> None:
    source = tmp_path / 'sample.wav'
    sf.write(source, np.zeros(SAMPLE_RATE), SAMPLE_RATE, subtype='PCM_16')

    info = get_audio_info(source)

    assert info['sample_rate'] == SAMPLE_RATE
    assert info['channels'] == 1
    assert info['duration'] == 1.0
    assert info['subtype'] == 'PCM_16'


def test_convert_audio_keeps_matching_wav(tmp_path: Path) -> None:
    source = tmp_path / 'sample.wav'
    sf.write(source, np.zeros(SAMPLE_RATE), SAMPLE_RATE, subtype='PCM_16')

    result_path, message = convert_audio_to_wav(str(source))

    assert Path(result_path) == source.resolve()
    assert 'уже соответствует' in message


def test_convert_audio_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / 'missing.wav'

    with pytest.raises(FileNotFoundError, match='не найден'):
        convert_audio_to_wav(str(missing))


def test_convert_audio_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'sample.txt'
    source.write_text('not audio', encoding='utf-8')

    with pytest.raises(ValueError, match='не поддерживается'):
        convert_audio_to_wav(str(source))
