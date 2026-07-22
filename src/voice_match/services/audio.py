"""Общее декодирование и нормализация аудиосигнала."""

from pathlib import Path

import librosa
import numpy as np

from voice_match.constants import SAMPLE_RATE
from voice_match.exceptions import AudioValidationError


def load_audio_signal(path: str | Path) -> np.ndarray:
    """Загрузить mono-сигнал 16 кГц и нормализовать амплитуду.

    Args:
        path: Путь к локальному аудиофайлу.

    Returns:
        Одномерный массив ``float32`` в диапазоне ``[-1, 1]``.

    Raises:
        AudioValidationError: Если файл отсутствует или не декодируется.
    """
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise AudioValidationError(
            f'Файл не найден: {audio_path.name}.',
        )

    try:
        signal, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioValidationError(
            f'Не удалось декодировать файл {audio_path.name}.',
        ) from exc

    audio = np.asarray(signal, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise AudioValidationError(
            f'Файл {audio_path.name} пуст.',
        )
    if not np.all(np.isfinite(audio)):
        raise AudioValidationError(
            f'Файл {audio_path.name} содержит некорректные значения.',
        )

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


__all__ = ['load_audio_signal']
