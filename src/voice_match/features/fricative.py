"""Извлечение признаков фрикативных согласных."""

import librosa
import numpy as np

from voice_match.constants import FRAME_DURATION_S, HOP_DURATION_S
from voice_match.log import setup_logger

log = setup_logger('fricative')


def extract_fricative_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает признаки фрикативных согласных (ш, с, ф, в, etc).
    Характеристики этих звуков зависят от анатомии речевого аппарата.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор признаков фрикативных звуков
    """
    try:
        # Параметры для анализа
        frame_length = int(FRAME_DURATION_S * sr)
        hop_length = int(HOP_DURATION_S * sr)

        # Спектральные признаки
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length).flatten()

        spec_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length).flatten()

        # Энергия в высокочастотных диапазонах (характерно для фрикативных)
        stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

        # Частотные полосы для фрикативных
        # 1. 2000-4000 Hz (s, z)
        # 2. 4000-8000 Hz (sh, zh)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        mask_s = (freq_bins >= 2000) & (freq_bins <= 4000)
        mask_sh = (freq_bins >= 4000) & (freq_bins <= 8000)

        energy_s = np.mean(stft[mask_s, :], axis=0)
        energy_sh = np.mean(stft[mask_sh, :], axis=0)

        # Отношение энергий - характеристика индивидуальных особенностей
        ratio = np.zeros_like(energy_s)
        mask = energy_s > 0
        ratio[mask] = energy_sh[mask] / energy_s[mask]

        # Формируем вектор признаков фрикативных
        features = np.array([
            np.mean(spec_centroid),
            np.std(spec_centroid),
            np.mean(spec_flatness),
            np.std(spec_flatness),
            np.mean(energy_s),
            np.mean(energy_sh),
            np.mean(ratio),
            np.std(ratio)
        ])

        return features
    except Exception as e:
        log.warning('Ошибка при извлечении признаков фрикативных: %s', e)
        return np.zeros(8)
