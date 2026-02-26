"""Извлечение признаков носовых звуков."""

import librosa
import numpy as np

from voice_match.constants import FRAME_DURATION_S, HOP_DURATION_S
from voice_match.log import setup_logger

log = setup_logger('nasal')


def extract_nasal_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает признаки носовых звуков (м, н).
    Носовые резонансы - уникальная характеристика голоса.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор признаков носовых звуков
    """
    try:
        # Параметры для анализа
        frame_length = int(FRAME_DURATION_S * sr)
        hop_length = int(HOP_DURATION_S * sr)

        # STFT для спектрального анализа
        spectrogram = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

        # Частотные полосы для носовых резонансов
        # Основной носовой резонанс: 250-450 Hz
        # Второй носовой резонанс: 1000-1200 Hz
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        mask_nasal1 = (freq_bins >= 250) & (freq_bins <= 450)
        mask_nasal2 = (freq_bins >= 1000) & (freq_bins <= 1200)

        # Средняя энергия в полосах
        energy_nasal1 = np.mean(spectrogram[mask_nasal1, :], axis=0)
        energy_nasal2 = np.mean(spectrogram[mask_nasal2, :], axis=0)

        # Отношение ко всему спектру
        energy_total = np.mean(spectrogram, axis=0)
        ratio1 = np.zeros_like(energy_nasal1)
        ratio2 = np.zeros_like(energy_nasal2)

        mask = energy_total > 0
        ratio1[mask] = energy_nasal1[mask] / energy_total[mask]
        ratio2[mask] = energy_nasal2[mask] / energy_total[mask]

        # Сбор признаков
        features = np.array([
            np.mean(energy_nasal1),
            np.std(energy_nasal1),
            np.mean(energy_nasal2),
            np.std(energy_nasal2),
            np.mean(ratio1),
            np.std(ratio1),
            np.mean(ratio2),
            np.std(ratio2)
        ])

        return features
    except Exception as e:
        log.warning('Ошибка при извлечении признаков носовых: %s', e)
        return np.zeros(8)
