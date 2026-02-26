"""Извлечение джиттера и шиммера — микровариации голоса."""

import librosa
import numpy as np

from voice_match.constants import FRAME_DURATION_S
from voice_match.log import setup_logger

log = setup_logger('jitter_shimmer')


def extract_jitter_shimmer(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает показатели микровариаций голоса: джиттер и шиммер.
    Эти параметры очень трудно подделать даже голосовым модификатором.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор признаков дрожания голоса
    """
    try:
        # Параметры
        frame_length = int(FRAME_DURATION_S * sr)
        hop_length = int(0.01 * sr)

        # Находим основной тон во всех фреймах
        pitches, magnitudes = librosa.core.piptrack(
            y=y, sr=sr,
            n_fft=frame_length,
            hop_length=hop_length,
            fmin=50,
            fmax=400
        )

        # Для каждого фрейма берем частоту с максимальной магнитудой
        pitch_values = []
        magnitude_values = []

        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            magnitude = magnitudes[index, t]

            # Записываем только ненулевые частоты (вокализованные участки)
            if pitch > 0 and magnitude > 0:
                pitch_values.append(pitch)
                magnitude_values.append(magnitude)

        # Если недостаточно вокализованных фреймов
        if len(pitch_values) < 5:
            return np.zeros(8)

        pitch_values = np.array(pitch_values)
        magnitude_values = np.array(magnitude_values)

        # Вычисление джиттера (вариации периода основного тона)
        periods = 1.0 / pitch_values
        period_diffs = np.abs(np.diff(periods))

        # Локальный джиттер (отношение средней разницы к среднему периоду)
        local_jitter = np.mean(period_diffs) / np.mean(periods) * 100

        # Абсолютный джиттер (средняя абсолютная разница)
        absolute_jitter = np.mean(period_diffs) * 1000  # в миллисекундах

        # PPQ5 (5-point period perturbation quotient)
        ppq5_values = []
        for i in range(2, len(periods) - 2):
            avg_period = np.mean(periods[i - 2:i + 3])
            ppq5_values.append(abs(periods[i] - avg_period))
        ppq5 = np.mean(ppq5_values) / np.mean(periods) * 100 if ppq5_values else 0

        # Вычисление шиммера (вариации амплитуды)
        amplitude_diffs = np.abs(np.diff(magnitude_values))

        # Локальный шиммер
        local_shimmer = np.mean(amplitude_diffs) / np.mean(magnitude_values) * 100

        # Абсолютный шиммер (в дБ)
        db_values = 20 * np.log10(magnitude_values / np.max(magnitude_values))
        db_diffs = np.abs(np.diff(db_values))
        absolute_shimmer_db = np.mean(db_diffs)

        # APQ5 (5-point amplitude perturbation quotient)
        apq5_values = []
        for i in range(2, len(magnitude_values) - 2):
            avg_amp = np.mean(magnitude_values[i - 2:i + 3])
            apq5_values.append(abs(magnitude_values[i] - avg_amp))
        apq5 = np.mean(apq5_values) / np.mean(magnitude_values) * 100 if apq5_values else 0

        # Вектор признаков
        features = np.array([
            local_jitter,
            absolute_jitter,
            ppq5,
            np.std(period_diffs) / np.mean(periods) * 100,  # вариабельность джиттера
            local_shimmer,
            absolute_shimmer_db,
            apq5,
            np.std(amplitude_diffs) / np.mean(magnitude_values) * 100  # вариабельность шиммера
        ])

        return features
    except Exception as e:
        log.warning('Ошибка при извлечении джиттера/шиммера: %s', e)
        return np.zeros(8)
