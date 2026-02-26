"""Извлечение эмбеддингов YAMNet."""

import librosa
import numpy as np
import tensorflow as tf

from voice_match.constants import SAMPLE_RATE
from voice_match.log import setup_logger
from voice_match.models.yamnet import get_yamnet

log = setup_logger('yamnet_features')


def extract_yamnet(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает перцептивные признаки с помощью YAMNet.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор embedding из YAMNet
    """
    try:
        # Привести частоту дискретизации к требуемой для YAMNet
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Преобразование в тензор
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)

        # Получение эмбеддингов из YAMNet
        _, embeddings, _ = get_yamnet()(waveform)

        # Возвращаем среднее значение эмбеддингов по времени
        return embeddings.numpy().mean(axis=0)
    except Exception as e:
        log.warning('Ошибка при извлечении YAMNet признаков: %s', e)
        return np.zeros(1024)  # YAMNet embeddings имеют размерность 1024
