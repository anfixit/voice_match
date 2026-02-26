"""Извлечение динамических признаков формант."""

import numpy as np

from voice_match.log import setup_logger

log = setup_logger('formant_dynamics')


def extract_formant_dynamics(formants: dict[str, np.ndarray]) -> np.ndarray:
    """
    Извлекает характеристики динамики формант, важные для идентификации.

    Args:
        formants: Словарь с формантами из extract_formants_advanced

    Returns:
        Вектор признаков динамики формант
    """
    if formants is None:
        return np.zeros(12)

    features = []

    # Для каждой форманты извлекаем статистические признаки
    for key in ["F1", "F2", "F3", "F4"]:
        values = formants[key]
        if len(values) > 2:  # Если есть достаточно данных
            # Среднее значение
            features.append(np.mean(values))

            # Стандартное отклонение (вариабельность)
            features.append(np.std(values))

            # Диапазон (размах)
            features.append(np.max(values) - np.min(values))
        else:
            # Если данных недостаточно, добавляем нули
            features.extend([0, 0, 0])

    return np.array(features)
