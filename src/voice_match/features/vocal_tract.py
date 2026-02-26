"""Оценка длины голосового тракта по формантам."""

import numpy as np

from voice_match.log import setup_logger

log = setup_logger('vocal_tract')


def extract_vocal_tract_length(formants: dict[str, np.ndarray]) -> float:
    """
    Оценивает длину голосового тракта на основе формант F1-F4.
    Длина тракта - биометрическая характеристика, не меняющаяся со временем.

    Args:
        formants: Словарь с формантами

    Returns:
        Оценка длины голосового тракта в см
    """
    if formants is None:
        return 0.0

    # Для оценки используем форманты F3 и F4 (наиболее стабильные)
    f3_values = formants["F3"]
    f4_values = formants["F4"]

    if len(f3_values) > 0 and len(f4_values) > 0:
        # Средние значения формант
        f3_mean = np.mean(f3_values)
        f4_mean = np.mean(f4_values)

        # Оценка длины голосового тракта
        # VTL (см) = c / (2 * F3), где c - скорость звука в воздухе (34400 см/с)
        vtl_from_f3 = 34400 / (2 * f3_mean)
        vtl_from_f4 = 34400 / (2 * f4_mean)

        # Итоговая оценка (среднее)
        return (vtl_from_f3 + vtl_from_f4) / 2

    return 0.0
