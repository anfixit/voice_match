"""
Модуль для трекинга формант.
Обертка над FormantAnalyzer для совместимости с интерфейсом приложения.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from models.formant_core import FormantAnalyzer
from models.formant_statistics import FormantStatistics
from app.log import setup_logger

log = setup_logger("formant_tracker")


class FormantTracker:
    """
    Класс для отслеживания формант во времени.
    Предоставляет унифицированный интерфейс для анализа формант.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Инициализирует трекер формант.

        Args:
            sample_rate: Частота дискретизации в Гц
        """
        self.sample_rate = sample_rate
        self.analyzer = FormantAnalyzer(sample_rate=sample_rate)
        self.statistics = FormantStatistics()

    def track_formants(
        self,
        audio: np.ndarray,
        frame_length: Optional[int] = None,
        hop_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Отслеживает форманты во времени для аудиосигнала.

        Args:
            audio: Аудиосигнал
            frame_length: Длина фрейма в сэмплах (опционально)
            hop_length: Шаг между фреймами в сэмплах (опционально)

        Returns:
            Словарь с массивами значений формант F1-F4 во времени
        """
        if len(audio) == 0:
            return {
                "F1": np.array([]),
                "F2": np.array([]),
                "F3": np.array([]),
                "F4": np.array([]),
                "timestamps": np.array([])
            }

        # Параметры окна по умолчанию
        if frame_length is None:
            frame_length = int(0.025 * self.sample_rate)  # 25 мс
        if hop_length is None:
            hop_length = int(0.010 * self.sample_rate)  # 10 мс

        formant_tracks = {
            "F1": [],
            "F2": [],
            "F3": [],
            "F4": [],
            "timestamps": []
        }

        # Обработка по фреймам
        num_frames = 1 + (len(audio) - frame_length) // hop_length

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length

            if end > len(audio):
                break

            frame = audio[start:end]
            timestamp = start / self.sample_rate

            # Извлечение формант для фрейма
            try:
                formants = self.analyzer.extract_formants(
                    frame,
                    method='lpc',
                    order=None  # Автоматический выбор порядка
                )

                if formants and "formants" in formants:
                    f_values = formants["formants"]
                    formant_tracks["F1"].append(f_values.get("F1", 0.0))
                    formant_tracks["F2"].append(f_values.get("F2", 0.0))
                    formant_tracks["F3"].append(f_values.get("F3", 0.0))
                    formant_tracks["F4"].append(f_values.get("F4", 0.0))
                    formant_tracks["timestamps"].append(timestamp)
            except Exception as e:
                log.warning(f"Ошибка при извлечении формант для фрейма {i}: {e}")
                continue

        # Преобразование в numpy arrays
        return {
            key: np.array(values)
            for key, values in formant_tracks.items()
        }

    def compute_formant_statistics(
        self,
        formant_tracks: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Вычисляет статистику для треков формант.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь со статистикой для каждой форманты
        """
        statistics = {}

        for formant_name in ["F1", "F2", "F3", "F4"]:
            if formant_name not in formant_tracks:
                continue

            values = formant_tracks[formant_name]

            # Фильтруем нулевые значения
            valid_values = values[values > 0]

            if len(valid_values) == 0:
                statistics[formant_name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "range": 0.0
                }
            else:
                statistics[formant_name] = {
                    "mean": float(np.mean(valid_values)),
                    "median": float(np.median(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "range": float(np.max(valid_values) - np.min(valid_values))
                }

        return statistics

    def estimate_vocal_tract_length(
        self,
        formant_tracks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Оценивает длину голосового тракта на основе формант.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с оценкой длины голосового тракта
        """
        # Используем форманты F3 и F4 для оценки
        f3_values = formant_tracks.get("F3", np.array([]))
        f4_values = formant_tracks.get("F4", np.array([]))

        # Фильтруем нулевые значения
        f3_valid = f3_values[f3_values > 0]
        f4_valid = f4_values[f4_values > 0]

        if len(f3_valid) == 0 and len(f4_valid) == 0:
            return {
                "mean": 0.0,
                "confidence": 0.0
            }

        # Скорость звука в воздухе: 34400 см/с
        c = 34400

        vtl_estimates = []

        # Оценка по F3
        if len(f3_valid) > 0:
            f3_mean = np.mean(f3_valid)
            vtl_f3 = c / (2 * f3_mean)
            vtl_estimates.append(vtl_f3)

        # Оценка по F4
        if len(f4_valid) > 0:
            f4_mean = np.mean(f4_valid)
            vtl_f4 = c / (2 * f4_mean)
            vtl_estimates.append(vtl_f4)

        if not vtl_estimates:
            return {
                "mean": 0.0,
                "confidence": 0.0
            }

        # Средняя оценка
        vtl_mean = np.mean(vtl_estimates)

        # Уверенность на основе согласованности оценок
        if len(vtl_estimates) > 1:
            vtl_std = np.std(vtl_estimates)
            confidence = max(0.0, 1.0 - (vtl_std / vtl_mean))
        else:
            confidence = 0.5

        return {
            "mean": float(vtl_mean),
            "std": float(np.std(vtl_estimates)) if len(vtl_estimates) > 1 else 0.0,
            "confidence": float(confidence)
        }

    def compare_formant_profiles(
        self,
        stats1: Dict[str, Dict[str, float]],
        stats2: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Сравнивает два формантных профиля.

        Args:
            stats1: Статистика формант первого голоса
            stats2: Статистика формант второго голоса

        Returns:
            Словарь с метриками сходства
        """
        similarities = []

        for formant_name in ["F1", "F2", "F3", "F4"]:
            if formant_name not in stats1 or formant_name not in stats2:
                continue

            mean1 = stats1[formant_name]["mean"]
            mean2 = stats2[formant_name]["mean"]

            if mean1 == 0.0 or mean2 == 0.0:
                continue

            # Относительная разница
            rel_diff = abs(mean1 - mean2) / max(mean1, mean2)
            similarity = 1.0 - min(1.0, rel_diff)
            similarities.append(similarity)

        if not similarities:
            return {
                "overall": 0.0,
                "F1": 0.0,
                "F2": 0.0,
                "F3": 0.0,
                "F4": 0.0
            }

        # Детальное сравнение по каждой форманте
        detailed_comparison = {}
        for i, formant_name in enumerate(["F1", "F2", "F3", "F4"]):
            if i < len(similarities):
                detailed_comparison[formant_name] = similarities[i]
            else:
                detailed_comparison[formant_name] = 0.0

        # Общая оценка
        detailed_comparison["overall"] = float(np.mean(similarities))

        return detailed_comparison


def get_formant_tracker(sample_rate: int = 16000) -> FormantTracker:
    """
    Возвращает экземпляр трекера формант.

    Args:
        sample_rate: Частота дискретизации в Гц

    Returns:
        Экземпляр FormantTracker
    """
    return FormantTracker(sample_rate=sample_rate)
