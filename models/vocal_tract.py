"""
Модуль: vocal_tract.py
Описание: Анализ параметров голосового тракта на основе формантных данных.
Содержит функции для оценки длины голосового тракта, анализа артикуляционных особенностей
и извлечения уникальных биометрических характеристик, связанных с анатомическими
особенностями говорящего.
"""

import numpy as np
import scipy.signal
import scipy.stats
from typing import Dict, List, Tuple, Union, Optional
from app.log import setup_logger

log = setup_logger("vocal_tract")


class VocalTractAnalyzer:
    """
    Класс для анализа голосового тракта на основе формантных данных.
    Обеспечивает извлечение физиологических параметров, которые являются
    уникальными биометрическими характеристиками говорящего.
    """

    def __init__(self):
        """
        Инициализирует анализатор голосового тракта.
        """
        # Скорость звука в воздухе при 20°C (в см/с)
        self.speed_of_sound = 34400.0

        # Типичные диапазоны длины голосового тракта (в см)
        self.tract_length_ranges = {
            "male": (15.5, 19.0),  # Мужской голосовой тракт
            "female": (13.0, 16.5),  # Женский голосовой тракт
            "child": (10.0, 14.0)  # Детский голосовой тракт
        }

        # Типичные диапазоны площади поперечного сечения в разных отделах тракта (в см²)
        self.tract_area_ranges = {
            "male": {
                "pharyngeal": (1.5, 5.0),  # Глоточная полость
                "oral": (2.0, 8.0),  # Ротовая полость
                "labial": (0.5, 3.0)  # Губная область
            },
            "female": {
                "pharyngeal": (1.2, 4.0),
                "oral": (1.5, 6.5),
                "labial": (0.4, 2.5)
            },
            "child": {
                "pharyngeal": (0.8, 3.0),
                "oral": (1.0, 5.0),
                "labial": (0.3, 2.0)
            }
        }

    def estimate_vocal_tract_length(self, formant_tracks: Dict[str, np.ndarray]) -> Dict[
        str, Union[float, List[float]]]:
        """
        Оценивает длину голосового тракта на основе формант F1-F4.
        Длина тракта - биометрическая характеристика, определяемая анатомией говорящего.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с оценками длины голосового тракта и доверительными интервалами
        """
        # Проверка наличия необходимых данных
        required_formants = ["F3", "F4"]
        if not all(key in formant_tracks for key in required_formants):
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "reliability": 0.0,
                "estimates": [],
                "method": "unknown"
            }

        # Извлечение данных формант
        f3_values = formant_tracks["F3"]
        f4_values = formant_tracks["F4"]
        is_voiced = formant_tracks.get("is_voiced", np.ones_like(f3_values, dtype=bool))

        # Маски надежности
        f3_reliability_key = "F3_reliability"
        f4_reliability_key = "F4_reliability"

        if f3_reliability_key in formant_tracks and f4_reliability_key in formant_tracks:
            f3_reliability = formant_tracks[f3_reliability_key]
            f4_reliability = formant_tracks[f4_reliability_key]
            reliability_threshold = 0.6  # Порог надежности
            reliable_f3_mask = f3_reliability > reliability_threshold
            reliable_f4_mask = f4_reliability > reliability_threshold
        else:
            reliable_f3_mask = np.ones_like(f3_values, dtype=bool)
            reliable_f4_mask = np.ones_like(f4_values, dtype=bool)

        # Маска для валидных F3
        valid_f3_mask = (f3_values > 0) & is_voiced & reliable_f3_mask
        valid_f3 = f3_values[valid_f3_mask]

        # Маска для валидных F4
        valid_f4_mask = (f4_values > 0) & is_voiced & reliable_f4_mask
        valid_f4 = f4_values[valid_f4_mask]

        # Если нет достаточного количества данных
        if len(valid_f3) < 5 and len(valid_f4) < 5:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "reliability": 0.0,
                "estimates": [],
                "method": "insufficient_data"
            }

        # Оценки длины тракта из разных формант
        vtl_estimates = []
        formant_weights = []
        method = "combined"

        # 1. Оценка по формуле для F3 (наиболее надежная)
        # Формула: VTL = (2n-1) * c / (4 * F)
        # Где n - номер форманты, c - скорость звука, F - частота форманты
        if len(valid_f3) >= 5:
            # Для F3: VTL = 5 * c / (4 * F3)
            vtl_from_f3 = 5 * self.speed_of_sound / (4 * valid_f3)
            vtl_estimates.extend(vtl_from_f3)

            # Надежность оценок из F3
            if f3_reliability_key in formant_tracks:
                f3_rel_values = formant_tracks[f3_reliability_key][valid_f3_mask]
                formant_weights.extend(f3_rel_values * 1.2)  # Повышенный вес для F3
            else:
                formant_weights.extend([1.2] * len(vtl_from_f3))  # Повышенный вес для F3

        # 2. Оценка по формуле для F4
        if len(valid_f4) >= 5:
            # Для F4: VTL = 7 * c / (4 * F4)
            vtl_from_f4 = 7 * self.speed_of_sound / (4 * valid_f4)
            vtl_estimates.extend(vtl_from_f4)

            # Надежность оценок из F4
            if f4_reliability_key in formant_tracks:
                f4_rel_values = formant_tracks[f4_reliability_key][valid_f4_mask]
                formant_weights.extend(f4_rel_values * 1.0)  # Стандартный вес для F4
            else:
                formant_weights.extend([1.0] * len(vtl_from_f4))

        # 3. Оценка по формуле для F1 и F2 (менее надежная из-за влияния артикуляции)
        f1_values = formant_tracks.get("F1", np.zeros_like(f3_values))
        f2_values = formant_tracks.get("F2", np.zeros_like(f3_values))

        if "F1" in formant_tracks and "F2" in formant_tracks:
            f1_reliability_key = "F1_reliability"
            f2_reliability_key = "F2_reliability"

            if f1_reliability_key in formant_tracks and f2_reliability_key in formant_tracks:
                f1_reliability = formant_tracks[f1_reliability_key]
                f2_reliability = formant_tracks[f2_reliability_key]
                reliability_threshold = 0.7  # Повышенный порог для F1/F2
                reliable_f1_mask = f1_reliability > reliability_threshold
                reliable_f2_mask = f2_reliability > reliability_threshold
            else:
                reliable_f1_mask = np.ones_like(f1_values, dtype=bool)
                reliable_f2_mask = np.ones_like(f2_values, dtype=bool)

            # Маски для валидных F1 и F2
            valid_f1_mask = (f1_values > 0) & is_voiced & reliable_f1_mask
            valid_f2_mask = (f2_values > 0) & is_voiced & reliable_f2_mask

            # Маска для одновременно валидных F1 и F2
            valid_f1f2_mask = valid_f1_mask & valid_f2_mask

            if np.sum(valid_f1f2_mask) >= 5:
                valid_f1 = f1_values[valid_f1f2_mask]
                valid_f2 = f2_values[valid_f1f2_mask]

                # Метод трех формант Фанта (только для непередних гласных)
                # Используем соотношение F2/F1 для фильтрации передних гласных
                f2_f1_ratio = valid_f2 / valid_f1
                back_vowels_mask = f2_f1_ratio < 2.0  # Непередние гласные обычно имеют F2/F1 < 2.0

                if np.sum(back_vowels_mask) >= 3:
                    back_vowel_f1 = valid_f1[back_vowels_mask]
                    back_vowel_f2 = valid_f2[back_vowels_mask]

                    # Формула Фанта: VTL ≈ c / (2 * sqrt(F1 * F2))
                    vtl_from_f1f2 = self.speed_of_sound / (2 * np.sqrt(back_vowel_f1 * back_vowel_f2))
                    vtl_estimates.extend(vtl_from_f1f2)

                    # Пониженные веса для оценок из F1/F2
                    formant_weights.extend([0.6] * len(vtl_from_f1f2))

        # Преобразование в массивы numpy
        vtl_estimates = np.array(vtl_estimates)
        formant_weights = np.array(formant_weights)

        # Исключение выбросов (очень маленьких или очень больших оценок)
        valid_range = (8.0, 25.0)  # Разумный диапазон длины голосового тракта в см
        valid_vtl_mask = (vtl_estimates >= valid_range[0]) & (vtl_estimates <= valid_range[1])

        if np.sum(valid_vtl_mask) < 3:  # Если осталось слишком мало оценок
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "reliability": 0.0,
                "estimates": [],
                "method": "outlier_removal_failed"
            }

        vtl_estimates = vtl_estimates[valid_vtl_mask]
        formant_weights = formant_weights[valid_vtl_mask]

        # Вычисление взвешенного среднего и доверительного интервала
        if len(formant_weights) > 0 and np.sum(formant_weights) > 0:
            # Взвешенное среднее
            vtl_mean = np.average(vtl_estimates, weights=formant_weights)

            # Взвешенная дисперсия
            weight_sum = np.sum(formant_weights)
            vtl_variance = np.sum(formant_weights * (vtl_estimates - vtl_mean) ** 2) / weight_sum
            vtl_std = np.sqrt(vtl_variance)

            # Эффективный размер выборки
            n_eff = (np.sum(formant_weights) ** 2) / np.sum(formant_weights ** 2)

            # Доверительный интервал (95%)
            t_crit = scipy.stats.t.ppf(0.975, n_eff - 1)  # Критическое значение t-распределения
            margin_of_error = t_crit * vtl_std / np.sqrt(n_eff)
            vtl_ci_lower = vtl_mean - margin_of_error
            vtl_ci_upper = vtl_mean + margin_of_error

            # Медиана (не взвешенная)
            vtl_median = np.median(vtl_estimates)

            # Общая надежность оценки
            reliability = min(1.0, n_eff / 20.0)  # Масштабирование до 1.0 при 20+ эффективных образцах
        else:
            # Если веса не заданы, используем обычные статистики
            vtl_mean = np.mean(vtl_estimates)
            vtl_std = np.std(vtl_estimates)
            vtl_median = np.median(vtl_estimates)

            # Доверительный интервал (95%)
            n = len(vtl_estimates)
            t_crit = scipy.stats.t.ppf(0.975, n - 1)  # Критическое значение t-распределения
            margin_of_error = t_crit * vtl_std / np.sqrt(n)
            vtl_ci_lower = vtl_mean - margin_of_error
            vtl_ci_upper = vtl_mean + margin_of_error

            # Надежность
            reliability = min(1.0, n / 20.0)  # Масштабирование до 1.0 при 20+ образцах

        # Инференс пола на основе длины тракта
        gender_probs = self._infer_gender_from_vtl(vtl_mean)

        return {
            "mean": float(vtl_mean),
            "median": float(vtl_median),
            "std": float(vtl_std),
            "ci_lower": float(vtl_ci_lower),
            "ci_upper": float(vtl_ci_upper),
            "reliability": float(reliability),
            "estimates": vtl_estimates.tolist(),
            "method": method,
            "gender_probabilities": gender_probs
        }

    def _infer_gender_from_vtl(self, vtl: float) -> Dict[str, float]:
        """
        Определяет вероятный пол и возраст на основе длины голосового тракта.

        Args:
            vtl: Длина голосового тракта в см

        Returns:
            Словарь с вероятностями для разных полов и возрастов
        """
        # Типичные диапазоны
        male_range = self.tract_length_ranges["male"]
        female_range = self.tract_length_ranges["female"]
        child_range = self.tract_length_ranges["child"]

        # Вероятности на основе принадлежности к диапазонам
        male_prob = 0.0
        female_prob = 0.0
        child_prob = 0.0

        # Если точно в диапазоне, высокая вероятность
        if male_range[0] <= vtl <= male_range[1]:
            male_prob = 0.8

            # Если близко к границе с женским диапазоном
            if vtl < (male_range[0] + (male_range[1] - male_range[0]) * 0.3):
                male_prob = 0.7
                female_prob = 0.3
        elif female_range[0] <= vtl <= female_range[1]:
            female_prob = 0.8

            # Если близко к границе с детским диапазоном
            if vtl < (female_range[0] + (female_range[1] - female_range[0]) * 0.3):
                female_prob = 0.7
                child_prob = 0.3
        elif child_range[0] <= vtl <= child_range[1]:
            child_prob = 0.9
        else:
            # Если вне диапазонов, оцениваем по близости
            if vtl > male_range[1]:  # Выше мужского диапазона
                male_prob = 0.8
            elif vtl < child_range[0]:  # Ниже детского диапазона
                child_prob = 0.8
            elif vtl > female_range[1]:  # Между мужским и женским
                dist_to_male = abs(vtl - male_range[0])
                dist_to_female = abs(vtl - female_range[1])

                if dist_to_male < dist_to_female:
                    male_prob = 0.7
                    female_prob = 0.3
                else:
                    male_prob = 0.3
                    female_prob = 0.7
            elif vtl > child_range[1]:  # Между женским и детским
                dist_to_female = abs(vtl - female_range[0])
                dist_to_child = abs(vtl - child_range[1])

                if dist_to_female < dist_to_child:
                    female_prob = 0.7
                    child_prob = 0.3
                else:
                    female_prob = 0.3
                    child_prob = 0.7

        # Нормализация вероятностей
        total_prob = male_prob + female_prob + child_prob
        if total_prob > 0:
            male_prob /= total_prob
            female_prob /= total_prob
            child_prob /= total_prob
        else:
            # По умолчанию, если нет информации
            male_prob = 0.4
            female_prob = 0.4
            child_prob = 0.2

        return {
            "male": float(male_prob),
            "female": float(female_prob),
            "child": float(child_prob)
        }

    def estimate_tract_cross_sections(self, formant_tracks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Оценивает площади поперечного сечения различных отделов голосового тракта.
        Эти параметры зависят от анатомических особенностей и являются
        биометрическими характеристиками говорящего.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с оценками площадей поперечного сечения
        """
        # Проверка наличия необходимых данных
        required_formants = ["F1", "F2", "F3"]
        if not all(key in formant_tracks for key in required_formants):
            return {
                "pharyngeal": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "oral": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "labial": {"mean": 0.0, "std": 0.0, "reliability": 0.0}
            }

        # Извлечение данных формант
        f1_values = formant_tracks["F1"]
        f2_values = formant_tracks["F2"]
        f3_values = formant_tracks["F3"]
        is_voiced = formant_tracks.get("is_voiced", np.ones_like(f1_values, dtype=bool))

        # Маски надежности
        rel_keys = [f"{f}_reliability" for f in required_formants]
        if all(key in formant_tracks for key in rel_keys):
            reliability_threshold = 0.6
            reliable_mask = (
                    (formant_tracks["F1_reliability"] > reliability_threshold) &
                    (formant_tracks["F2_reliability"] > reliability_threshold) &
                    (formant_tracks["F3_reliability"] > reliability_threshold)
            )
        else:
            reliable_mask = np.ones_like(f1_values, dtype=bool)

        # Маска для валидных значений
        valid_mask = (
                (f1_values > 0) & (f2_values > 0) & (f3_values > 0) &
                is_voiced & reliable_mask
        )

        # Если нет достаточного количества данных
        if np.sum(valid_mask) < 5:
            return {
                "pharyngeal": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "oral": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "labial": {"mean": 0.0, "std": 0.0, "reliability": 0.0}
            }

        # Действительные значения формант
        valid_f1 = f1_values[valid_mask]
        valid_f2 = f2_values[valid_mask]
        valid_f3 = f3_values[valid_mask]

        # Оценка длины голосового тракта (нужна для вычисления площадей)
        vtl_info = self.estimate_vocal_tract_length(formant_tracks)
        vtl = vtl_info["mean"]

        if vtl <= 0:
            return {
                "pharyngeal": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "oral": {"mean": 0.0, "std": 0.0, "reliability": 0.0},
                "labial": {"mean": 0.0, "std": 0.0, "reliability": 0.0}
            }

        # 1. Оценка площади глоточной полости
        # Используем соотношение F1 и VTL
        # Формула: A_pharyngeal ≈ (VTL^2 * pi) / (4 * F1^2)
        pharyngeal_areas = (vtl ** 2 * np.pi) / (4 * (valid_f1 ** 2)) * (self.speed_of_sound ** 2)

        # 2. Оценка площади ротовой полости
        # Используем соотношение F2 и VTL
        # Формула: A_oral ≈ (VTL^2 * pi) / (4 * (F2 - F1)^2)
        oral_areas = (vtl ** 2 * np.pi) / (4 * ((valid_f2 - valid_f1) ** 2)) * (self.speed_of_sound ** 2)

        # 3. Оценка площади губной области
        # Используем соотношение F3 и F2
        # Формула: A_labial ≈ (VTL^2 * pi) / (4 * (F3 - F2)^2)
        labial_areas = (vtl ** 2 * np.pi) / (4 * ((valid_f3 - valid_f2) ** 2)) * (self.speed_of_sound ** 2)

        # Фильтрация выбросов
        pharyngeal_areas = self._filter_area_outliers(pharyngeal_areas)
        oral_areas = self._filter_area_outliers(oral_areas)
        labial_areas = self._filter_area_outliers(labial_areas)

        # Статистика площадей
        results = {}

        if len(pharyngeal_areas) > 0:
            results["pharyngeal"] = {
                "mean": float(np.mean(pharyngeal_areas)),
                "std": float(np.std(pharyngeal_areas)),
                "reliability": float(min(1.0, len(pharyngeal_areas) / 20.0))
            }
        else:
            results["pharyngeal"] = {"mean": 0.0, "std": 0.0, "reliability": 0.0}

        if len(oral_areas) > 0:
            results["oral"] = {
                "mean": float(np.mean(oral_areas)),
                "std": float(np.std(oral_areas)),
                "reliability": float(min(1.0, len(oral_areas) / 20.0))
            }
        else:
            results["oral"] = {"mean": 0.0, "std": 0.0, "reliability": 0.0}

        if len(labial_areas) > 0:
            results["labial"] = {
                "mean": float(np.mean(labial_areas)),
                "std": float(np.std(labial_areas)),
                "reliability": float(min(1.0, len(labial_areas) / 20.0))
            }
        else:
            results["labial"] = {"mean": 0.0, "std": 0.0, "reliability": 0.0}

        return results

    def _filter_area_outliers(self, areas: np.ndarray) -> np.ndarray:
        """
        Фильтрует выбросы в оценках площадей поперечного сечения.

        Args:
            areas: Массив оценок площадей

        Returns:
            Отфильтрованный массив оценок
        """
        if len(areas) < 3:
            return areas

        # Исключение отрицательных и нулевых значений
        positive_areas = areas[areas > 0]

        if len(positive_areas) < 3:
            return positive_areas

        # Логарифмическое преобразование для лучшей робастности
        log_areas = np.log(positive_areas)

        # Медиана и MAD для робастной оценки центра и разброса
        median_log_area = np.median(log_areas)
        mad = np.median(np.abs(log_areas - median_log_area))

        # Масштабирование MAD для соответствия стандартному отклонению
        mad_to_std = 1.4826

        # Порог для выбросов (3 стандартных отклонения)
        threshold = 3.0 * mad_to_std * mad

        # Маска для невыбросов
        inlier_mask = np.abs(log_areas - median_log_area) <= threshold

        # Возвращаемся к исходной шкале
        filtered_areas = positive_areas[inlier_mask]

        return filtered_areas

    def analyze_articulation_profile(self, formant_tracks: Dict[str, np.ndarray]) -> Dict[
        str, Union[float, Dict[str, float]]]:
        """
        Анализирует артикуляционный профиль говорящего на основе динамики формант.
        Выявляет индивидуальные особенности артикуляции, которые являются
        уникальными биометрическими характеристиками.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с артикуляционными характеристиками
        """
        # Проверка наличия необходимых данных
        required_formants = ["F1", "F2"]
        if not all(key in formant_tracks for key in required_formants):
            return {
                "articulation_space": {"area": 0.0, "density": 0.0},
                "vowel_targets": {},
                "transition_speed": 0.0,
                "overall_score": 0.0
            }

        # Извлечение данных формант
        f1_values = formant_tracks["F1"]
        f2_values = formant_tracks["F2"]
        is_voiced = formant_tracks.get("is_voiced", np.ones_like(f1_values, dtype=bool))

        # Маски надежности
        rel_keys = [f"{f}_reliability" for f in required_formants]
        if all(key in formant_tracks for key in rel_keys):
            reliability_threshold = 0.6
            reliable_mask = (
                    (formant_tracks["F1_reliability"] > reliability_threshold) &
                    (formant_tracks["F2_reliability"] > reliability_threshold)
            )
        else:
            reliable_mask = np.ones_like(f1_values, dtype=bool)

        # Маска для валидных значений
        valid_mask = (
                (f1_values > 0) & (f2_values > 0) &
                is_voiced & reliable_mask
        )

        # Если нет достаточного количества данных
        if np.sum(valid_mask) < 10:
            return {
                "articulation_space": {"area": 0.0, "density": 0.0},
                "vowel_targets": {},
                "transition_speed": 0.0,
                "overall_score": 0.0
            }

        # Действительные значения формант
        valid_f1 = f1_values[valid_mask]
        valid_f2 = f2_values[valid_mask]

        # 1. Анализ артикуляционного пространства
        # Оценка площади артикуляционного пространства
        # Используем выпуклую оболочку точек в пространстве F1-F2
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack((valid_f1, valid_f2))

            # Если точек слишком мало для выпуклой оболочки, используем прямоугольную оценку
            if len(points) < 4:
                area = (np.max(valid_f1) - np.min(valid_f1)) * (np.max(valid_f2) - np.min(valid_f2))
                else:
                # Вычисление выпуклой оболочки
                hull = ConvexHull(points)
                area = hull.volume  # Для 2D это площадь

            # Плотность точек в артикуляционном пространстве
            density = len(points) / area if area > 0 else 0.0

        except Exception as e:
            log.warning(f"Ошибка при вычислении площади артикуляционного пространства: {str(e)}")
            # Прямоугольная оценка площади
            area = (np.max(valid_f1) - np.min(valid_f1)) * (np.max(valid_f2) - np.min(valid_f2))
            density = len(valid_f1) / area if area > 0 else 0.0

        articulation_space = {
            "area": float(area),
            "density": float(density),
            "f1_range": (float(np.min(valid_f1)), float(np.max(valid_f1))),
            "f2_range": (float(np.min(valid_f2)), float(np.max(valid_f2)))
        }

        # 2. Анализ целевых точек гласных (vowel targets)
        # Кластеризация в пространстве F1-F2 для выявления основных целевых точек артикуляции
        vowel_targets = self._identify_vowel_targets(valid_f1, valid_f2)

        # 3. Анализ скорости перехода между артикуляционными целями
        # Если последовательные фреймы
        transition_speeds = []
        frame_indices = np.where(valid_mask)[0]

        for i in range(1, len(frame_indices)):
            prev_idx = frame_indices[i - 1]
            curr_idx = frame_indices[i]

            # Если это последовательные фреймы
            if curr_idx - prev_idx == 1:
                # Евклидово расстояние в пространстве F1-F2
                f1_diff = f1_values[curr_idx] - f1_values[prev_idx]
                f2_diff = f2_values[curr_idx] - f2_values[prev_idx]

                # Расстояние в Гц
                distance = np.sqrt(f1_diff ** 2 + f2_diff ** 2)

                # Скорость перехода (Гц/фрейм)
                transition_speeds.append(distance)

        # Средняя скорость перехода
        if transition_speeds:
            mean_transition_speed = np.mean(transition_speeds)
            std_transition_speed = np.std(transition_speeds)
        else:
            mean_transition_speed = 0.0
            std_transition_speed = 0.0

        # 4. Интегральная оценка артикуляционного профиля
        # Нормированная площадь (относительно типичного диапазона)
        # Типичная площадь артикуляционного пространства: 200000-700000 Гц²
        normalized_area = min(1.0, area / 500000)

        # Нормированная скорость перехода
        # Типичная скорость: 50-200 Гц/фрейм
        normalized_speed = min(1.0, mean_transition_speed / 150) if mean_transition_speed > 0 else 0.0

        # Интегральная оценка
        overall_score = 0.5 * normalized_area + 0.3 * normalized_speed + 0.2 * min(1.0, len(vowel_targets) / 5.0)

        return {
            "articulation_space": articulation_space,
            "vowel_targets": vowel_targets,
            "transition_speed": {
                "mean": float(mean_transition_speed),
                "std": float(std_transition_speed)
            },
            "overall_score": float(overall_score)
        }

    def _identify_vowel_targets(self, f1_values: np.ndarray, f2_values: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Идентифицирует основные целевые точки артикуляции (гласные) в пространстве F1-F2.

        Args:
            f1_values: Значения F1
            f2_values: Значения F2

        Returns:
            Словарь с целевыми точками артикуляции и их характеристиками
        """
        try:
            # Используем кластеризацию для выявления основных целевых точек
            from sklearn.cluster import KMeans, DBSCAN

            # Объединение F1 и F2 в матрицу признаков
            X = np.column_stack((f1_values, f2_values))

            # Определение оптимального числа кластеров
            max_clusters = min(8, len(X) // 5)  # Максимум 8 кластеров, минимум 5 точек на кластер
            max_clusters = max(2, max_clusters)  # Минимум 2 кластера

            # Простая эвристика для определения числа кластеров на основе силуэтного коэффициента
            best_score = -1
            best_n_clusters = 2

            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)

                # Если возникла ошибка или все точки в одном кластере, пропускаем
                if len(np.unique(cluster_labels)) < 2:
                    continue

                try:
                    from sklearn.metrics import silhouette_score
                    # Вычисление силуэтного коэффициента
                    score = silhouette_score(X, cluster_labels)

                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except Exception:
                    # Если не удалось вычислить силуэтный коэффициент, используем стандартное количество
                    pass

            # Кластеризация с оптимальным числом кластеров
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Вычисление характеристик каждого кластера
            vowel_targets = {}
            for i in range(best_n_clusters):
                # Точки в текущем кластере
                cluster_points = X[cluster_labels == i]

                if len(cluster_points) == 0:
                    continue

                # Центр кластера
                center = np.mean(cluster_points, axis=0)

                # Стандартное отклонение
                std_dev = np.std(cluster_points, axis=0)

                # Определение наиболее вероятного гласного по положению в пространстве F1-F2
                vowel = self._identify_vowel(center[0], center[1])

                # Компактность кластера (среднее расстояние до центра)
                distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
                compactness = np.mean(distances)

                vowel_targets[f"cluster_{i + 1}"] = {
                    "f1_center": float(center[0]),
                    "f2_center": float(center[1]),
                    "f1_std": float(std_dev[0]),
                    "f2_std": float(std_dev[1]),
                    "probable_vowel": vowel,
                    "compactness": float(compactness),
                    "point_count": int(len(cluster_points))
                }

            return vowel_targets

        except Exception as e:
            log.warning(f"Ошибка при идентификации целевых точек артикуляции: {str(e)}")
            return {}

    def _identify_vowel(self, f1: float, f2: float) -> str:
        """
        Определяет наиболее вероятный гласный звук по значениям F1 и F2.

        Args:
            f1: Значение первой форманты
            f2: Значение второй форманты

        Returns:
            Наиболее вероятный гласный звук
        """
        # Типичные значения формант для гласных (Гц)
        # Значения могут различаться в зависимости от диалекта и пола говорящего
        vowel_formants = {
            "а": {"F1": (700, 1000), "F2": (1100, 1500)},  # а как в "мама"
            "о": {"F1": (500, 700), "F2": (900, 1100)},  # о как в "дом"
            "у": {"F1": (300, 500), "F2": (600, 900)},  # у как в "суп"
            "э": {"F1": (500, 700), "F2": (1700, 2100)},  # э как в "это"
            "и": {"F1": (300, 400), "F2": (2200, 2700)},  # и как в "мир"
            "ы": {"F1": (300, 500), "F2": (1600, 1900)}  # ы как в "мы"
        }

        # Оценка вероятности для каждого гласного
        probabilities = {}

        for vowel, ranges in vowel_formants.items():
            f1_range = ranges["F1"]
            f2_range = ranges["F2"]

            # Проверка попадания в диапазон
            in_f1_range = f1_range[0] <= f1 <= f1_range[1]
            in_f2_range = f2_range[0] <= f2 <= f2_range[1]

            if in_f1_range and in_f2_range:
                # Если значения точно в диапазоне, высокая вероятность
                probabilities[vowel] = 1.0
            else:
                # Иначе оценка близости к центру диапазона
                f1_center = (f1_range[0] + f1_range[1]) / 2
                f2_center = (f2_range[0] + f2_range[1]) / 2

                f1_width = (f1_range[1] - f1_range[0]) / 2
                f2_width = (f2_range[1] - f2_range[0]) / 2

                f1_distance = abs(f1 - f1_center) / f1_width
                f2_distance = abs(f2 - f2_center) / f2_width

                # Общая дистанция в нормализованном пространстве
                distance = np.sqrt(f1_distance ** 2 + f2_distance ** 2)

                # Вероятность обратно пропорциональна расстоянию
                probabilities[vowel] = 1.0 / (1.0 + distance)

        # Выбор наиболее вероятного гласного
        if probabilities:
            best_vowel = max(probabilities, key=probabilities.get)
            return best_vowel
        else:
            return "unknown"

    def compare_vocal_tract_parameters(self, params1: Dict[str, Dict[str, float]],
                                       params2: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Сравнивает параметры голосового тракта двух говорящих.

        Args:
            params1: Параметры голосового тракта первого говорящего
            params2: Параметры голосового тракта второго говорящего

        Returns:
            Словарь с мерами сходства параметров
        """
        # Проверка наличия данных
        if not params1 or not params2:
            return {
                "vtl_similarity": 0.0,
                "area_similarity": 0.0,
                "articulation_similarity": 0.0,
                "overall_similarity": 0.0
            }

        # 1. Сравнение длины голосового тракта
        vtl_similarity = 0.0

        if "mean" in params1 and "mean" in params2:
            vtl1 = params1["mean"]
            vtl2 = params2["mean"]

            if vtl1 > 0 and vtl2 > 0:
                # Относительная разница
                vtl_diff = abs(vtl1 - vtl2)

                # Нормализация: разница в 1 см дает сходство 0.5
                vtl_similarity = 1.0 - min(1.0, vtl_diff / 2.0)

                # Учет доверительных интервалов
                if "ci_lower" in params1 and "ci_upper" in params1 and \
                        "ci_lower" in params2 and "ci_upper" in params2:
                    ci1 = (params1["ci_lower"], params1["ci_upper"])
                    ci2 = (params2["ci_lower"], params2["ci_upper"])

                    # Перекрытие интервалов
                    overlap = max(0, min(ci1[1], ci2[1]) - max(ci1[0], ci2[0]))

                    # Общая длина объединенного интервала
                    total_range = max(ci1[1], ci2[1]) - min(ci1[0], ci2[0])

                    # Доля перекрытия
                    overlap_ratio = overlap / total_range if total_range > 0 else 0

                    # Учет перекрытия в оценке сходства
                    vtl_similarity = 0.7 * vtl_similarity + 0.3 * overlap_ratio

        # 2. Сравнение площадей поперечного сечения
        area_similarities = []

        # Общие разделы для сравнения
        sections = ["pharyngeal", "oral", "labial"]

        for section in sections:
            if section in params1 and section in params2:
                section1 = params1[section]
                section2 = params2[section]

                if "mean" in section1 and "mean" in section2:
                    area1 = section1["mean"]
                    area2 = section2["mean"]

                    if area1 > 0 and area2 > 0:
                        # Отношение площадей (меньшая к большей)
                        area_ratio = min(area1, area2) / max(area1, area2)

                        # Взвешивание по надежности
                        reliability1 = section1.get("reliability", 0.5)
                        reliability2 = section2.get("reliability", 0.5)

                        avg_reliability = (reliability1 + reliability2) / 2
                        weighted_similarity = area_ratio * avg_reliability

                        area_similarities.append(weighted_similarity)

        # Среднее сходство площадей
        area_similarity = np.mean(area_similarities) if area_similarities else 0.0

        # 3. Сравнение артикуляционных характеристик
        articulation_similarity = 0.0

        if "articulation_space" in params1 and "articulation_space" in params2:
            space1 = params1["articulation_space"]
            space2 = params2["articulation_space"]

            # Сравнение площадей артикуляционного пространства
            if "area" in space1 and "area" in space2:
                area1 = space1["area"]
                area2 = space2["area"]

                if area1 > 0 and area2 > 0:
                    # Отношение площадей (меньшая к большей)
                    area_ratio = min(area1, area2) / max(area1, area2)

                    # Учет плотности точек
                    density1 = space1.get("density", 0.0)
                    density2 = space2.get("density", 0.0)

                    density_ratio = min(density1, density2) / max(density1, density2) if max(density1,
                                                                                             density2) > 0 else 0.0

                    # Комбинированная оценка сходства артикуляционного пространства
                    articulation_similarity = 0.7 * area_ratio + 0.3 * density_ratio

        # 4. Общая оценка сходства
        # Взвешенное среднее разных аспектов сходства
        weights = {
            "vtl": 0.5,  # Длина голосового тракта - важнейший параметр
            "area": 0.3,  # Площади поперечного сечения
            "articulation": 0.2  # Артикуляционные характеристики
        }

        overall_similarity = (
                weights["vtl"] * vtl_similarity +
                weights["area"] * area_similarity +
                weights["articulation"] * articulation_similarity
        )

        return {
            "vtl_similarity": float(vtl_similarity),
            "area_similarity": float(area_similarity),
            "articulation_similarity": float(articulation_similarity),
            "overall_similarity": float(overall_similarity)
        }

    def get_vocal_tract_analyzer() -> VocalTractAnalyzer:
        """
        Создает и возвращает анализатор голосового тракта.

        Returns:
            Экземпляр VocalTractAnalyzer
        """
        return VocalTractAnalyzer()
