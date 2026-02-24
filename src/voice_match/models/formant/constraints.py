"""
Модуль: formant_constraints.py
Описание: Функции для применения физиологических ограничений к формантам.
Обеспечивает проверку и корректировку значений формант в соответствии с
анатомическими особенностями голосового тракта человека.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from voice_match.log import setup_logger

log = setup_logger("formant_constraints")


class FormantConstraintHandler:
    """
    Класс для проверки и корректировки формант в соответствии с
    физиологическими ограничениями голосового тракта человека.
    """

    def __init__(self):
        """
        Инициализирует обработчик ограничений.
        """
        # Диапазоны допустимых значений формант (в Гц)
        self.formant_ranges = {
            "male": {
                "F1": (250, 900),  # Первая форманта
                "F2": (800, 2300),  # Вторая форманта
                "F3": (1800, 3000),  # Третья форманта
                "F4": (3000, 4500)  # Четвертая форманта
            },
            "female": {
                "F1": (300, 1100),  # Первая форманта
                "F2": (900, 2800),  # Вторая форманта
                "F3": (2300, 3500),  # Третья форманта
                "F4": (3400, 5000)  # Четвертая форманта
            },
            "child": {
                "F1": (350, 1200),  # Первая форманта
                "F2": (1000, 3000),  # Вторая форманта
                "F3": (2500, 4000),  # Третья форманта
                "F4": (3700, 5500)  # Четвертая форманта
            }
        }

        # Допустимые соотношения формант для разных гласных
        # Эти значения важны для проверки согласованности формант
        self.vowel_formant_ratios = {
            # Соотношения F2/F1 для разных гласных звуков
            "a": (1.4, 2.1),  # а
            "e": (2.3, 3.2),  # э
            "i": (3.3, 4.7),  # и
            "o": (0.9, 1.6),  # о
            "u": (0.7, 1.3)  # у
        }

        # Типичные интервалы между формантами (в Гц)
        self.formant_intervals = {
            "male": {
                "F2-F1": (500, 1200),  # Интервал между F2 и F1
                "F3-F2": (700, 1500),  # Интервал между F3 и F2
                "F4-F3": (700, 1800)  # Интервал между F4 и F3
            },
            "female": {
                "F2-F1": (600, 1400),
                "F3-F2": (800, 1700),
                "F4-F3": (800, 2000)
            },
            "child": {
                "F2-F1": (700, 1600),
                "F3-F2": (900, 1800),
                "F4-F3": (900, 2200)
            }
        }

    def enforce_physiological_constraints(self, formant_tracks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Применяет физиологические ограничения к формантным трекам.
        Проверяет, что F1 < F2 < F3 < F4 и все форманты находятся
        в физиологически достоверных диапазонах.

        Args:
            formant_tracks: Словарь с треками формант и сопутствующей информацией

        Returns:
            Обновленные формантные треки с примененными ограничениями
        """
        # Создаем копию для безопасного изменения
        updated_tracks = formant_tracks.copy()

        num_frames = len(updated_tracks["F1"])
        gender = updated_tracks.get("gender", "male")
        ranges = self.formant_ranges.get(gender, self.formant_ranges["male"])
        intervals = self.formant_intervals.get(gender, self.formant_intervals["male"])

        for t in range(num_frames):
            # Применяем ограничения только к фреймам с речью
            if updated_tracks.get("is_voiced", np.ones(num_frames, dtype=bool))[t] and \
                    updated_tracks.get("energy", np.ones(num_frames))[t] > 0:

                # 1. Проверка на корректность порядка формант
                modified = False
                for i in range(1, 4):
                    current_formant = f"F{i}"
                    next_formant = f"F{i + 1}"

                    # Если обе форманты определены
                    if updated_tracks[current_formant][t] > 0 and updated_tracks[next_formant][t] > 0:
                        # Проверка на следование правилу F1 < F2 < F3 < F4
                        if updated_tracks[next_formant][t] <= updated_tracks[current_formant][t]:
                            # Нарушен порядок формант - исправляем на основе надежности
                            current_reliability = updated_tracks[f"{current_formant}_reliability"][t]
                            next_reliability = updated_tracks[f"{next_formant}_reliability"][t]

                            # Для нарушения порядка формант устанавливаем мин. разделение 100 Hz
                            min_separation = 100.0

                            if current_reliability >= next_reliability:
                                # Более надежна текущая форманта - корректируем следующую
                                updated_tracks[next_formant][t] = updated_tracks[current_formant][t] + min_separation
                                updated_tracks[f"{next_formant}_reliability"][t] *= 0.7  # Понижаем надежность
                            else:
                                # Более надежна следующая форманта - корректируем текущую
                                updated_tracks[current_formant][t] = updated_tracks[next_formant][t] - min_separation
                                updated_tracks[f"{current_formant}_reliability"][t] *= 0.7  # Понижаем надежность

                            modified = True

                # 2. Проверка на попадание в допустимые диапазоны
                for formant_name, (min_freq, max_freq) in ranges.items():
                    formant_value = updated_tracks[formant_name][t]
                    if formant_value > 0:
                        # Коррекция, если значение выходит за допустимые пределы
                        if formant_value < min_freq:
                            # Слишком низкое значение - коррекция к нижней границе
                            correction_factor = min(1.0, (min_freq - formant_value) / 100.0)
                            updated_tracks[formant_name][t] = min_freq
                            updated_tracks[f"{formant_name}_reliability"][t] *= (1.0 - 0.5 * correction_factor)
                            modified = True
                        elif formant_value > max_freq:
                            # Слишком высокое значение - коррекция к верхней границе
                            correction_factor = min(1.0, (formant_value - max_freq) / 200.0)
                            updated_tracks[formant_name][t] = max_freq
                            updated_tracks[f"{formant_name}_reliability"][t] *= (1.0 - 0.5 * correction_factor)
                            modified = True

                # 3. Проверка физиологически вероятных соотношений формант
                # Например, соотношение F2/F1 обычно находится в определенном диапазоне для разных гласных
                if updated_tracks["F1"][t] > 0 and updated_tracks["F2"][t] > 0:
                    f2_f1_ratio = updated_tracks["F2"][t] / updated_tracks["F1"][t]

                    # Типичное соотношение F2/F1 для человеческого голоса: 1.3-4.5
                    if f2_f1_ratio < 1.3:
                        # Слишком низкое соотношение - возможно, ошибка определения
                        updated_tracks["F1_reliability"][t] *= 0.6
                        updated_tracks["F2_reliability"][t] *= 0.6
                    elif f2_f1_ratio > 4.5:
                        # Слишком высокое соотношение - возможно, ошибка определения
                        updated_tracks["F1_reliability"][t] *= 0.7
                        updated_tracks["F2_reliability"][t] *= 0.7

                # Если были модификации, пересчитываем некоторые взаимозависимые параметры
                if modified:
                    # Пересчет надежности, если были изменения
                    for formant_name in ["F1", "F2", "F3", "F4"]:
                        # Ограничение минимального уровня надежности
                        reliability_key = f"{formant_name}_reliability"
                        updated_tracks[reliability_key][t] = max(0.1, updated_tracks[reliability_key][t])

        # Анализ полной физиологической согласованности
        self._check_formant_intervals(updated_tracks)

        return updated_tracks

    def _check_formant_intervals(self, formant_tracks: Dict[str, np.ndarray]) -> None:
        """
        Проверяет интервалы между формантами на соответствие физиологическим нормам.

        Args:
            formant_tracks: Словарь с треками формант
        """
        num_frames = len(formant_tracks["F1"])
        gender = formant_tracks.get("gender", "male")
        intervals = self.formant_intervals.get(gender, self.formant_intervals["male"])

        for t in range(num_frames):
            if formant_tracks.get("formant_count", np.zeros(num_frames, dtype=int))[t] >= 3:  # Минимум F1, F2, F3
                # Вычисляем типичные соотношения формант для проверки общей согласованности
                # Например, F2-F1 и F3-F2 обычно имеют характерные интервалы
                f1 = formant_tracks["F1"][t]
                f2 = formant_tracks["F2"][t]
                f3 = formant_tracks["F3"][t]
                f4 = formant_tracks.get("F4", np.zeros(num_frames))[t]

                if f1 > 0 and f2 > 0 and f3 > 0:
                    # Вычисляем интервалы
                    f2_f1_interval = f2 - f1
                    f3_f2_interval = f3 - f2
                    f4_f3_interval = f4 - f3 if f4 > 0 else 0

                    # Проверяем на очень нестандартные интервалы, которые могут быть ошибкой
                    if (f2_f1_interval < intervals["F2-F1"][0] or
                        f2_f1_interval > intervals["F2-F1"][1] or
                        f3_f2_interval < intervals["F3-F2"][0] or
                        f3_f2_interval > intervals["F3-F2"][1]) or \
                            (f4 > 0 and (f4_f3_interval < intervals["F4-F3"][0] or
                                         f4_f3_interval > intervals["F4-F3"][1])):
                        # Понижаем общую надежность всех формант
                        for formant_name in ["F1", "F2", "F3", "F4"]:
                            formant_tracks[f"{formant_name}_reliability"][t] *= 0.8

    def check_formant_consistency(self, formant_values: Dict[str, float]) -> Dict[str, Union[bool, float, str]]:
        """
        Проверяет согласованность набора формант и пытается определить фонему.

        Args:
            formant_values: Словарь с значениями формант F1-F4

        Returns:
            Словарь с результатами проверки: согласованность, вероятная фонема и т.д.
        """
        # Проверка на наличие необходимых формант
        if "F1" not in formant_values or "F2" not in formant_values or \
                formant_values["F1"] <= 0 or formant_values["F2"] <= 0:
            return {
                "is_consistent": False,
                "probable_vowel": "unknown",
                "consistency_score": 0.0,
                "formant_ratio": 0.0
            }

        # Вычисление соотношения F2/F1 для определения гласного
        f2_f1_ratio = formant_values["F2"] / formant_values["F1"]

        # Определение наиболее вероятного гласного звука
        probable_vowel = "unknown"
        best_match_score = 0.0

        for vowel, (min_ratio, max_ratio) in self.vowel_formant_ratios.items():
            if min_ratio <= f2_f1_ratio <= max_ratio:
                # Находимся в диапазоне для этого гласного
                # Вычисляем оценку соответствия
                center_ratio = (min_ratio + max_ratio) / 2
                dist_from_center = abs(f2_f1_ratio - center_ratio)
                max_dist = (max_ratio - min_ratio) / 2

                # Нормализованное расстояние от центра диапазона
                match_score = 1.0 - min(1.0, dist_from_center / max_dist)

                if match_score > best_match_score:
                    best_match_score = match_score
                    probable_vowel = vowel

        # Проверка на корректный порядок формант
        correct_order = True
        if "F1" in formant_values and "F2" in formant_values and \
                formant_values["F1"] > 0 and formant_values["F2"] > 0:
            correct_order = formant_values["F1"] < formant_values["F2"]

        if "F2" in formant_values and "F3" in formant_values and \
                formant_values["F2"] > 0 and formant_values["F3"] > 0:
            correct_order = correct_order and formant_values["F2"] < formant_values["F3"]

        if "F3" in formant_values and "F4" in formant_values and \
                formant_values["F3"] > 0 and formant_values["F4"] > 0:
            correct_order = correct_order and formant_values["F3"] < formant_values["F4"]

        # Итоговая оценка согласованности
        consistency_score = best_match_score * (1.0 if correct_order else 0.5)

        return {
            "is_consistent": correct_order and best_match_score > 0.5,
            "probable_vowel": probable_vowel,
            "consistency_score": consistency_score,
            "formant_ratio": f2_f1_ratio
        }

    def detect_formant_manipulation(self, formant_tracks: Dict[str, np.ndarray]) -> Dict[str, Union[bool, float, str]]:
        """
        Обнаруживает признаки искусственной манипуляции формантами.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с результатами обнаружения
        """
        # Проверка на наличие необходимых данных
        if not all(key in formant_tracks for key in ["F1", "F2", "F3", "is_voiced"]):
            return {
                "is_manipulated": False,
                "confidence": 0.0,
                "type": "unknown"
            }

        # Находим вокализованные фреймы с надежными формантами
        voiced_mask = formant_tracks["is_voiced"].astype(bool)
        formant_mask = (formant_tracks["F1"] > 0) & (formant_tracks["F2"] > 0) & (formant_tracks["F3"] > 0)
        valid_mask = voiced_mask & formant_mask

        if np.sum(valid_mask) < 10:  # Недостаточно данных
            return {
                "is_manipulated": False,
                "confidence": 0.0,
                "type": "insufficient_data"
            }

        # 1. Проверка на неестественную стабильность формант
        # Искусственно измененные голоса часто имеют аномально стабильные форманты
        f1_values = formant_tracks["F1"][valid_mask]
        f2_values = formant_tracks["F2"][valid_mask]
        f3_values = formant_tracks["F3"][valid_mask]

        # Коэффициенты вариации формант
        f1_cv = np.std(f1_values) / np.mean(f1_values) if np.mean(f1_values) > 0 else 0
        f2_cv = np.std(f2_values) / np.mean(f2_values) if np.mean(f2_values) > 0 else 0
        f3_cv = np.std(f3_values) / np.mean(f3_values) if np.mean(f3_values) > 0 else 0

        # Нормальный диапазон вариативности формант
        # Коэффициент вариации для натуральной речи обычно: F1: 0.15-0.30, F2: 0.10-0.25, F3: 0.05-0.15
        f1_stable = f1_cv < 0.10
        f2_stable = f2_cv < 0.08
        f3_stable = f3_cv < 0.04

        formant_stability_score = (f1_stable * 1.0 + f2_stable * 1.0 + f3_stable * 1.0) / 3.0
        extremely_stable = formant_stability_score > 0.7

        # 2. Проверка на нетипичные соотношения формант
        # Формантные сдвиги часто приводят к нетипичным соотношениям F2/F1
        f2_f1_ratios = f2_values / f1_values
        f3_f2_ratios = f3_values / f2_values

        # Статистика соотношений
        mean_f2_f1 = np.mean(f2_f1_ratios)
        mean_f3_f2 = np.mean(f3_f2_ratios)

        # Проверка на выход за нормальные пределы
        abnormal_f2_f1 = mean_f2_f1 < 1.2 or mean_f2_f1 > 5.0
        abnormal_f3_f2 = mean_f3_f2 < 1.1 or mean_f3_f2 > 3.0

        ratio_abnormality_score = (abnormal_f2_f1 * 1.0 + abnormal_f3_f2 * 1.0) / 2.0

        # 3. Оценка итогового результата
        is_manipulated = extremely_stable or ratio_abnormality_score > 0.5

        # Тип манипуляции
        manipulation_type = "unknown"
        if extremely_stable:
            manipulation_type = "synthetic_voice"
        elif abnormal_f2_f1 and not abnormal_f3_f2:
            manipulation_type = "formant_shift_f1f2"
        elif abnormal_f3_f2 and not abnormal_f2_f1:
            manipulation_type = "formant_shift_f2f3"
        elif abnormal_f2_f1 and abnormal_f3_f2:
            manipulation_type = "complete_formant_shift"

        # Уверенность в обнаружении
        confidence = max(formant_stability_score, ratio_abnormality_score)

        return {
            "is_manipulated": is_manipulated,
            "confidence": confidence,
            "type": manipulation_type,
            "stability_details": {
                "f1_cv": f1_cv,
                "f2_cv": f2_cv,
                "f3_cv": f3_cv,
                "formant_stability_score": formant_stability_score
            },
            "ratio_details": {
                "mean_f2_f1": mean_f2_f1,
                "mean_f3_f2": mean_f3_f2,
                "ratio_abnormality_score": ratio_abnormality_score
            }
        }


def get_formant_constraint_handler() -> FormantConstraintHandler:
    """
    Создает и возвращает обработчик формантных ограничений.

    Returns:
        Экземпляр FormantConstraintHandler
    """
    return FormantConstraintHandler()
