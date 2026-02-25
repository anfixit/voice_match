"""
Модуль: formant_statistics.py
Описание: Статистический анализ формант и вычисление доверительных интервалов.
Обеспечивает статистическую обработку результатов формантного анализа и
предоставляет инструменты для оценки надежности полученных данных.
"""


import numpy as np
import scipy.stats

from voice_match.log import setup_logger

log = setup_logger("formant_statistics")


class FormantStatistics:
    """
    Класс для статистической обработки формантных данных
    и построения доверительных интервалов.
    """

    def __init__(self):
        """
        Инициализирует анализатор статистики формант.
        """
        # Уровень доверия по умолчанию для доверительных интервалов
        self.confidence_level = 0.95

    def compute_formant_statistics(self, formant_tracks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        """
        Вычисляет статистические характеристики формант с учетом надежности измерений.

        Args:
            formant_tracks: Словарь с треками формант и сопутствующей информацией

        Returns:
            Словарь со статистическими характеристиками для каждой форманты
        """
        statistics = {}

        # Базовые форманты для анализа
        formant_names = ["F1", "F2", "F3", "F4"]

        for formant_name in formant_names:
            # Проверка наличия данных
            if formant_name not in formant_tracks:
                continue

            # Получение данных форманты и надежности
            formant_values = formant_tracks[formant_name]
            reliability_key = f"{formant_name}_reliability"

            # Если есть информация о надежности, используем её как веса
            if reliability_key in formant_tracks:
                reliability_values = formant_tracks[reliability_key]

                # Применяем маску для валидных значений
                valid_mask = (formant_values > 0) & (reliability_values > 0)
                valid_formants = formant_values[valid_mask]
                valid_reliability = reliability_values[valid_mask]

                # Если нет валидных значений, пропускаем
                if len(valid_formants) == 0:
                    statistics[formant_name] = self._empty_statistics()
                    continue

                # Вычисление взвешенного среднего
                weighted_mean = np.average(valid_formants, weights=valid_reliability)

                # Вычисление взвешенной дисперсии
                weight_sum = np.sum(valid_reliability)
                weighted_variance = np.sum(valid_reliability * (valid_formants - weighted_mean) ** 2) / weight_sum
                weighted_std = np.sqrt(weighted_variance)

                # Вычисление доверительного интервала
                # Взвешенная эффективная выборка
                effective_sample_size = (np.sum(valid_reliability) ** 2) / np.sum(valid_reliability ** 2)

                # Стандартная ошибка для взвешенного среднего
                weighted_sem = weighted_std / np.sqrt(effective_sample_size)

                # t-критическое значение для заданного уровня доверия
                t_critical = scipy.stats.t.ppf((1 + self.confidence_level) / 2, effective_sample_size - 1)

                # Доверительный интервал
                ci_half_width = t_critical * weighted_sem
                ci_lower = weighted_mean - ci_half_width
                ci_upper = weighted_mean + ci_half_width

                # Дополнительные статистики
                formant_min = np.min(valid_formants)
                formant_max = np.max(valid_formants)
                formant_median = np.median(valid_formants)

                # Квартили
                q1 = np.percentile(valid_formants, 25)
                q3 = np.percentile(valid_formants, 75)

                # Робастная оценка стандартного отклонения (MAD)
                mad = np.median(np.abs(valid_formants - formant_median))

                # Интерквартильный размах
                iqr = q3 - q1

                # Асимметрия и эксцесс
                skewness = scipy.stats.skew(valid_formants)
                kurtosis = scipy.stats.kurtosis(valid_formants)

                # Средняя надежность
                mean_reliability = np.mean(valid_reliability)

                # Сохранение результатов
                statistics[formant_name] = {
                    "count": len(valid_formants),
                    "mean": weighted_mean,
                    "std": weighted_std,
                    "median": formant_median,
                    "min": formant_min,
                    "max": formant_max,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "mad": mad,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_upper - ci_lower,
                    "mean_reliability": mean_reliability,
                    "effective_sample_size": effective_sample_size
                }
            else:
                # Обработка без учета надежности
                valid_mask = formant_values > 0
                valid_formants = formant_values[valid_mask]

                # Если нет валидных значений, пропускаем
                if len(valid_formants) == 0:
                    statistics[formant_name] = self._empty_statistics()
                    continue

                # Базовые статистики
                formant_mean = np.mean(valid_formants)
                formant_std = np.std(valid_formants)
                formant_median = np.median(valid_formants)
                formant_min = np.min(valid_formants)
                formant_max = np.max(valid_formants)

                # Доверительный интервал
                n = len(valid_formants)
                sem = formant_std / np.sqrt(n)

                # t-критическое значение для заданного уровня доверия
                t_critical = scipy.stats.t.ppf((1 + self.confidence_level) / 2, n - 1)

                # Доверительный интервал
                ci_half_width = t_critical * sem
                ci_lower = formant_mean - ci_half_width
                ci_upper = formant_mean + ci_half_width

                # Квартили
                q1 = np.percentile(valid_formants, 25)
                q3 = np.percentile(valid_formants, 75)

                # Робастная оценка стандартного отклонения (MAD)
                mad = np.median(np.abs(valid_formants - formant_median))

                # Интерквартильный размах
                iqr = q3 - q1

                # Асимметрия и эксцесс
                skewness = scipy.stats.skew(valid_formants)
                kurtosis = scipy.stats.kurtosis(valid_formants)

                # Сохранение результатов
                statistics[formant_name] = {
                    "count": n,
                    "mean": formant_mean,
                    "std": formant_std,
                    "median": formant_median,
                    "min": formant_min,
                    "max": formant_max,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "mad": mad,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_upper - ci_lower,
                    "mean_reliability": 1.0,
                    "effective_sample_size": n
                }

        # Вычисление статистик отношений формант
        self._compute_formant_ratio_statistics(formant_tracks, statistics)

        return statistics

    def _empty_statistics(self) -> dict[str, float]:
        """
        Возвращает структуру с пустыми статистиками.

        Returns:
            Словарь с нулевыми значениями статистик
        """
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "mad": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_width": 0.0,
            "mean_reliability": 0.0,
            "effective_sample_size": 0
        }

    def _compute_formant_ratio_statistics(self, formant_tracks: dict[str, np.ndarray],
                                          statistics: dict[str, dict[str, float]]) -> None:
        """
        Вычисляет статистику соотношений формант и добавляет её в словарь статистик.

        Args:
            formant_tracks: Словарь с треками формант
            statistics: Словарь статистик для обновления
        """
        # Отношения формант, которые мы хотим вычислить
        formant_ratios = [
            ("F2/F1", "F2", "F1"),
            ("F3/F2", "F3", "F2"),
            ("F4/F3", "F4", "F3"),
            ("F3/F1", "F3", "F1")
        ]

        for ratio_name, numerator, denominator in formant_ratios:
            # Проверка наличия данных
            if numerator not in formant_tracks or denominator not in formant_tracks:
                continue

            # Получение данных
            numerator_values = formant_tracks[numerator]
            denominator_values = formant_tracks[denominator]

            # Маска для валидных значений
            valid_mask = (numerator_values > 0) & (denominator_values > 0)

            # Если есть информация о надежности, используем среднюю надежность
            numerator_rel_key = f"{numerator}_reliability"
            denominator_rel_key = f"{denominator}_reliability"

            if numerator_rel_key in formant_tracks and denominator_rel_key in formant_tracks:
                numerator_rel = formant_tracks[numerator_rel_key]
                denominator_rel = formant_tracks[denominator_rel_key]

                # Маска для надежных значений
                reliability_mask = (numerator_rel > 0) & (denominator_rel > 0)
                valid_mask = valid_mask & reliability_mask

                # Средняя надежность для отношения
                ratio_reliability = 0.5 * (numerator_rel[valid_mask] + denominator_rel[valid_mask])

                # Вычисление отношений
                ratios = numerator_values[valid_mask] / denominator_values[valid_mask]

                # Если нет валидных значений, пропускаем
                if len(ratios) == 0:
                    statistics[ratio_name] = self._empty_statistics()
                    continue

                # Вычисление взвешенного среднего
                weighted_mean = np.average(ratios, weights=ratio_reliability)

                # Вычисление взвешенной дисперсии
                weight_sum = np.sum(ratio_reliability)
                weighted_variance = np.sum(ratio_reliability * (ratios - weighted_mean) ** 2) / weight_sum
                weighted_std = np.sqrt(weighted_variance)

                # Вычисление доверительного интервала
                # Взвешенная эффективная выборка
                effective_sample_size = (np.sum(ratio_reliability) ** 2) / np.sum(ratio_reliability ** 2)

                # Стандартная ошибка для взвешенного среднего
                weighted_sem = weighted_std / np.sqrt(effective_sample_size)

                # t-критическое значение для заданного уровня доверия
                t_critical = scipy.stats.t.ppf((1 + self.confidence_level) / 2, effective_sample_size - 1)

                # Доверительный интервал
                ci_half_width = t_critical * weighted_sem
                ci_lower = weighted_mean - ci_half_width
                ci_upper = weighted_mean + ci_half_width

                # Дополнительные статистики
                ratio_min = np.min(ratios)
                ratio_max = np.max(ratios)
                ratio_median = np.median(ratios)

                # Квартили
                q1 = np.percentile(ratios, 25)
                q3 = np.percentile(ratios, 75)

                # Интерквартильный размах
                iqr = q3 - q1

                # Сохранение результатов
                statistics[ratio_name] = {
                    "count": len(ratios),
                    "mean": weighted_mean,
                    "std": weighted_std,
                    "median": ratio_median,
                    "min": ratio_min,
                    "max": ratio_max,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_upper - ci_lower,
                    "mean_reliability": np.mean(ratio_reliability),
                    "effective_sample_size": effective_sample_size
                }
            else:
                # Обработка без учета надежности
                # Вычисление отношений
                valid_ratios = numerator_values[valid_mask] / denominator_values[valid_mask]

                # Если нет валидных значений, пропускаем
                if len(valid_ratios) == 0:
                    statistics[ratio_name] = self._empty_statistics()
                    continue

                # Базовые статистики
                ratio_mean = np.mean(valid_ratios)
                ratio_std = np.std(valid_ratios)
                ratio_median = np.median(valid_ratios)
                ratio_min = np.min(valid_ratios)
                ratio_max = np.max(valid_ratios)

                # Доверительный интервал
                n = len(valid_ratios)
                sem = ratio_std / np.sqrt(n)

                # t-критическое значение для заданного уровня доверия
                t_critical = scipy.stats.t.ppf((1 + self.confidence_level) / 2, n - 1)

                # Доверительный интервал
                ci_half_width = t_critical * sem
                ci_lower = ratio_mean - ci_half_width
                ci_upper = ratio_mean + ci_half_width

                # Квартили
                q1 = np.percentile(valid_ratios, 25)
                q3 = np.percentile(valid_ratios, 75)

                # Интерквартильный размах
                iqr = q3 - q1

                # Сохранение результатов
                statistics[ratio_name] = {
                    "count": n,
                    "mean": ratio_mean,
                    "std": ratio_std,
                    "median": ratio_median,
                    "min": ratio_min,
                    "max": ratio_max,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_upper - ci_lower,
                    "mean_reliability": 1.0,
                    "effective_sample_size": n
                }

    def analyze_formant_stability(self, formant_tracks: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Анализирует стабильность формант во времени, что важно для обнаружения
        синтетического голоса или подделок.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с метриками стабильности
        """
        stability_metrics = {}

        # Форманты для анализа
        formant_names = ["F1", "F2", "F3", "F4"]

        for formant_name in formant_names:
            if formant_name not in formant_tracks:
                continue

            formant_values = formant_tracks[formant_name]

            # Маска для валидных значений
            valid_mask = formant_values > 0
            valid_formants = formant_values[valid_mask]

            if len(valid_formants) < 2:
                stability_metrics[f"{formant_name}_stability"] = 0.0
                continue

            # Коэффициент вариации - основная метрика стабильности
            cv = np.std(valid_formants) / np.mean(valid_formants) if np.mean(valid_formants) > 0 else 0

            # Относительное среднее абсолютное отклонение (RMAD)
            rmad = np.mean(np.abs(valid_formants - np.mean(valid_formants))) / np.mean(valid_formants) \
                if np.mean(valid_formants) > 0 else 0

            # Относительный интерквартильный размах (RIQR)
            q1 = np.percentile(valid_formants, 25)
            q3 = np.percentile(valid_formants, 75)
            iqr = q3 - q1
            riqr = iqr / np.median(valid_formants) if np.median(valid_formants) > 0 else 0

            # Вычисление автокорреляции для оценки временной зависимости
            if len(valid_formants) > 10:
                # Вычисляем автокорреляцию с лагом 1
                autocorr = np.corrcoef(valid_formants[:-1], valid_formants[1:])[0, 1]
            else:
                autocorr = 0.0

            # Добавляем метрики стабильности
            stability_metrics[f"{formant_name}_cv"] = cv
            stability_metrics[f"{formant_name}_rmad"] = rmad
            stability_metrics[f"{formant_name}_riqr"] = riqr
            stability_metrics[f"{formant_name}_autocorr"] = autocorr

            # Интегральная оценка стабильности (обратно пропорциональна вариабельности)
            # Чем ближе к 1, тем более стабильна форманта
            formant_stability = 1.0 - min(1.0, cv * 3.0)
            stability_metrics[f"{formant_name}_stability"] = formant_stability

        # Средняя стабильность всех формант
        formant_stability_values = [value for key, value in stability_metrics.items()
                                    if key.endswith("_stability")]

        if formant_stability_values:
            stability_metrics["overall_stability"] = np.mean(formant_stability_values)
        else:
            stability_metrics["overall_stability"] = 0.0

        return stability_metrics

    def compare_formant_profiles(self, profile1: dict[str, dict[str, float]],
                                 profile2: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Сравнивает статистические профили формант двух голосов.

        Args:
            profile1: Статистики формант первого голоса
            profile2: Статистики формант второго голоса

        Returns:
            Словарь с мерами сходства по разным аспектам
        """
        comparison_results = {}

        # Форманты и отношения для сравнения
        formant_keys = ["F1", "F2", "F3", "F4", "F2/F1", "F3/F2", "F3/F1"]

        for key in formant_keys:
            # Проверка наличия данных в обоих профилях
            if key not in profile1 or key not in profile2:
                continue

            stats1 = profile1[key]
            stats2 = profile2[key]

            # Если нет достаточно данных, пропускаем
            if stats1["count"] < 5 or stats2["count"] < 5:
                comparison_results[f"{key}_similarity"] = 0.0
                continue

            # 1. Сравнение средних значений
            # Масштабированное абсолютное различие
            mean_diff = abs(stats1["mean"] - stats2["mean"])

            # Нормализация разницы относительно диапазона возможных значений
            # Для формант и отношений используем разные методы нормализации
            if key in ["F1", "F2", "F3", "F4"]:
                # Для формант используем % различие от среднего
                mean_avg = (stats1["mean"] + stats2["mean"]) / 2
                norm_mean_diff = mean_diff / mean_avg if mean_avg > 0 else 1.0

                # Сходство по средним (обратно пропорционально различию)
                mean_similarity = 1.0 - min(1.0, norm_mean_diff * 2.0)  # Ограничиваем до 1.0
            else:
                # Для отношений используем абсолютную разницу
                norm_mean_diff = min(1.0, mean_diff / 1.0)  # Ограничиваем до 1.0 при разнице 1.0

                # Сходство по средним
                mean_similarity = 1.0 - norm_mean_diff

            # 2. Проверка перекрытия доверительных интервалов
            # Вычисляем перекрытие интервалов
            ci1 = (stats1["ci_lower"], stats1["ci_upper"])
            ci2 = (stats2["ci_lower"], stats2["ci_upper"])

            # Перекрытие = max(0, min(upper) - max(lower))
            overlap = max(0, min(ci1[1], ci2[1]) - max(ci1[0], ci2[0]))

            # Общая длина объединенного интервала
            total_range = max(ci1[1], ci2[1]) - min(ci1[0], ci2[0])

            # Доля перекрытия
            overlap_ratio = overlap / total_range if total_range > 0 else 0

            # 3. Сравнение распределений (с помощью KL-дивергенции)
            # Мы не имеем полных распределений, но можем использовать моменты
            # (среднее, стд, асимметрия, эксцесс) для приближенной оценки

            # Сравнение стандартных отклонений
            std_ratio = min(stats1["std"], stats2["std"]) / max(stats1["std"], stats2["std"]) \
                if max(stats1["std"], stats2["std"]) > 0 else 1.0

            # Взвешенное итоговое сходство для этой характеристики
            # Больший вес среднему значению, меньший - перекрытию интервалов и стд
            similarity = 0.6 * mean_similarity + 0.3 * overlap_ratio + 0.1 * std_ratio

            # Сохраняем результат
            comparison_results[f"{key}_similarity"] = similarity

        # Вычисление общего сходства на основе всех характеристик
        # Разные веса для разных характеристик
        weights = {
            "F1_similarity": 1.0,
            "F2_similarity": 1.0,
            "F3_similarity": 1.2,  # F3 более постоянна для человека
            "F4_similarity": 0.8,
            "F2/F1_similarity": 1.5,  # Отношения особенно важны
            "F3/F2_similarity": 1.3,
            "F3/F1_similarity": 1.2
        }

        weighted_sum = 0.0
        weight_sum = 0.0

        for key, weight in weights.items():
            if key in comparison_results:
                weighted_sum += comparison_results[key] * weight
                weight_sum += weight

        # Общая оценка сходства
        if weight_sum > 0:
            comparison_results["overall"] = weighted_sum / weight_sum
        else:
            comparison_results["overall"] = 0.0

        return comparison_results

    def detect_formant_outliers(self, formant_tracks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Обнаруживает выбросы в формантных треках и возвращает маски выбросов.

        Args:
            formant_tracks: Словарь с треками формант

        Returns:
            Словарь с масками выбросов для каждой форманты
        """
        outlier_masks = {}

        # Форманты для анализа
        formant_names = ["F1", "F2", "F3", "F4"]

        for formant_name in formant_names:
            if formant_name not in formant_tracks:
                continue

            formant_values = formant_tracks[formant_name]

            # Маска для валидных значений
            valid_mask = formant_values > 0
            valid_indices = np.where(valid_mask)[0]
            valid_formants = formant_values[valid_mask]

            if len(valid_formants) < 5:  # Недостаточно данных
                outlier_masks[formant_name] = np.zeros_like(formant_values, dtype=bool)
                continue

            # Робастное определение выбросов с помощью MAD
            # (Median Absolute Deviation) - более устойчиво к выбросам, чем std
            median = np.median(valid_formants)
            mad = np.median(np.abs(valid_formants - median))

            # MAD к стандартному отклонению для нормального распределения
            mad_to_std = 1.4826  # Константа для нормального распределения

            # Порог для выбросов (по умолчанию 3 стандартных отклонения)
            threshold = 3.0 * mad_to_std * mad

            # Определение выбросов
            upper_bound = median + threshold
            lower_bound = median - threshold

            # Маска выбросов для валидных значений
            valid_outliers = (valid_formants < lower_bound) | (valid_formants > upper_bound)

            # Маска выбросов для всех значений
            outlier_mask = np.zeros_like(formant_values, dtype=bool)
            outlier_mask[valid_indices[valid_outliers]] = True

            outlier_masks[formant_name] = outlier_mask

        return outlier_masks

    def set_confidence_level(self, level: float) -> None:
        """
        Устанавливает уровень доверия для доверительных интервалов.

        Args:
            level: Уровень доверия (0-1), например 0.95 для 95% интервалов
        """
        if 0 < level < 1:
            self.confidence_level = level
        else:
            log.warning(f"Неверный уровень доверия: {level}. Используется значение по умолчанию 0.95.")
            self.confidence_level = 0.95


def get_formant_statistics() -> FormantStatistics:
    """
    Создает и возвращает анализатор статистики формант.

    Returns:
        Экземпляр FormantStatistics
    """
    return FormantStatistics()
