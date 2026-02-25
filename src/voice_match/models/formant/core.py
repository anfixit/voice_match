"""
Модуль: formant_core.py
Описание: Базовые функции для LPC-анализа и извлечения формант для системы voice_match.
Содержит основные инструменты для спектрального анализа и извлечения формант из речевого сигнала.
"""


import librosa
import numpy as np
import scipy.linalg
import scipy.signal

from voice_match.log import setup_logger

log = setup_logger("formant_core")


class FormantAnalyzer:
    """
    Класс для анализа формант речевого сигнала.
    Форманты - это резонансные частоты голосового тракта, уникальные биометрические
    характеристики человека, которые сложно подделать.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Инициализирует анализатор формант.

        Args:
            sample_rate: Частота дискретизации в Гц
        """
        self.sample_rate = sample_rate
        # Параметры временного окна анализа
        self.frame_length = int(0.025 * sample_rate)  # 25 мс
        self.hop_length = int(0.010 * sample_rate)  # 10 мс

        # Диапазоны допустимых значений формант (в Гц)
        # Диапазоны разделены по мужскому и женскому голосу
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
            # Детские голоса (для полноты анализа)
            "child": {
                "F1": (350, 1200),  # Первая форманта
                "F2": (1000, 3000),  # Вторая форманта
                "F3": (2500, 4000),  # Третья форманта
                "F4": (3700, 5500)  # Четвертая форманта
            }
        }

        # Параметры LPC-анализа
        self.lpc_order_range = (12, 36)  # Диапазон порядков LPC для адаптивного анализа
        self.lpc_window_types = ['hamming', 'hann', 'blackman', 'bartlett']  # Типы окон

    def adaptive_lpc_order(self, frame: np.ndarray) -> int:
        """
        Адаптивно определяет оптимальный порядок LPC-анализа для конкретного фрейма
        на основе спектрального анализа и критерия Акаике.

        Алгоритм учитывает:
        1. Спектральную сложность фрейма
        2. Соотношение сигнал/шум
        3. Количество значимых пиков в спектре

        Args:
            frame: Фрейм аудиосигнала

        Returns:
            Оптимальный порядок LPC-анализа
        """
        frame_len = len(frame)
        max_order = min(self.lpc_order_range[1], frame_len // 3)
        min_order = self.lpc_order_range[0]

        # Применяем оконную функцию для FFT
        frame_windowed = frame * np.hamming(frame_len)

        # Получаем спектр и находим локальные максимумы
        fft_spectrum = np.abs(np.fft.rfft(frame_windowed))
        fft_freqs = np.fft.rfftfreq(frame_len, 1.0 / self.sample_rate)

        # Количество значимых пиков в спектре
        significant_peaks = 0
        if len(fft_spectrum) > 3:  # Проверка на минимальную длину
            # Нормализация спектра для обнаружения пиков
            normalized_spectrum = fft_spectrum / np.max(fft_spectrum) if np.max(fft_spectrum) > 0 else fft_spectrum
            # Находим пики (только в полосе 100-5000 Гц)
            freq_mask = (fft_freqs >= 100) & (fft_freqs <= 5000)
            speech_spectrum = normalized_spectrum[freq_mask]

            if len(speech_spectrum) > 3:  # Снова проверяем длину
                # Находим пики с амплитудой выше порога
                peaks, _ = scipy.signal.find_peaks(speech_spectrum, height=0.15, distance=3)
                significant_peaks = len(peaks)

        # Базовое правило для определения порядка LPC в зависимости от количества пиков
        # 2 пика на форманту + 4-6 для общей формы спектра
        initial_order = min(max_order, max(min_order, 4 + 2 * significant_peaks))

        # Проверяем несколько порядков вокруг initial_order с шагом 2
        orders_to_test = list(range(
            max(min_order, initial_order - 4),
            min(max_order, initial_order + 6),
            2
        ))

        best_aic = float('inf')
        best_order = initial_order

        # Применяем критерий Акаике для выбора оптимального порядка
        for order in orders_to_test:
            # Вычисляем коэффициенты LPC
            lpc_coeffs = librosa.lpc(frame_windowed, order=order)

            # Вычисляем ошибку предсказания
            pred_error = self._compute_prediction_error(frame_windowed, lpc_coeffs)

            # Критерий Акаике: AIC = N * log(MSE) + 2 * p
            # где N - длина сигнала, MSE - средняя квадратичная ошибка, p - порядок LPC
            mse = np.mean(pred_error ** 2) if len(pred_error) > 0 else float('inf')
            aic = frame_len * np.log(mse) + 2 * order

            # Добавляем штраф для слишком высоких порядков, чтобы избежать переобучения
            overfitting_penalty = 0.05 * (order - min_order) ** 2
            aic += overfitting_penalty

            if aic < best_aic:
                best_aic = aic
                best_order = order

        # Логирование для отладки
        log.debug('Адаптивный LPC порядок: %s (пики: %s)', best_order, significant_peaks)

        return best_order

    def _compute_prediction_error(self, frame: np.ndarray, lpc_coeffs: np.ndarray) -> np.ndarray:
        """
        Вычисляет ошибку предсказания для данного фрейма и LPC-коэффициентов.
        Используется улучшенный алгоритм для повышения точности.

        Args:
            frame: Фрейм аудиосигнала
            lpc_coeffs: LPC-коэффициенты

        Returns:
            Массив ошибок предсказания
        """
        # Создаем фильтр из LPC-коэффициентов
        n_lpc = len(lpc_coeffs)
        n_frame = len(frame)

        if n_lpc > n_frame:
            # Обработка краевого случая
            return np.array([])

        # Создаем буфер для предсказанного сигнала
        pred_signal = np.zeros(n_frame)

        # Блочная реализация для повышения эффективности
        for i in range(n_lpc - 1, n_frame):
            # Вектор предыдущих значений в обратном порядке
            history = frame[i - (n_lpc - 1):i + 1][::-1]
            # Умножение вектора на коэффициенты
            pred_signal[i] = -np.sum(lpc_coeffs[1:] * history[:-1])

        # Ограничиваем ошибку только валидной частью
        valid_indices = np.arange(n_lpc - 1, n_frame)
        error = frame[valid_indices] - pred_signal[valid_indices]

        return error

    def compute_lpc_with_multiple_windows(self, frame: np.ndarray, order: int) -> list[tuple[np.ndarray, float]]:
        """
        Вычисляет LPC-коэффициенты с разными оконными функциями
        для повышения робастности анализа. Возвращает коэффициенты и качество модели.

        Args:
            frame: Фрейм аудиосигнала
            order: Порядок LPC-анализа

        Returns:
            Список кортежей (коэффициенты LPC, качество модели)
        """
        lpc_results = []
        frame_len = len(frame)

        for window_type in self.lpc_window_types:
            # Выбор оконной функции
            if window_type == 'hamming':
                window = np.hamming(frame_len)
            elif window_type == 'hann':
                window = np.hann(frame_len)
            elif window_type == 'blackman':
                window = np.blackman(frame_len)
            elif window_type == 'bartlett':
                window = np.bartlett(frame_len)
            else:
                window = np.hamming(frame_len)  # По умолчанию

            # Применяем окно
            frame_windowed = frame * window

            # Вычисляем LPC-коэффициенты
            lpc_coeffs = librosa.lpc(frame_windowed, order=order)

            # Оценка качества модели
            pred_error = self._compute_prediction_error(frame_windowed, lpc_coeffs)
            if len(pred_error) > 0:
                # Нормализованная ошибка предсказания (меньше = лучше)
                mse = np.mean(pred_error ** 2)
                frame_energy = np.mean(frame_windowed ** 2)
                normalized_error = mse / (frame_energy + 1e-10)

                # Проверка на стабильность фильтра (все полюса должны быть внутри единичного круга)
                roots = np.roots(lpc_coeffs)
                is_stable = np.all(np.abs(roots) < 1.0)

                # Качество модели (обратно пропорционально ошибке, с учетом стабильности)
                model_quality = (1.0 / (1.0 + 20.0 * normalized_error)) * (0.8 if is_stable else 0.3)

                lpc_results.append((lpc_coeffs, model_quality))
            else:
                # Если не удалось вычислить ошибку, качество низкое
                lpc_results.append((lpc_coeffs, 0.1))

        # Сортировка по качеству (от лучшего к худшему)
        lpc_results.sort(key=lambda x: x[1], reverse=True)

        return lpc_results

    def extract_formants_from_lpc(self, lpc_coeffs: np.ndarray, gender: str = 'male',
                                  model_quality: float = 1.0) -> dict[str, float | dict]:
        """
        Извлекает частоты формант из LPC-коэффициентов с учетом пола говорящего.
        Улучшенная версия с оценкой надежности и дополнительными параметрами.

        Args:
            lpc_coeffs: LPC-коэффициенты
            gender: Пол говорящего ('male', 'female' или 'child')
            model_quality: Качество LPC-модели (от 0 до 1)

        Returns:
            Словарь с частотами формант F1-F4, их полосами пропускания и оценками надежности
        """
        # Нахождение корней характеристического полинома
        roots = np.roots(lpc_coeffs)

        # Выбираем только корни с положительной мнимой частью (комплексно-сопряженные пары)
        roots = roots[np.imag(roots) > 0]

        if len(roots) == 0:
            # Если нет подходящих корней, возвращаем пустые значения
            empty_result = {
                "F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0,
                "F1_bandwidth": 0.0, "F2_bandwidth": 0.0,
                "F3_bandwidth": 0.0, "F4_bandwidth": 0.0,
                "F1_reliability": 0.0, "F2_reliability": 0.0,
                "F3_reliability": 0.0, "F4_reliability": 0.0,
                "formant_count": 0,
                "model_quality": model_quality
            }
            return empty_result

        # Вычисляем частоты и полосы пропускания
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * self.sample_rate / (2 * np.pi)

        # Вычисляем ширину полосы для каждого резонанса
        # Ширина полосы связана с затуханием корня в Z-плоскости
        radii = np.abs(roots)
        bandwidths = -np.log(radii) * self.sample_rate / np.pi

        # Сортировка по возрастанию частот
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        bandwidths = bandwidths[idx]
        radii = radii[idx]

        # Оценка надежности каждого формантного пика
        # Более узкие форманты (малая полоса) и более выраженные (большой радиус) более надежны
        reliabilities = 1.0 - bandwidths / 500.0  # Чем меньше полоса, тем выше надежность
        reliabilities = np.clip(reliabilities, 0.1, 0.95)  # Ограничение в разумных пределах

        # Корректировка надежности с учетом качества модели
        reliabilities *= model_quality

        # Структура для результатов с начальными нулевыми значениями
        formants = {
            "F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0,
            "F1_bandwidth": 0.0, "F2_bandwidth": 0.0,
            "F3_bandwidth": 0.0, "F4_bandwidth": 0.0,
            "F1_reliability": 0.0, "F2_reliability": 0.0,
            "F3_reliability": 0.0, "F4_reliability": 0.0,
            "formant_count": 0,
            "model_quality": model_quality
        }

        # Используем соответствующие диапазоны в зависимости от пола
        # Если пол неизвестен или не соответствует ключам, используем мужской по умолчанию
        ranges = self.formant_ranges.get(gender, self.formant_ranges["male"])

        # Создаем список кандидатов в форманты
        formant_candidates = []
        for i, (freq, bw, reliability) in enumerate(zip(freqs, bandwidths, reliabilities, strict=False)):
            # Проверяем, что полоса пропускания в разумных пределах (не слишком широкая)
            if bw < 500:  # ограничение на слишком широкие форманты
                for formant_name, (min_freq, max_freq) in ranges.items():
                    if min_freq <= freq <= max_freq:
                        # Добавляем кандидата с именем форманты и мерой соответствия
                        # Мера соответствия зависит от положения в диапазоне и полосы
                        center_freq = (min_freq + max_freq) / 2
                        rel_dist = abs(freq - center_freq) / ((max_freq - min_freq) / 2)
                        match_score = (1.0 - rel_dist) * reliability

                        formant_candidates.append({
                            "formant_name": formant_name,
                            "frequency": freq,
                            "bandwidth": bw,
                            "reliability": reliability,
                            "match_score": match_score,
                            "original_index": i
                        })
                        break

        # Сортировка кандидатов по формантам и соответствию
        formant_candidates.sort(key=lambda x: (x["formant_name"], -x["match_score"]))

        # Выбираем лучшего кандидата для каждой форманты
        assigned_formants = set()
        for candidate in formant_candidates:
            formant_name = candidate["formant_name"]
            if formant_name not in assigned_formants:
                formants[formant_name] = candidate["frequency"]
                formants[f"{formant_name}_bandwidth"] = candidate["bandwidth"]
                formants[f"{formant_name}_reliability"] = candidate["reliability"]
                assigned_formants.add(formant_name)
                formants["formant_count"] += 1

                # Прекращаем если нашли все нужные форманты
                if len(assigned_formants) == 4:  # F1-F4
                    break

        # Проверка на физиологическую согласованность формант
        if (formants["F1"] > 0 and formants["F2"] > 0 and formants["F1"] >= formants["F2"]) or \
                (formants["F2"] > 0 and formants["F3"] > 0 and formants["F2"] >= formants["F3"]) or \
                (formants["F3"] > 0 and formants["F4"] > 0 and formants["F3"] >= formants["F4"]):
            # Если нарушен порядок формант, снижаем надежность
            for i in range(1, 5):
                formant_name = f"F{i}"
                if formants[formant_name] > 0:
                    formants[f"{formant_name}_reliability"] *= 0.5

        return formants

    def evaluate_spectrum_quality(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """
        Оценивает качество спектра для формантного анализа.

        Args:
            spectrum: Спектр сигнала
            freqs: Соответствующие частоты

        Returns:
            Оценка качества от 0 до 1
        """
        # Нормализация спектра
        norm_spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

        # Выделение речевого диапазона (200-5000 Hz)
        speech_mask = (freqs >= 200) & (freqs <= 5000)
        speech_spectrum = norm_spectrum[speech_mask]
        freqs[speech_mask]

        if len(speech_spectrum) == 0:
            return 0.0

        # Проверка на наличие явных пиков (формантная структура)
        peaks, properties = scipy.signal.find_peaks(
            speech_spectrum, height=0.1, distance=5, prominence=0.05
        )

        # Оценка количества и качества пиков
        if len(peaks) == 0:
            return 0.1  # Слишком мало пиков

        # Средняя амплитуда и ширина пиков
        mean_prominence = np.mean(properties["prominences"]) if "prominences" in properties else 0.0

        # Оценка соотношения сигнал/шум
        spectral_snr = mean_prominence / (np.mean(speech_spectrum) + 1e-10)

        # Итоговая оценка качества
        quality = min(1.0,
                      0.2 + 0.4 * spectral_snr +
                      0.2 * min(1.0, len(peaks) / 8) +  # Поощряем наличие 3-4 пиков (форманты)
                      0.2 * (mean_prominence / 0.3)  # Нормализация к ожидаемой выраженности
                      )

        return quality

    def estimate_gender(self, y: np.ndarray) -> dict[str, float]:
        """
        Оценивает вероятный пол говорящего на основе распределения основного тона
        и формантной структуры. Возвращает вероятности для каждой категории.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с вероятностями для разных полов и возрастов
        """
        # Оценка основного тона (pitch, F0)
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=50, fmax=500  # Расширенный диапазон для разных полов/возрастов
        )

        # Извлечение питча с максимальной магнитудой для каждого фрейма
        pitch_values = []
        for t in range(magnitudes.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            magnitude = magnitudes[index, t]

            if pitch > 0 and magnitude > 0:
                pitch_values.append(pitch)

        # Если не удалось определить питч, считаем мужской голос с низким уровнем уверенности
        if not pitch_values:
            return {"male": 0.6, "female": 0.3, "child": 0.1}

        # Статистический анализ распределения основного тона
        pitch_values = np.array(pitch_values)
        median_pitch = np.median(pitch_values)
        mean_pitch = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)

        # Создаем гистограмму питча
        hist, bin_edges = np.histogram(pitch_values, bins=20, range=(50, 500))
        hist_prob = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        # Находим моду (пик распределения)
        mode_bin = np.argmax(hist_prob)
        mode_pitch = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2

        # Вычисление вероятностей на основе статистик
        # Типичные диапазоны F0:
        # Мужской: 85-180 Hz (медиана ~120 Hz)
        # Женский: 160-255 Hz (медиана ~210 Hz)
        # Детский: 250-400 Hz (медиана ~300 Hz)

        # Вероятности на основе медианы питча
        male_prob_from_median = np.exp(-0.5 * ((median_pitch - 120) / 30) ** 2)
        female_prob_from_median = np.exp(-0.5 * ((median_pitch - 210) / 30) ** 2)
        child_prob_from_median = np.exp(-0.5 * ((median_pitch - 300) / 40) ** 2)

        # Вероятности на основе моды питча
        male_prob_from_mode = np.exp(-0.5 * ((mode_pitch - 120) / 30) ** 2)
        female_prob_from_mode = np.exp(-0.5 * ((mode_pitch - 210) / 30) ** 2)
        child_prob_from_mode = np.exp(-0.5 * ((mode_pitch - 300) / 40) ** 2)

        # Учет вариабельности (стабильность питча)
        # Детский голос часто более вариабельный
        variability_factor = pitch_std / mean_pitch if mean_pitch > 0 else 0
        child_bonus = max(0, min(0.2, variability_factor - 0.15))  # Бонус для детского при высокой вариабельности

        # Комбинирование вероятностей с весами
        probs = {
            "male": 0.6 * male_prob_from_median + 0.4 * male_prob_from_mode,
            "female": 0.6 * female_prob_from_median + 0.4 * female_prob_from_mode,
            "child": 0.5 * child_prob_from_median + 0.3 * child_prob_from_mode + 0.2 * child_bonus
        }

        # Нормализация вероятностей
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        else:
            probs = {"male": 0.6, "female": 0.3, "child": 0.1}  # По умолчанию

        return probs


def get_formant_analyzer() -> FormantAnalyzer:
    """
    Создает и возвращает экземпляр анализатора формант.

    Returns:
        Экземпляр FormantAnalyzer
    """
    return FormantAnalyzer()
