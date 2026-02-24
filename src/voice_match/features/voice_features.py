import numpy as np
import librosa
import scipy.signal
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union
import os
import tempfile
from voice_match.log import setup_logger

from voice_match.constants import (
    FRAME_DURATION_S,
    HOP_DURATION_S,
    JITTER_NORMAL_RANGE,
    PITCH_RANGE_FEMALE,
    PITCH_RANGE_MALE,
    SHIMMER_NORMAL_RANGE,
)


log = setup_logger("voice_features")


class VoiceFeatureExtractor:
    """
    Извлекает специализированные голосовые биометрические характеристики для судебной фоноскопии.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Инициализирует экстрактор биометрических признаков голоса.

        Args:
            sample_rate: Частота дискретизации в Гц
        """
        self.sample_rate = sample_rate
        # Настройки для анализа
        self.frame_length = int(FRAME_DURATION_S * sample_rate)
        self.hop_length = int(HOP_DURATION_S * sample_rate)

        # Диапазоны голосовых характеристик для взрослого мужчины/женщины
        self.pitch_range = {
            'male': PITCH_RANGE_MALE,
            'female': PITCH_RANGE_FEMALE,
        }

        # Нормальный диапазон джиттера (в %) для здорового голоса
        self.jitter_normal_range = JITTER_NORMAL_RANGE

        # Нормальный диапазон шиммера (в %) для здорового голоса
        self.shimmer_normal_range = SHIMMER_NORMAL_RANGE

    def extract_all_features(self, y: np.ndarray) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Извлекает полный набор голосовых характеристик для биометрической идентификации.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с извлеченными характеристиками
        """
        features = {}

        # 1. Извлечение характеристик основного тона (F0)
        pitch_features = self.extract_pitch_features(y)
        features["pitch"] = pitch_features

        # 2. Извлечение джиттера и шиммера
        jitter_shimmer = self.extract_jitter_shimmer(y)
        features["jitter_shimmer"] = jitter_shimmer

        # 3. Извлечение характеристик фрикативных звуков
        fricative_features = self.extract_fricative_features(y)
        features["fricative"] = fricative_features

        # 4. Извлечение характеристик носовых звуков
        nasal_features = self.extract_nasal_features(y)
        features["nasal"] = nasal_features

        # 5. Вычисление перцептивных характеристик
        perceptual_features = self.extract_perceptual_features(y)
        features["perceptual"] = perceptual_features

        # 6. Оценка качества голоса (ненапряженный/напряженный, хриплый и т.д.)
        voice_quality = self.assess_voice_quality(y, pitch_features, jitter_shimmer)
        features["voice_quality"] = voice_quality

        # 7. Темпоральные характеристики (паузация, скорость речи)
        temporal_features = self.extract_temporal_features(y)
        features["temporal"] = temporal_features

        # 8. Биометрический вектор признаков
        biometric_vector = self.create_biometric_vector(
            pitch_features,
            jitter_shimmer,
            fricative_features,
            nasal_features,
            voice_quality
        )
        features["biometric_vector"] = biometric_vector

        return features

    def extract_pitch_features(self, y: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Извлекает характеристики основного тона (pitch, F0).

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с характеристиками основного тона
        """
        # 1. Алгоритм YIN для детекции pitch
        pitches, voiced_flags, voiced_probs = librosa.pyin(
            y,
            fmin=50,
            fmax=400,
            sr=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        # Фильтрация только вокализованных фреймов с высокой уверенностью
        reliable_pitches = pitches[voiced_flags & (voiced_probs > 0.7)]

        # Если нет надежно определенных питчей
        if len(reliable_pitches) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "values": np.array([]),
                "times": np.array([]),
                "gender_probability": {"male": 0.5, "female": 0.5}
            }

        # 2. Базовые статистические характеристики
        pitch_stats = {
            "mean": np.mean(reliable_pitches),
            "median": np.median(reliable_pitches),
            "std": np.std(reliable_pitches),
            "min": np.min(reliable_pitches),
            "max": np.max(reliable_pitches),
            "range": np.max(reliable_pitches) - np.min(reliable_pitches),
            "values": reliable_pitches,
            "times": librosa.frames_to_time(
                np.where(voiced_flags & (voiced_probs > 0.7))[0],
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        }

        # 3. Определение вероятного пола по основному тону
        mean_pitch = pitch_stats["mean"]

        male_range = self.pitch_range["male"]
        female_range = self.pitch_range["female"]

        # Вероятность принадлежности к мужскому/женскому голосу
        male_prob = 0.0
        female_prob = 0.0

        if male_range[0] <= mean_pitch <= male_range[1]:
            # В мужском диапазоне
            male_prob = 1.0
        elif female_range[0] <= mean_pitch <= female_range[1]:
            # В женском диапазоне
            female_prob = 1.0
        elif mean_pitch > male_range[1] and mean_pitch < female_range[0]:
            # В переходной зоне
            total_range = female_range[0] - male_range[1]
            distance_from_male = mean_pitch - male_range[1]

            male_prob = 1.0 - (distance_from_male / total_range)
            female_prob = distance_from_male / total_range
        elif mean_pitch < male_range[0]:
            # Ниже мужского диапазона - вероятно мужской
            male_prob = 0.9
            female_prob = 0.1
        elif mean_pitch > female_range[1]:
            # Выше женского диапазона - вероятно женский
            male_prob = 0.1
            female_prob = 0.9

        pitch_stats["gender_probability"] = {
            "male": male_prob,
            "female": female_prob
        }

        # 4. Вычисление контуров pitch
        # Для судебной экспертизы важны характерные мелодические контуры

        # Вычисление первой производной (скорость изменения)
        pitch_derivative = np.gradient(reliable_pitches)

        # Вычисление второй производной (ускорение изменения)
        pitch_second_derivative = np.gradient(pitch_derivative)

        # Характеристики контуров
        pitch_stats["contour"] = {
            "derivative_mean": np.mean(np.abs(pitch_derivative)),
            "derivative_std": np.std(pitch_derivative),
            "acceleration_mean": np.mean(np.abs(pitch_second_derivative)),
            "acceleration_std": np.std(pitch_second_derivative),
            "pitch_reset": self._compute_pitch_reset(reliable_pitches)
        }

        return pitch_stats

    def _compute_pitch_reset(self, pitches: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет характеристики сброса основного тона.
        Сброс тона - индивидуальная характеристика говорящего.

        Args:
            pitches: Массив значений основного тона

        Returns:
            Словарь с характеристиками сброса тона
        """
        if len(pitches) < 10:
            return {"magnitude": 0.0, "rate": 0.0, "frequency": 0.0}

        # Поиск точек сброса (значительное падение питча)
        reset_threshold = 0.1 * (np.max(pitches) - np.min(pitches))
        reset_points = []

        for i in range(1, len(pitches)):
            drop = pitches[i - 1] - pitches[i]
            if drop > reset_threshold:
                reset_points.append((i, drop))

        # Анализ сбросов
        if not reset_points:
            return {"magnitude": 0.0, "rate": 0.0, "frequency": 0.0}

        # Средняя величина сброса
        reset_magnitudes = [drop for _, drop in reset_points]

        # Средняя скорость сброса (изменение F0 / длительность)
        # Предполагаем, что сброс происходит за 2 фрейма
        frame_duration = self.hop_length / self.sample_rate
        reset_rates = [mag / (2 * frame_duration) for mag in reset_magnitudes]

        return {
            "magnitude": np.mean(reset_magnitudes),
            "rate": np.mean(reset_rates),
            "frequency": len(reset_points) / len(pitches)
        }

    def extract_jitter_shimmer(self, y: np.ndarray) -> Dict[str, float]:
        """
        Извлекает параметры микроколебаний голоса: джиттер и шиммер.
        Эти параметры являются важными биометрическими характеристиками.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с параметрами джиттера и шиммера
        """
        # Детекция питча
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=50,
            fmax=400
        )

        # Выделение основного питча для каждого фрейма
        pitch_values = []
        magnitude_values = []

        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            magnitude = magnitudes[index, t]

            # Учитываем только вокализованные фреймы
            if pitch > 0 and magnitude > 0:
                pitch_values.append(pitch)
                magnitude_values.append(magnitude)

        # Если недостаточно вокализованных фреймов
        if len(pitch_values) < 5:
            return {
                "local_jitter": 0.0,
                "absolute_jitter": 0.0,
                "ppq5": 0.0,
                "jitter_variability": 0.0,
                "local_shimmer": 0.0,
                "absolute_shimmer": 0.0,
                "apq5": 0.0,
                "shimmer_variability": 0.0
            }

        # 1. Вычисление джиттера

        # Преобразование частоты в периоды
        periods = 1.0 / np.array(pitch_values)
        period_diffs = np.abs(np.diff(periods))

        # Локальный джиттер (%)
        local_jitter = 100 * np.mean(period_diffs) / np.mean(periods)

        # Абсолютный джиттер (мс)
        absolute_jitter = 1000 * np.mean(period_diffs)

        # PPQ5 (5-point Period Perturbation Quotient)
        ppq5_values = []
        for i in range(2, len(periods) - 2):
            avg_period = np.mean(periods[i - 2:i + 3])
            ppq5_values.append(abs(periods[i] - avg_period))

        ppq5 = 100 * np.mean(ppq5_values) / np.mean(periods) if ppq5_values else 0

        # Вариабельность джиттера
        jitter_variability = 100 * np.std(period_diffs) / np.mean(periods)

        # 2. Вычисление шиммера

        # Преобразование магнитуд в амплитуды
        amplitudes = np.array(magnitude_values)
        amplitude_diffs = np.abs(np.diff(amplitudes))

        # Локальный шиммер (%)
        local_shimmer = 100 * np.mean(amplitude_diffs) / np.mean(amplitudes)

        # Абсолютный шиммер (дБ)
        # Преобразование в дБ для корректного вычисления
        db_values = 20 * np.log10(amplitudes / np.mean(amplitudes))
        db_diffs = np.abs(np.diff(db_values))
        absolute_shimmer = np.mean(db_diffs)

        # APQ5 (5-point Amplitude Perturbation Quotient)
        apq5_values = []
        for i in range(2, len(amplitudes) - 2):
            avg_amp = np.mean(amplitudes[i - 2:i + 3])
            apq5_values.append(abs(amplitudes[i] - avg_amp))

        apq5 = 100 * np.mean(apq5_values) / np.mean(amplitudes) if apq5_values else 0

        # Вариабельность шиммера
        shimmer_variability = 100 * np.std(amplitude_diffs) / np.mean(amplitudes)

        return {
            "local_jitter": local_jitter,
            "absolute_jitter": absolute_jitter,
            "ppq5": ppq5,
            "jitter_variability": jitter_variability,
            "local_shimmer": local_shimmer,
            "absolute_shimmer": absolute_shimmer,
            "apq5": apq5,
            "shimmer_variability": shimmer_variability
        }

    def extract_fricative_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Извлекает характеристики фрикативных согласных.
        Спектральные особенности фрикативных звуков (ш, с, ф, в и т.д.)
        зависят от анатомии речевого аппарата говорящего.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с характеристиками фрикативных звуков
        """
        # STFT для анализа в частотной области
        stft = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))

        # Частотные диапазоны для разных фрикативных
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)

        # Диапазоны фрикативных звуков
        # 1. s, z (2000-4000 Hz)
        s_z_mask = (freq_bins >= 2000) & (freq_bins <= 4000)

        # 2. sh, zh (4000-8000 Hz)
        sh_zh_mask = (freq_bins >= 4000) & (freq_bins <= 8000)

        # 3. f, v (1000-2000 Hz)
        f_v_mask = (freq_bins >= 1000) & (freq_bins <= 2000)

        # Вычисление средней энергии в каждом диапазоне
        s_z_energy = np.mean(stft[s_z_mask, :], axis=0)
        sh_zh_energy = np.mean(stft[sh_zh_mask, :], axis=0)
        f_v_energy = np.mean(stft[f_v_mask, :], axis=0)

        # Общая энергия
        total_energy = np.mean(stft, axis=0)

        # Фильтрация для выделения фреймов с фрикативными звуками
        # Пороги выбраны эмпирически
        s_z_frames = s_z_energy > 0.2 * np.max(s_z_energy)
        sh_zh_frames = sh_zh_energy > 0.2 * np.max(sh_zh_energy)
        f_v_frames = f_v_energy > 0.2 * np.max(f_v_energy)

        # Вычисление отношений для фреймов с фрикативными
        s_z_ratio = np.zeros_like(s_z_energy)
        sh_zh_ratio = np.zeros_like(sh_zh_energy)
        f_v_ratio = np.zeros_like(f_v_energy)

        # Маска ненулевой общей энергии
        nonzero_mask = total_energy > 0

        s_z_ratio[nonzero_mask] = s_z_energy[nonzero_mask] / total_energy[nonzero_mask]
        sh_zh_ratio[nonzero_mask] = sh_zh_energy[nonzero_mask] / total_energy[nonzero_mask]
        f_v_ratio[nonzero_mask] = f_v_energy[nonzero_mask] / total_energy[nonzero_mask]

        # Статистика только для фреймов с фрикативными
        s_z_stats = {
            "mean_energy": np.mean(s_z_energy[s_z_frames]) if np.any(s_z_frames) else 0,
            "mean_ratio": np.mean(s_z_ratio[s_z_frames]) if np.any(s_z_frames) else 0,
            "std_ratio": np.std(s_z_ratio[s_z_frames]) if np.any(s_z_frames) else 0,
            "frame_count": np.sum(s_z_frames)
        }

        sh_zh_stats = {
            "mean_energy": np.mean(sh_zh_energy[sh_zh_frames]) if np.any(sh_zh_frames) else 0,
            "mean_ratio": np.mean(sh_zh_ratio[sh_zh_frames]) if np.any(sh_zh_frames) else 0,
            "std_ratio": np.std(sh_zh_ratio[sh_zh_frames]) if np.any(sh_zh_frames) else 0,
            "frame_count": np.sum(sh_zh_frames)
        }

        f_v_stats = {
            "mean_energy": np.mean(f_v_energy[f_v_frames]) if np.any(f_v_frames) else 0,
            "mean_ratio": np.mean(f_v_ratio[f_v_frames]) if np.any(f_v_frames) else 0,
            "std_ratio": np.std(f_v_ratio[f_v_frames]) if np.any(f_v_frames) else 0,
            "frame_count": np.sum(f_v_frames)
        }

        # Вычисление спектрального центроида для фрикативных фреймов
        all_fricative_frames = s_z_frames | sh_zh_frames | f_v_frames

        spectral_centroid = librosa.feature.spectral_centroid(
            S=stft, sr=self.sample_rate
        )[0]

        centroid_mean = np.mean(spectral_centroid[all_fricative_frames]) if np.any(all_fricative_frames) else 0
        centroid_std = np.std(spectral_centroid[all_fricative_frames]) if np.any(all_fricative_frames) else 0

        # Объединение статистик
        return {
            "s_z": s_z_stats,
            "sh_zh": sh_zh_stats,
            "f_v": f_v_stats,
            "centroid_mean": centroid_mean,
            "centroid_std": centroid_std,
            "overall_fricative_ratio": np.sum(all_fricative_frames) / len(all_fricative_frames)
        }

    def extract_nasal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Извлекает характеристики носовых звуков (м, н).
        Носовой резонанс - важная индивидуальная характеристика голоса.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с характеристиками носовых звуков
        """
        # STFT для спектрального анализа
        stft = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))

        # Частотные диапазоны
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)

        # Диапазоны носовых резонансов
        # 1. Основной носовой резонанс: 250-450 Hz
        nasal1_mask = (freq_bins >= 250) & (freq_bins <= 450)

        # 2. Второй носовой резонанс: 1000-1200 Hz
        nasal2_mask = (freq_bins >= 1000) & (freq_bins <= 1200)

        # 3. Антирезонанс: 700-900 Hz (провал энергии)
        antires_mask = (freq_bins >= 700) & (freq_bins <= 900)

        # Вычисление энергии в каждом диапазоне
        nasal1_energy = np.mean(stft[nasal1_mask, :], axis=0)
        nasal2_energy = np.mean(stft[nasal2_mask, :], axis=0)
        antires_energy = np.mean(stft[antires_mask, :], axis=0)

        # Общая энергия
        total_energy = np.mean(stft, axis=0)

        # Фильтрация для выделения фреймов с носовыми звуками
        # Для носовых характерен сильный первый резонанс и провал антирезонанса
        nonzero_mask = total_energy > 0

        nasal1_ratio = np.zeros_like(nasal1_energy)
        nasal2_ratio = np.zeros_like(nasal2_energy)
        antires_ratio = np.zeros_like(antires_energy)

        nasal1_ratio[nonzero_mask] = nasal1_energy[nonzero_mask] / total_energy[nonzero_mask]
        nasal2_ratio[nonzero_mask] = nasal2_energy[nonzero_mask] / total_energy[nonzero_mask]
        antires_ratio[nonzero_mask] = antires_energy[nonzero_mask] / total_energy[nonzero_mask]

        # Детекция носовых звуков - высокий первый резонанс и низкий антирезонанс
        nasal_mask = (nasal1_ratio > 1.5 * np.mean(nasal1_ratio)) & (antires_ratio < 0.7 * np.mean(antires_ratio))

        # Если нет обнаруженных носовых звуков
        if not np.any(nasal_mask):
            return {
                "nasal1_resonance_mean": np.mean(nasal1_ratio),
                "nasal2_resonance_mean": np.mean(nasal2_ratio),
                "antiresonance_mean": np.mean(antires_ratio),
                "nasal_presence": 0.0,
                "nasal_quality": 0.0,
                "nasal_resonance_ratio": 0.0
            }

        # Статистика только для носовых фреймов
        nasal1_resonance_mean = np.mean(nasal1_ratio[nasal_mask])
        nasal2_resonance_mean = np.mean(nasal2_ratio[nasal_mask])
        antiresonance_mean = np.mean(antires_ratio[nasal_mask])

        # Доля носовых фреймов (measure of nasal presence)
        nasal_presence = np.sum(nasal_mask) / len(nasal_mask)

        # Качество назализации (отношение первого резонанса к антирезонансу)
        nasal_quality = np.mean(nasal1_ratio[nasal_mask] / (antires_ratio[nasal_mask] + 1e-10))

        # Отношение первого и второго резонансов (индивидуальная характеристика)
        nasal_resonance_ratio = np.mean(nasal1_ratio[nasal_mask] / (nasal2_ratio[nasal_mask] + 1e-10))

        return {
            "nasal1_resonance_mean": nasal1_resonance_mean,
            "nasal2_resonance_mean": nasal2_resonance_mean,
            "antiresonance_mean": antiresonance_mean,
            "nasal_presence": nasal_presence,
            "nasal_quality": nasal_quality,
            "nasal_resonance_ratio": nasal_resonance_ratio
        }

    def extract_perceptual_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Извлекает перцептивные (воспринимаемые) характеристики голоса.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с перцептивными характеристиками
        """
        # Мел-спектрограмма - приближена к человеческому восприятию
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=128,
            fmin=50,
            fmax=8000
        )

        # Преобразование в дБ
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Усредненный мел-спектр
        mean_mel_spec = np.mean(mel_spec_db, axis=1)

        # 1. Спектральный баланс (соотношение НЧ и ВЧ компонентов)
        low_freq = np.mean(mean_mel_spec[:43])  # 0-1000 Hz
        mid_freq = np.mean(mean_mel_spec[43:86])  # 1000-4000 Hz
        high_freq = np.mean(mean_mel_spec[86:])  # 4000-8000 Hz

        spectral_balance = {
            "low_energy": low_freq,
            "mid_energy": mid_freq,
            "high_energy": high_freq,
            "low_to_high_ratio": low_freq / (high_freq + 1e-10),
            "mid_to_high_ratio": mid_freq / (high_freq + 1e-10)
        }

        # 2. Спектральная яркость
        brightness = np.sum(mean_mel_spec[86:]) / np.sum(mean_mel_spec)

        # 3. Спектральный спад (rolloff)
        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=self.sample_rate,
            roll_percent=0.85
        )[0]

        # 4. Спектральный центроид
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]

        # 5. Контрастность спектра
        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=self.sample_rate,
            n_bands=6
        )

        # Средние значения контрастности по полосам
        mean_contrast = np.mean(contrast, axis=1)

        # Общая перцептивная оценка характеристик
        perceptual_features = {
            "spectral_balance": spectral_balance,
            "brightness": brightness,
            "rolloff_mean": np.mean(rolloff),
            "rolloff_std": np.std(rolloff),
            "centroid_mean": np.mean(centroid),
            "centroid_std": np.std(centroid),
            "contrast_mean": mean_contrast.tolist(),

            # Производные характеристики
            "warmth": low_freq / (mid_freq + high_freq + 1e-10),
            "timbre_richness": np.mean(mean_contrast)
        }

        return perceptual_features

    def assess_voice_quality(self, y: np.ndarray,
                             pitch_features: Dict[str, Union[float, np.ndarray]],
                             jitter_shimmer: Dict[str, float]) -> Dict[str, float]:
        """
        Оценивает качество голоса и его специфические характеристики.

        Args:
            y: Аудиосигнал
            pitch_features: Характеристики основного тона
            jitter_shimmer: Параметры джиттера и шиммера

        Returns:
            Словарь с оценками качества голоса
        """
        # 1. Хриплость (breathiness)
        # Высокий джиттер и шиммер, высокий шум в высоких частотах
        breathiness = 0.0

        if "local_jitter" in jitter_shimmer and "local_shimmer" in jitter_shimmer:
            # Нормализация параметров джиттер/шиммер к диапазону 0-1
            norm_jitter = min(1.0, jitter_shimmer["local_jitter"] / self.jitter_normal_range[1])
            norm_shimmer = min(1.0, jitter_shimmer["local_shimmer"] / self.shimmer_normal_range[1])

            # Оценка хриплости
            breathiness = 0.5 * norm_jitter + 0.5 * norm_shimmer

        # 2. Скрипучесть (creakiness/vocal fry)
        # Характеризуется низким питчем и нерегулярной амплитудной модуляцией
        creakiness = 0.0

        if "mean" in pitch_features and "std" in pitch_features:
            # Низкий средний питч
            low_pitch_factor = 1.0 if pitch_features["mean"] < 100 else max(0.0, 1.0 - (
                        pitch_features["mean"] - 100) / 50)

            # Высокая вариативность питча
            high_variability = min(1.0, pitch_features["std"] / 20.0)

            creakiness = 0.6 * low_pitch_factor + 0.4 * high_variability

        # 3. Напряженность (tenseness)
        # Характеризуется высоким питчем, высоким спектральным наклоном
        tenseness = 0.0

        if "mean" in pitch_features:
            # Высокий питч для соответствующего пола
            gender_prob = pitch_features.get("gender_probability", {"male": 0.5, "female": 0.5})

            if gender_prob["male"] > gender_prob["female"]:
                # Для мужского голоса
                high_pitch_factor = min(1.0, max(0.0, (pitch_features["mean"] - 100) / 80))
            else:
                # Для женского голоса
                high_pitch_factor = min(1.0, max(0.0, (pitch_features["mean"] - 180) / 120))

            # Спектральный наклон (быстрое убывание амплитуды с частотой)
            spectral_slope = librosa.feature.spectral_slope(y=y)[0]
            steep_slope_factor = min(1.0, abs(np.mean(spectral_slope)) * 10)

            tenseness = 0.5 * high_pitch_factor + 0.5 * steep_slope_factor

        # 4. Назальность (nasality)
        # Высокая энергия в носовом резонансе (250-450 Hz)
        nasality = 0.0

        # Вычисление носового резонанса
        s = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)

        # Носовой диапазон 250-450 Hz
        nasal_mask = (freqs >= 250) & (freqs <= 450)

        # Энергия в носовом диапазоне относительно общей энергии
        nasal_energy = np.mean(s[nasal_mask, :])
        total_energy = np.mean(s)

        if total_energy > 0:
            nasality = min(1.0, nasal_energy / total_energy * 3)

        # 5. Степень звонкости (voicing degree)
        # Доля вокализованных фреймов
        voicing_degree = 0.0

        if "is_voiced" in pitch_features:
            voiced_frames = np.sum(pitch_features["is_voiced"])
            total_frames = len(pitch_features["is_voiced"])

            if total_frames > 0:
                voicing_degree = voiced_frames / total_frames

        # 6. Стабильность голоса (voice stability)
        # Низкий джиттер и шиммер, стабильный питч
        stability = 0.0

        if "local_jitter" in jitter_shimmer and "local_shimmer" in jitter_shimmer and "std" in pitch_features:
            # Инвертируем показатели нестабильности
            jitter_stability = 1.0 - min(1.0, jitter_shimmer["local_jitter"] / self.jitter_normal_range[1])
            shimmer_stability = 1.0 - min(1.0, jitter_shimmer["local_shimmer"] / self.shimmer_normal_range[1])

            # Нормализованная стабильность питча
            pitch_stability = 1.0 - min(1.0, pitch_features["std"] / (0.2 * pitch_features["mean"]))

            stability = 0.3 * jitter_stability + 0.3 * shimmer_stability + 0.4 * pitch_stability

        return {
            "breathiness": breathiness,
            "creakiness": creakiness,
            "tenseness": tenseness,
            "nasality": nasality,
            "voicing_degree": voicing_degree,
            "stability": stability
        }

    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Извлекает темпоральные характеристики речи.

        Args:
            y: Аудиосигнал

        Returns:
            Словарь с темпоральными характеристиками
        """
        # Обнаружение речевой активности (VAD)
        # Вычисление энергии
        energy = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        energy_threshold = 0.05 * np.max(energy)

        # Детекция речи
        speech_frames = energy > energy_threshold

        # Преобразование к временным отметкам
        frame_times = librosa.frames_to_time(
            np.arange(len(speech_frames)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # Если нет речи
        if not np.any(speech_frames):
            return {
                "speech_rate": 0.0,
                "speech_ratio": 0.0,
                "pause_ratio": 1.0,
                "mean_phrase_duration": 0.0,
                "mean_pause_duration": 0.0,
                "speaking_tempo": 0.0
            }

        # 1. Определение фраз (последовательностей речевых фреймов)
        phrase_start_idxs = np.where(np.diff(np.concatenate(([0], speech_frames.astype(int)))) == 1)[0]
        phrase_end_idxs = np.where(np.diff(np.concatenate((speech_frames.astype(int), [0]))) == -1)[0]

        # 2. Определение пауз (последовательностей неречевых фреймов)
        pause_start_idxs = np.where(np.diff(np.concatenate(([1], speech_frames.astype(int)))) == -1)[0]
        pause_end_idxs = np.where(np.diff(np.concatenate((speech_frames.astype(int), [1]))) == 1)[0]

        # 3. Вычисление длительностей фраз и пауз
        phrase_durations = frame_times[phrase_end_idxs] - frame_times[phrase_start_idxs]
        pause_durations = frame_times[pause_end_idxs] - frame_times[pause_start_idxs]

        # 4. Основные темпоральные характеристики

        # Доля речи
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        pause_ratio = 1.0 - speech_ratio

        # Средняя длительность фраз и пауз
        mean_phrase_duration = np.mean(phrase_durations) if len(phrase_durations) > 0 else 0
        mean_pause_duration = np.mean(pause_durations) if len(pause_durations) > 0 else 0

        # Скорость речи (фраз в секунду)
        total_duration = frame_times[-1]
        speech_rate = len(phrase_durations) / total_duration if total_duration > 0 else 0

        # Темп речи (отношение времени речи к общему времени)
        speaking_tempo = speech_ratio * (1.0 / mean_phrase_duration) if mean_phrase_duration > 0 else 0

        return {
            "speech_rate": speech_rate,
            "speech_ratio": speech_ratio,
            "pause_ratio": pause_ratio,
            "mean_phrase_duration": mean_phrase_duration,
            "mean_pause_duration": mean_pause_duration,
            "speaking_tempo": speaking_tempo,
            "phrase_count": len(phrase_durations),
            "pause_count": len(pause_durations)
        }

    def create_biometric_vector(self,
                                    pitch_features: Dict[str, Union[float, np.ndarray]],
                                    jitter_shimmer: Dict[str, float],
                                    fricative_features: Dict[str, Union[float, dict]],
                                    nasal_features: Dict[str, float],
                                    voice_quality: Dict[str, float]) -> np.ndarray:
        """
        Создает единый биометрический вектор признаков для идентификации говорящего.

        Args:
            pitch_features: Характеристики основного тона
            jitter_shimmer: Параметры джиттера и шиммера
            fricative_features: Характеристики фрикативных звуков
            nasal_features: Характеристики носовых звуков
            voice_quality: Оценки качества голоса

        Returns:
            Биометрический вектор признаков
        """
        # Компоненты вектора биометрических признаков
        biometric_vector = []

        # 1. Признаки основного тона
        if "mean" in pitch_features and "median" in pitch_features and "std" in pitch_features:
            biometric_vector.extend([
                pitch_features["mean"],
                pitch_features["median"],
                pitch_features["std"],
                pitch_features.get("contour", {}).get("derivative_mean", 0.0),
                pitch_features.get("contour", {}).get("acceleration_mean", 0.0)
            ])
        else:
            biometric_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 2. Признаки джиттера и шиммера (микроколебания)
        biometric_vector.extend([
            jitter_shimmer.get("local_jitter", 0.0),
            jitter_shimmer.get("absolute_jitter", 0.0),
            jitter_shimmer.get("ppq5", 0.0),
            jitter_shimmer.get("local_shimmer", 0.0),
            jitter_shimmer.get("absolute_shimmer", 0.0),
            jitter_shimmer.get("apq5", 0.0)
        ])

        # 3. Признаки фрикативных звуков
        if "s_z" in fricative_features and "sh_zh" in fricative_features:
            biometric_vector.extend([
                fricative_features["s_z"].get("mean_ratio", 0.0),
                fricative_features["sh_zh"].get("mean_ratio", 0.0),
                fricative_features["centroid_mean"]
            ])
        else:
            biometric_vector.extend([0.0, 0.0, 0.0])

        # 4. Признаки носовых звуков
        biometric_vector.extend([
            nasal_features.get("nasal1_resonance_mean", 0.0),
            nasal_features.get("nasal2_resonance_mean", 0.0),
            nasal_features.get("nasal_resonance_ratio", 0.0)
        ])

        # 5. Признаки качества голоса
        biometric_vector.extend([
            voice_quality.get("breathiness", 0.0),
            voice_quality.get("creakiness", 0.0),
            voice_quality.get("tenseness", 0.0),
            voice_quality.get("nasality", 0.0),
            voice_quality.get("stability", 0.0)
        ])

        return np.array(biometric_vector)

    def compare_voice_features(self, features1: Dict[str, any], features2: Dict[str, any]) -> Dict[str, float]:
        """
        Сравнивает характеристики двух голосов и возвращает меры сходства.

        Args:
            features1: Характеристики первого голоса
            features2: Характеристики второго голоса

        Returns:
            Словарь с мерами сходства по разным аспектам
        """
        # Сходство по биометрическим векторам
        biometric_sim = 0.0

        if "biometric_vector" in features1 and "biometric_vector" in features2:
            v1 = features1["biometric_vector"]
            v2 = features2["biometric_vector"]

            # Косинусное сходство
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                biometric_sim = np.dot(v1, v2) / (norm1 * norm2)

        # Сходство по основному тону
        pitch_sim = self._compare_pitch_features(features1.get("pitch", {}), features2.get("pitch", {}))

        # Сходство по джиттеру/шиммеру
        jitter_shimmer_sim = self._compare_jitter_shimmer(
            features1.get("jitter_shimmer", {}),
            features2.get("jitter_shimmer", {})
        )

        # Сходство по фрикативным
        fricative_sim = self._compare_fricative_features(
            features1.get("fricative", {}),
            features2.get("fricative", {})
        )

        # Сходство по носовым
        nasal_sim = self._compare_nasal_features(
            features1.get("nasal", {}),
            features2.get("nasal", {})
        )

        # Сходство по качеству голоса
        voice_quality_sim = self._compare_voice_quality(
            features1.get("voice_quality", {}),
            features2.get("voice_quality", {})
        )

        # Общая оценка сходства (взвешенная)
        weights = {
            "biometric": 1.5,
            "pitch": 1.0,
            "jitter_shimmer": 1.2,
            "fricative": 0.8,
            "nasal": 1.0,
            "voice_quality": 0.7
        }

        similarities = {
            "biometric": biometric_sim,
            "pitch": pitch_sim,
            "jitter_shimmer": jitter_shimmer_sim,
            "fricative": fricative_sim,
            "nasal": nasal_sim,
            "voice_quality": voice_quality_sim
        }

        # Вычисление взвешенного среднего
        weighted_sum = 0.0
        weight_sum = 0.0

        for key, value in similarities.items():
            weight = weights.get(key, 1.0)
            weighted_sum += value * weight
            weight_sum += weight

        overall_similarity = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Итоговые результаты с общей оценкой
        similarities["overall"] = overall_similarity

        return similarities

    def _compare_pitch_features(self, pitch1: Dict[str, any], pitch2: Dict[str, any]) -> float:
        """
        Сравнивает характеристики основного тона двух голосов.

        Args:
            pitch1: Характеристики основного тона первого голоса
            pitch2: Характеристики основного тона второго голоса

        Returns:
            Мера сходства по основному тону
        """
        if not pitch1 or not pitch2:
            return 0.0

        # Сходство по средним значениям питча
        mean_sim = 0.0

        if "mean" in pitch1 and "mean" in pitch2:
            p1 = pitch1["mean"]
            p2 = pitch2["mean"]

            if p1 > 0 and p2 > 0:
                # Относительная разница
                rel_diff = abs(p1 - p2) / max(p1, p2)
                mean_sim = 1.0 - min(1.0, 2.0 * rel_diff)

        # Сходство по вариативности питча
        variability_sim = 0.0

        if "std" in pitch1 and "std" in pitch2 and "mean" in pitch1 and "mean" in pitch2:
            # Нормализованные стандартные отклонения (коэффициент вариации)
            cv1 = pitch1["std"] / pitch1["mean"] if pitch1["mean"] > 0 else 0
            cv2 = pitch2["std"] / pitch2["mean"] if pitch2["mean"] > 0 else 0

            cv_diff = abs(cv1 - cv2)
            variability_sim = 1.0 - min(1.0, 5.0 * cv_diff)

        # Сходство по контурам питча
        contour_sim = 0.0

        if "contour" in pitch1 and "contour" in pitch2:
            c1 = pitch1["contour"]
            c2 = pitch2["contour"]

            if "derivative_mean" in c1 and "derivative_mean" in c2:
                deriv_diff = abs(c1["derivative_mean"] - c2["derivative_mean"])
                deriv_sim = 1.0 - min(1.0, deriv_diff / 5.0)

                accel_diff = abs(c1.get("acceleration_mean", 0) - c2.get("acceleration_mean", 0))
                accel_sim = 1.0 - min(1.0, accel_diff / 10.0)

                reset_diff = abs(c1.get("pitch_reset", {}).get("frequency", 0) -
                                 c2.get("pitch_reset", {}).get("frequency", 0))
                reset_sim = 1.0 - min(1.0, 2.0 * reset_diff)

                contour_sim = (deriv_sim + accel_sim + reset_sim) / 3.0

        # Общее сходство питча (взвешенное)
        return 0.5 * mean_sim + 0.3 * variability_sim + 0.2 * contour_sim

    def _compare_jitter_shimmer(self, js1: Dict[str, float], js2: Dict[str, float]) -> float:
        """
        Сравнивает джиттер и шиммер двух голосов.

        Args:
            js1: Параметры джиттера и шиммера первого голоса
            js2: Параметры джиттера и шиммера второго голоса

        Returns:
            Мера сходства по джиттеру и шиммеру
        """
        if not js1 or not js2:
            return 0.0

        # Сравнение локального джиттера
        jitter_sim = 0.0

        if "local_jitter" in js1 and "local_jitter" in js2:
            j1 = js1["local_jitter"]
            j2 = js2["local_jitter"]

            # Абсолютная разница, нормализованная к допустимому диапазону
            jitter_diff = abs(j1 - j2) / self.jitter_normal_range[1]
            jitter_sim = 1.0 - min(1.0, jitter_diff)

        # Сравнение локального шиммера
        shimmer_sim = 0.0

        if "local_shimmer" in js1 and "local_shimmer" in js2:
            s1 = js1["local_shimmer"]
            s2 = js2["local_shimmer"]

            # Аналогично джиттеру
            shimmer_diff = abs(s1 - s2) / self.shimmer_normal_range[1]
            shimmer_sim = 1.0 - min(1.0, shimmer_diff)

        # Сравнение по другим параметрам
        ppq5_sim = 0.0
        apq5_sim = 0.0

        if "ppq5" in js1 and "ppq5" in js2:
            ppq5_diff = abs(js1["ppq5"] - js2["ppq5"]) / 2.0  # Нормализация к типичному диапазону
            ppq5_sim = 1.0 - min(1.0, ppq5_diff)

        if "apq5" in js1 and "apq5" in js2:
            apq5_diff = abs(js1["apq5"] - js2["apq5"]) / 4.0  # Нормализация к типичному диапазону
            apq5_sim = 1.0 - min(1.0, apq5_diff)

        # Взвешенное среднее
        return 0.3 * jitter_sim + 0.3 * shimmer_sim + 0.2 * ppq5_sim + 0.2 * apq5_sim

    def _compare_fricative_features(self, fric1: Dict[str, any], fric2: Dict[str, any]) -> float:
        """
        Сравнивает характеристики фрикативных звуков двух голосов.

        Args:
            fric1: Характеристики фрикативных первого голоса
            fric2: Характеристики фрикативных второго голоса

        Returns:
            Мера сходства по фрикативным
        """
        if not fric1 or not fric2:
            return 0.0

        # Сравнение соотношений энергии в разных типах фрикативных
        s_z_sim = 0.0
        sh_zh_sim = 0.0

        if "s_z" in fric1 and "s_z" in fric2:
            # Сравнение средних соотношений
            s_z_ratio_diff = abs(
                fric1["s_z"].get("mean_ratio", 0.0) -
                fric2["s_z"].get("mean_ratio", 0.0)
            )
            s_z_sim = 1.0 - min(1.0, 5.0 * s_z_ratio_diff)

        if "sh_zh" in fric1 and "sh_zh" in fric2:
            sh_zh_ratio_diff = abs(
                fric1["sh_zh"].get("mean_ratio", 0.0) -
                fric2["sh_zh"].get("mean_ratio", 0.0)
            )
            sh_zh_sim = 1.0 - min(1.0, 5.0 * sh_zh_ratio_diff)

        # Сравнение спектрального центроида
        centroid_sim = 0.0

        if "centroid_mean" in fric1 and "centroid_mean" in fric2:
            cent_diff = abs(fric1["centroid_mean"] - fric2["centroid_mean"])
            centroid_sim = 1.0 - min(1.0, cent_diff / 1000.0)  # Нормализация к типичному диапазону разницы

        # Взвешенное среднее
        return 0.35 * s_z_sim + 0.35 * sh_zh_sim + 0.3 * centroid_sim

    def _compare_nasal_features(self, nasal1: Dict[str, float], nasal2: Dict[str, float]) -> float:
        """
        Сравнивает характеристики носовых звуков двух голосов.

        Args:
            nasal1: Характеристики носовых первого голоса
            nasal2: Характеристики носовых второго голоса

        Returns:
            Мера сходства по носовым
        """
        if not nasal1 or not nasal2:
            return 0.0

        # Сравнение резонансов
        res1_sim = 0.0

        if "nasal1_resonance_mean" in nasal1 and "nasal1_resonance_mean" in nasal2:
            res1_diff = abs(nasal1["nasal1_resonance_mean"] - nasal2["nasal1_resonance_mean"])
            res1_sim = 1.0 - min(1.0, 5.0 * res1_diff)

        # Сравнение отношения резонансов (особенно важно для биометрии)
        ratio_sim = 0.0

        if "nasal_resonance_ratio" in nasal1 and "nasal_resonance_ratio" in nasal2:
            ratio_diff = abs(nasal1["nasal_resonance_ratio"] - nasal2["nasal_resonance_ratio"])
            ratio_sim = 1.0 - min(1.0, 2.0 * ratio_diff)

        # Сравнение общей назализации
        nasal_sim = 0.0

        if "nasal_quality" in nasal1 and "nasal_quality" in nasal2:
            nasal_diff = abs(nasal1["nasal_quality"] - nasal2["nasal_quality"])
            nasal_sim = 1.0 - min(1.0, 2.0 * nasal_diff)

        # Взвешенное среднее
        return 0.3 * res1_sim + 0.5 * ratio_sim + 0.2 * nasal_sim

    def _compare_voice_quality(self, vq1: Dict[str, float], vq2: Dict[str, float]) -> float:
        """
        Сравнивает качества голоса двух говорящих.

        Args:
            vq1: Качества голоса первого говорящего
            vq2: Качества голоса второго говорящего

        Returns:
            Мера сходства по качеству голоса
        """
        if not vq1 or not vq2:
            return 0.0

        # Сравнение по каждому параметру качества
        similarities = []

        for param in ["breathiness", "creakiness", "tenseness", "nasality", "stability"]:
            if param in vq1 and param in vq2:
                diff = abs(vq1[param] - vq2[param])
                sim = 1.0 - min(1.0, 2.0 * diff)  # Двойной вес для разницы
                similarities.append(sim)

        # Среднее сходство
        return np.mean(similarities) if similarities else 0.0

def get_voice_feature_extractor():
    """
    Возвращает экземпляр экстрактора голосовых признаков.

    Returns:
        Экземпляр VoiceFeatureExtractor
    """
    return VoiceFeatureExtractor()
