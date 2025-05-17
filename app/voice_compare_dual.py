import os
import json
import librosa
import numpy as np
import tensorflow as tf
import torch
import webrtcvad
import soundfile as sf
import noisereduce as nr
import scipy.signal
import scipy.stats
import logging
from app.log import setup_logger
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union
from audiomentations import Compose, Normalize, BandPassFilter

from models.yamnet import get_yamnet
from models.xvector import get_xvector
from models.ecapa import get_ecapa
from models.resemblyzer import get_resemblyzer

# Импорты новых моделей
from models.antispoofing import get_antispoofing_detector
from models.formant_tracker import get_formant_tracker
from models.voice_features import get_voice_feature_extractor

# ─────────────────────── Логирование ───────────────────────
log = setup_logger("voice_match")

# ─────────────────────── Константы ───────────────────────
SAMPLE_RATE = 16000
SEGMENT_DURATION = 30.0
SEGMENT_COUNT = 5
CONFIDENCE_LEVEL = 0.95  # Уровень доверия для интервальных оценок
# Анатомические ограничения на форманты в Hz для взрослого мужчины
FORMANT_LIMITS = {
    "F1": (100, 1000),  # Гласные звуки
    "F2": (800, 2500),  # Гласные звуки
    "F3": (1500, 3500),  # Индивидуальные особенности тракта
    "F4": (3000, 5000)  # Анатомия голосового тракта, фиксировано для человека
}
# Диапазоны ожидаемого джиттера/шиммера для нормального голоса
JITTER_NORMAL_RANGE = (0.0, 1.04)  # в процентах
SHIMMER_NORMAL_RANGE = (0.0, 3.81)  # в процентах
# Типичные параметры голосовых модификаторов
VOICE_MODIFIERS = {
    "pitch_shift": (-12, 12),  # Полутоны
    "formant_shift": (0.5, 2.0),  # Коэффициент
    "robot": {"harmonic_intensity": (0.7, 1.0)}
}

# ─────────────────────── Аугментация ───────────────────────
augment = Compose([
    Normalize(p=1.0),
    BandPassFilter(min_center_freq=100, max_center_freq=7500, p=1.0),
])

# ─────────────────────── Весовые коэффициенты ───────────────────────
DEFAULT_WEIGHTS = {
    "ecapa": 1.5,  # Высокоуровневые речевые признаки
    "xvec": 1.5,  # Нейросетевые дискриминативные признаки
    "res": 1.2,  # Общее сходство голоса
    "formant": 1.0,  # Анатомические особенности
    "formant_dynamics": 1.5,  # Динамические особенности формант
    "fricative": 1.2,  # Фрикативные согласные
    "nasal": 1.3,  # Носовые резонансы
    "vocal_tract": 1.8,  # Длина голосового тракта
    "jitter_shimmer": 1.1,  # Микровариации голоса
    "yamnet": 0.8  # Перцептивные признаки
}

try:
    with open("../weights.json", "r") as f:
        weights = json.load(f)
        log.info("Весовые коэффициенты загружены из weights.json")
except Exception:
    weights = DEFAULT_WEIGHTS
    log.warning("weights.json не найден, используются значения по умолчанию")

# ─────────────────────── Детектор речи ───────────────────────
vad = webrtcvad.Vad(3)  # Максимальная чувствительность


# ─────────────────────── Модели (ленивая загрузка) ───────────────────────
@lru_cache(maxsize=1)
def lazy_ecapa():
    """Загружает ECAPA-TDNN модель при первом обращении и кэширует результат."""
    return get_ecapa()


@lru_cache(maxsize=1)
def lazy_xvector():
    """Загружает x-vector модель при первом обращении и кэширует результат."""
    return get_xvector()


@lru_cache(maxsize=1)
def lazy_yamnet():
    """Загружает YAMNet модель при первом обращении и кэширует результат."""
    return get_yamnet()


@lru_cache(maxsize=1)
def lazy_res():
    """Загружает Resemblyzer при первом обращении и кэширует результат."""
    return get_resemblyzer()


@lru_cache(maxsize=1)
def lazy_antispoofing():
    """Загружает детектор подделок голоса при первом обращении и кэширует результат."""
    from models.antispoofing import get_antispoofing_detector
    return get_antispoofing_detector()


@lru_cache(maxsize=1)
def lazy_formant_tracker():
    """Загружает трекер формант при первом обращении и кэширует результат."""
    from models.formant_tracker import get_formant_tracker
    return get_formant_tracker()


@lru_cache(maxsize=1)
def lazy_voice_features():
    """Загружает экстрактор голосовых характеристик при первом обращении и кэширует результат."""
    from models.voice_features import get_voice_feature_extractor
    return get_voice_feature_extractor()


# ─────────────────────── Обработка ───────────────────────
def preprocess(y: np.ndarray) -> np.ndarray:
    """
    Предобработка аудиосигнала:
    1. Нормализация амплитуды
    2. Шумоподавление
    3. Полосовая фильтрация голосового диапазона

    Args:
        y: Аудиосигнал в формате numpy array

    Returns:
        Обработанный аудиосигнал
    """
    # Нормализация амплитуды
    y = librosa.util.normalize(y)

    # Шумоподавление с сохранением речевых характеристик
    y = nr.reduce_noise(
        y=y,
        sr=SAMPLE_RATE,
        stationary=False,  # Нестационарный шум (более точно для реальных записей)
        prop_decrease=0.75  # Сохраняем 25% шума для достоверности
    )

    # Полосовая фильтрация в диапазоне человеческого голоса
    y = augment(samples=y, sample_rate=SAMPLE_RATE)

    return y


def get_segments(y: np.ndarray, sr: int, duration: float = SEGMENT_DURATION,
                 count: int = SEGMENT_COUNT) -> List[np.ndarray]:
    """
    Извлекает сегменты речи из аудиосигнала с перекрытием.
    Выбирает только сегменты с обнаруженной речью.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации
        duration: Длительность сегмента в секундах
        count: Максимальное количество сегментов

    Returns:
        Список сегментов с речью
    """
    window_size = int(sr * duration)
    hop = int(sr * duration / 2)  # 50% перекрытие
    segments = []

    # Вычисляем энергию сигнала для каждого фрейма
    energy = librosa.feature.rms(y=y)[0]
    energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy

    # Находим сегменты с высокой энергией
    energy_segments = []
    for start in range(0, len(y) - window_size, hop):
        end = start + window_size
        segment = y[start:end]

        # Проверяем наличие речи с помощью VAD
        if is_voiced(segment, sr):
            # Вычисляем среднюю энергию сегмента
            start_frame = start // hop
            end_frame = min(start_frame + (window_size // hop), len(energy_norm))
            mean_energy = np.mean(energy_norm[start_frame:end_frame])

            energy_segments.append((segment, mean_energy, start))

    # Сортируем сегменты по энергии (от высокой к низкой)
    energy_segments.sort(key=lambda x: x[1], reverse=True)

    # Берем сегменты с наибольшей энергией, но стараемся выбрать
    # из разных частей записи (не подряд идущие)
    selected_segments = []
    selected_starts = set()

    for segment, _, start in energy_segments:
        # Проверяем, не перекрывается ли с уже выбранными
        is_overlapping = False
        for sel_start in selected_starts:
            if abs(start - sel_start) < window_size // 2:
                is_overlapping = True
                break

        if not is_overlapping:
            selected_segments.append(segment)
            selected_starts.add(start)

            if len(selected_segments) >= count:
                break

    # Если нашли меньше сегментов, чем нужно, используем все что есть
    if len(selected_segments) < count and energy_segments:
        remaining = count - len(selected_segments)
        for segment, _, _ in energy_segments:
            if segment not in selected_segments:
                selected_segments.append(segment)
                if len(selected_segments) >= count:
                    break

    return selected_segments


def is_voiced(segment: np.ndarray, sr: int) -> bool:
    """
    Определяет наличие речи в сегменте с помощью WebRTC VAD.

    Args:
        segment: Сегмент аудиосигнала
        sr: Частота дискретизации

    Returns:
        True если обнаружена речь, иначе False
    """
    # Преобразуем в 16-битный формат для VAD
    int16_audio = (segment * 32767).astype(np.int16)

    # Размер фрейма для VAD (30 мс рекомендовано)
    frame_size = int(sr * 30 / 1000)
    voice_frames = 0
    total_frames = 0

    # Анализируем фреймы
    for i in range(0, len(int16_audio) - frame_size, frame_size):
        frame = int16_audio[i:i + frame_size].tobytes()
        if vad.is_speech(frame, sr):
            voice_frames += 1
        total_frames += 1

    # Если более 15% фреймов содержат речь, считаем сегмент речевым
    return voice_frames > 0.15 * total_frames if total_frames > 0 else False


def detect_voice_modification(y: np.ndarray, sr: int) -> Tuple[bool, Optional[str]]:
    """
    Обнаруживает признаки использования голосовых модификаторов.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        (is_modified, modifier_type): Флаг модификации и тип модификатора
    """
    # Извлекаем основные признаки
    pitched_segments = 0
    total_segments = 0
    frame_length = 2048
    hop_length = 512

    # Анализ основного тона
    pitches, magnitudes = librosa.core.piptrack(
        y=y, sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=50,
        fmax=400
    )

    # Ищем неестественные скачки основного тона
    pitch_changes = []
    prev_pitch = None

    # Для каждого фрейма находим максимальную магнитуду
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]

        # Если тон определен (не ноль)
        if pitch > 0:
            if prev_pitch is not None and prev_pitch > 0:
                # Вычисляем изменение в полутонах
                semitones = 12 * np.log2(pitch / prev_pitch)
                pitch_changes.append(abs(semitones))
            prev_pitch = pitch
            pitched_segments += 1
        total_segments += 1

    # Проверяем признаки механического изменения голоса
    if pitched_segments > 0:
        # 1. Неестественные скачки основного тона
        if pitch_changes and np.percentile(pitch_changes, 95) > 5.0:
            return True, "pitch_shift"

        # 2. Признаки "робота": слишком стабильный тон
        if np.std(pitch_changes) < 0.1 and pitched_segments > 0.5 * total_segments:
            return True, "robot_voice"

        # 3. Искусственная модуляция формант
        formants = extract_formants_advanced(y, sr)
        if formants is not None:
            f1_std = np.std(formants["F1"]) if formants["F1"].size > 0 else 0
            f2_std = np.std(formants["F2"]) if formants["F2"].size > 0 else 0

            # Слишком стабильные форманты - признак Voice Changer'а
            if (f1_std < 10 or f2_std < 20) and pitched_segments > 0.5 * total_segments:
                return True, "formant_modification"

    return False, None


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами.

    Args:
        v1: Первый вектор
        v2: Второй вектор

    Returns:
        Значение косинусного сходства от -1 до 1
    """
    # Предварительная нормализация векторов
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # Проверка на нулевые векторы
    if v1_norm == 0 or v2_norm == 0:
        return 0.0

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Вычисление косинусного сходства
    return np.dot(v1, v2)


def extract_formants_advanced(y: np.ndarray, sr: int, order: int = 16) -> Dict[str, np.ndarray]:
    """
    Расширенное извлечение формант с отслеживанием динамики.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации
        order: Порядок LPC-анализа

    Returns:
        Словарь с формантами F1-F4 и их динамикой
    """
    try:
        # Параметры оконного анализа
        frame_length = int(0.025 * sr)  # 25 мс окно
        hop_length = int(0.010 * sr)  # 10 мс шаг

        # Подготовка результатов
        formants_result = {
            "F1": np.array([]),
            "F2": np.array([]),
            "F3": np.array([]),
            "F4": np.array([]),
            "F1_bandwidth": np.array([]),
            "F2_bandwidth": np.array([]),
            "F3_bandwidth": np.array([]),
            "F4_bandwidth": np.array([]),
        }

        # Проход по фреймам
        for i in range(0, len(y) - frame_length, hop_length):
            # Извлекаем фрейм
            frame = y[i:i + frame_length]

            # Применяем оконную функцию для уменьшения краевых эффектов
            frame = frame * np.hamming(len(frame))

            # Выполняем LPC-анализ
            A = librosa.lpc(frame, order=order)

            # Рассчитываем частотную характеристику
            w, h = scipy.signal.freqz(1, A, worN=2000)
            freqs = w * sr / (2 * np.pi)

            # Преобразуем к амплитудам
            magnitude = np.abs(h)

            # Находим пики (форманты)
            peaks, properties = scipy.signal.find_peaks(
                magnitude,
                height=0.1,
                distance=5,
                prominence=0.1
            )

            # Сортируем по частоте
            sorted_peaks = sorted(peaks, key=lambda x: freqs[x])

            # Фильтруем по известным диапазонам формант
            valid_formants = []
            for peak in sorted_peaks:
                freq = freqs[peak]
                # Проверяем, попадает ли в диапазон какой-либо форманты
                for i, (formant, (fmin, fmax)) in enumerate(FORMANT_LIMITS.items(), 1):
                    if fmin <= freq <= fmax:
                        valid_formants.append((i, freq, peak))
                        break

            # Группируем по номеру форманты
            grouped_formants = {}
            for num, freq, peak in valid_formants:
                if num not in grouped_formants:
                    grouped_formants[num] = []
                grouped_formants[num].append((freq, peak))

            # Для каждой форманты выбираем по одному значению с максимальной амплитудой
            for i in range(1, 5):  # F1-F4
                formant_key = f"F{i}"
                bandwidth_key = f"F{i}_bandwidth"

                if i in grouped_formants and grouped_formants[i]:
                    # Выбираем пик с наибольшей амплитудой
                    best_peak = max(grouped_formants[i], key=lambda x: magnitude[x[1]])
                    formants_result[formant_key] = np.append(formants_result[formant_key], best_peak[0])

                    # Оценка ширины полосы (bandwidth)
                    peak_idx = best_peak[1]
                    peak_value = magnitude[peak_idx]
                    half_power = peak_value / np.sqrt(2)

                    # Ищем точки пересечения с уровнем половинной мощности
                    left_idx = peak_idx
                    while left_idx > 0 and magnitude[left_idx] > half_power:
                        left_idx -= 1

                    right_idx = peak_idx
                    while right_idx < len(magnitude) - 1 and magnitude[right_idx] > half_power:
                        right_idx += 1

                    # Вычисляем ширину полосы пропускания
                    bandwidth = freqs[right_idx] - freqs[left_idx]
                    formants_result[bandwidth_key] = np.append(formants_result[bandwidth_key], bandwidth)

        return formants_result
    except Exception as e:
        log.warning(f"Ошибка при извлечении формант: {e}")
        return None


def extract_formant_dynamics(formants: Dict[str, np.ndarray]) -> np.ndarray:
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


def extract_vocal_tract_length(formants: Dict[str, np.ndarray]) -> float:
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


def extract_fricative_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает признаки фрикативных согласных (ш, с, ф, в, etc).
    Характеристики этих звуков зависят от анатомии речевого аппарата.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор признаков фрикативных звуков
    """
    try:
        # Параметры для анализа
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        # Спектральные признаки
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length).flatten()

        spec_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length).flatten()

        # Энергия в высокочастотных диапазонах (характерно для фрикативных)
        stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

        # Частотные полосы для фрикативных
        # 1. 2000-4000 Hz (s, z)
        # 2. 4000-8000 Hz (sh, zh)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        mask_s = (freq_bins >= 2000) & (freq_bins <= 4000)
        mask_sh = (freq_bins >= 4000) & (freq_bins <= 8000)

        energy_s = np.mean(stft[mask_s, :], axis=0)
        energy_sh = np.mean(stft[mask_sh, :], axis=0)

        # Отношение энергий - характеристика индивидуальных особенностей
        ratio = np.zeros_like(energy_s)
        mask = energy_s > 0
        ratio[mask] = energy_sh[mask] / energy_s[mask]

        # Формируем вектор признаков фрикативных
        features = np.array([
            np.mean(spec_centroid),
            np.std(spec_centroid),
            np.mean(spec_flatness),
            np.std(spec_flatness),
            np.mean(energy_s),
            np.mean(energy_sh),
            np.mean(ratio),
            np.std(ratio)
        ])

        return features
    except Exception as e:
        log.warning(f"Ошибка при извлечении признаков фрикативных: {e}")
        return np.zeros(8)


def extract_nasal_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Извлекает признаки носовых звуков (м, н).
    Носовые резонансы - уникальная характеристика голоса.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Вектор признаков носовых звуков
    """
    try:
        # Параметры для анализа
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        # STFT для спектрального анализа
        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

        # Частотные полосы для носовых резонансов
        # Основной носовой резонанс: 250-450 Hz
        # Второй носовой резонанс: 1000-1200 Hz
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        mask_nasal1 = (freq_bins >= 250) & (freq_bins <= 450)
        mask_nasal2 = (freq_bins >= 1000) & (freq_bins <= 1200)

        # Средняя энергия в полосах
        energy_nasal1 = np.mean(S[mask_nasal1, :], axis=0)
        energy_nasal2 = np.mean(S[mask_nasal2, :], axis=0)

        # Отношение ко всему спектру
        energy_total = np.mean(S, axis=0)
        ratio1 = np.zeros_like(energy_nasal1)
        ratio2 = np.zeros_like(energy_nasal2)

        mask = energy_total > 0
        ratio1[mask] = energy_nasal1[mask] / energy_total[mask]
        ratio2[mask] = energy_nasal2[mask] / energy_total[mask]

        # Сбор признаков
        features = np.array([
            np.mean(energy_nasal1),
            np.std(energy_nasal1),
            np.mean(energy_nasal2),
            np.std(energy_nasal2),
            np.mean(ratio1),
            np.std(ratio1),
            np.mean(ratio2),
            np.std(ratio2)
        ])

        return features
    except Exception as e:
        log.warning(f"Ошибка при извлечении признаков носовых: {e}")
        return np.zeros(8)


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
        frame_length = int(0.025 * sr)
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
        log.warning(f"Ошибка при извлечении джиттера/шиммера: {e}")
        return np.zeros(8)


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
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Преобразование в тензор
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)

        # Получение эмбеддингов из YAMNet
        _, embeddings, _ = lazy_yamnet()(waveform)

        # Возвращаем среднее значение эмбеддингов по времени
        return embeddings.numpy().mean(axis=0)
    except Exception as e:
        log.warning(f"Ошибка при извлечении YAMNet признаков: {e}")
        return np.zeros(1024)  # YAMNet embeddings имеют размерность 1024


def calculate_confidence_interval(similarities: List[float]) -> Tuple[float, float]:
    """
    Вычисляет доверительный интервал для средней оценки сходства.

    Args:
        similarities: Список оценок сходства

    Returns:
        (lower_bound, upper_bound): Границы доверительного интервала
    """
    # Если оценок меньше 2, доверительный интервал не имеет смысла
    if len(similarities) < 2:
        return (0.0, 1.0)

    # Вычисляем среднее и стандартное отклонение
    mean = np.mean(similarities)
    std_dev = np.std(similarities, ddof=1)  # Несмещенная оценка

    # Степени свободы
    df = len(similarities) - 1

    # Критическое значение t-распределения для выбранного уровня доверия
    t_crit = scipy.stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df)

    # Стандартная ошибка среднего
    sem = std_dev / np.sqrt(len(similarities))

    # Доверительный интервал
    margin_of_error = t_crit * sem
    lower_bound = max(0.0, mean - margin_of_error)
    upper_bound = min(1.0, mean + margin_of_error)

    return (lower_bound, upper_bound)


def compare_voices_dual(file1: str, file2: str, weights: dict = weights) -> Tuple[str, str]:
    """
    Выполняет комплексное сравнение двух голосовых файлов.

    Args:
        file1: Путь к первому аудиофайлу
        file2: Путь ко второму аудиофайлу
        weights: Весовые коэффициенты для каждой метрики

    Returns:
        (verdict, summary): Вердикт о сходстве и подробный отчет
    """
    log.info(f"Сравнение файлов: {file1} и {file2}")

    # Загрузка аудиофайлов
    y1, _ = librosa.load(file1, sr=SAMPLE_RATE)
    y2, _ = librosa.load(file2, sr=SAMPLE_RATE)

    # Предобработка
    y1 = preprocess(y1)
    y2 = preprocess(y2)

    # === НОВОЕ: Проверка на синтетический голос с помощью модели AntiSpoofing ===
    antispoofing = lazy_antispoofing()
    spoof_result1 = antispoofing.detect(y1, SAMPLE_RATE)
    spoof_result2 = antispoofing.detect(y2, SAMPLE_RATE)

    synthetic_warning = ""
    is_synthetic = False

    if spoof_result1["is_synthetic"] > 0.7 or spoof_result2["is_synthetic"] > 0.7:
        synthetic_warning = (
            f"🚨 ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ: Обнаружены признаки синтетического голоса или deepfake!\n"
            f"Файл 1: Вероятность синтетического голоса {spoof_result1['is_synthetic']:.1%}\n"
            f"Файл 2: Вероятность синтетического голоса {spoof_result2['is_synthetic']:.1%}\n"
            f"Результаты сравнения крайне ненадежны при наличии синтетического голоса.\n\n"
        )
        is_synthetic = True

    # Проверка на модификацию голоса
    mod1, mod_type1 = detect_voice_modification(y1, SAMPLE_RATE)
    mod2, mod_type2 = detect_voice_modification(y2, SAMPLE_RATE)

    modification_warning = ""
    if mod1 or mod2:
        modification_warning = (
            f"⚠️ ВНИМАНИЕ: Обнаружены признаки модификации голоса!\n"
            f"Файл 1: {'Да, тип: ' + mod_type1 if mod1 else 'Нет'}\n"
            f"Файл 2: {'Да, тип: ' + mod_type2 if mod2 else 'Нет'}\n"
            f"Результаты сравнения могут быть искажены.\n\n"
        )

    # Получение речевых сегментов
    segments1 = get_segments(y1, SAMPLE_RATE)
    segments2 = get_segments(y2, SAMPLE_RATE)

    # Если сегменты не найдены, выдаем ошибку
    if not segments1 or not segments2:
        return (
            "⚠️ Ошибка анализа: в одном или обоих файлах не обнаружено достаточно речи",
            "Убедитесь, что файлы содержат голос и не повреждены."
        )

    # Инициализация результатов сравнения
    sims = {
        "ecapa": [],  # Нейросетевое сравнение (ECAPA-TDNN)
        "xvec": [],  # Нейросетевое сравнение (X-vector)
        "res": [],  # Нейросетевое сравнение (Resemblyzer)
        "formant": [],  # Базовые форманты
        "formant_dynamics": [],  # Динамика формант
        "fricative": [],  # Фрикативные звуки
        "nasal": [],  # Носовые резонансы
        "vocal_tract": [],  # Длина голосового тракта
        "jitter_shimmer": [],  # Микровариации голоса
        "yamnet": [],  # Перцептивные признаки
        "voice_features": [],  # Биометрические вектора (новая модель)
        "formant_tracker": []  # Отслеживание формант (новая модель)
    }

    # === НОВОЕ: Запуск расширенного анализа формант через FormantTracker ===
    formant_tracker = lazy_formant_tracker()
    formant_tracks1 = formant_tracker.track_formants(y1)
    formant_tracks2 = formant_tracker.track_formants(y2)

    # Получение статистики формант для обоих голосов
    formant_stats1 = formant_tracker.compute_formant_statistics(formant_tracks1)
    formant_stats2 = formant_tracker.compute_formant_statistics(formant_tracks2)

    # Оценка вокального тракта для обоих голосов
    vocal_tract_estimate1 = formant_tracker.estimate_vocal_tract_length(formant_tracks1)
    vocal_tract_estimate2 = formant_tracker.estimate_vocal_tract_length(formant_tracks2)

    # Сравнение формантных профилей
    formant_comparison = formant_tracker.compare_formant_profiles(formant_stats1, formant_stats2)

    # === НОВОЕ: Извлечение полных голосовых характеристик через VoiceFeatureExtractor ===
    voice_features = lazy_voice_features()
    features1 = voice_features.extract_all_features(y1)
    features2 = voice_features.extract_all_features(y2)

    # Сравнение голосовых характеристик
    voice_features_comparison = voice_features.compare_voice_features(features1, features2)

    # Загрузка остальных моделей
    ecapa = lazy_ecapa()
    xvector = lazy_xvector()
    res = lazy_res()

    # Сравнение сегментов
    for s1, s2 in zip(segments1, segments2):
        # Преобразование в тензоры для нейросетевых моделей
        t1 = torch.tensor(s1).unsqueeze(0)
        t2 = torch.tensor(s2).unsqueeze(0)

        # Сравнение ECAPA-TDNN
        sims["ecapa"].append(cosine_similarity(
            ecapa.encode_batch(t1).squeeze().detach().numpy(),
            ecapa.encode_batch(t2).squeeze().detach().numpy()
        ))

        # Сравнение Resemblyzer
        sims["res"].append(cosine_similarity(
            res.embed_utterance(s1), res.embed_utterance(s2)))

        # Сравнение X-vector
        sims["xvec"].append(cosine_similarity(
            xvector(t1).squeeze().numpy(), xvector(t2).squeeze().numpy()))

        # Извлечение и сравнение признаков формант
        formants1 = extract_formants_advanced(s1, SAMPLE_RATE)
        formants2 = extract_formants_advanced(s2, SAMPLE_RATE)

        if formants1 is not None and formants2 is not None:
            # Формантный анализ (сходство F1-F4)
            formant_vector1 = np.concatenate([
                np.mean(formants1["F1"]) if formants1["F1"].size > 0 else np.array([0]),
                np.mean(formants1["F2"]) if formants1["F2"].size > 0 else np.array([0]),
                np.mean(formants1["F3"]) if formants1["F3"].size > 0 else np.array([0]),
                np.mean(formants1["F4"]) if formants1["F4"].size > 0 else np.array([0])
            ])

            formant_vector2 = np.concatenate([
                np.mean(formants2["F1"]) if formants2["F1"].size > 0 else np.array([0]),
                np.mean(formants2["F2"]) if formants2["F2"].size > 0 else np.array([0]),
                np.mean(formants2["F3"]) if formants2["F3"].size > 0 else np.array([0]),
                np.mean(formants2["F4"]) if formants2["F4"].size > 0 else np.array([0])
            ])

            sims["formant"].append(cosine_similarity(formant_vector1, formant_vector2))

            # Динамика формант
            dynamics1 = extract_formant_dynamics(formants1)
            dynamics2 = extract_formant_dynamics(formants2)
            sims["formant_dynamics"].append(cosine_similarity(dynamics1, dynamics2))

            # Длина голосового тракта
            vtl1 = extract_vocal_tract_length(formants1)
            vtl2 = extract_vocal_tract_length(formants2)
            # Нормализованное сходство для длины тракта (абсолютная разница)
            vtl_sim = 1.0 - min(abs(vtl1 - vtl2) / 5.0, 1.0)  # Нормализация по 5 см максимальной разницы
            sims["vocal_tract"].append(vtl_sim)

        # Фрикативные звуки
        fricative1 = extract_fricative_features(s1, SAMPLE_RATE)
        fricative2 = extract_fricative_features(s2, SAMPLE_RATE)
        sims["fricative"].append(cosine_similarity(fricative1, fricative2))

        # Носовые резонансы
        nasal1 = extract_nasal_features(s1, SAMPLE_RATE)
        nasal2 = extract_nasal_features(s2, SAMPLE_RATE)
        sims["nasal"].append(cosine_similarity(nasal1, nasal2))

        # Джиттер и шиммер
        jitter_shimmer1 = extract_jitter_shimmer(s1, SAMPLE_RATE)
        jitter_shimmer2 = extract_jitter_shimmer(s2, SAMPLE_RATE)
        # Специальное сравнение для джиттера/шиммера - чем ближе, тем выше сходство
        js_diff = np.abs(jitter_shimmer1 - jitter_shimmer2)
        js_sim = 1.0 - np.mean(np.minimum(js_diff / np.array([2.0, 5.0, 2.0, 3.0, 5.0, 3.0, 5.0, 5.0]), 1.0))
        sims["jitter_shimmer"].append(js_sim)

        # YAMNet перцептивные признаки
        yamnet1 = extract_yamnet(s1, SAMPLE_RATE)
        yamnet2 = extract_yamnet(s2, SAMPLE_RATE)
        sims["yamnet"].append(cosine_similarity(yamnet1, yamnet2))

        # === НОВОЕ: Извлечение и сравнение голосовых биометрических характеристик для сегментов ===
        segment_features1 = voice_features.extract_all_features(s1)
        segment_features2 = voice_features.extract_all_features(s2)

        # Сравнение биометрических векторов
        segment_comparison = voice_features.compare_voice_features(segment_features1, segment_features2)
        sims["voice_features"].append(segment_comparison["overall"])

        # === НОВОЕ: Извлечение и сравнение формантных треков для сегментов ===
        segment_formant_tracks1 = formant_tracker.track_formants(s1)
        segment_formant_tracks2 = formant_tracker.track_formants(s2)

        # Статистика и сравнение формант для сегментов
        segment_formant_stats1 = formant_tracker.compute_formant_statistics(segment_formant_tracks1)
        segment_formant_stats2 = formant_tracker.compute_formant_statistics(segment_formant_tracks2)

        segment_formant_comparison = formant_tracker.compare_formant_profiles(
            segment_formant_stats1, segment_formant_stats2)

        if "overall" in segment_formant_comparison:
            sims["formant_tracker"].append(segment_formant_comparison["overall"])

    # === НОВОЕ: Добавляем результаты из общего сравнения полных файлов ===
    # Эти результаты важны для глобального анализа речевой идентичности
    if "overall" in formant_comparison:
        sims["formant_tracker"].append(formant_comparison["overall"])

    if "overall" in voice_features_comparison:
        sims["voice_features"].append(voice_features_comparison["overall"])

    # Формирование итогового отчета
    summary = synthetic_warning + modification_warning
    weighted_total = 0
    weighted_score = 0

    # Таблица результатов
    summary += "| Метрика | Медиана | 95% CI | >0.90 | Вес |\n"
    summary += "| ------- | ------- | ------ | ----- | --- |\n"

    all_medians = []  # Для оценки согласованности результатов

    for key, values in sims.items():
        if not values:  # Пропускаем, если нет значений
            continue

        # Статистика по метрике
        med = np.median(values)
        all_medians.append(med)
        count_high = sum(1 for x in values if x > 0.9)
        ci_low, ci_high = calculate_confidence_interval(values)

        # Определение уверенности по ширине доверительного интервала
        ci_width = ci_high - ci_low
        confidence = "🟢" if ci_width < 0.1 else "🟡" if ci_width < 0.2 else "🔴"

        # Метка для медианы
        label = "🟢" if med > 0.85 else "🟡" if med > 0.7 else "🔴"

        # Весовой коэффициент
        weight = weights.get(key, 1.0)
        weighted_total += weight
        weighted_score += weight * med

        # Добавление в таблицу
        summary += f"| {label} {key.upper()} | {med:.3f} | {ci_low:.2f}-{ci_high:.2f} {confidence} | {count_high}/{len(values)} | {weight} |\n"

    # Расчет итоговой оценки
    final_score = weighted_score / weighted_total if weighted_total > 0 else 0

    # Определение согласованности результатов разных методов
    consistency = np.std(all_medians)
    consistency_label = "🟢" if consistency < 0.05 else "🟡" if consistency < 0.15 else "🔴"

    # Доверительный интервал для итоговой оценки
    confidence_range = f"{final_score:.2f} ± {consistency:.2f}"

    # === НОВОЕ: Добавление информации о вокальном тракте ===
    if vocal_tract_estimate1 and vocal_tract_estimate2 and "mean" in vocal_tract_estimate1 and "mean" in vocal_tract_estimate2:
        vtl1_mean = vocal_tract_estimate1["mean"]
        vtl2_mean = vocal_tract_estimate2["mean"]
        vtl_diff = abs(vtl1_mean - vtl2_mean)

        vtl_assessment = (
            f"\n**Анализ длины голосового тракта:**\n"
            f"Файл 1: {vtl1_mean:.1f} см\n"
            f"Файл 2: {vtl2_mean:.1f} см\n"
            f"Разница: {vtl_diff:.2f} см\n"
        )

        vtl_conclusion = ""
        if vtl_diff < 0.5:
            vtl_conclusion = "✅ Длины голосовых трактов очень близки, что характерно для одного человека"
        elif vtl_diff < 1.0:
            vtl_conclusion = "🟡 Небольшая разница в длине голосовых трактов, возможно один человек с разной артикуляцией"
        else:
            vtl_conclusion = "❌ Значительная разница в длине голосовых трактов, характерная для разных людей"

        summary += vtl_assessment + vtl_conclusion + "\n\n"

    # Добавляем итоговую оценку в отчет
    summary += f"\n**Итоговая оценка: {final_score:.3f}** (консистентность методов: {consistency_label} ±{consistency:.2f})\n"
    summary += f"**Доверительный интервал: {confidence_range}**\n\n"

    # Формирование вердикта с учетом консистентности методов и наличия синтетической речи
    verdict = ""

    # Проверка на синтетический голос перед вынесением вердикта
    if is_synthetic:
        verdict = "⚠️ РЕЗУЛЬТАТ НЕНАДЕЖЕН: Обнаружены признаки синтетического голоса (deepfake)"
        if final_score >= 0.85:
            verdict += "\nПри игнорировании признаков deepfake сходство высокое, но не может быть использовано как доказательство"
    else:
        if final_score >= 0.95 and consistency < 0.1:
            verdict = "✅ ЗАКЛЮЧЕНИЕ ЭКСПЕРТА: Голоса с высочайшей вероятностью принадлежат одному и тому же человеку"
        elif final_score >= 0.88 and consistency < 0.12:
            verdict = "✅ ЗАКЛЮЧЕНИЕ ЭКСПЕРТА: Голоса с высокой вероятностью принадлежат одному и тому же человеку"
        elif final_score >= 0.80 and consistency < 0.15:
            verdict = "🟡 ЗАКЛЮЧЕНИЕ ЭКСПЕРТА: Голоса, вероятно, принадлежат одному человеку, но требуются дополнительные подтверждения"
        elif final_score >= 0.70:
            verdict = "⚠️ ЗАКЛЮЧЕНИЕ ЭКСПЕРТА: Имеется некоторое сходство голосов, но недостаточно для надежного вывода"
        else:
            verdict = "❌ ЗАКЛЮЧЕНИЕ ЭКСПЕРТА: Голоса с высокой вероятностью принадлежат разным людям"

    # Добавляем пояснения к вердикту
    if mod1 or mod2:
        verdict += "\n⚠️ ВНИМАНИЕ: Обнаружены признаки искусственной модификации голоса"

    if consistency > 0.15:
        verdict += "\n⚠️ ВНИМАНИЕ: Высокая несогласованность между методами анализа, результаты могут быть ненадежными"

    return verdict, summary
