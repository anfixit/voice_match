
import librosa
import numpy as np
import scipy.signal
import scipy.stats

from voice_match.constants import (
    FRAME_LENGTH,
    HOP_LENGTH,
)


def detect_pitch_shift(y: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Обнаруживает признаки искусственного изменения основного тона.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        (is_shifted, shift_amount): Флаг смещения и оценка величины в полутонах
    """
    # Параметры для анализа
    frame_length = FRAME_LENGTH
    hop_length = HOP_LENGTH

    # Оценка основного тона
    pitches, magnitudes = librosa.core.piptrack(
        y=y, sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=50,
        fmax=400
    )

    # Фильтрация питчей по магнитуде
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        magnitude = magnitudes[index, t]

        # Добавляем только питчи с существенной магнитудой
        if pitch > 0 and magnitude > 0.1 * np.max(magnitudes):
            pitch_values.append(pitch)

    if len(pitch_values) < 10:
        return False, 0.0

    # Анализ распределения питчей
    pitch_values = np.array(pitch_values)

    # 1. Проверка на неестественно стабильный питч (равномерное распределение)
    hist, _bin_edges = np.histogram(pitch_values, bins=100)

    # Энтропия гистограммы (оценка равномерности распределения)
    hist_prob = hist / np.sum(hist)
    hist_prob = hist_prob[hist_prob > 0]  # Убираем нулевые вероятности
    entropy = -np.sum(hist_prob * np.log2(hist_prob))

    # Нормализованная энтропия (0 = один пик, 1 = равномерное распределение)
    max_entropy = np.log2(len(hist_prob)) if len(hist_prob) > 0 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 2. Проверка на аномальный диапазон питча
    mean_pitch = np.mean(pitch_values)

    # Мужской голос обычно 85-180 Hz, женский 160-255 Hz
    if mean_pitch < 80 or mean_pitch > 300:
        shift_amount = 12 * np.log2(mean_pitch / 150)  # Относительно средней частоты
        return True, shift_amount

    # 3. Проверка на неестественно стабильный питч
    if normalized_entropy < 0.4:  # Неестественно стабильный питч
        return True, 0.0

    return False, 0.0


def detect_formant_shift(formants: dict[str, np.ndarray]) -> tuple[bool, float]:
    """
    Обнаруживает признаки искусственного изменения формант.

    Args:
        formants: Словарь с формантами

    Returns:
        (is_shifted, shift_ratio): Флаг смещения и оценка коэффициента смещения
    """
    if formants is None or "F1" not in formants or "F2" not in formants:
        return False, 0.0

    f1_values = formants["F1"]
    f2_values = formants["F2"]

    if len(f1_values) < 10 or len(f2_values) < 10:
        return False, 0.0

    # Средние значения формант
    f1_mean = np.mean(f1_values)
    f2_mean = np.mean(f2_values)

    # Проверка на аномальные значения формант
    # Первая форманта обычно 270-860 Hz (средняя ~500)
    # Вторая форманта обычно 840-2790 Hz (средняя ~1500)
    f1_expected = 500
    f2_expected = 1500

    f1_ratio = f1_mean / f1_expected
    f2_ratio = f2_mean / f2_expected

    # Если оба соотношения близки, это может быть признаком масштабирования формант
    if abs(f1_ratio - f2_ratio) < 0.2 and (f1_ratio < 0.7 or f1_ratio > 1.3):
        return True, f1_ratio

    return False, 0.0


def detect_robot_voice(y: np.ndarray, sr: int) -> bool:
    """
    Обнаруживает признаки механического/роботизированного голоса.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        is_robot: Флаг обнаружения
    """
    # Параметры для анализа
    frame_length = FRAME_LENGTH
    hop_length = HOP_LENGTH

    # Спектральные признаки
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=frame_length, hop_length=hop_length).flatten()

    # 1. Неестественная равномерность спектра (робот)
    if np.mean(flatness) > 0.4:
        return True

    # 2. Анализ монотонности (роботы часто имеют монотонный тон)
    pitches, magnitudes = librosa.core.piptrack(
        y=y, sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=50,
        fmax=400
    )

    # Фильтрация питчей по магнитуде
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        magnitude = magnitudes[index, t]

        if pitch > 0 and magnitude > 0.1 * np.max(magnitudes):
            pitch_values.append(pitch)

    if len(pitch_values) < 10:
        return False

    # Стандартное отклонение питча (низкое значение = монотонный голос)
    pitch_std = np.std(pitch_values)
    pitch_mean = np.mean(pitch_values)

    # Коэффициент вариации (CV) - отношение стд к среднему
    pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0

    # Обычно CV человеческого голоса > 0.05 (5%)
    if pitch_cv < 0.05:
        return True

    # 3. Проверка на равные интервалы между гармониками (признак синтеза)
    spectrogram = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Автокорреляция спектра для поиска регулярных паттернов
    autocorr = []
    for frame in range(spectrogram.shape[1]):
        frame_auto = np.correlate(spectrogram[:, frame], spectrogram[:, frame], mode='full')
        frame_auto = frame_auto[frame_length - 1:]
        frame_auto = frame_auto / np.max(frame_auto) if np.max(frame_auto) > 0 else frame_auto
        autocorr.append(frame_auto)

    # Усреднение по всем фреймам
    mean_autocorr = np.mean(autocorr, axis=0)

    # Поиск пиков в автокорреляции (регулярные паттерны)
    peaks, _ = scipy.signal.find_peaks(mean_autocorr, height=0.3, distance=5)

    # Оценка равномерности интервалов между пиками
    if len(peaks) > 3:
        peak_diffs = np.diff(peaks)
        peak_diffs_std = np.std(peak_diffs)
        peak_diffs_mean = np.mean(peak_diffs)

        # Коэффициент вариации интервалов (низкое значение = равномерные интервалы)
        peak_diffs_cv = peak_diffs_std / peak_diffs_mean if peak_diffs_mean > 0 else 0

        if peak_diffs_cv < 0.1:  # Очень регулярные интервалы = синтетический голос
            return True

    return False


def detect_voice_modification(y: np.ndarray, sr: int, formants: dict[str, np.ndarray] | None = None) -> tuple[
    bool, str | None, float]:
    """
    Комплексное обнаружение голосовых модификаторов.

    Args:
        y: Аудиосигнал
        sr: Частота дискретизации
        formants: Предварительно извлеченные форманты (опционально)

    Returns:
        (is_modified, modifier_type, confidence): Флаг модификации, тип и уверенность
    """
    # Проверка на смещение тона
    pitch_shifted, pitch_amount = detect_pitch_shift(y, sr)

    # Проверка на смещение формант
    if formants is None:
        from voice_match.services.comparison import extract_formants_advanced
        formants = extract_formants_advanced(y, sr)

    formant_shifted, formant_ratio = detect_formant_shift(formants)

    # Проверка на робота
    robot_detected = detect_robot_voice(y, sr)

    # Определение итогового результата с уверенностью
    if pitch_shifted and abs(pitch_amount) > 3:
        return True, "pitch_shift", min(abs(pitch_amount) / 6.0, 1.0)
    elif formant_shifted:
        return True, "formant_shift", min(abs(formant_ratio - 1.0) / 0.5, 1.0)
    elif robot_detected:
        return True, "robot_voice", 0.8

    # Если есть слабые признаки нескольких модификаций
    if (pitch_shifted and formant_shifted) or (pitch_shifted and robot_detected) or (
            formant_shifted and robot_detected):
        return True, "combined_modification", 0.7

    return False, None, 0.0
