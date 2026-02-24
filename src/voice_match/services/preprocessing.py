import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import subprocess
import scipy.signal
import scipy.io.wavfile
import noisereduce as nr
from voice_match.log import setup_logger
from typing import Tuple, List, Dict, Optional, Union, Any
from pydub import AudioSegment
from voice_match.constants import SUPPORTED_EXTENSIONS, TARGET_SAMPLE_RATE, TARGET_CHANNELS

# ──────────────── Логгер ────────────────
log = setup_logger("utils")


class AudioSegmentInfo:
    """Класс для хранения информации о сегменте аудио."""

    def __init__(self, start: float, end: float, energy: float, snr: float,
                 is_speech: bool, is_voiced: bool):
        """
        Инициализирует информацию о сегменте аудио.

        Args:
            start: Начальное время сегмента (в секундах)
            end: Конечное время сегмента (в секундах)
            energy: Средняя энергия сегмента
            snr: Отношение сигнал/шум
            is_speech: Флаг наличия речи
            is_voiced: Флаг наличия вокализованных звуков
        """
        self.start = start
        self.end = end
        self.duration = end - start
        self.energy = energy
        self.snr = snr
        self.is_speech = is_speech
        self.is_voiced = is_voiced
        self.quality_score = 0.0  # Будет рассчитано позже

    def calculate_quality_score(self) -> float:
        """
        Вычисляет общую оценку качества сегмента.

        Returns:
            Оценка качества от 0 до 1
        """
        if not self.is_speech:
            return 0.0

        # Нормализация SNR (типичный диапазон от 5 до 30 дБ)
        snr_score = min(1.0, max(0.0, (self.snr - 5) / 25))

        # Вклады разных компонентов
        self.quality_score = 0.5 * snr_score + 0.3 * float(self.is_voiced) + 0.2 * min(1.0, self.energy)
        return self.quality_score


def convert_audio_to_wav(file_path: str, force_resample: bool = True) -> Tuple[str, str]:
    """
    Конвертирует аудиофайл в WAV формат с контролем качества.
    Поддерживает нормализацию, ресемплирование и установку стерео/моно.

    Args:
        file_path: Путь к исходному аудиофайлу
        force_resample: Принудительное ресемплирование даже если частота соответствует

    Returns:
        (wav_path, log_msg): Путь к WAV файлу и лог-сообщение
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}. Поддерживаются: {SUPPORTED_EXTENSIONS}")

    # Проверка целостности файла и его параметров
    try:
        file_info = get_audio_info(file_path)
        log.info(f"Информация о файле: {file_info}")
    except Exception as exc:
        log.error(f"Не удалось получить информацию о файле: {exc}")
        file_info = {}

    # Проверка требуемых параметров
    sample_rate = file_info.get("sample_rate", 0)
    channels = file_info.get("channels", 0)

    # Определяем, нужна ли конвертация
    needs_conversion = (
            ext != ".wav" or
            force_resample or
            sample_rate != TARGET_SAMPLE_RATE or
            channels != TARGET_CHANNELS
    )

    if not needs_conversion and ext == ".wav":
        log.info(f"Файл {file_path} уже в WAV формате с требуемыми параметрами")
        return file_path, f"Файл уже в требуемом формате: {TARGET_SAMPLE_RATE} Hz, {TARGET_CHANNELS} канал(ов)"

    # Загрузка и конвертация
    try:
        # Сначала пробуем использовать pydub
        try:
            audio = AudioSegment.from_file(file_path)

            # Применяем требуемые параметры
            audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)

            # Нормализация уровня громкости
            if audio.dBFS < -30:
                # Если файл слишком тихий, нормализуем до -3dB
                audio = audio.normalize(headroom=3.0)
                log.info(f"Файл {file_path} нормализован (был слишком тихим)")

            # Обработка некорректной продолжительности
            if len(audio) < 1000:  # Меньше 1 секунды
                log.warning(f"Файл {file_path} слишком короткий: {len(audio) / 1000:.2f} сек")

        except Exception as pydub_error:
            # Если pydub не справляется, используем ffmpeg напрямую
            log.warning(f"Ошибка при использовании pydub: {pydub_error}. Используем ffmpeg напрямую")

            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)

            # Формируем команду ffmpeg с нужными параметрами
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", file_path,
                "-ac", str(TARGET_CHANNELS),
                "-ar", str(TARGET_SAMPLE_RATE),
                "-sample_fmt", "s16",  # 16-бит PCM
                "-af", "dynaudnorm=f=150:g=15",  # Нормализация динамического диапазона
                temp_path
            ]

            # Выполняем команду
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                log.error(f"Ошибка при выполнении ffmpeg: {result.stderr}")
                raise RuntimeError(f"Ошибка при конвертации с помощью ffmpeg: {result.stderr}")

            # Загружаем результат через pydub для дальнейшей обработки
            audio = AudioSegment.from_file(temp_path)
            os.unlink(temp_path)  # Удаляем временный файл

        # Создаем итоговый WAV файл
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        # Экспортируем файл
        audio.export(temp_path, format="wav", parameters=["-q:a", "0"])

    except Exception as exc:
        log.error(f"Ошибка при конвертации файла {file_path}: {exc}")
        raise RuntimeError(
            f"Ошибка при конвертации файла {file_path}\n"
            f"Убедитесь, что файл корректный и ffmpeg поддерживает его кодек.\n"
            f"Детали: {exc}"
        )

    # Проверяем результат конвертации
    try:
        converted_info = get_audio_info(temp_path)
        log.info(f"Информация о сконвертированном файле: {converted_info}")

        if (converted_info.get("sample_rate") != TARGET_SAMPLE_RATE or
                converted_info.get("channels") != TARGET_CHANNELS):
            log.warning(f"Параметры сконвертированного файла не соответствуют ожидаемым")
    except Exception as exc:
        log.warning(f"Не удалось проверить параметры сконвертированного файла: {exc}")

    log_msg = (
        f"Файл {file_path} успешно конвертирован в WAV: {temp_path} "
        f"с {TARGET_SAMPLE_RATE} Hz, {TARGET_CHANNELS} канал(ом)."
    )
    return temp_path, log_msg


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Извлекает информацию о аудиофайле.

    Args:
        file_path: Путь к аудиофайлу

    Returns:
        Словарь с информацией о файле
    """
    info = {}

    try:
        # Для WAV файлов используем soundfile
        if file_path.lower().endswith('.wav'):
            with sf.SoundFile(file_path) as f:
                info["sample_rate"] = f.samplerate
                info["channels"] = f.channels
                info["frames"] = f.frames
                info["duration"] = f.frames / f.samplerate
                info["format"] = f.format
                info["subtype"] = f.subtype
        else:
            # Для других форматов используем librosa
            y, sr = librosa.load(file_path, sr=None, mono=False)
            channels = 1 if y.ndim == 1 else y.shape[0]
            info["sample_rate"] = sr
            info["channels"] = channels
            info["frames"] = y.shape[-1]
            info["duration"] = y.shape[-1] / sr
            info["format"] = os.path.splitext(file_path)[1][1:]
    except Exception as exc:
        # Если не удалось, используем ffprobe
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout:
                import json
                probe_data = json.loads(result.stdout)

                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        info["sample_rate"] = int(stream.get("sample_rate", 0))
                        info["channels"] = int(stream.get("channels", 0))
                        info["duration"] = float(stream.get("duration", 0))
                        info["codec"] = stream.get("codec_name", "unknown")
                        break

                format_info = probe_data.get("format", {})
                if "duration" in format_info:
                    info["duration"] = float(format_info["duration"])
                info["format"] = format_info.get("format_name", "unknown")
        except Exception as ffprobe_exc:
            log.error(f"Не удалось получить информацию о файле через ffprobe: {ffprobe_exc}")
            raise RuntimeError(f"Не удалось получить информацию о файле: {exc}. {ffprobe_exc}")

    return info


def extract_voice_segment(wav_path: str, max_duration: int = 58, min_duration: int = 5,
                          snr_threshold: float = 10.0) -> Tuple[str, str]:
    """
    Извлекает наиболее качественный сегмент речи из WAV файла.
    Анализирует энергию, SNR и наличие речи для выбора оптимального сегмента.

    Args:
        wav_path: Путь к WAV файлу
        max_duration: Максимальная длительность сегмента в секундах
        min_duration: Минимальная длительность сегмента в секундах
        snr_threshold: Минимальное SNR для качественного сегмента

    Returns:
        (segment_path, log_msg): Путь к извлеченному сегменту и лог-сообщение
    """
    try:
        # Проверка длительности файла
        duration = librosa.get_duration(filename=wav_path)
        if duration <= min_duration:
            return wav_path, f"⚠️ Файл слишком короткий для анализа (длительность: {duration:.1f} сек)"

        if duration <= max_duration:
            return wav_path, f"Файл не требует нарезки (длительность: {duration:.1f} сек)"

        # Загрузка аудио
        y, sr = librosa.load(wav_path, sr=None)

        # 1. Шумоподавление для более точного анализа
        y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75)

        # 2. Сегментация аудио с перекрытием
        frame_length = int(1.0 * sr)  # 1-секундные фреймы
        hop_length = int(0.5 * sr)  # 50% перекрытие

        # 3. Анализ энергии и SNR каждого фрейма
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_denoised = librosa.feature.rms(y=y_denoised, frame_length=frame_length, hop_length=hop_length)[0]

        # 4. Оценка SNR (отношение сигнал/шум)
        snr = np.zeros_like(energy)
        mask = energy > 0
        snr[mask] = 10 * np.log10(energy_denoised[mask] ** 2 / (energy[mask] ** 2 - energy_denoised[mask] ** 2 + 1e-10))

        # 5. Детектирование речи (VAD)
        from voice_match.services.comparison import is_voiced

        segments_info = []

        for i in range(0, len(y) - sr * min_duration, hop_length):
            end = min(i + sr * max_duration, len(y))
            segment = y[i:end]

            if len(segment) < sr * min_duration:
                continue

            # Индексы фреймов для текущего сегмента
            start_frame = i // hop_length
            end_frame = min(start_frame + (len(segment) // hop_length), len(energy))

            if start_frame >= end_frame:
                continue

            # Средние значения для сегмента
            segment_energy = np.mean(energy[start_frame:end_frame])
            segment_snr = np.mean(snr[start_frame:end_frame])

            # Проверка наличия речи
            is_speech = False
            is_voiced_segment = False

            # Проверяем каждые 5 секунд на наличие речи
            speech_frames = 0
            total_frames = 0

            for j in range(0, len(segment) - sr, sr * 5):
                chunk = segment[j:j + sr * 5]
                if is_voiced(chunk, sr):
                    speech_frames += 1
                total_frames += 1

            if total_frames > 0 and speech_frames / total_frames >= 0.4:
                is_speech = True

                # Дополнительная проверка на вокализованные звуки
                pitch_detection = librosa.yin(segment, fmin=80, fmax=400, sr=sr)
                voiced_frames = np.sum(pitch_detection > 0) / len(pitch_detection)
                is_voiced_segment = voiced_frames >= 0.3

            # Сохраняем информацию о сегменте
            segment_info = AudioSegmentInfo(
                start=i / sr,
                end=end / sr,
                energy=segment_energy,
                snr=segment_snr,
                is_speech=is_speech,
                is_voiced=is_voiced_segment
            )
            segment_info.calculate_quality_score()
            segments_info.append(segment_info)

        # 6. Выбор лучшего сегмента
        if not segments_info:
            return wav_path, "Не удалось найти подходящие речевые сегменты"

        # Сортировка по оценке качества
        segments_info.sort(key=lambda x: x.quality_score, reverse=True)
        best_segment = segments_info[0]

        # 7. Извлечение и сохранение лучшего сегмента
        start_sample = int(best_segment.start * sr)
        end_sample = int(best_segment.end * sr)
        segment_y = y[start_sample:end_sample]

        # 8. Финальная обработка выбранного сегмента
        # Применяем плавное затухание (фейдинг) в начале и конце для устранения щелчков
        fade_samples = int(0.05 * sr)  # 50 мс
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        if len(segment_y) > 2 * fade_samples:
            segment_y[:fade_samples] *= fade_in
            segment_y[-fade_samples:] *= fade_out

        # Сохраняем сегмент
        temp_fd, segment_path = tempfile.mkstemp(suffix="_segment.wav")
        os.close(temp_fd)

        sf.write(segment_path, segment_y, sr)

        # 9. Формируем подробное описание
        log_msg = (
            f"Из файла {os.path.basename(wav_path)} (длительность: {duration:.1f} сек) "
            f"извлечен сегмент {best_segment.duration:.1f} сек, "
            f"начиная с {best_segment.start:.1f} сек.\n"
            f"Качество сегмента: {best_segment.quality_score:.2f}, SNR: {best_segment.snr:.1f} дБ"
        )

        log.info(log_msg)
        return segment_path, log_msg

    except Exception as e:
        log.warning(f"Не удалось извлечь сегмент из {wav_path}: {e}")
        return wav_path, f"Не удалось извлечь сегмент: {str(e)}"


def remove_silence(wav_path: str, min_silence_duration: float = 0.3,
                   silence_threshold: float = 0.01) -> Tuple[str, str]:
    """
    Удаляет длительные участки тишины из аудиофайла.

    Args:
        wav_path: Путь к WAV файлу
        min_silence_duration: Минимальная длительность тишины для удаления (в секундах)
        silence_threshold: Порог энергии для определения тишины (0-1)

    Returns:
        (processed_path, log_msg): Путь к обработанному файлу и лог-сообщение
    """
    try:
        # Загрузка аудио
        y, sr = librosa.load(wav_path, sr=None)

        # Вычисление энергии
        frame_length = int(0.025 * sr)  # 25 мс
        hop_length = int(0.010 * sr)  # 10 мс

        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_threshold = silence_threshold * np.max(energy)

        # Определение сегментов с речью
        speech_frames = (energy > energy_threshold)

        # Преобразование фреймов в отметки времени
        timestamps = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sr, hop_length=hop_length)

        # Обнаружение сегментов с речью
        speech_segments = []
        is_speech = False
        start_time = 0

        for i, speech in enumerate(speech_frames):
            if speech and not is_speech:
                # Начало речевого сегмента
                is_speech = True
                start_time = timestamps[i]
            elif not speech and is_speech:
                # Конец речевого сегмента
                is_speech = False
                # Если сегмент достаточно длинный
                if timestamps[i] - start_time >= 0.1:  # Минимум 100 мс
                    speech_segments.append((start_time, timestamps[i]))

        # Добавляем последний сегмент, если он есть
        if is_speech and timestamps[-1] - start_time >= 0.1:
            speech_segments.append((start_time, timestamps[-1]))

        # Если нет речевых сегментов или весь файл - один сегмент
        if not speech_segments:
            return wav_path, "Не обнаружено речевых сегментов"

        # Объединение близких сегментов
        merged_segments = []
        current_start, current_end = speech_segments[0]

        for start, end in speech_segments[1:]:
            if start - current_end <= min_silence_duration:
                # Объединяем с текущим сегментом
                current_end = end
            else:
                # Сохраняем текущий сегмент и начинаем новый
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end

        # Добавляем последний сегмент
        merged_segments.append((current_start, current_end))

        # Создаем новый аудиофайл без тишины
        processed_audio = np.zeros(0)

        for start, end in merged_segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Безопасное извлечение сегмента
            start_sample = max(0, min(start_sample, len(y) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(y)))

            segment = y[start_sample:end_sample]
            processed_audio = np.concatenate((processed_audio, segment))

        # Сохраняем результат
        temp_fd, processed_path = tempfile.mkstemp(suffix="_nosilence.wav")
        os.close(temp_fd)

        sf.write(processed_path, processed_audio, sr)

        # Формируем отчет
        original_duration = len(y) / sr
        processed_duration = len(processed_audio) / sr
        reduction = 100 * (original_duration - processed_duration) / original_duration

        log_msg = (
            f"Удалены паузы из файла {os.path.basename(wav_path)}\n"
            f"Исходная длительность: {original_duration:.2f} сек\n"
            f"Новая длительность: {processed_duration:.2f} сек\n"
            f"Сокращение: {reduction:.1f}%"
        )

        return processed_path, log_msg

    except Exception as e:
        log.warning(f"Ошибка при удалении тишины из {wav_path}: {e}")
        return wav_path, f"Не удалось удалить тишину: {str(e)}"


def enhance_speech(wav_path: str, enhance_formants: bool = True,
                   reduce_noise: bool = True) -> Tuple[str, str]:
    """
    Улучшает качество речи, усиливая форманты и подавляя шум.

    Args:
        wav_path: Путь к WAV файлу
        enhance_formants: Флаг усиления формант для улучшения разборчивости
        reduce_noise: Флаг шумоподавления

    Returns:
        (enhanced_path, log_msg): Путь к улучшенному файлу и лог-сообщение
    """
    try:
        # Загрузка аудио
        y, sr = librosa.load(wav_path, sr=None)

        # Журнал обработки
        processing_log = []

        # 1. Шумоподавление
        if reduce_noise:
            y = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=False,
                prop_decrease=0.75
            )
            processing_log.append("Применено шумоподавление")

        # 2. Предусиление речевого диапазона
        if enhance_formants:
            # Параметры фильтра для усиления речевых частот
            # Полосовой фильтр для усиления формант (300-3400 Hz)
            b, a = scipy.signal.butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype='band')
            y_filtered = scipy.signal.filtfilt(b, a, y)

            # Смешиваем с оригиналом для баланса
            y = 0.7 * y + 0.3 * y_filtered
            processing_log.append("Применено усиление формант")

        # 3. Нормализация уровня
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 0 and max_amplitude < 0.5:
            y = y / max_amplitude * 0.9  # Нормализация до 90% от максимума
            processing_log.append("Применена нормализация громкости")

        # Сохранение результата
        temp_fd, enhanced_path = tempfile.mkstemp(suffix="_enhanced.wav")
        os.close(temp_fd)

        sf.write(enhanced_path, y, sr)

        # Формирование отчета
        log_msg = (
            f"Улучшено качество файла {os.path.basename(wav_path)}\n"
            f"Примененные обработки: {', '.join(processing_log)}"
        )

        return enhanced_path, log_msg

    except Exception as e:
        log.warning(f"Ошибка при улучшении качества файла {wav_path}: {e}")
        return wav_path, f"Не удалось улучшить качество: {str(e)}"


def detect_voice_modifications(wav_path: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Обнаруживает признаки искусственной модификации голоса.

    Args:
        wav_path: Путь к WAV файлу

    Returns:
        (is_modified, details, log_msg): Флаг модификации, детали и лог-сообщение
    """
    try:
        # Загрузка аудио
        y, sr = librosa.load(wav_path, sr=None)

        details = {
            "pitch_shifted": False,
            "pitch_shift_amount": 0.0,
            "formant_shifted": False,
            "formant_shift_ratio": 0.0,
            "robot_voice": False,
            "confidence": 0.0
        }

        # 1. Анализ основного тона
        pitches, magnitudes = librosa.core.piptrack(
            y=y, sr=sr,
            n_fft=2048,
            hop_length=512,
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
            log_msg = "Недостаточно вокализованных фреймов для анализа"
            return False, details, log_msg

        # Анализ распределения питчей
        pitch_values = np.array(pitch_values)

        # Статистика питчей
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        cv_pitch = std_pitch / mean_pitch if mean_pitch > 0 else 0

        # 2. Проверка на неестественную стабильность питча
        if cv_pitch < 0.05:  # Коэффициент вариации меньше 5%
            details["robot_voice"] = True
            details["confidence"] = 0.8
            log_msg = "Обнаружены признаки синтетического голоса: аномально стабильный питч"
            return True, details, log_msg

        # 3. Проверка на аномальный диапазон питча
        if mean_pitch < 80 or mean_pitch > 300:
            details["pitch_shifted"] = True
            # Оценка сдвига относительно нормального диапазона
            reference_pitch = 120 if mean_pitch < 160 else 220  # M/F
            shift_amount = 12 * np.log2(mean_pitch / reference_pitch)

            details["pitch_shift_amount"] = shift_amount
            details["confidence"] = min(0.7 + abs(shift_amount) / 10.0, 0.95)

            log_msg = (
                f"Обнаружены признаки изменения основного тона: "
                f"средний питч {mean_pitch:.1f} Hz вне нормального диапазона, "
                f"сдвиг примерно {shift_amount:.1f} полутонов"
            )
            return True, details, log_msg

            # 4. Проверка на формантные сдвиги
            # Извлечение формант
        from voice_match.services.comparison import extract_formants_advanced
        formants = extract_formants_advanced(y, sr)

        if formants is not None and "F1" in formants and "F2" in formants:
            f1_values = formants["F1"]
            f2_values = formants["F2"]

            if len(f1_values) > 10 and len(f2_values) > 10:
                f1_mean = np.mean(f1_values)
                f2_mean = np.mean(f2_values)

                # Проверка на неестественное соотношение формант
                f2_f1_ratio = f2_mean / f1_mean if f1_mean > 0 else 0

                # Типичное отношение F2/F1 для взрослого человека: 2.5-4
                if f2_f1_ratio < 1.5 or f2_f1_ratio > 5.0:
                    details["formant_shifted"] = True
                    details["formant_shift_ratio"] = f2_f1_ratio
                    details["confidence"] = 0.75

                    log_msg = (
                        f"Обнаружены признаки искажения формант: "
                        f"соотношение F2/F1 = {f2_f1_ratio:.2f} "
                        f"вне типичного диапазона (1.5-5.0)"
                    )
                    return True, details, log_msg

        # 5. Проверка на признаки робота/синтеза
        # Анализ спектральной плоскости
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # Высокая спектральная плоскость - признак синтеза
        if flatness > 0.4:
            details["robot_voice"] = True
            details["confidence"] = 0.7

            log_msg = (
                f"Обнаружены признаки синтетического голоса: "
                f"высокая спектральная плоскость ({flatness:.3f})"
            )
            return True, details, log_msg

        # Проверка на признаки Voice Changer
        # 6. Проверка на равномерные гармоники
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

        # Среднее спектральное распределение
        mean_spectrum = np.mean(S, axis=1)

        # Автокорреляция для обнаружения регулярных гармоник
        autocorr = np.correlate(mean_spectrum, mean_spectrum, mode='full')
        autocorr = autocorr[len(mean_spectrum):]
        autocorr = autocorr / np.max(autocorr)

        # Поиск пиков в автокорреляции
        peaks, _ = scipy.signal.find_peaks(autocorr, height=0.3, distance=5)

        if len(peaks) > 3:
            peak_diffs = np.diff(peaks)
            peak_cv = np.std(peak_diffs) / np.mean(peak_diffs) if np.mean(peak_diffs) > 0 else 0

            # Слишком регулярные гармоники
            if peak_cv < 0.1:
                details["robot_voice"] = True
                details["confidence"] = 0.8

                log_msg = (
                    f"Обнаружены признаки синтетического голоса: "
                    f"равномерные гармоники с вариацией {peak_cv:.3f}"
                )
                return True, details, log_msg

        # Если ничего не обнаружено
        log_msg = "Признаков искусственной модификации голоса не обнаружено"
        return False, details, log_msg

    except Exception as e:
        log.warning(f"Ошибка при анализе модификаций голоса в {wav_path}: {e}")
        return False, {}, f"Не удалось проанализировать модификации голоса: {str(e)}"

    def cut_speech_segments(wav_path: str, min_segment_duration: float = 3.0,
                            max_segments: int = 5) -> Tuple[List[str], str]:
        """
        Нарезает аудиофайл на отдельные сегменты с речью для последующего анализа.

        Args:
            wav_path: Путь к WAV файлу
            min_segment_duration: Минимальная длительность сегмента (в секундах)
            max_segments: Максимальное количество сегментов

        Returns:
            (segment_paths, log_msg): Список путей к сегментам и лог-сообщение
        """
        try:
            # Загрузка аудио
            y, sr = librosa.load(wav_path, sr=None)

            # Применение VAD для обнаружения речи
            # Вычисление энергии
            frame_length = int(0.025 * sr)  # 25 мс
            hop_length = int(0.010 * sr)  # 10 мс

            # Энергия
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            energy_threshold = 0.05 * np.max(energy)

            # Детектирование речи
            speech_frames = (energy > energy_threshold)

            # Сглаживание для устранения дрожания
            from scipy.ndimage import binary_opening, binary_closing
            speech_frames = binary_closing(binary_opening(speech_frames, np.ones(5)), np.ones(10))

            # Преобразование фреймов в отметки времени
            timestamps = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sr, hop_length=hop_length)

            # Обнаружение сегментов с речью
            speech_segments = []
            is_speech = False
            start_time = 0

            for i, speech in enumerate(speech_frames):
                if speech and not is_speech:
                    # Начало речевого сегмента
                    is_speech = True
                    start_time = timestamps[i]
                elif not speech and is_speech:
                    # Конец речевого сегмента
                    is_speech = False
                    # Если сегмент достаточно длинный
                    if timestamps[i] - start_time >= min_segment_duration:
                        speech_segments.append((start_time, timestamps[i]))

            # Добавляем последний сегмент, если он есть
            if is_speech and timestamps[-1] - start_time >= min_segment_duration:
                speech_segments.append((start_time, timestamps[-1]))

            # Если нет речевых сегментов
            if not speech_segments:
                # Разбиваем файл на равные части
                total_duration = len(y) / sr
                if total_duration >= min_segment_duration:
                    segment_duration = min(SEGMENT_DURATION, total_duration)
                    num_segments = min(max_segments, int(total_duration / segment_duration))

                    for i in range(num_segments):
                        start_time = i * segment_duration
                        end_time = min((i + 1) * segment_duration, total_duration)
                        speech_segments.append((start_time, end_time))
                else:
                    return [wav_path], "Файл слишком короткий для сегментации"

            # Ограничение количества сегментов
            if len(speech_segments) > max_segments:
                # Сортировка по длительности
                speech_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
                speech_segments = speech_segments[:max_segments]
                # Сортировка по времени начала
                speech_segments.sort(key=lambda x: x[0])

            # Сохранение сегментов
            segment_paths = []
            segment_descriptions = []

            for i, (start, end) in enumerate(speech_segments):
                start_sample = int(start * sr)
                end_sample = int(end * sr)

                # Извлечение сегмента
                segment = y[start_sample:end_sample]

                # Добавляем фейдинг
                fade_samples = min(int(0.05 * sr), len(segment) // 10)
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)

                if len(segment) > 2 * fade_samples:
                    segment[:fade_samples] *= fade_in
                    segment[-fade_samples:] *= fade_out

                # Сохранение сегмента
                temp_fd, segment_path = tempfile.mkstemp(suffix=f"_segment_{i + 1}.wav")
                os.close(temp_fd)

                sf.write(segment_path, segment, sr)
                segment_paths.append(segment_path)

                # Добавление описания
                segment_descriptions.append(
                    f"Сегмент {i + 1}: {start:.1f}-{end:.1f} сек (длительность: {end - start:.1f} сек)"
                )

            # Формирование отчета
            log_msg = (
                    f"Из файла {os.path.basename(wav_path)} выделено {len(segment_paths)} сегментов с речью:\n" +
                    "\n".join(segment_descriptions)
            )

            return segment_paths, log_msg

        except Exception as e:
            log.warning(f"Ошибка при выделении речевых сегментов из {wav_path}: {e}")
            return [wav_path], f"Не удалось выделить речевые сегменты: {str(e)}"

    def preprocess_for_forensic_analysis(file_path: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Выполняет полную предобработку аудиофайла для судебного фоноскопического анализа.

        Args:
            file_path: Путь к исходному аудиофайлу

        Returns:
            (processed_path, metadata, log_msg): Путь к обработанному файлу, метаданные и лог-сообщение
        """
        log_messages = []
        metadata = {}

        try:
            # 1. Проверка формата и конвертация в WAV
            log.info(f"Начало обработки файла: {file_path}")
            wav_path, log_msg = convert_audio_to_wav(file_path)
            log_messages.append(log_msg)

            # 2. Проверка на модификации голоса
            is_modified, modification_details, mod_log = detect_voice_modifications(wav_path)
            metadata["voice_modified"] = is_modified
            metadata["modification_details"] = modification_details
            log_messages.append(mod_log)

            if is_modified:
                log.warning(f"Обнаружены признаки модификации голоса в файле {file_path}")

            # 3. Удаление тишины
            nosilence_path, log_msg = remove_silence(wav_path)
            log_messages.append(log_msg)

            # 4. Улучшение качества речи
            enhanced_path, log_msg = enhance_speech(nosilence_path)
            log_messages.append(log_msg)

            # 5. Извлечение оптимального сегмента
            segment_path, log_msg = extract_voice_segment(enhanced_path)
            log_messages.append(log_msg)

            # 6. Сбор метаданных о файле
            audio_info = get_audio_info(segment_path)
            metadata["original_file"] = os.path.basename(file_path)
            metadata["processed_file"] = os.path.basename(segment_path)
            metadata["audio_info"] = audio_info
            metadata["preprocessing_steps"] = log_messages

            # Формирование итогового отчета
            full_log = "\n".join(log_messages)
            log.info(f"Завершена обработка файла: {file_path}")

            return segment_path, metadata, full_log

        except Exception as e:
            error_msg = f"Критическая ошибка при обработке файла {file_path}: {e}"
            log.error(error_msg)
            return file_path, {"error": str(e)}, error_msg

    def prepare_segments_for_comparison(wav_path: str, segment_count: int = 5,
                                        segment_duration: float = 30.0) -> Tuple[List[np.ndarray], str]:
        """
        Подготавливает оптимальные сегменты для сравнения голосов.

        Args:
            wav_path: Путь к WAV файлу
            segment_count: Количество сегментов
            segment_duration: Длительность сегмента в секундах

        Returns:
            (segments, log_msg): Список сегментов в виде numpy arrays и лог-сообщение
        """
        try:
            # Загрузка аудио
            y, sr = librosa.load(wav_path, sr=None)

            # Предобработка для анализа
            from voice_match.services.comparison import preprocess
            y_processed = preprocess(y)

            # Определение количества фреймов для сегмента
            segment_frames = int(segment_duration * sr)

            # Удаление тишины для лучшего использования аудио
            # Вычисление энергии
            energy = librosa.feature.rms(y=y_processed)[0]
            energy_threshold = 0.05 * np.max(energy)

            # Получение индексов фреймов с речью
            hop_length = 512  # Стандартное значение для RMS
            speech_frames = np.where(energy > energy_threshold)[0]

            if len(speech_frames) == 0:
                # Если речи не обнаружено, используем весь файл
                segments = [y_processed]
                log_msg = "Не обнаружено достаточно речи, используется весь файл"
                return segments, log_msg

            # Преобразование индексов фреймов в сэмплы
            speech_samples = librosa.frames_to_samples(speech_frames, hop_length=hop_length)

            # Определение речевых сегментов
            segments = []

            # Если файл слишком короткий
            if len(y_processed) < segment_frames:
                segments = [y_processed]
                log_msg = f"Файл короче требуемой длительности сегмента ({segment_duration} сек)"
                return segments, log_msg

            # Формирование сегментов с максимальной речью
            # Расчет средней энергии в потенциальных сегментах
            potential_segments = []

            window_hop = max(1, (len(y_processed) - segment_frames) // (segment_count * 2))

            for start in range(0, len(y_processed) - segment_frames, window_hop):
                end = start + segment_frames
                segment = y_processed[start:end]

                # Проверка наличия речи
                segment_energy = librosa.feature.rms(y=segment)[0]
                speech_ratio = np.sum(segment_energy > energy_threshold) / len(segment_energy)

                # Проверка вокализации (наличие основного тона)
                pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
                voiced_frames = np.sum(np.max(magnitudes, axis=0) > 0) / magnitudes.shape[1]

                # Качество сегмента
                quality_score = 0.6 * speech_ratio + 0.4 * voiced_frames

                potential_segments.append((start, end, quality_score))

            # Сортировка сегментов по оценке качества
            potential_segments.sort(key=lambda x: x[2], reverse=True)

            # Выбор лучших сегментов с минимальным перекрытием
            selected_segments = []
            overlap_threshold = segment_frames // 4  # Допустимое перекрытие 25%

            for start, end, score in potential_segments:
                # Проверка на перекрытие с уже выбранными сегментами
                is_overlapping = False

                for sel_start, sel_end, _ in selected_segments:
                    # Проверка на перекрытие
                    if (start < sel_end and end > sel_start and
                            min(end, sel_end) - max(start, sel_start) > overlap_threshold):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    selected_segments.append((start, end, score))

                    if len(selected_segments) >= segment_count:
                        break

            # Если не хватает сегментов, добавляем с перекрытием
            if len(selected_segments) < segment_count and len(potential_segments) > len(selected_segments):
                remaining = segment_count - len(selected_segments)

                for start, end, score in potential_segments:
                    if (start, end, score) not in selected_segments:
                        selected_segments.append((start, end, score))

                        if len(selected_segments) >= segment_count:
                            break

            # Сортировка сегментов по времени начала
            selected_segments.sort(key=lambda x: x[0])

            # Извлечение сегментов
            segments = [y_processed[start:end] for start, end, _ in selected_segments]

            # Лог-сообщение
            segments_info = [
                f"Сегмент {i + 1}: {start / sr:.1f}-{end / sr:.1f} сек (качество: {score:.2f})"
                for i, (start, end, score) in enumerate(selected_segments)
            ]

            log_msg = (
                    f"Подготовлено {len(segments)} сегментов для сравнения:\n" +
                    "\n".join(segments_info)
            )

            return segments, log_msg

        except Exception as e:
            log.warning(f"Ошибка при подготовке сегментов из {wav_path}: {e}")
            # В случае ошибки возвращаем весь файл как один сегмент
            y, sr = librosa.load(wav_path, sr=None)
            return [y], f"Ошибка при подготовке сегментов: {str(e)}"

    def plot_spectral_features(wav_path: str) -> Tuple[str, str]:
        """
        Создает расширенную визуализацию спектральных характеристик голоса.

        Args:
            wav_path: Путь к WAV файлу

        Returns:
            (plot_path, log_msg): Путь к изображению и лог-сообщение
        """
        try:
            # Загрузка аудио
            y, sr = librosa.load(wav_path, sr=None)

            # Нормализация
            y = librosa.util.normalize(y)

            # Предобработка
            from voice_match.services.comparison import preprocess
            y_processed = preprocess(y)

            # Создание визуализации
            plt.figure(figsize=(14, 12))

            # 1. Спектрограмма
            plt.subplot(3, 2, 1)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_processed)), ref=np.max)
            librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Спектрограмма')

            # 2. Мел-спектрограмма (подчеркивает речевые особенности)
            plt.subplot(3, 2, 2)
            S = librosa.feature.melspectrogram(y=y_processed, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Мел-спектрограмма')

            # 3. MFCC (ключевые коэффициенты)
            plt.subplot(3, 2, 3)
            mfcc = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=20)
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title('MFCC коэффициенты')

            # 4. Основной тон (F0)
            plt.subplot(3, 2, 4)
            pitches, magnitudes = librosa.piptrack(y=y_processed, sr=sr, fmin=60, fmax=400)

            # Выделение питча с максимальной магнитудой для каждого фрейма
            pitch_frames = []
            for t in range(magnitudes.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # ноль для невокализованных фреймов
                    pitch_frames.append((t, pitch))

            if pitch_frames:
                frames, pitches_val = zip(*pitch_frames)
                plt.scatter(
                    librosa.frames_to_time(frames, sr=sr),
                    pitches_val,
                    alpha=0.5,
                    s=5,
                    color='blue'
                )
                plt.plot(
                    librosa.frames_to_time(frames, sr=sr),
                    pitches_val,
                    color='blue',
                    alpha=0.25
                )
            plt.title('Основной тон (F0)')
            plt.xlabel('Время (сек)')
            plt.ylabel('Частота (Hz)')
            plt.grid(True, alpha=0.3)

            # 5. Энергия сигнала
            plt.subplot(3, 2, 5)
            rms = librosa.feature.rms(y=y_processed)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=512)
            plt.plot(times, rms, color='green')
            plt.title('Энергия сигнала (RMS)')
            plt.xlabel('Время (сек)')
            plt.ylabel('Амплитуда')
            plt.grid(True, alpha=0.3)

            # 6. Спектральный центроид
            plt.subplot(3, 2, 6)
            cent = librosa.feature.spectral_centroid(y=y_processed, sr=sr)[0]
            times = librosa.times_like(cent, sr=sr, hop_length=512)
            plt.plot(times, cent, color='orange')
            plt.title('Спектральный центроид')
            plt.xlabel('Время (сек)')
            plt.ylabel('Частота (Hz)')
            plt.grid(True, alpha=0.3)

            # Сохранение графика
            plt.tight_layout()
            temp_fd, plot_path = tempfile.mkstemp(suffix="_spectral.png")
            os.close(temp_fd)
            plt.savefig(plot_path, dpi=150)
            plt.close()

            log_msg = f"Создана визуализация спектральных характеристик для {os.path.basename(wav_path)}"
            return plot_path, log_msg

        except Exception as e:
            log.warning(f"Ошибка при создании визуализации для {wav_path}: {e}")
            return "", f"Не удалось создать визуализацию: {str(e)}"
