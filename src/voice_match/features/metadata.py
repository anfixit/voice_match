import datetime
import json
import os
import subprocess

from typing import Any

from voice_match.log import setup_logger

log = setup_logger("metadata")


def extract_file_metadata(file_path: str) -> dict[str, Any]:
    """
    Извлекает метаданные из аудиофайла для аналитических целей.

    Args:
        file_path: Путь к аудиофайлу

    Returns:
        Словарь с метаданными
    """
    if not os.path.exists(file_path):
        log.error(f"Файл не найден: {file_path}")
        return {}

    # Базовые метаданные из файловой системы
    file_stat = os.stat(file_path)

    metadata = {
        "filename": os.path.basename(file_path),
        "format": os.path.splitext(file_path)[1].lower()[1:],
        "size_bytes": file_stat.st_size,
        "size_mb": file_stat.st_size / (1024 * 1024),
        "creation_time": datetime.datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modification_time": datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "audio_info": {},
        "codec_info": {},
        "creation_tool": "Unknown",
        "modification_history": []
    }

    # Извлечение детальных аудио метаданных с помощью ffprobe
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
            probe_data = json.loads(result.stdout)

            # Аудио информация
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    metadata["audio_info"] = {
                        "codec": stream.get("codec_name", "Unknown"),
                        "sample_rate": stream.get("sample_rate", "Unknown"),
                        "channels": stream.get("channels", "Unknown"),
                        "bit_rate": stream.get("bit_rate", "Unknown"),
                        "duration": stream.get("duration", "Unknown")
                    }

                    metadata["codec_info"] = {
                        "codec_tag": stream.get("codec_tag", "Unknown"),
                        "codec_tag_string": stream.get("codec_tag_string", "Unknown"),
                        "codec_time_base": stream.get("codec_time_base", "Unknown"),
                        "codec_long_name": stream.get("codec_long_name", "Unknown")
                    }

                    # Проверка на потенциальные проблемы с кодеком
                    if "pcm" in stream.get("codec_name", "").lower():
                        metadata["codec_notes"] = "Несжатый PCM формат - низкая вероятность артефактов сжатия"
                    elif "mp3" in stream.get("codec_name", "").lower():
                        metadata["codec_notes"] = "MP3 формат - возможны артефакты сжатия с потерями"

                    break

            # Информация о формате и происхождении
            format_info = probe_data.get("format", {})

            if "tags" in format_info:
                tags = format_info["tags"]

                # Извлечение информации о создании/модификации
                if "creation_time" in tags:
                    metadata["embedded_creation_time"] = tags["creation_time"]

                # Программа, создавшая файл
                for tag in ["encoder", "vendor", "product", "artist", "comment"]:
                    if tag in tags:
                        metadata["creation_tool"] = tags[tag]
                        break
    except Exception as e:
        log.warning(f"Ошибка при извлечении метаданных: {e}")

    # Проверка целостности файла
    metadata["integrity_check"] = check_file_integrity(file_path)

    return metadata


def check_file_integrity(file_path: str) -> dict[str, Any]:
    """
    Проверяет целостность аудиофайла на наличие повреждений.

    Args:
        file_path: Путь к аудиофайлу

    Returns:
        Словарь с результатами проверки
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    try:
        # Проверка с помощью ffmpeg с опцией проверки ошибок
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", file_path,
            "-f", "null",
            "-"
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0 or proc.stderr:
            result["valid"] = False
            result["errors"].append(proc.stderr.strip())
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Ошибка проверки: {e!s}")

    return result


def detect_recording_environment(audio_data: dict[str, Any]) -> dict[str, Any]:
    """
    Анализирует параметры записи для определения условий записи.

    Args:
        audio_data: Словарь с характеристиками аудио

    Returns:
        Оценка условий записи
    """
    environment = {
        "likely_device_type": "Unknown",
        "noise_level": "Unknown",
        "recording_quality": "Unknown",
        "likelihood_of_tampering": "Low",
        "notes": []
    }

    # Оценка типа устройства по параметрам
    if "sample_rate" in audio_data and "channels" in audio_data and "bit_rate" in audio_data:
        sample_rate = int(audio_data["sample_rate"]) if audio_data["sample_rate"] != "Unknown" else 0
        channels = int(audio_data["channels"]) if audio_data["channels"] != "Unknown" else 0
        bit_rate = int(audio_data["bit_rate"]) if audio_data["bit_rate"] != "Unknown" else 0

        # Профессиональная запись
        if sample_rate >= 44100 and bit_rate >= 320000:
            environment["likely_device_type"] = "Professional recording equipment"
            environment["recording_quality"] = "High"
        # Мобильный телефон
        elif 8000 <= sample_rate <= 16000 and channels == 1:
            environment["likely_device_type"] = "Mobile phone or voice recorder"
            environment["recording_quality"] = "Medium"
        # Телефонный звонок
        elif sample_rate <= 8000:
            environment["likely_device_type"] = "Telephone call"
            environment["recording_quality"] = "Low"
        # Веб-сервис или мессенджер
        elif 16000 <= sample_rate <= 24000:
            environment["likely_device_type"] = "VoIP service or messenger"
            environment["recording_quality"] = "Medium"

    return environment


def compare_file_metadata(metadata1: dict[str, Any], metadata2: dict[str, Any]) -> dict[str, Any]:
    """
    Сравнивает метаданные двух файлов для выявления несоответствий.

    Args:
        metadata1: Метаданные первого файла
        metadata2: Метаданные второго файла

    Returns:
        Результаты сравнения
    """
    comparison = {
        "same_format": metadata1.get("format") == metadata2.get("format"),
        "same_codec": metadata1.get("audio_info", {}).get("codec") == metadata2.get("audio_info", {}).get("codec"),
        "same_sample_rate": metadata1.get("audio_info", {}).get("sample_rate") == metadata2.get("audio_info", {}).get(
            "sample_rate"),
        "same_channels": metadata1.get("audio_info", {}).get("channels") == metadata2.get("audio_info", {}).get(
            "channels"),
        "likely_same_device": False,
        "warnings": []
    }

    # Оценка вероятности записи на одном устройстве
    env1 = metadata1.get("environment", {}).get("likely_device_type")
    env2 = metadata2.get("environment", {}).get("likely_device_type")

    if env1 == env2 and env1 != "Unknown":
        comparison["likely_same_device"] = True

    # Проверка на подозрительные несоответствия
    if not comparison["same_sample_rate"]:
        comparison["warnings"].append("Разная частота дискретизации может указывать на различные источники записи")

    if abs(float(metadata1.get("audio_info", {}).get("duration", 0)) - float(
            metadata2.get("audio_info", {}).get("duration", 0))) > 300:
        comparison["warnings"].append("Значительная разница в длительности записей")

    return comparison
