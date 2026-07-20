"""Безопасная проверка и конвертация входных аудиофайлов."""

import shutil
import subprocess  # noqa: S404
import tempfile
from pathlib import Path
from typing import TypedDict

import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from voice_match.constants import (
    SAMPLE_RATE,
    SUPPORTED_EXTENSIONS,
    TARGET_CHANNELS,
)
from voice_match.log import setup_logger

log = setup_logger('preprocessing')

_FFMPEG_TIMEOUT_SECONDS = 120


class AudioInfo(TypedDict):
    """Технические параметры аудиофайла."""

    sample_rate: int
    channels: int
    frames: int
    duration: float
    format: str
    subtype: str


def convert_audio_to_wav(
    file_path: str,
    *,
    force_resample: bool = False,
) -> tuple[str, str]:
    """Преобразовать аудио в mono PCM WAV 16 кГц.

    Возвращаемый временный файл должен удалить вызывающий код.

    Raises:
        FileNotFoundError: Если исходный файл отсутствует.
        ValueError: Если формат не поддерживается.
        RuntimeError: Если декодирование или конвертация не удались.
    """
    source = Path(file_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Аудиофайл не найден: {source}')
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f'Формат {source.suffix!r} не поддерживается. '
            f'Допустимы: {supported}.'
        )

    info = get_audio_info(source)
    needs_conversion = (
        force_resample
        or source.suffix.lower() != '.wav'
        or info['sample_rate'] != SAMPLE_RATE
        or info['channels'] != TARGET_CHANNELS
        or info['subtype'] not in {'PCM_16', 'PCM_24', 'FLOAT'}
    )
    if not needs_conversion:
        return str(source), (
            f'Файл уже соответствует формату: {SAMPLE_RATE} Гц, mono.'
        )

    destination = _new_temporary_wav()
    try:
        _convert_with_pydub(source, destination)
    except (CouldntDecodeError, OSError, RuntimeError) as exc:
        log.warning(
            'Pydub не декодировал %s: %s. Используется ffmpeg.',
            source.name,
            exc,
        )
        _convert_with_ffmpeg(source, destination)

    converted = get_audio_info(destination)
    if (
        converted['sample_rate'] != SAMPLE_RATE
        or converted['channels'] != TARGET_CHANNELS
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            'Сконвертированный файл не соответствует ожидаемому '
            'формату.'
        )

    return str(destination), (
        f'Файл преобразован в WAV: {SAMPLE_RATE} Гц, mono, '
        f'{converted["duration"]:.1f} с.'
    )


def get_audio_info(file_path: str | Path) -> AudioInfo:
    """Прочитать параметры аудио через libsndfile."""
    path = Path(file_path)
    try:
        with sf.SoundFile(path) as audio:
            return AudioInfo(
                sample_rate=int(audio.samplerate),
                channels=int(audio.channels),
                frames=int(audio.frames),
                duration=float(audio.frames / audio.samplerate),
                format=str(audio.format),
                subtype=str(audio.subtype),
            )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f'Не удалось прочитать параметры аудиофайла {path.name}.'
        ) from exc


def _new_temporary_wav() -> Path:
    file_handle = tempfile.NamedTemporaryFile(
        suffix='.wav',
        prefix='voice-match-',
        delete=False,
    )
    file_handle.close()
    return Path(file_handle.name)


def _convert_with_pydub(source: Path, destination: Path) -> None:
    audio = AudioSegment.from_file(source)
    audio = audio.set_channels(TARGET_CHANNELS)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio.export(destination, format='wav', codec='pcm_s16le')


def _convert_with_ffmpeg(source: Path, destination: Path) -> None:
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        destination.unlink(missing_ok=True)
        raise RuntimeError('Исполняемый файл ffmpeg не найден.')

    command = [
        ffmpeg_path,
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        str(source),
        '-vn',
        '-ac',
        str(TARGET_CHANNELS),
        '-ar',
        str(SAMPLE_RATE),
        '-c:a',
        'pcm_s16le',
        str(destination),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg превысил лимит времени.') from exc

    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        error = result.stderr.strip() or 'неизвестная ошибка ffmpeg'
        raise RuntimeError(f'ffmpeg не смог преобразовать файл: {error}')


__all__ = ['AudioInfo', 'convert_audio_to_wav', 'get_audio_info']
