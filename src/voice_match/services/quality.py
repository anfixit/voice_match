"""Контроль качества и извлечение речевых сегментов."""

from dataclasses import dataclass

import numpy as np
import webrtcvad

from voice_match.constants import SAMPLE_RATE, VAD_FRAME_MS

_PCM_SCALE = 32_767
_DEFAULT_AGGRESSIVENESS = 2


@dataclass(frozen=True, slots=True)
class AudioQuality:
    """Измеримые характеристики пригодности аудио."""

    duration_seconds: float
    speech_seconds: float
    speech_ratio: float
    clipping_ratio: float
    rms: float
    issues: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        """Вернуть True, если критические проблемы не обнаружены."""
        return not self.issues


def analyze_audio_quality(
    signal: np.ndarray,
    sample_rate: int,
    *,
    min_duration_seconds: float,
    min_speech_seconds: float,
    min_speech_ratio: float,
    max_clipping_ratio: float,
) -> AudioQuality:
    """Оценить пригодность записи для speaker verification."""
    audio = _validate_signal(signal, sample_rate)
    duration = audio.size / sample_rate
    voiced_mask = _voiced_frame_mask(audio, sample_rate)
    frame_seconds = VAD_FRAME_MS / 1_000
    speech_seconds = float(np.count_nonzero(voiced_mask) * frame_seconds)
    speech_ratio = speech_seconds / duration if duration else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    issues: list[str] = []
    if duration < min_duration_seconds:
        issues.append(
            f'Длительность {duration:.1f} с меньше минимальных '
            f'{min_duration_seconds:.1f} с.'
        )
    if speech_seconds < min_speech_seconds:
        issues.append(
            f'Обнаружено только {speech_seconds:.1f} с речи; '
            f'требуется не менее {min_speech_seconds:.1f} с.'
        )
    if speech_ratio < min_speech_ratio:
        issues.append(
            f'Доля речи {speech_ratio:.0%} ниже минимальных '
            f'{min_speech_ratio:.0%}.'
        )
    if clipping_ratio > max_clipping_ratio:
        issues.append(
            f'Клиппинг затрагивает {clipping_ratio:.1%} отсчётов; '
            f'допустимо не более {max_clipping_ratio:.1%}.'
        )
    if rms < 1e-4:
        issues.append('Сигнал практически не содержит полезной энергии.')

    return AudioQuality(
        duration_seconds=duration,
        speech_seconds=speech_seconds,
        speech_ratio=speech_ratio,
        clipping_ratio=clipping_ratio,
        rms=rms,
        issues=tuple(issues),
    )


def extract_speech_segments(
    signal: np.ndarray,
    sample_rate: int,
    *,
    segment_seconds: float,
    max_segments: int,
    min_segment_seconds: float,
) -> list[np.ndarray]:
    """Извлечь сегменты, состоящие только из VAD-речи."""
    audio = _validate_signal(signal, sample_rate)
    frame_samples = sample_rate * VAD_FRAME_MS // 1_000
    voiced_mask = _voiced_frame_mask(audio, sample_rate)
    voiced_frames: list[np.ndarray] = []

    for index, is_voiced in enumerate(voiced_mask):
        if not is_voiced:
            continue
        start = index * frame_samples
        voiced_frames.append(audio[start:start + frame_samples])

    if not voiced_frames:
        return []

    speech = np.concatenate(voiced_frames)
    target_samples = max(1, int(segment_seconds * sample_rate))
    minimum_samples = max(1, int(min_segment_seconds * sample_rate))
    hop_samples = max(1, target_samples // 2)
    segments: list[np.ndarray] = []

    for start in range(0, speech.size, hop_samples):
        segment = speech[start:start + target_samples]
        if segment.size < minimum_samples:
            break
        segments.append(segment.astype(np.float32, copy=False))
        if len(segments) >= max_segments:
            break

    return segments


def _validate_signal(
    signal: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f'VAD ожидает {SAMPLE_RATE} Гц, получено {sample_rate} Гц.'
        )

    audio = np.asarray(signal, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError('Аудиосигнал пуст.')
    if not np.all(np.isfinite(audio)):
        raise ValueError('Аудиосигнал содержит NaN или бесконечность.')

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio


def _voiced_frame_mask(
    signal: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    vad = webrtcvad.Vad(_DEFAULT_AGGRESSIVENESS)
    frame_samples = sample_rate * VAD_FRAME_MS // 1_000
    frame_count = signal.size // frame_samples
    mask = np.zeros(frame_count, dtype=bool)

    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * _PCM_SCALE).astype(np.int16)
    for index in range(frame_count):
        start = index * frame_samples
        frame = pcm[start:start + frame_samples]
        mask[index] = vad.is_speech(frame.tobytes(), sample_rate)

    return mask
