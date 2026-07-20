"""Fail-closed интерфейс anti-spoofing модели."""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
import torch

from voice_match.config import settings
from voice_match.constants import SAMPLE_RATE
from voice_match.exceptions import ModelUnavailableError
from voice_match.log import setup_logger

log = setup_logger('antispoofing')


class AntiSpoofingResult(TypedDict):
    """Некалиброванный score валидированной модели."""

    spoof_score: float
    segment_count: int
    model_name: str


class AntiSpoofingDetector:
    """Загружает только явно предоставленную обученную модель.

    Случайно инициализированная сеть никогда не используется.
    Текущий репозиторий не поставляет валидированные веса, поэтому
    функция по умолчанию недоступна и завершается безопасно.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or (
            settings.models_dir / 'antispoofing' / 'model.pt'
        )
        self._model: torch.jit.ScriptModule | None = None

    def load(self) -> None:
        """Загрузить TorchScript-модель anti-spoofing."""
        if self._model is not None:
            return
        if not self._model_path.is_file():
            raise ModelUnavailableError(
                'Anti-spoofing отключён: валидированные веса не '
                f'найдены в {self._model_path}.'
            )

        try:
            self._model = torch.jit.load(  # type: ignore[no-untyped-call]
                str(self._model_path),
                map_location='cpu',
            )
            self._model.eval()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelUnavailableError(
                'Не удалось загрузить валидированную anti-spoofing '
                'модель.'
            ) from exc

        log.info('Anti-spoofing модель загружена: %s', self._model_path)

    def detect(
        self,
        signal: np.ndarray,
        sample_rate: int,
    ) -> AntiSpoofingResult:
        """Получить сырой model score без вероятностных заявлений."""
        self.load()
        model = self._model
        if model is None:
            raise ModelUnavailableError(
                'Anti-spoofing модель не инициализирована.'
            )

        audio = np.asarray(signal, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
            )

        segment_samples = 4 * SAMPLE_RATE
        if audio.size < segment_samples:
            audio = np.pad(audio, (0, segment_samples - audio.size))

        scores: list[float] = []
        for start in range(
            0,
            audio.size - segment_samples + 1,
            segment_samples,
        ):
            segment = torch.from_numpy(
                audio[start:start + segment_samples],
            ).unsqueeze(0)
            with torch.inference_mode():
                output = model(segment)
            score = float(torch.sigmoid(output.reshape(-1)[0]).item())
            scores.append(score)

        return AntiSpoofingResult(
            spoof_score=float(np.mean(scores)),
            segment_count=len(scores),
            model_name=self._model_path.stem,
        )


@lru_cache(maxsize=1)
def get_antispoofing_detector() -> AntiSpoofingDetector:
    """Вернуть общий fail-closed anti-spoofing detector."""
    return AntiSpoofingDetector()


__all__ = [
    'AntiSpoofingDetector',
    'AntiSpoofingResult',
    'get_antispoofing_detector',
]
