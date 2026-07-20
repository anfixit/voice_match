"""ECAPA-TDNN encoder для speaker verification."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from voice_match.config import settings
from voice_match.exceptions import ModelUnavailableError
from voice_match.log import setup_logger

log = setup_logger('ecapa')

_MODEL_SOURCE = 'speechbrain/spkrec-ecapa-voxceleb'


class EcapaEncoder:
    """Извлекает нормализованные speaker embeddings.

    Модель возвращает эмбеддинги, а не вероятность совпадения.
    Калибровка и решение same/different должны выполняться
    отдельным, обученным на целевом домене слоем.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        use_gpu: bool,
    ) -> None:
        self._model_dir = model_dir
        self._device = self._resolve_device(use_gpu)
        self._model: EncoderClassifier | None = None

    def load(self) -> None:
        """Загрузить предобученную модель SpeechBrain."""
        if self._model is not None:
            return

        try:
            self._model = EncoderClassifier.from_hparams(
                source=_MODEL_SOURCE,
                savedir=str(self._model_dir),
                run_opts={'device': self._device},
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelUnavailableError(
                'Не удалось загрузить ECAPA-TDNN. Для первого запуска '
                'нужен доступ к Hugging Face либо заранее загруженные '
                f'веса в {self._model_dir}.'
            ) from exc

        log.info(
            'ECAPA-TDNN загружена на устройство %s',
            self._device,
        )

    def encode_segments(
        self,
        segments: list[np.ndarray],
    ) -> np.ndarray:
        """Извлечь один L2-нормализованный эмбеддинг на сегмент."""
        if not segments:
            raise ValueError('Для кодирования нужен хотя бы один сегмент.')

        self.load()
        model = self._model
        if model is None:
            raise ModelUnavailableError(
                'ECAPA-TDNN не инициализирована.'
            )

        embeddings: list[np.ndarray] = []
        for segment in segments:
            waveform = torch.from_numpy(
                np.asarray(segment, dtype=np.float32),
            ).unsqueeze(0)
            waveform = waveform.to(self._device)

            with torch.inference_mode():
                embedding = model.encode_batch(waveform)
                embedding = embedding.reshape(-1)
                embedding = torch.nn.functional.normalize(
                    embedding,
                    p=2,
                    dim=0,
                )

            embeddings.append(
                embedding.detach().cpu().numpy().astype(np.float64),
            )

        return np.vstack(embeddings)

    @staticmethod
    def _resolve_device(use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'


@lru_cache(maxsize=1)
def get_ecapa() -> EcapaEncoder:
    """Вернуть общий экземпляр ECAPA-кодировщика."""
    model_dir = settings.models_dir / 'speechbrain-ecapa'
    model_dir.mkdir(parents=True, exist_ok=True)
    return EcapaEncoder(
        model_dir=model_dir,
        use_gpu=settings.use_gpu,
    )


__all__ = ['EcapaEncoder', 'get_ecapa']
