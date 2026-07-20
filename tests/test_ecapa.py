"""Тесты ECAPA adapter без сетевой загрузки модели."""

from pathlib import Path

import numpy as np
import pytest
import torch

from voice_match.models import ecapa


class FakeClassifier:
    """Минимальный совместимый SpeechBrain classifier."""

    def encode_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 2
        return torch.tensor([[[3.0, 4.0]]])


def test_ecapa_loads_and_normalizes_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ecapa.EncoderClassifier,
        'from_hparams',
        lambda **_: FakeClassifier(),
    )
    encoder = ecapa.EcapaEncoder(
        model_dir=tmp_path,
        use_gpu=False,
    )

    result = encoder.encode_segments(
        [np.ones(16000, dtype=np.float32)],
    )

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(0.6)
    assert result[0, 1] == pytest.approx(0.8)


def test_ecapa_rejects_empty_segments(tmp_path: Path) -> None:
    encoder = ecapa.EcapaEncoder(
        model_dir=tmp_path,
        use_gpu=False,
    )

    try:
        encoder.encode_segments([])
    except ValueError as exc:
        assert 'хотя бы один сегмент' in str(exc)
    else:
        raise AssertionError('Ожидался ValueError.')
