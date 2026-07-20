"""Тесты fail-closed anti-spoofing."""

from pathlib import Path

import numpy as np
import pytest
import torch

from voice_match.constants import SAMPLE_RATE
from voice_match.detection.antispoofing import AntiSpoofingDetector
from voice_match.exceptions import ModelUnavailableError


class FakeAntiSpoofingModel:
    """Детерминированная TorchScript-подобная модель."""

    def __call__(self, segment: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0]], dtype=segment.dtype)


def test_antispoofing_refuses_missing_weights(tmp_path: Path) -> None:
    detector = AntiSpoofingDetector(tmp_path / 'missing.pt')

    with pytest.raises(ModelUnavailableError, match='валидированные веса'):
        detector.load()


def test_antispoofing_returns_raw_score_without_probability(
    tmp_path: Path,
) -> None:
    detector = AntiSpoofingDetector(tmp_path / 'model.pt')
    detector._model = FakeAntiSpoofingModel()  # type: ignore[assignment]
    signal = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)

    result = detector.detect(signal, SAMPLE_RATE)

    assert result['spoof_score'] == pytest.approx(0.5)
    assert result['segment_count'] == 1
    assert result['model_name'] == 'model'
    assert 'probability' not in result
